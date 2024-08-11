import open3d as o3d
import numpy as np
import settings

class Lidar_Reconstruction:
    # Generate Yaw and Pitch value of the cylindrical projected points
    def yaw_pitch_generation(self, xyz):
        x = -xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        r = np.sqrt(x**2 + y**2 + z**2)
        yaw = np.arctan2(y, x)
        pitch = np.arcsin(z/r)

        yaw = np.nan_to_num(yaw, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)

        return yaw, pitch, r

    def project_uv(self, yaw, pitch):
        yaw[yaw < 0] += 2 * np.pi
        pitch += np.pi / 4
        
        u = np.floor(yaw / (2 * np.pi / settings.LIDAR_HOR_RES))
        v = np.floor(abs(settings.LIDAR_VER_RES - pitch / (np.pi / 2 / settings.LIDAR_VER_RES)))

        u = np.nan_to_num(u, nan=-1.0).astype(int)
        v = np.nan_to_num(v, nan=-1.0).astype(int)

        return u, v

    def voxel_filtering(self, voxel_u, voxel_v, voxel_r):
        mask_proj = (voxel_r > settings.depth_min) & (voxel_u >= 0) & (voxel_v >= 0) & \
                    (voxel_u < settings.LIDAR_HOR_RES) & (voxel_v < settings.LIDAR_VER_RES)
        voxel_u = voxel_u[mask_proj]
        voxel_v = voxel_v[mask_proj]
        voxel_r = voxel_r[mask_proj]

        return voxel_u, voxel_v, voxel_r, mask_proj

    def integrate(self, voxel_grid, point_cloud, range_img):
        range_img = range_img.astype(np.float64)
        range_img /= 1000.0
        frustum_block_coords = voxel_grid.compute_unique_block_coordinates(point_cloud, \
                                                                           settings.trunc_voxel_multiplier)

        voxel_grid.hashmap().activate(frustum_block_coords)

        buf_indices, masks = voxel_grid.hashmap().find(frustum_block_coords)

        voxel_coords, voxel_indices = voxel_grid.voxel_coordinates_and_flattened_indices(buf_indices)

        yaw, pitch, voxel_r = self.yaw_pitch_generation(voxel_coords.to(settings.cpu_device).numpy())
        voxel_u, voxel_v = self.project_uv(yaw, pitch)

        voxel_u, voxel_v, voxel_r, mask_proj = self.voxel_filtering(voxel_u, voxel_v, voxel_r)

        depth = o3d.core.Tensor(range_img[voxel_v, voxel_u]).to(o3d.core.Dtype.Float32)

        sdf = depth - voxel_r

        mask_inlier = (depth > 0) & (sdf >= -settings.trunc) & (depth <= settings.depth_max)

        sdf[sdf >= settings.trunc] = settings.trunc
        sdf = sdf / settings.trunc

        weight = voxel_grid.attribute('weight').reshape((-1, 1))
        tsdf = voxel_grid.attribute('tsdf').reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
        w = weight[valid_voxel_indices]
        wp = w + 1

        tsdf[valid_voxel_indices] = (tsdf[valid_voxel_indices] * w + sdf[mask_inlier].reshape(w.shape)) / wp
        weight[valid_voxel_indices] = wp

        return voxel_grid

    def lidar_reconstruction(self, point_cloud, range_img):
        voxel_grid = o3d.t.geometry.VoxelBlockGrid(
                        attr_names=('tsdf', 'weight'),
                        attr_dtypes=(o3d.core.float32, o3d.core.float32),
                        attr_channels=((1), (1)),
                        voxel_size = settings.voxel_size,
                        block_resolution = 8,
                        block_count = settings.block_count,
                        device = settings.cpu_device)

        voxel_grid = self.integrate(voxel_grid, point_cloud, range_img)

        mesh = voxel_grid.extract_triangle_mesh(weight_threshold = 0).to(settings.cpu_device)

        return mesh
