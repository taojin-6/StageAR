import numpy as np
import open3d as o3d
import settings

class Depth_Processing:
    def __init__(self, img_width, img_height):
        self.depth_img_list = []

        self.compute_device = None
        self.set_compute_device(settings.use_gpu)

        self.voxel_grid = self.create_voxel_block_grid()

        self.mesh = None
        self.raycast_scene = None

        self.img_width = img_width
        self.img_height = img_height

        self.intrinsic_t = None
        self.extrinsics = None

    def set_compute_device(self, use_gpu_compute=False):
        if use_gpu_compute:
            self.compute_device = settings.gpu_device
        else:
            self.compute_device = settings.cpu_device

    def set_intrinsic(self, tensor_intrinsic):
        self.intrinsic_t = tensor_intrinsic

    def set_extrinsics(self, extrinsics):
        self.extrinsics = extrinsics

    def create_voxel_block_grid(self):
        self.voxel_grid = o3d.t.geometry.VoxelBlockGrid(
                        attr_names=('tsdf', 'weight'),
                        attr_dtypes=(o3d.core.float32, o3d.core.float32),
                        attr_channels=((1), (1)),
                        voxel_size = settings.voxel_size,
                        block_resolution = 16,
                        block_count = settings.block_count,
                        device = self.compute_device)

    def extract_mesh(self):
        self.mesh = self.voxel_grid.extract_triangle_mesh(weight_threshold = 0).to(self.compute_device)

        self.raycast_scene = self.create_raycast_scene(decimation_level = 0.0)

    def get_mesh(self):
        return self.mesh

    def get_raycast_scene(self):
        return self.raycast_scene

    def set_mesh(self, mesh):
        self.mesh = mesh

    def create_raycast_scene(self, decimation_level = 0.0):
        if decimation_level != 0.0:
            self.mesh = self.mesh.simplify_quadric_decimation(decimation_level)

        self.raycast_scene = o3d.t.geometry.RaycastingScene()
        self.raycast_scene.add_triangles(self.mesh)

    def raycast(self, objPts_2d, camera_extrinsic):
        rays = self.raycast_scene.create_rays_pinhole(self.intrinsic_t, camera_extrinsic,
                                                        width_px=self.img_width,
                                                        height_px=self.img_height)

        rays = rays[objPts_2d[:,1], objPts_2d[:,0]]

        result = self.raycast_scene.cast_rays(rays, nthreads=0)
        hit_mask = result['t_hit'].isfinite()

        # points = rays[hit_mask][:,:3] + rays[hit_mask][:,3:]*result['t_hit'][hit_mask].reshape((-1,1))
        points = rays[:,:3] + rays[:,3:]*result['t_hit'].reshape((-1,1))

        return points.numpy().astype('float32'), hit_mask.numpy()
