import open3d as o3d
import numpy as np
from visualize_cloud import create_cloud, visualize_cloud, create_camera_intrinsic
import settings


def create_voxel_block_grid():
    voxel_grid = o3d.t.geometry.VoxelBlockGrid(
                    attr_names=('tsdf', 'weight'),
                    attr_dtypes=(o3d.core.float32, o3d.core.float32),
                    attr_channels=((1), (1)),
                    voxel_size = settings.voxel_size,
                    block_resolution = 16,
                    block_count = settings.block_count,
                    device = settings.gpu_device)
    return voxel_grid

def integrate(voxel_grid, tensor_intrinsic, gt_depth_list, depth_imgs):
    for depth_cam_num in range(settings.num_fixed_depth):
        frustum_block_coords = voxel_grid.compute_unique_block_coordinates(
                                    depth_imgs[depth_cam_num], tensor_intrinsic, gt_depth_list[depth_cam_num], 
                                    settings.depth_scale, settings.depth_max)

        voxel_grid.integrate(frustum_block_coords, depth_imgs[depth_cam_num], tensor_intrinsic,
                                gt_depth_list[depth_cam_num], settings.depth_scale, 
                                settings.depth_max, settings.trunc_voxel_multiplier)

    return voxel_grid

def create_mesh(voxel_grid, persistence_frame = 0, decimation_level = 0.0):
    mesh = voxel_grid.extract_triangle_mesh(weight_threshold = persistence_frame).to(settings.cpu_device)

    scene = create_raycast_scene_given_mesh(mesh, decimation_level = 0.0)

    return scene, mesh

def create_raycast_scene_given_mesh(mesh, decimation_level = 0.0):
    if decimation_level != 0.0:
        mesh = mesh.simplify_quadric_decimation(decimation_level)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    return scene

def raycast(scene, uv, intrinsic, extrinsic, width, height):
    rays = scene.create_rays_pinhole(intrinsic, extrinsic,
                                        width_px=width,
                                        height_px=height)

    rays = rays[uv[:,1], uv[:,0]]

    result = scene.cast_rays(rays, nthreads=0)
    hit_mask = result['t_hit'].isfinite()

    points = rays[hit_mask][:,:3] + rays[hit_mask][:,3:]*result['t_hit'][hit_mask].reshape((-1,1))

    return points.numpy().astype('float32'), hit_mask.numpy()
