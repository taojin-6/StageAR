import open3d as o3d
import numpy as np

voxel_size = 0.1
depth_scale = 1.0
block_count = 10000
depth_max = 10000.0
depth_min = 0.3
trunc_voxel_multiplier = 4.0
trunc = voxel_size * trunc_voxel_multiplier

cpu_device = o3d.core.Device('CPU:0')
gpu_device = o3d.core.Device('CUDA:0')
use_gpu = True

# LiDAR config
LIDAR_HOR_RES = 2048
LIDAR_VER_RES = 128
LIDAR_FOV_UP = 45 / 180 * np.pi
LIDAR_FOV_DOWN = -45 / 180 * np.pi
LIDAR_FOV = LIDAR_FOV_UP + abs(LIDAR_FOV_DOWN)

num_integrate_frame = 1
