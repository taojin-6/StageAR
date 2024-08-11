import open3d as o3d
import numpy as np

def tf_blender_o3d(pose):
    rotation = np.radians(np.array([180, 0, 0]).reshape((3,1)))
    rot_m = o3d.geometry.get_rotation_matrix_from_xyz(rotation)

    tf = np.identity(4)
    tf[0:3, 0:3] = rot_m

    pose = pose @ tf

    return pose

def get_gt_pose(gt_filepath):
    pose = np.loadtxt(gt_filepath)
    pose = tf_blender_o3d(pose)
    pose = np.linalg.inv(pose)

    return pose


def create_camera_intrinsic(width, height, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    tensor_intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, dtype = o3d.core.Dtype.Float64)

    return intrinsic, tensor_intrinsic
