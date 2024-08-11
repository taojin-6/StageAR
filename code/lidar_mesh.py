#! /usr/bin/env python3

import numpy as np
import open3d as o3d
import time
import argparse
import sys
sys.path.insert(0, 'PATH')

from utils import io_util
from depth_processing_util import Depth_Processing

from superglue_feature_extractor import Superglue_Matcher
from feature_filtering_utils import Feature_Filter
from real_lidar_reconstruction import Lidar_Reconstruction

from socket_util import Socket_Util
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process mesh, reference image, and camera intrinsic and extrinsic paths.")
    parser.add_argument("--img_path", required=True, help="Path to the reference image")
    parser.add_argument("--pcd_path", required=True, help="Path to the lidar point clouds")
    parser.add_argument("--depth_path", required=True, help="Path to the depth images")
    parser.add_argument("--camera_intrinsic", required=True, help="Path to the camera intrinsic file")
    parser.add_argument("--camera_extrinsic", required=True, help="Path to the camera extrinsic file")
    parser.add_argument("--image_width", required=True, help="image_width")
    parser.add_argument("--image_height", required=True, help="image_width")
    return parser.parse_args()


def main():

    args = parse_arguments()
    img_path = args.img_path
    pcd_path = args.pcd_path
    depth_path = args.depth_path
    camera_intrinsic = args.camera_intrinsic
    camera_extrinsic = args.camera_extrinsic
    image_width = args.image_width
    image_height = args.image_height

    # initialize SuperGlue Matcher
    sg_matcher = Superglue_Matcher()
    feature_filter = Feature_Filter()

    lidar_cam_intrinsic = np.loadtxt(camera_intrinsic)
    lidar_cam_extrinsic = io_util.get_gt_pose(camera_extrinsic)

    # lidar_cam_intrinsic[0:2, 2] = lidar_cam_intrinsic[0:2, 2] / 2.0

    tensor_intrinsic = o3d.core.Tensor(lidar_cam_intrinsic, 
                                        dtype = o3d.core.Dtype.Float64)
    
    epoch_time = int(time.time())
    lidar_img_ref = img_path + '/' + str(epoch_time) + '.png'
    lidar_point_cloud_ref = o3d.t.io.read_point_cloud(pcd_path + '/' + str(epoch_time) + ".pcd")
    depth_img_ref = np.load(depth_path + '/' + str(epoch_time) + ".npy")
    
    socket = Socket_Util()
    socket.init_socket()

    while True:
    #figure out how to get est and ref frame (number)
        epoch_time += 1
        lidar_point_cloud_est = o3d.t.io.read_point_cloud(pcd_path + '/' + str(epoch_time) + ".pcd")
        depth_img_est = np.load(depth_path + '/' + str(epoch_time) + ".npy")
        depth_processor_ref = Depth_Processing(image_width,image_height)
        depth_processor_ref.set_intrinsic(tensor_intrinsic)
        depth_processor_ref.set_extrinsics(lidar_cam_extrinsic)

        depth_processor_est = Depth_Processing(image_width,image_height)
        depth_processor_est.set_intrinsic(tensor_intrinsic)
        depth_processor_est.set_extrinsics(lidar_cam_extrinsic)

        lidar_img_est = img_path + '/' + str(epoch_time) + '.png'
        lidar_reconstructor = Lidar_Reconstruction()

        mesh_ref = lidar_reconstructor.lidar_reconstruction(lidar_point_cloud_ref, 
                                                            depth_img_ref)
        depth_processor_ref.set_mesh(mesh_ref)
        depth_processor_ref.create_raycast_scene()

        mesh_est = lidar_reconstructor.lidar_reconstruction(lidar_point_cloud_est, 
                                                            depth_img_est)
        depth_processor_est.set_mesh(mesh_est)
        depth_processor_est.create_raycast_scene()

        # o3d.visualization.draw([mesh_ref])

        ##### MOTION FILTER #####
        sg_matcher.update_ref_img(lidar_img_ref)
        sg_matcher.update_est_img(lidar_img_est)
        sg_matcher.feature_extraction()
        mkpts_lidar_ref, mkpts_lidar_est = sg_matcher.get_filtered_feature()
        desc_ref, desc_est = sg_matcher.get_descriptor()
        confidence = sg_matcher.get_confidence()

        objPts_ref, hit_mask_ref = depth_processor_ref.raycast(mkpts_lidar_ref, lidar_cam_extrinsic)
        objPts_est, hit_mask_est = depth_processor_est.raycast(mkpts_lidar_est, lidar_cam_extrinsic)

        feature_filter.set_reference_feature_3d(objPts_ref)
        feature_filter.set_feature_3d(objPts_est)
        motion_mask = feature_filter.motion_filter()

        objPts_ref = objPts_ref[motion_mask]
        mkpts_lidar_ref = mkpts_lidar_ref[motion_mask]
        desc_ref = desc_ref[:,motion_mask]
        confidence = confidence[motion_mask]

        lidar_img_ref = lidar_img_est
        lidar_point_cloud_ref = lidar_point_cloud_est
        depth_img_ref = depth_img_est

        num_keypoints = len(objPts_ref[:,0])
        if num_keypoints < 4:
            print("too few points")
        else:
            data = {
                "num_points" : num_keypoints,
                "objPts" : objPts_ref.tolist(),
                "confidence" : confidence.tolist(),
                "descriptors" : desc_ref.tolist()
            }
            serialized_data = json.dumps(data)
            socket.send_data(serialized_data.encode("ascii"))

                


if __name__ == '__main__':
    main()
