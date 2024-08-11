#! /usr/bin/env python3

import numpy as np
import argparse
import time
import sys
sys.path.insert(0, 'ADDR')

from utils import io_util

from superglue_feature_extractor import Superglue_Matcher
from stereo_depth_matcher import Stereo_Depth_Matcher
from feature_filtering_utils import Feature_Filter


from socket_util import Socket_Util
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process mesh, reference image, and camera intrinsic and extrinsic paths.")
    parser.add_argument("--cam_1", required=True, help="Path to the images from fixed camera 1 (stereo left)")
    parser.add_argument("--cam_2", required=True, help="Path to the images from fixed camera 2 (stereo right)")
    parser.add_argument("--camera_intrinsic", required=True, help="Path to the camera intrinsic folder")
    parser.add_argument("--camera_extrinsic", required=True, help="Path to the camera extrinsic folder")
    return parser.parse_args()

def main():

    args = parse_arguments()
    stereo_left = args.cam_1
    stereo_right = args.cam_2
    camera_intrinsic_path = args.camera_intrinsic
    camera_extrinsic_path = args.camera_extrinsic


    # initialize SuperGlue Matcher
    sg_matcher = Superglue_Matcher()
    depth_estimator = Stereo_Depth_Matcher()
    feature_filter = Feature_Filter()

    # get all the intrinsics and extrinsics
    stereo_left_intrinsic = np.loadtxt(camera_intrinsic_path+'/1.txt')
    stereo_left_pose = io_util.get_gt_pose(camera_extrinsic_path+'/1.txt')

    stereo_right_intrinsic = np.loadtxt(camera_intrinsic_path+'/2.txt')
    stereo_right_pose = io_util.get_gt_pose(camera_extrinsic_path+'/2.txt')

    depth_estimator.set_stereo_intrinsics(stereo_left_intrinsic, 
                                            stereo_right_intrinsic)
    depth_estimator.set_projection_matrix(real_cam_pose_left=stereo_left_pose, 
                                            real_cam_pose_right=stereo_right_pose)

    epoch_time = int(time.time())
    stereo_img_left_ref = stereo_left + '/' + str(epoch_time) + '.png'
    stereo_img_right_ref = stereo_right + '/' + str(epoch_time) + '.png'
    
    socket = Socket_Util()
    socket.init_socket()

    while True:
        epoch_time = int(time.time())
        stereo_img_left_est = stereo_left + '/' + str(epoch_time) + '.png'
        stereo_img_right_est = stereo_right + '/' + str(epoch_time) + '.png'

        ##### MOTION FILTER #####
        # Set current features
        sg_matcher.update_ref_img(stereo_img_left_est)
        sg_matcher.update_est_img(stereo_img_right_est)

        sg_matcher.feature_extraction()
        sg_matcher.superglue_visualize()

        mkpts_left_est, mkpts_right_est = sg_matcher.get_filtered_feature()

        # Set reference features
        sg_matcher.update_ref_img(stereo_img_left_ref)
        sg_matcher.update_est_img(stereo_img_right_ref)
        sg_matcher.feature_extraction()
        mkpts_left_ref, mkpts_right_ref = sg_matcher.get_filtered_feature()

        confidence = sg_matcher.get_confidence()
        desc_ref, desc_est = sg_matcher.get_descriptor()
        common_indices_ref = np.where((mkpts_left_ref[:, None] == \
                                        mkpts_left_est).all(-1))[0]
        mkpts_left_ref = mkpts_left_ref[common_indices_ref]
        mkpts_right_ref = mkpts_right_ref[common_indices_ref]
        confidence = confidence[common_indices_ref]
        desc_ref = desc_ref[:,common_indices_ref]

        common_indices_est = np.where((mkpts_left_est[:, None] == \
                                        mkpts_left_ref).all(-1))[0]
        mkpts_left_est = mkpts_left_est[common_indices_est]
        mkpts_right_est = mkpts_right_est[common_indices_est]

        #MOTION FILTER

        depth_estimator.set_stereo_feature(mkpts_left_est, mkpts_right_est)
        depth_estimator.triangulate_and_normalize()

        feature_3d = depth_estimator.get_objPts()
        feature_filter.set_feature_3d(feature_3d)

        depth_estimator.set_stereo_feature(mkpts_left_ref, mkpts_right_ref)
        depth_estimator.triangulate_and_normalize()

        reference_feature_3d = depth_estimator.get_objPts()
        feature_filter.set_reference_feature_3d(reference_feature_3d)

        motion_mask = feature_filter.motion_filter()

        reference_feature_3d = reference_feature_3d[motion_mask]
        confidence = confidence[motion_mask]
        desc_ref = desc_ref[:,motion_mask]

        stereo_img_left_ref = stereo_img_left_est
        stereo_img_right_ref = stereo_img_right_est

        num_keypoints = len(reference_feature_3d[:,0])
    
        if num_keypoints < 4:
            print("Too few points\n")
        else:
            data = {
                "num_points" : num_keypoints,
                "objPts" : reference_feature_3d.tolist(),
                "confidence" : confidence.tolist(),
                "descriptors" : desc_ref.tolist()
            }
            serialized_data = json.dumps(data)
            socket.send_data(serialized_data.encode("ascii"))



if __name__ == '__main__':
    main()
