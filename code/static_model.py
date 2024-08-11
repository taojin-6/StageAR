#! /usr/bin/env python3

import numpy as np
import open3d as o3d
import sys
sys.path.insert(0, 'ADDRESS_OF_SOURCE_DIRECTORY')
import argparse

from utils import io_util
import time
import settings

from superglue_feature_extractor import Superglue_Matcher
from depth_processing_util import Depth_Processing

from socket_util import Socket_Util
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process mesh, reference image, and camera intrinsic and extrinsic paths.")
    parser.add_argument("--mesh", required=True, help="Path to the mesh file")
    parser.add_argument("--ref_image", required=True, help="Path to the reference image (from pre-scanned model))")
    parser.add_argument("--est_image", required=True, help="Path to the est images folder (image from ground truth camera)")
    parser.add_argument("--camera_intrinsic", required=True, help="Path to the camera intrinsic file (fixed camera)")
    parser.add_argument("--camera_extrinsic", required=True, help="Path to the camera extrinsic file (apriltag images)")
    parser.add_argument("--image_width", required=True, help="image_width")
    parser.add_argument("--image_height", required=True, help="image_width")
    return parser.parse_args()

def main():

    args = parse_arguments()

    mesh_path = args.mesh
    ref_img = args.ref_image
    camera_intrinsic_path = args.camera_intrinsic
    camera_extrinsic_path = args.camera_extrinsic
    image_width = args.image_width
    image_height = args.image_height
    est_img_path = args.est_image

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).to(settings.cpu_device)

    cam_intrinsic = np.loadtxt(camera_intrinsic_path)
    cam_extrinsic = io_util.get_gt_pose(camera_extrinsic_path)

    sg_matcher = Superglue_Matcher()
    depth_processor = Depth_Processing(image_width,image_height)

    sg_matcher.update_ref_img(ref_img)
    intrinsic, tensor_intrinsic = io_util.create_camera_intrinsic(image_width, image_height, cam_intrinsic)

    depth_processor.set_intrinsic(tensor_intrinsic)
    depth_processor.set_extrinsics(cam_extrinsic)
    depth_processor.set_mesh(tensor_mesh)
    depth_processor.create_raycast_scene()
    
    socket = Socket_Util()
    socket.init_socket()

    while True: 
        epoch_time = int(time.time())
        est_img = est_img_path + '/' + str(epoch_time) + '.png'

        #MOTION FILTER
        sg_matcher.update_est_img(est_img)
        sg_matcher.feature_extraction()
        mkpts_lidar_ref, mkpts_lidar_est = sg_matcher.get_filtered_feature()

        objPts, hit_mask = depth_processor.raycast(mkpts_lidar_ref, cam_extrinsic)
        desc_ref, desc_est = sg_matcher.get_descriptor()
        confidence = sg_matcher.get_confidence()

        num_keypoints = len(objPts[:,0])

        if num_keypoints < 4:
            print("Too few points\n")
        else:
            data = {
                "num_points" : num_keypoints,
                "objPts" : objPts.tolist(),
                "confidence" : confidence.tolist(),
                "descriptors" : desc_ref.tolist()
            }
            serialized_data = json.dumps(data)
            socket.send_data(serialized_data.encode("ascii"))

if __name__ == '__main__':
    main()
