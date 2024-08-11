import numpy as np
import open3d as o3d
import sys
import cv2
sys.path.insert(0, '/home/sc/Desktop/crowd_slam')

from utils import io_util

from superglue_feature_extractor import Superglue_Matcher

def compute_pose(objPts, imagePts, cam_intrinsic, distortion=None):
    retval, rvec, tvec, inlier = cv2.solvePnPRansac(objPts, imagePts, cam_intrinsic.astype(float), distortion, 
                                                    iterationsCount = 1000, reprojectionError = 2.0, flags=cv2.SOLVEPNP_EPNP)
    
    rmat = cv2.Rodrigues(rvec)[0]

    estimated_pose = np.zeros((4, 4))
    estimated_pose[0:3, 0:3] = rmat
    estimated_pose[0:3, 3] = tvec.flatten()
    estimated_pose[3, 3] = 1

    return estimated_pose

def pixel_error_eval(apriltag_pose,est_pose,intrinsic,img_width):
    point = np.array([1,1,1,1]).T
    ap = (intrinsic @ apriltag_pose[0:3]) @ point
    ap /= ap[2]
    
    est = (intrinsic @ est_pose[0:3]) @ point
    est /= est[2]

    diff = np.linalg.norm(ap[:2] - est[:2])
    px_error = [1.0 * diff / (img_width)] #WIDTH

    print("pixel errror: ", px_error)

    return px_error

def pose_error_eval(gt_pose, estimated_pose):
    # translation error
    gt_trans = gt_pose[0:3, 3]
    est_trans = estimated_pose[0:3, 3]
    trans_err = np.linalg.norm(gt_trans - est_trans)
    print("translation error (meters): " + str(trans_err))

    # rotation error
    est_rot = estimated_pose[0:3, 0:3]
    gt_rot = gt_pose[0:3, 0:3]
    rel_rot = est_rot @ gt_rot.T
    rel_rot = cv2.Rodrigues(rel_rot)[0]
    rot_error = np.linalg.norm(rel_rot)
    deg_error = np.degrees(rot_error)
    print("rotation error (degree): " + str(deg_error))

    return [trans_err, deg_error]

def localize(objPts, desc, conf):
    user_img = 'USER_IMG_PATH'
    user_intrinsic = np.loadtxt('USER_INTRINSIC_PATH')
    user_extrinsic = io_util.get_gt_pose('USER_EXTRINSIC_PATH')
    img_width = 1920

    sg_matcher = Superglue_Matcher()
    sg_matcher.update_est_img(user_img)
    objPts2D = objPts[:,:2]
    sg_matcher.feature_extraction_with_known_ref(objPts2D,desc,conf)
    mkpts_ref, mkpts_est = sg_matcher.get_filtered_feature()
    valid = sg_matcher.get_valid()
    matched_objPts = objPts[valid]
    estimated_position = compute_pose(matched_objPts, mkpts_est, user_intrinsic)

    metric_pose_error = pose_error_eval(user_extrinsic, estimated_position)
    px_pose_error = pixel_error_eval(user_extrinsic, estimated_position, 
                                     user_intrinsic, img_width)
