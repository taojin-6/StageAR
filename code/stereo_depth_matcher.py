import numpy as np
import cv2

class Stereo_Depth_Matcher:
    def __init__(self):
        self.stereo_left_kpts = None
        self.stereo_right_kpts = None
        self.user_kpts = None
        self.stereo_left_user_kpts = None
        self.confidence = None

        self.intrinsic_left = None
        self.intrinsic_right = None
        self.proj_m_left = None
        self.proj_m_right = None

        self.objPts = None
        self.imagePts = None

        self.descs_left = None
        self.descs_right = None

    def set_stereo_feature(self, stereo_left_kpts, stereo_right_kpts):
        self.stereo_left_kpts = stereo_left_kpts
        self.stereo_right_kpts = stereo_right_kpts
    
    def set_user_stereo_feature(self, user_kpts, stereo_left_user_kpts):
        self.user_kpts = user_kpts
        self.stereo_left_user_kpts = stereo_left_user_kpts

    def set_stereo_intrinsics(self, intrinsic_left, intrinsic_right):
        self.intrinsic_left = intrinsic_left
        self.intrinsic_right = intrinsic_right

    def set_projection_matrix(self, real_cam_pose_left=None, real_cam_pose_right=None):
        extrinsic_left = real_cam_pose_left
        extrinsic_right = real_cam_pose_right
        
        self.proj_m_left = self.intrinsic_left @ extrinsic_left[0:3]
        self.proj_m_right = self.intrinsic_right @ extrinsic_right[0:3]

    def get_num_keypoints(self):
        return self.user_kpts.shape[0]

    def get_objPts(self):
        return self.objPts

    def get_imagePts(self):
        return self.user_kpts
    
    def get_confidence(self):
        return self.confidence
    
    def set_descriptors(self, descs_left, descs_right):
        self.descs_left = descs_left
        self.descs_right = descs_right

    def get_descriptors(self):
        return self.descs_left, self.descs_right

    def triangulate_and_normalize(self):
        self.objPts = cv2.triangulatePoints(self.proj_m_left, 
                                            self.proj_m_right, 
                                            self.stereo_left_kpts.T, 
                                            self.stereo_right_kpts.T).T

        self.objPts = self.objPts[:, 0:3] / \
                        self.objPts[:, 3].reshape(self.objPts.shape[0], 1)

    def get_common_features(self):
        # Get the indices of the common features between left matching user
        # and left matching right
        common_indices = np.where((self.stereo_left_kpts[:, None] == \
                                   self.stereo_left_user_kpts).all(-1))[0]

        common_indices_user = np.where((self.stereo_left_user_kpts[:, None] == \
                                   self.stereo_left_kpts).all(-1))[0]

        self.descs_left = self.descs_left[common_indices]
        self.descs_right = self.descs_right[common_indices]

        self.stereo_left_kpts = self.stereo_left_kpts[common_indices]
        self.stereo_right_kpts = self.stereo_right_kpts[common_indices]
        self.confidence = self.confidence[common_indices]

        self.user_kpts = self.user_kpts[common_indices_user]
