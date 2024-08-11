import numpy as np

class Feature_Filter:
    def __init__(self):
        self.max_motion = 1.0

        self.reference_feature_3d = None
        self.feature_3d = None
    
    def set_reference_feature_3d(self, objPts):
        self.reference_feature_3d = objPts

    def set_feature_3d(self, objPts):
        self.feature_3d = objPts

    def motion_filter(self):
        motion = np.linalg.norm(self.reference_feature_3d - self.feature_3d, axis=1)
        motion_mask = np.where((motion != np.nan) & \
                                (motion != np.inf) & \
                                (motion < self.max_motion))[0]
        print(np.average(motion))

        return motion_mask
