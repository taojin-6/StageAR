import numpy as np
import torch

from models.matching import Matching
from models.utils import read_image, make_matching_plot

from superglue_utils import get_opt

torch.set_grad_enabled(False)

visualize = True

class Superglue_Matcher():
    def __init__(self):
        self.opt = get_opt()
        print(self.opt)

        self.device = 'cuda' if torch.cuda.is_available() and not self.opt.force_cpu else 'cpu'
        # self.device = torch.device("mps")
        print('Running inference on device \"{}\"'.format(self.device))

        self.config = {
            'superpoint': {
                'nms_radius': self.opt.nms_radius,
                'keypoint_threshold': self.opt.keypoint_threshold,
                'max_keypoints': self.opt.max_keypoints
            },
            'superglue': {
                'weights': self.opt.superglue,
                'sinkhorn_iterations': self.opt.sinkhorn_iterations,
                'match_threshold': 0.7,
            }
        }

        self.matching = Matching(self.config).eval().to(self.device)

        self.image_ref = None
        self.inp_ref = None
        self.scales_ref = None

        self.image_est = None
        self.inp_est = None
        self.scales_est = None

        self.kpts0 = None
        self.kpts1 = None

        self.matches = None
        self.conf = None
        self.mconf = None

        self.mkpts0 = None
        self.mkpts1 = None

        self.descriptor0 = None
        self.descriptor1 = None

        self.score0 = None
        self.score1 = None

        self.valid = None

    def update_ref_img(self, img_path):
        self.image_ref, self.inp_ref, self.scales_ref = read_image(
            img_path, self.device, self.opt.resize, 
            0, self.opt.resize_float)

        if self.image_ref is None:
            print('Problem reading image ref')
            exit(1)

    def update_est_img(self, img_path):
        self.image_est, self.inp_est, self.scales_est = read_image(
            img_path, self.device, self.opt.resize, 
            0, self.opt.resize_float)

        if self.image_est is None:
            print('Problem reading image est')
            exit(1)

    def feature_extraction(self):
        # Perform the matching.
        pred = self.matching({'image0': self.inp_ref, 'image1': self.inp_est})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        self.kpts0, self.kpts1 = pred['keypoints0'], pred['keypoints1']

        self.descriptor0, self.descriptor1 = pred['descriptors0'], pred['descriptors1']
        self.score0, self.score1 = pred['scores0'], pred['scores1']
        self.matches, self.conf = pred['matches0'], pred['matching_scores0']
        # Keep the matching keypoints.
        valid = self.matches > -1
        self.mkpts0 = self.kpts0[valid]
        self.mkpts1 = self.kpts1[self.matches[valid]]
        self.mconf = self.conf[valid]
        
        self.descriptor0 = self.descriptor0[:,valid]
        self.descriptor1 = self.descriptor1[:, self.matches[valid]]

        self.score0 = self.score0[valid]
        self.score1 = self.score1[self.matches[valid]]

    def feature_extraction_with_known_ref(self,kpts, descriptor, conf):
        # Perform the matching.
        mkpts0_tensor = torch.from_numpy(kpts).to(self.device)
        desc0_tensor = torch.from_numpy(descriptor).to(self.device)
        scores0_tensor = torch.from_numpy(conf).to(self.device)

        last_data = {}
        last_data['keypoints0'] = mkpts0_tensor
        last_data['descriptors0'] = desc0_tensor
        last_data['scores0'] = scores0_tensor.unsqueeze(0)

        pred = self.matching({**last_data,'image1': self.inp_est})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        self.kpts0, self.kpts1 = last_data['keypoints0'], pred['keypoints1']

        self.descriptor0, self.descriptor1 = last_data['descriptors0'], pred['descriptors1']

        self.matches, self.conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = self.matches > -1
        self.mkpts0 = self.kpts0[valid]
        self.mkpts1 = self.kpts1[self.matches[valid]]
        self.mconf = self.conf[valid]

        self.valid = valid
        
        self.descriptor0 = self.descriptor0[:,valid]
        self.descriptor1 = self.descriptor1[:, self.matches[valid]]

    def get_raw_feature(self):
        return self.kpts0, self.kpts1, self.matches

    def get_filtered_feature(self):
        return self.mkpts0, self.mkpts1

    def get_objPts_2d(self):
        return self.mkpts0
    
    def get_imagePts(self):
        return self.mkpts1
    
    def get_confidence(self):
        return self.mconf
    
    def get_descriptor(self):
        return self.descriptor0, self.descriptor1
    
    def get_score(self):
        return self.score0, self.score1
    
    def get_valid(self):
        return self.valid

    def superglue_visualize(self):
        import matplotlib.cm as cm
        # Visualize the matches.
        color = cm.jet(self.mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(self.kpts0), len(self.kpts1)),
            'Matches: {}'.format(len(self.mkpts0)),
        ]

        # Display extra parameter info.
        k_thresh = self.matching.superpoint.config['keypoint_threshold']
        m_thresh = self.matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
        ]

        make_matching_plot(
            self.image_ref, self.image_est, self.kpts0, self.kpts1, 
            self.mkpts0, self.mkpts1, color,
            text, 'test', self.opt.show_keypoints,
            self.opt.fast_viz, self.opt.opencv_display, 'Matches', small_text)