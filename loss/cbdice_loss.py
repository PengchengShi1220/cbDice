import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import distance_transform_edt
from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize

class CBDC_loss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(CBDC_loss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)
        
    def combine_tensors(self, A, B, C):
        A_C = A * C
        B_C = B * C
        D = B_C.clone()
        mask_AC = (A != 0) & (B == 0)
        D[mask_AC] = A_C[mask_AC]
        return D

    def get_weights(self, mask, skel, dim):
        distances = distance_transform_edt(mask).float()

        smooth = 1e-7
        mask_inv = mask == 0
        distances[mask_inv] = 0

        skel_radius = torch.zeros_like(distances, dtype=torch.float32)
        skel_radius[skel == 1] = distances[skel == 1]

        skel_radius_max = skel_radius.reshape(skel_radius.shape[0], -1).max(dim=1).values.view(-1, 1, 1, 1)
        valid_max_mask = skel_radius_max > 0
        skel_radius_max[~valid_max_mask] = 1

        dist_map_norm = torch.where(valid_max_mask, distances / skel_radius_max, distances)
        skel_R_norm = torch.where(valid_max_mask, skel_radius / skel_radius_max, skel_radius)

        if dim == 2:
            skel_1_R_norm = (1 + smooth) / (skel_R_norm + smooth)
            skel_1_R_norm[skel != 1] = 0
            return dist_map_norm, skel_R_norm, skel_1_R_norm
        else:
            skel_1_R2_norm = (1 + smooth) / (skel_R_norm ** 2 + smooth)
            skel_1_R2_norm[skel != 1] = 0
            return dist_map_norm, skel_R_norm, skel_1_R2_norm

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred = torch.where(y_prob_binary[:, 1] > 0.5, 1, 0).float()
        y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()

        if t_skeletonize_flage:
            skel_pred = self.t_skeletonize(y_pred.unsqueeze(1)).squeeze(1)
            skel_true = self.t_skeletonize(y_true.unsqueeze(1).detach()).squeeze(1)
        else:
            skel_pred = self.m_skeletonize(y_pred.unsqueeze(1)).squeeze(1)
            skel_true = self.m_skeletonize(y_true.unsqueeze(1).detach()).squeeze(1)

        q_vl, q_slvl, q_sl = self.get_weights(y_true, skel_true, dim)
        q_vp, q_spvp, q_sp = self.get_weights(y_pred, skel_pred, dim)

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(self.combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(self.combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)
        cb_dice = 1. - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)

        return cb_dice
    
class clMdice_loss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(clMdice_loss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)
        
    def combine_tensors(self, A, B, C):
        A_C = A * C
        B_C = B * C
        D = B_C.clone()
        mask_AC = (A != 0) & (B == 0)
        D[mask_AC] = A_C[mask_AC]
        return D

    def get_weights(self, mask, skel, dim):
        distances = distance_transform_edt(mask).float()

        smooth = 1e-7
        mask_inv = mask == 0
        distances[mask_inv] = 0

        skel_radius = torch.zeros_like(distances, dtype=torch.float32)
        skel_radius[skel == 1] = distances[skel == 1]

        skel_radius_max = skel_radius.reshape(skel_radius.shape[0], -1).max(dim=1).values.view(-1, 1, 1, 1)
        valid_max_mask = skel_radius_max > 0
        skel_radius_max[~valid_max_mask] = 1

        dist_map_norm = torch.where(valid_max_mask, distances / skel_radius_max, distances)
        skel_R_norm = torch.where(valid_max_mask, skel_radius / skel_radius_max, skel_radius)

        if dim == 2:
            skel_1_R_norm = (1 + smooth) / (skel_R_norm + smooth)
            skel_1_R_norm[skel != 1] = 0
            return dist_map_norm, skel_R_norm, skel_1_R_norm
        else:
            skel_1_R2_norm = (1 + smooth) / (skel_R_norm ** 2 + smooth)
            skel_1_R2_norm[skel != 1] = 0
            return dist_map_norm, skel_R_norm, skel_1_R2_norm

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred = torch.where(y_prob_binary[:, 1] > 0.5, 1, 0).float()
        y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()

        if t_skeletonize_flage:
            skel_pred = self.t_skeletonize(y_pred.unsqueeze(1)).squeeze(1)
            skel_true = self.t_skeletonize(y_true.unsqueeze(1).detach()).squeeze(1)
        else:
            skel_pred = self.m_skeletonize(y_pred.unsqueeze(1)).squeeze(1)
            skel_true = self.m_skeletonize(y_true.unsqueeze(1).detach()).squeeze(1)

        q_vl, q_slvl, _ = self.get_weights(y_true, skel_true, dim)
        q_vp, q_spvp, _ = self.get_weights(y_pred, skel_pred, dim)

        q_sl = skel_true
        q_sp = skel_pred

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(self.combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(self.combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)
        cl_m_dice = 1. - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        
        return cl_m_dice
