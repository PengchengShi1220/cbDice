import torch
from monai.transforms import distance_transform_edt
from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize

class SoftcbDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftcbDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

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
        y_pred_prob = y_prob_binary[:, 1] # predicted probability map of foreground
        
        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob

        q_vl, q_slvl, q_sl = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)

        cb_dice_loss = - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        
        return cb_dice_loss
    
class SoftclMDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftclMDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

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
        y_pred_prob = y_prob_binary[:, 1] # predicted probability map of foreground
        
        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
        
        skel_pred_prob = skel_pred_hard * y_pred_prob

        q_vl, q_slvl, _ = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, _ = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        q_sl = skel_true
        q_sp = skel_pred_prob

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)

        cl_m_dice_loss = - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)

        return cl_m_dice_loss

def combine_tensors(A, B, C):
    A_C = A * C
    B_C = B * C
    D = B_C.clone()
    mask_AC = (A != 0) & (B == 0)
    D[mask_AC] = A_C[mask_AC]
    return D

def get_weights(mask_input, skel_input, dim, prob_flag=True):
    if prob_flag:
        mask_prob = mask_input
        skel_prob = skel_input

        mask = (mask_prob > 0.5).int()
        skel = (skel_prob > 0.5).int()
    else:
        mask = mask_input
        skel = skel_input

    distances = distance_transform_edt(mask).float()

    smooth = 1e-7
    distances[mask == 0] = 0

    skel_radius = torch.zeros_like(distances, dtype=torch.float32)
    skel_radius[skel == 1] = distances[skel == 1]

    dist_map_norm = torch.zeros_like(distances, dtype=torch.float32)
    skel_R_norm = torch.zeros_like(skel_radius, dtype=torch.float32)
    I_norm = torch.zeros_like(mask, dtype=torch.float32)
    for i in range(skel_radius.shape[0]):
        distances_i = distances[i]
        skel_i = skel_radius[i]
        skel_radius_max = max(skel_i.max(), 1)
        skel_radius_min = max(skel_i.min(), 1)
    
        distances_i[distances_i > skel_radius_max] = skel_radius_max
        dist_map_norm[i] = distances_i / skel_radius_max
        skel_R_norm[i] = skel_i / skel_radius_max

        # subtraction-based inverse (linear)：
        if dim == 2:
            I_norm[i] = (skel_radius_max - skel_i + skel_radius_min) / skel_radius_max
        else:
            I_norm[i] = ((skel_radius_max - skel_i + skel_radius_min) / skel_radius_max) ** 2

        # division-based inverse (nonlinear):
        # if dim == 2:
        #     I_norm[i] = (1 + smooth) / (skel_R_norm[i] + smooth) # weight for skel
        # else:
        #     I_norm[i] = (1 + smooth) / (skel_R_norm[i] ** 2 + smooth)

    I_norm[skel == 0] = 0 # 0 for non-skeleton pixels

    if prob_flag:
        return dist_map_norm * mask_prob, skel_R_norm * mask_prob, I_norm * skel_prob
    else:
        return dist_map_norm * mask, skel_R_norm * mask, I_norm * skel
