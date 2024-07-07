import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from nnunetv2.training.loss.skeletonize import Skeletonize

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta-skel*delta)
    return skel

class CBDC_loss(torch.nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(CBDC_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.skeletonization_module = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
    
    def combine_tensors(self, A, B, C):
        A_C = A * C
        B_C = B * C
        D = B_C.clone()
        mask_AC = (A != 0) & (B == 0)
        D[mask_AC] = A_C[mask_AC]
        return D

    def get_weights(self, mask, skel, dim):
        distances = monai.transforms.utils.distance_transform_edt(mask)

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

    def forward(self, y_pred, y_true, skeletonization_flage=True):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_prob = torch.softmax(y_pred, 1)
        y_pre = torch.argmax(y_prob, dim=1)
        y_pred = torch.where(y_pre > 0, 1, 0).float()
        y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()

        if skeletonization_flage:
            skel_pred = self.skeletonization_module(y_pred.unsqueeze(1)).squeeze(1)
            skel_true = self.skeletonization_module(y_true.unsqueeze(1).detach()).squeeze(1)
        else:
            skel_pred = soft_skel(y_pred.unsqueeze(1), self.iter).squeeze(1)
            skel_true = soft_skel(y_true.unsqueeze(1).detach(), self.iter).squeeze(1)

        q_vl, q_slvl, q_sl = self.get_weights(y_true, skel_true, dim)
        q_vp, q_spvp, q_sp = self.get_weights(y_pred, skel_pred, dim)

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(self.combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(self.combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)
        cb_dice = 1. - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)

        return cb_dice
    
class clMdice_loss(torch.nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(clMdice_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.skeletonization_module = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
    
    def combine_tensors(self, A, B, C):
        A_C = A * C
        B_C = B * C
        D = B_C.clone()
        mask_AC = (A != 0) & (B == 0)
        D[mask_AC] = A_C[mask_AC]
        return D

    def get_weights(self, mask, skel, dim):
        distances = monai.transforms.utils.distance_transform_edt(mask)

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

    def forward(self, y_pred, y_true, skeletonization_flage=True):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_prob = torch.softmax(y_pred, 1)
        y_pre = torch.argmax(y_prob, dim=1)
        y_pred = torch.where(y_pre > 0, 1, 0).float()
        y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()

        if skeletonization_flage:
            skel_pred = self.skeletonization_module(y_pred.unsqueeze(1)).squeeze(1)
            skel_true = self.skeletonization_module(y_true.unsqueeze(1).detach()).squeeze(1)
        else:
            skel_pred = soft_skel(y_pred.unsqueeze(1), self.iter).squeeze(1)
            skel_true = soft_skel(y_true.unsqueeze(1).detach(), self.iter).squeeze(1)

        q_vl, q_slvl, _ = self.get_weights(y_true, skel_true, dim)
        q_vp, q_spvp, _ = self.get_weights(y_pred, skel_pred, dim)

        q_sl = skel_true
        q_sp = skel_pred

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(self.combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(self.combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)
        cl_m_dice = 1. - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        
        return cl_m_dice
