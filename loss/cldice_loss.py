import torch
from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize

class SoftclDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftclDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]

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

        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_true))+self.smooth)/(torch.sum(skel_pred_prob)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice_loss = - 2.0 * (tprec*tsens)/(tprec+tsens)

        return cl_dice_loss
