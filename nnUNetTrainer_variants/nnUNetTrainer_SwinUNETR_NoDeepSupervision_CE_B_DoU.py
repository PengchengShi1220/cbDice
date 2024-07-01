import numpy as np
from nnunetv2.training.loss.compound_losses import B_DoU_and_CE_loss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_SwinUNETR_NoDeepSupervision import nnUNetTrainer_SwinUNETR_NoDeepSupervision

class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_B_DoU(nnUNetTrainer_SwinUNETR_NoDeepSupervision):
    
    def _build_loss(self):
        
        lambda_bdou = 1.0
        lambda_ce = lambda_bdou

        num_classes = self.label_manager.num_segmentation_heads
        loss = B_DoU_and_CE_loss({}, {'n_classes': num_classes}, weight_ce=lambda_ce, weight_bdou=lambda_bdou, ignore_label=self.label_manager.ignore_label)

        self.print_to_log_file("lambda_bdou: %s" % str(lambda_bdou))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        return loss
    