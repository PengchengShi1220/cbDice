import numpy as np
from nnunetv2.training.loss.compound_losses import DC_CE_and_B_DoU_loss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision

class nnUNetTrainer_CE_DC_B_DoU_NoDeepSupervision(nnUNetTrainerNoDeepSupervision):

    def _build_loss(self):
        
        lambda_bdou = 1.0
        lambda_dc = 1.0
        lambda_ce = lambda_bdou + lambda_dc

        num_classes = self.label_manager.num_segmentation_heads
        loss = DC_CE_and_B_DoU_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, \
                                    {}, {'n_classes': num_classes}, weight_ce=lambda_ce, weight_dice=lambda_dc, weight_bdou=lambda_bdou, ignore_label=self.label_manager.ignore_label)

        self.print_to_log_file("lambda_bdou: %s" % str(lambda_bdou))
        self.print_to_log_file("lambda_dc: %s" % str(lambda_dc))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        return loss
    