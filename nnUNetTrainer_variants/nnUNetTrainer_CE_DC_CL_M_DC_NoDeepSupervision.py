import numpy as np
from nnunetv2.training.loss.compound_cbdice_loss import DC_and_CE_and_CL_M_DC_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision

class nnUNetTrainer_CE_DC_CL_M_DC_NoDeepSupervision(nnUNetTrainerNoDeepSupervision):

    def _build_loss(self):
        
        lambda_clMdice = 1.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_clMdice

        loss = DC_and_CE_and_CL_M_DC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    {'iter_': 10, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_clMdice=lambda_clMdice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        self.print_to_log_file("lambda_clMdice: %s" % str(lambda_clMdice))
        self.print_to_log_file("lambda_dice: %s" % str(lambda_dice))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        return loss
    
