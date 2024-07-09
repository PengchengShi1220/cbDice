import numpy as np
from nnunetv2.training.loss.compound_cbdice_loss import DC_and_CE_and_CBDC_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_SwinUNETR_NoDeepSupervision import nnUNetTrainer_SwinUNETR_NoDeepSupervision

class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC(nnUNetTrainer_SwinUNETR_NoDeepSupervision):

    def _build_loss(self):
        
        lambda_cbdice = 2.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_cbdice

        loss = DC_and_CE_and_CBDC_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    {'iter_': 10, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_cbdice=lambda_cbdice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        self.print_to_log_file("lambda_cbdice: %s" % str(lambda_cbdice))
        self.print_to_log_file("lambda_dice: %s" % str(lambda_dice))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        return loss
    
