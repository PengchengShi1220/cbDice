from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CB_DICE import nnUNetTrainer_CB_DICE

class nnUNetTrainer_CB_DICE_NoMirroring(nnUNetTrainer_CB_DICE):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
        
