import os
import torch
from torch import autocast, nn
from typing import Union, Tuple, List
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.monai.swin_unetr import SwinUNETR
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels

class nnUNetTrainer_SwinUNETR_NoDeepSupervision(nnUNetTrainerNoDeepSupervision):
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            label_manager = self.plans_manager.get_label_manager(self.dataset_json)

            patch_size = self.configuration_manager.patch_size
            self.num_input_channels, label_manager.num_segmentation_heads
            
            patch_size_tuple = tuple(patch_size)
            self.network = SwinUNETR(img_size=patch_size_tuple, in_channels=self.num_input_channels, out_channels=label_manager.num_segmentation_heads, spatial_dims=len(patch_size), use_v2=False).to(self.device)

            # compile network for free speedup
            if ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager,
                                                                   dataset_json)
        
        label_manager = plans_manager.get_label_manager(dataset_json)

        patch_size = configuration_manager.patch_size
        patch_size_tuple = tuple(patch_size)
        network = SwinUNETR(img_size=patch_size_tuple, in_channels=num_input_channels, out_channels=label_manager.num_segmentation_heads, spatial_dims=len(patch_size), use_v2=False)

        return network