import inspect
import multiprocessing
import warnings
import torch
import os
import numpy as np
# import distmap
import cupy as cp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy
import nibabel as nib

from skimage.morphology import skeletonize_3d
from scipy import ndimage
from scipy.spatial import cKDTree
from torch import autocast, nn
from time import time, sleep
from datetime import datetime
from typing import Union, Tuple, List
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.PlainConvUNet_DC_CLDC_skeletonize import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.skeletonize import Skeletonize
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data_skeletonize import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss, CE_loss, BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.logging.nnunet_logger_vesselgrapher import nnUNetLogger_vesselgrapher
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
    
class nnUNetTrainer_CB_DICE(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried. http://tiny.cc/gzgwuz

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(
            configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
            if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self._get_network()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger_vesselgrapher()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 2  # 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)
    
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            # print("self.dataset_json: ", self.dataset_json)
            # max_label_value = max(self.dataset_json["labels"].values())
            # self.dataset_json["labels"]["skel_vessel"] = max_label_value + 1
            # self.dataset_json["labels"]["skel"] = max_label_value + 2

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)
            
            self.skeletonization_module_binary = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic').to(self.device)
            self.skeletonization_module_multi = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic').to(self.device)

            self.skeletonization_module = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic').to(self.device)

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

            # self.seg_loss = self._build_loss()
            self.seg_loss_0_bin, self.seg_loss_0_mul, self.seg_loss_weights, self.seg_loss_deep = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    
    def _build_loss(self):
        # if self.label_manager.has_regions:
        #     loss = DC_and_BCE_loss({},
        #                            {'batch_dice': self.configuration_manager.batch_dice,
        #                             'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
        #                            use_ignore_label=self.label_manager.ignore_label is not None,
        #                            dice_class=MemoryEfficientSoftDiceLoss)
        # else:
        loss_0_bin = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=0.5, weight_dice=0.5,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        loss_0_mul = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=0.5, weight_dice=0.5,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        loss_deep = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=0.5, weight_dice=0.5,
                                    ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss_deep = DeepSupervisionWrapper(loss_deep, weights[1:])

        return loss_0_bin, loss_0_mul, weights, loss_deep
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        # configuration_manager.UNet_class_name
        segmentation_network_class_name = 'PlainConvUNet'
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
            }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model

    def batch_id2pos_indexs_k(self, id, size):
        x, y, z = size

        pos_x = (id // (y * z)).type(torch.int32)
        pos_y = ((id // z) % y).type(torch.int32)
        pos_z = (id % z).type(torch.int32)

        pos = torch.stack((pos_x, pos_y, pos_z))

        return pos

    def combine_tensors(self, A, B, C):
        D = A.clone()        # Clone A to D
        B_C = B * C          # Element-wise multiply B and C
        mask = (A != 0) & (B != 0)  # Non-zero mask for A and B

        D[mask] = B_C[mask]  # Update D based on mask
        return D

    def get_radius_weights(self, y_true, skel_true, H, W, D):
        # dist_map_3d = distmap.euclidean_distance_transform(y_true)
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
        y_true_cupy_array = cp.from_dlpack(to_dlpack(y_true))
        dist_map_3d_cupy_array = distance_transform_edt_cupy(y_true_cupy_array)
        dist_map_3d = from_dlpack(dist_map_3d_cupy_array.toDlpack())
    
        dist_map_3d[y_true == 0] = 0
        vessel_radius = dist_map_3d[skel_true == 1]

        if vessel_radius.shape[0] == 0 or vessel_radius.min() == vessel_radius.max():
            return y_true, skel_true.clone(), skel_true.clone(), skel_true.clone()

        smooth = 1e-7
        vessel_radius_max = vessel_radius.max()
        dist_map_3d[dist_map_3d > vessel_radius_max] = vessel_radius_max
        # print("vessel_radius_max: ", vessel_radius_max)
        # print("dist_map_3d_max: ", dist_map_3d.max())
        vessel_radius_0_1 = vessel_radius / vessel_radius_max
        vessel_radius_1_R2 = (1 + smooth) / (vessel_radius_0_1 ** 2 + smooth)
        vessel_radius_1_R = (1 + smooth) / (vessel_radius_0_1 + smooth)
        y_dist_map_norm = dist_map_3d / vessel_radius_max
        
        vessel_radius_weights_1_R2 = torch.zeros_like(skel_true, dtype=torch.float32)
        vessel_radius_weights_1_R = torch.zeros_like(skel_true, dtype=torch.float32)
        vessel_radius_weights_1 = torch.zeros_like(skel_true, dtype=torch.float32)
        N = H * W * D
        skel_N = skel_true.reshape(N)
        nodes = (skel_N == 1).nonzero(as_tuple=False).squeeze()
        nodes_pos = self.batch_id2pos_indexs_k(nodes, (H, W, D)).T

        vessel_radius_weights_1_R2[nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2]] = vessel_radius_1_R2
        vessel_radius_weights_1_R[nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2]] = vessel_radius_1_R
        vessel_radius_weights_1[nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2]] = vessel_radius_0_1

        return y_dist_map_norm, vessel_radius_weights_1_R2, vessel_radius_weights_1_R, vessel_radius_weights_1
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            target_0 = target[0].clone()
            target_0[target_0 > 0] = 1
            skel_target_0 = self.skeletonization_module(target_0.float())

            output = self.network(data)
            output_0 = output[0]

            seg_binary_output_0 = output_0[:, -2:, :, :, :]
            seg_output_0 = output_0[:, :-2, :, :, :]

            loss_binary_dc = self.seg_loss_0_bin(seg_binary_output_0, target_0)
                        
            loss_seg_0 = self.seg_loss_0_mul(seg_output_0, target[0])
            loss_seg_deep = self.seg_loss_deep(output[1:], target[1:])

            seg_loss_weight_0 = self.seg_loss_weights[0]
            seg_loss_weight_deep = self.seg_loss_weights[1:]
            
            results_prob = torch.softmax(seg_binary_output_0, 1)
            seg_pre = torch.argmax(results_prob, dim=1)
            skel_pred_binary = self.skeletonization_module_binary(seg_pre.unsqueeze(1).float()).squeeze(1)
            skel_true = skel_target_0.detach().squeeze(1)
            y_pred_binary = torch.where(seg_pre > 0, 1, 0)
            y_true = target_0.detach().squeeze(1)

            results_prob_fore = torch.softmax(seg_output_0, 1)
            seg_pre_fore = torch.argmax(results_prob_fore, dim=1)
            y_pred_multi = torch.where(seg_pre_fore > 0, 1, 0)
            #skel_pred_multi = self.skeletonization_module_multi(y_pred_multi.unsqueeze(1).float()).squeeze(1)
            skel_pred_multi = skel_pred_binary

            self.smooth = 1e-3
            Batch = y_true.shape[0]
            H, W, D = y_true.shape[1], y_true.shape[2], y_true.shape[3]

            if loss_seg_0 > 0.3:
                radii_weights_1_R2_true_multi = skel_true.float()
                radii_weights_1_R2_pred_multi = skel_pred_multi.float()
                radii_weights_1_R2_pred_binary = skel_pred_binary.float()
                radii_weights_1_R2_true_binary = skel_true.float()
                radii_weights_1_R_true_multi = skel_true.float()
                radii_weights_1_R_pred_multi = skel_pred_multi.float()
                radii_weights_1_R_pred_binary = skel_pred_binary.float()
                radii_weights_1_R_true_binary = skel_true.float()
                radii_weights_1_true_multi = skel_true.float()
                radii_weights_1_pred_multi = skel_pred_multi.float()
                radii_weights_1_pred_binary = skel_pred_binary.float()
                radii_weights_1_true_binary = skel_true.float()
                y_true_dist_map_norm = y_true.float()
                y_pred_multi_dist_map_norm = y_pred_multi.float()
                y_pred_binary_dist_map_norm = y_pred_binary.float()
            else:
                radii_weights_1_R2_true_multi = torch.zeros_like(skel_true).float()
                radii_weights_1_R2_pred_multi = torch.zeros_like(skel_pred_multi).float()
                radii_weights_1_R2_pred_binary = torch.zeros_like(skel_pred_binary).float()
                radii_weights_1_R_true_multi = torch.zeros_like(skel_true).float()
                radii_weights_1_R_pred_multi = torch.zeros_like(skel_pred_multi).float()
                radii_weights_1_R_pred_binary = torch.zeros_like(skel_pred_binary).float()
                radii_weights_1_true_multi = torch.zeros_like(skel_true).float()
                radii_weights_1_pred_multi = torch.zeros_like(skel_pred_multi).float()
                radii_weights_1_pred_binary = torch.zeros_like(skel_pred_binary).float()
                y_true_dist_map_norm = torch.zeros_like(y_true).float()
                y_pred_multi_dist_map_norm = torch.zeros_like(y_pred_multi).float()
                y_pred_binary_dist_map_norm = torch.zeros_like(y_pred_binary).float()

                for b_i in range(Batch):
                    y_true_dist_map_norm[b_i], radii_weights_1_R2_true_multi[b_i], radii_weights_1_R_true_multi[b_i], radii_weights_1_true_multi[b_i] = self.get_radius_weights(y_true[b_i], skel_true[b_i], H, W, D)
                    y_pred_multi_dist_map_norm[b_i], radii_weights_1_R2_pred_multi[b_i], radii_weights_1_R_pred_multi[b_i], radii_weights_1_pred_multi[b_i] = self.get_radius_weights(y_pred_multi[b_i], skel_pred_multi[b_i], H, W, D)
                    y_pred_binary_dist_map_norm[b_i], radii_weights_1_R2_pred_binary[b_i], radii_weights_1_R_pred_binary[b_i], radii_weights_1_pred_binary[b_i] = self.get_radius_weights(y_pred_binary[b_i], skel_pred_binary[b_i], H, W, D)

                radii_weights_1_R_true_binary = radii_weights_1_R_true_multi
                radii_weights_1_R2_true_binary = radii_weights_1_R2_true_multi
                radii_weights_1_true_binary = radii_weights_1_true_multi
            
            weighted_tprec = (torch.sum(torch.multiply(radii_weights_1_R2_pred_binary, y_true_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_pred_binary, radii_weights_1_true_binary, radii_weights_1_R2_pred_binary))+self.smooth)
            weighted_tsens = (torch.sum(torch.multiply(radii_weights_1_R2_true_binary, y_pred_binary_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_true_binary, radii_weights_1_pred_binary, radii_weights_1_R2_true_binary))+self.smooth) 
            skel_cl_dice = - 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)
            print("skel cl_dice: ", skel_cl_dice)

            weighted_tprec = (torch.sum(torch.multiply(radii_weights_1_R2_pred_multi, y_true_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_pred_multi, radii_weights_1_true_multi, radii_weights_1_R2_pred_multi))+self.smooth)
            weighted_tsens = (torch.sum(torch.multiply(radii_weights_1_R2_true_multi, y_pred_multi_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_true_multi, radii_weights_1_pred_multi, radii_weights_1_R2_true_multi))+self.smooth)
            seg_cl_dice = - 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)
            print("seg_output cl_dice: ", seg_cl_dice)

            loss_skel_dc_cldc = loss_binary_dc + 0.5 * skel_cl_dice

            loss_0_dc_cldc = loss_seg_0 + 0.5 * seg_cl_dice
            loss_seg_deep_dc_cldc = loss_seg_deep + np.sum(seg_loss_weight_deep) * 0.5 * seg_cl_dice

            l = loss_seg_deep_dc_cldc + seg_loss_weight_0 * loss_0_dc_cldc + seg_loss_weight_0 * loss_skel_dc_cldc
            print("loss_seg_deep_dc_cldc, loss_0_dc_cldc, loss_skel_dc_cldc, l: ", loss_seg_deep_dc_cldc, loss_0_dc_cldc, loss_skel_dc_cldc, l)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
    
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            target_0 = target[0].clone()
            target_0[target_0 > 0] = 1
            skel_target_0 = self.skeletonization_module(target_0.float())

            output = self.network(data)
            output_0 = output[0]

            seg_binary_output_0 = output_0[:, -2:, :, :, :]
            seg_output_0 = output_0[:, :-2, :, :, :]

            loss_binary_dc = self.seg_loss_0_bin(seg_binary_output_0, target_0)
                        
            loss_seg_0 = self.seg_loss_0_mul(seg_output_0, target[0])
            loss_seg_deep = self.seg_loss_deep(output[1:], target[1:])

            seg_loss_weight_0 = self.seg_loss_weights[0]
            seg_loss_weight_deep = self.seg_loss_weights[1:]
            
            results_prob = torch.softmax(seg_binary_output_0, 1)
            seg_pre = torch.argmax(results_prob, dim=1)
            skel_pred_binary = self.skeletonization_module_binary(seg_pre.unsqueeze(1).float()).squeeze(1)
            skel_true = skel_target_0.detach().squeeze(1)
            y_pred_binary = torch.where(seg_pre > 0, 1, 0)
            y_true = target_0.detach().squeeze(1)

            results_prob_fore = torch.softmax(seg_output_0, 1)
            seg_pre_fore = torch.argmax(results_prob_fore, dim=1)
            y_pred_multi = torch.where(seg_pre_fore > 0, 1, 0)
            skel_pred_multi = self.skeletonization_module_multi(y_pred_multi.unsqueeze(1).float()).squeeze(1)

            self.smooth = 1e-3
            Batch = y_true.shape[0]
            H, W, D = y_true.shape[1], y_true.shape[2], y_true.shape[3]

            if loss_seg_0 > 0.3:
                radii_weights_1_R2_true_multi = skel_true.float()
                radii_weights_1_R2_pred_multi = skel_pred_multi.float()
                radii_weights_1_R2_pred_binary = skel_pred_binary.float()
                radii_weights_1_R2_true_binary = skel_true.float()
                radii_weights_1_R_true_multi = skel_true.float()
                radii_weights_1_R_pred_multi = skel_pred_multi.float()
                radii_weights_1_R_pred_binary = skel_pred_binary.float()
                radii_weights_1_R_true_binary = skel_true.float()
                radii_weights_1_true_multi = skel_true.float()
                radii_weights_1_pred_multi = skel_pred_multi.float()
                radii_weights_1_pred_binary = skel_pred_binary.float()
                radii_weights_1_true_binary = skel_true.float()
                y_true_dist_map_norm = y_true.float()
                y_pred_multi_dist_map_norm = y_pred_multi.float()
                y_pred_binary_dist_map_norm = y_pred_binary.float()
            else:
                radii_weights_1_R2_true_multi = torch.zeros_like(skel_true).float()
                radii_weights_1_R2_pred_multi = torch.zeros_like(skel_pred_multi).float()
                radii_weights_1_R2_pred_binary = torch.zeros_like(skel_pred_binary).float()
                radii_weights_1_R_true_multi = torch.zeros_like(skel_true).float()
                radii_weights_1_R_pred_multi = torch.zeros_like(skel_pred_multi).float()
                radii_weights_1_R_pred_binary = torch.zeros_like(skel_pred_binary).float()
                radii_weights_1_true_multi = torch.zeros_like(skel_true).float()
                radii_weights_1_pred_multi = torch.zeros_like(skel_pred_multi).float()
                radii_weights_1_pred_binary = torch.zeros_like(skel_pred_binary).float()
                y_true_dist_map_norm = torch.zeros_like(y_true).float()
                y_pred_multi_dist_map_norm = torch.zeros_like(y_pred_multi).float()
                y_pred_binary_dist_map_norm = torch.zeros_like(y_pred_binary).float()
                    
                for b_i in range(Batch):
                    y_true_dist_map_norm[b_i], radii_weights_1_R2_true_multi[b_i], radii_weights_1_R_true_multi[b_i], radii_weights_1_true_multi[b_i] = self.get_radius_weights(y_true[b_i], skel_true[b_i], H, W, D)
                    y_pred_multi_dist_map_norm[b_i], radii_weights_1_R2_pred_multi[b_i], radii_weights_1_R_pred_multi[b_i], radii_weights_1_pred_multi[b_i] = self.get_radius_weights(y_pred_multi[b_i], skel_pred_multi[b_i], H, W, D)
                    y_pred_binary_dist_map_norm[b_i], radii_weights_1_R2_pred_binary[b_i], radii_weights_1_R_pred_binary[b_i], radii_weights_1_pred_binary[b_i] = self.get_radius_weights(y_pred_binary[b_i], skel_pred_binary[b_i], H, W, D)

                radii_weights_1_R_true_binary = radii_weights_1_R_true_multi
                radii_weights_1_R2_true_binary = radii_weights_1_R2_true_multi
                radii_weights_1_true_binary = radii_weights_1_true_multi
            
            weighted_tprec = (torch.sum(torch.multiply(radii_weights_1_R2_pred_binary, y_true_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_pred_binary, radii_weights_1_true_binary, radii_weights_1_R2_pred_binary))+self.smooth)
            weighted_tsens = (torch.sum(torch.multiply(radii_weights_1_R2_true_binary, y_pred_binary_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_true_binary, radii_weights_1_pred_binary, radii_weights_1_R2_true_binary))+self.smooth) 
            skel_cl_dice = - 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)
            print("skel cl_dice: ", skel_cl_dice)

            weighted_tprec = (torch.sum(torch.multiply(radii_weights_1_R2_pred_multi, y_true_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_pred_multi, radii_weights_1_true_multi, radii_weights_1_R2_pred_multi))+self.smooth)
            weighted_tsens = (torch.sum(torch.multiply(radii_weights_1_R2_true_multi, y_pred_multi_dist_map_norm))+self.smooth)/(torch.sum(self.combine_tensors(radii_weights_1_R_true_multi, radii_weights_1_pred_multi, radii_weights_1_R2_true_multi))+self.smooth)
            seg_cl_dice = - 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)
            print("seg_output cl_dice: ", seg_cl_dice)

            loss_skel_dc_cldc = loss_binary_dc + 0.5 * skel_cl_dice

            if torch.sum(skel_true) > 0 and torch.sum(skel_pred_multi) == 0:
                cl_dice = 0
            else:
                cl_dice = -seg_cl_dice.detach().cpu().numpy()

            loss_0_dc_cldc = loss_seg_0 + 0.5 * seg_cl_dice
            loss_seg_deep_dc_cldc = loss_seg_deep + np.sum(seg_loss_weight_deep) * 0.5 * seg_cl_dice

            l = loss_seg_deep_dc_cldc + seg_loss_weight_0 * loss_0_dc_cldc + seg_loss_weight_0 * loss_skel_dc_cldc
            print("loss_seg_deep_dc_cldc, loss_0_dc_cldc, loss_skel_dc_cldc, l: ", loss_seg_deep_dc_cldc, loss_0_dc_cldc, loss_skel_dc_cldc, l)
            
        target = target[0]
        output = seg_output_0
        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard, 'cl_dice': cl_dice}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        mean_fg_cl_dice = np.mean(outputs_collated['cl_dice'])
        
        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                        zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        mean_fg_dc_cldc = 0.5 * mean_fg_dice + 0.5 * mean_fg_cl_dice
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('mean_fg_cl_dice', mean_fg_cl_dice, self.current_epoch)
        self.logger.log('mean_fg_dc_cldc', mean_fg_dc_cldc, self.current_epoch)
        self.logger.log('dice_per_class_or_region',
                        global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
    
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('Pseudo cl dice', np.round(self.logger.my_fantastic_logging['mean_fg_cl_dice'][-1], decimals=4))
        self.print_to_log_file('Pseudo dc_cldc', np.round(self.logger.my_fantastic_logging['mean_fg_dc_cldc'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dc_cldc'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dc_cldc'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            for k in dataset_val.keys():
                proceed = not check_workers_busy(segmentation_export_pool, results,
                                                 allowed_num_queued=2 * len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_busy(segmentation_export_pool, results,
                                                     allowed_num_queued=2 * len(segmentation_export_pool._pool))

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                try:
                    prediction = predictor.predict_sliding_window_return_logits(data)
                except RuntimeError:
                    predictor.perform_everything_on_gpu = False
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    predictor.perform_everything_on_gpu = True

                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


