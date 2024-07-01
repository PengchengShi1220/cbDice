import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC import nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC

class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_5epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_1epoch(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_10epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_20epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 20


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_50epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_100epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100

class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_200epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200

class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_250epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_2000epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

    
class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_4000epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000


class nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC_8000epochs(nnUNetTrainer_SwinUNETR_NoDeepSupervision_CE_DC_CBDC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000
