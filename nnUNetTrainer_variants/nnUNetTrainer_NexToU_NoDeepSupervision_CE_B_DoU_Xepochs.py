import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU import nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU

class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_5epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_1epoch(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_10epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_20epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 20


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_50epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_100epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_250epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250

class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_500epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_2000epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

    
class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_4000epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU_8000epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_B_DoU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000
