import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC import nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC

class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_5epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_1epoch(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_10epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_20epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 20


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_50epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_100epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_250epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_2000epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

    
class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_4000epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000


class nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC_8000epochs(nnUNetTrainer_NexToU_NoDeepSupervision_CE_DC_CL_M_DC):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000
