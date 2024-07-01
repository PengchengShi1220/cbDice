import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision import nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_5iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)
        self.save_every = 100

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_10iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)
        self.save_every = 100

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_20iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)
        self.save_every = 50

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_50iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 50
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)
        self.save_every = 20

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_100iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)
        self.save_every = 10

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_500iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_1000iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 1000
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)

class nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision_2000iterations(nnUNetTrainer_CE_DC_CBDC_NoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 2000
        self.num_val_iterations_per_epoch = int(self.num_iterations_per_epoch * 0.2)
