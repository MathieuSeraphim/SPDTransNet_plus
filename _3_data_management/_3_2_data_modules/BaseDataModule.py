from typing import Optional
from torch import Generator
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from _3_data_management._3_2_data_modules import utils


class BaseDataModule(LightningDataModule):

    def __init__(self, dataset_name: str, batch_size: int, cross_validation_fold_index: int,
                 dataloader_num_workers: int, random_seed: int, alt_fold_folder_name: Optional[str] = None):
        super(BaseDataModule, self).__init__()

        self.training_set = None
        self.validation_set = None
        self.test_set = None

        self.generator = Generator()
        if random_seed is not None:
            self.generator.manual_seed(random_seed)

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.cross_validation_fold_index = cross_validation_fold_index
        self.dataloader_num_workers = dataloader_num_workers
        self.random_seed = random_seed

        self.fold_folder_name = dataset_name
        if alt_fold_folder_name is not None:
            self.fold_folder_name = alt_fold_folder_name

    def train_dataloader(self, shuffled: bool = True):
        assert self.training_set is not None
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=shuffled,
                          num_workers=self.dataloader_num_workers, generator=self.generator)

    def val_dataloader(self):
        assert self.validation_set is not None
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        assert self.test_set is not None
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers)

    def get_cross_validation_recording_indices(self):
        return utils.get_cross_validation_recording_indices(self.fold_folder_name, self.cross_validation_fold_index)
