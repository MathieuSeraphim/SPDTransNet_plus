from copy import deepcopy
from typing import Union, Optional
from _1_configs._1_z_miscellaneous.channel_wise_transformations.utils import get_channel_wise_transformations
from _3_data_management._3_2_data_modules.BaseDataModule import BaseDataModule
from _3_data_management._3_2_data_modules.DatasetWrapper import get_dataset_from_config
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset


class SPDFromEEGDataModule(BaseDataModule):

    COMPATIBLE_DATASET_CLASSES = (SPDFromEEGDataset,)

    def __init__(self, dataset_config_file: str, batch_size: int, cross_validation_fold_index: int,
                 extra_epochs_on_each_side: int, signal_preprocessing_strategy: str,
                 channel_wise_transformations_config_file: str, channel_wise_transformations_as_hyphenated_list: str,
                 covariance_estimator: str, statistic_vectors_for_matrix_augmentation_as_hyphenated_list: str,
                 transfer_recording_wise_matrices: bool, rebalance_training_set_by_oversampling: bool = False,
                 clip_test_set_recordings_by_amount: Union[int, None] = None, dataloader_num_workers: int = 0,
                 no_covariances: bool = False, use_recording_wise_simple_covariances: bool = False,
                 get_epoch_eeg_signals: bool = False, random_seed: int = 42,
                 alt_fold_folder_name: Optional[str] = None):
        self.save_hyperparameters(logger=False)

        self.__setup_done_flag = False

        dataset = get_dataset_from_config(dataset_config_file)
        assert isinstance(dataset, self.COMPATIBLE_DATASET_CLASSES)
        self.base_dataset = dataset
        dataset_name = dataset.data_reader.dataset_name

        super(SPDFromEEGDataModule, self).__init__(dataset_name, batch_size, cross_validation_fold_index,
                                                   dataloader_num_workers, random_seed, alt_fold_folder_name)

        statistic_vectors_for_matrix_augmentation\
            = statistic_vectors_for_matrix_augmentation_as_hyphenated_list.split("-")

        channel_wise_transformations_list = channel_wise_transformations_as_hyphenated_list.split("-")
        channel_wise_transformations = get_channel_wise_transformations(channel_wise_transformations_config_file,
                                                                        channel_wise_transformations_list)

        self.dataset_kwargs = {"extra_epochs_on_each_side": extra_epochs_on_each_side,
                               "signal_preprocessing_strategy": signal_preprocessing_strategy,
                               "channel_wise_transformations": channel_wise_transformations,
                               "covariance_estimator": covariance_estimator,
                               "statistic_vectors_for_matrix_augmentation": statistic_vectors_for_matrix_augmentation,
                               "transfer_recording_wise_matrices": transfer_recording_wise_matrices,
                               "no_covariances": no_covariances,
                               "use_recording_wise_simple_covariances": use_recording_wise_simple_covariances,
                               "get_epoch_eeg_signals": get_epoch_eeg_signals,
                               "random_seed": random_seed}
        self.rebalance_training_set_by_oversampling = rebalance_training_set_by_oversampling
        self.clip_test_set_recordings_by_amount = clip_test_set_recordings_by_amount

        self.reinitialize_datasets(cross_validation_fold_index)

    def send_hparams_to_logger(self):
        hparams_dict = dict(self.hparams)

        datamodule_class_path = self.__module__ + "." + self.__class__.__name__
        datamodule_dict = {"class_path": datamodule_class_path, "init_args": hparams_dict}

        output_dict = {"datamodule": datamodule_dict}
        self.hparams.clear()
        self.hparams.update(output_dict)
        self.save_hyperparameters()

    def setup(self, stage: str):
        if not self.__setup_done_flag:
            self.send_hparams_to_logger()
            self.__setup_done_flag = True

    def reinitialize_datasets(self, cross_validation_fold_index: int):
        self.training_set = deepcopy(self.base_dataset)
        self.validation_set = deepcopy(self.base_dataset)
        self.test_set = deepcopy(self.base_dataset)
        self.cross_validation_fold_index = cross_validation_fold_index

        recording_indices_per_set_dict = self.get_cross_validation_recording_indices()

        self.training_set.setup(**self.dataset_kwargs, recording_indices=recording_indices_per_set_dict["training"],
                                rebalance_set_by_oversampling=self.rebalance_training_set_by_oversampling)
        self.validation_set.setup(**self.dataset_kwargs, recording_indices=recording_indices_per_set_dict["validation"])
        self.test_set.setup(**self.dataset_kwargs, recording_indices=recording_indices_per_set_dict["test"],
                            clip_recordings_by_amount=self.clip_test_set_recordings_by_amount)

