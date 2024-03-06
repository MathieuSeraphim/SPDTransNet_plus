import numpy as np
from torch.utils.data import Dataset
from _3_data_management._3_1_data_readers.DataReaderWrapper import get_data_reader_from_config
from _3_data_management._3_1_data_readers.SPD_matrices_from_EEG_signals.SPDFromEEGDataReader import SPDFromEEGDataReader
from typing import List, Tuple, Any
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


class EEGEpochsDataset(Dataset):

    COMPATIBLE_DATAREADER_CLASS = SPDFromEEGDataReader

    def __init__(self, eeg_signals: List[str], labels: List[str], data_reader_config_file: str):

        self.__setup_done_flag = False

        self.data_reader = get_data_reader_from_config(data_reader_config_file)
        assert isinstance(self.data_reader, self.COMPATIBLE_DATAREADER_CLASS)

        self.data_reader_setup_kwargs = {"eeg_signals": eeg_signals,
                                         "labels": labels,
                                         "statistic_vectors_to_return": [],
                                         "return_epoch_eeg_signals": True,
                                         "return_recording_wise_matrices": "none",
                                         "return_recording_eeg_signals": False}

        self.labels_list = labels
        self.number_of_classes = len(labels)
        assert self.number_of_classes > 0
        self.label_ids_list = list(range(self.number_of_classes))

        self.recording_indices = []
        self.recording_wise_data_list = []
        self.dataset_epochs_global_indices_list = []
        self.dataset_epochs_labels_list = []
        self.dataset_length = -1

    def setup(self, recording_indices: List[int], signal_preprocessing_strategy: str,
              channel_wise_transformations: List[Tuple[str, Any]], obligatory_token_covariance_estimator: str,
              rebalance_set_by_oversampling: bool = False, random_seed: int = 42):
        assert not self.__setup_done_flag

        self.recording_indices = recording_indices
        assert len(self.recording_indices) > 0

        all_epochs_labels_list, recording_wise_data_list = self.data_reader.setup(
            **self.data_reader_setup_kwargs,
            signal_preprocessing_strategy=signal_preprocessing_strategy,
            channel_wise_transformations=channel_wise_transformations,
            covariance_estimator=obligatory_token_covariance_estimator
        )

        self.recording_wise_data_list = [recording_wise_data_list[recording_index] for recording_index in self.recording_indices]

        for recording_wise_data_dict in self.recording_wise_data_list:
            recording_range_start, recording_range_stop = recording_wise_data_dict["epoch ids range"]
            number_of_epochs_in_recording = recording_range_stop - recording_range_start
            assert number_of_epochs_in_recording > 0
            recording_epochs_global_indices_list = list(range(recording_range_start, recording_range_stop))
            recording_epochs_labels_list = all_epochs_labels_list[recording_range_start:recording_range_stop]
            assert len(recording_epochs_global_indices_list) == len(recording_epochs_labels_list) == number_of_epochs_in_recording

            self.dataset_epochs_global_indices_list += recording_epochs_global_indices_list
            self.dataset_epochs_labels_list += recording_epochs_labels_list

        self.dataset_length = len(self.dataset_epochs_global_indices_list)
        assert self.dataset_length == len(self.dataset_epochs_labels_list)

        if rebalance_set_by_oversampling:
            pre_oversampling_elements_per_class_counter = Counter(self.dataset_epochs_labels_list)
            assert sorted(list(pre_oversampling_elements_per_class_counter.keys())) == self.label_ids_list
            pre_oversampling_most_common_class_size = pre_oversampling_elements_per_class_counter.most_common(1)[0][1]

            dataset_epochs_global_indices_list = np.array(self.dataset_epochs_global_indices_list)
            dataset_epochs_global_indices_list = np.expand_dims(dataset_epochs_global_indices_list, axis=1)
            oversampler = RandomOverSampler(sampling_strategy="auto", random_state=random_seed)
            dataset_epochs_global_indices_list, self.dataset_epochs_labels_list = oversampler.fit_resample(dataset_epochs_global_indices_list, self.dataset_epochs_labels_list)

            dataset_epochs_global_indices_list = np.squeeze(dataset_epochs_global_indices_list)
            assert len(dataset_epochs_global_indices_list.shape) == 1
            self.dataset_epochs_global_indices_list = dataset_epochs_global_indices_list.tolist()

            self.dataset_length = len(self.dataset_epochs_global_indices_list)
            post_oversampling_elements_per_class_counter = Counter(self.dataset_epochs_labels_list)
            assert sorted(list(post_oversampling_elements_per_class_counter.keys())) == self.label_ids_list
            assert self.dataset_length == len(self.dataset_epochs_labels_list) == pre_oversampling_most_common_class_size * self.number_of_classes
            for value in post_oversampling_elements_per_class_counter.values():
                assert value == pre_oversampling_most_common_class_size

        self.__setup_done_flag = True

    def __len__(self):
        assert self.__setup_done_flag
        return self.dataset_length

    def __getitem__(self, item):
        assert self.__setup_done_flag
        assert item < self.__len__()
        output_dict = {}

        epoch_global_index = self.dataset_epochs_global_indices_list[item]
        epoch_data_dict = self.data_reader.get_element_data(epoch_global_index)

        epoch_label_id = epoch_data_dict["label id"]
        epoch_label = epoch_data_dict["label"]
        assert epoch_label in self.labels_list
        assert epoch_label_id == self.labels_list.index(epoch_label)

        output_dict["sampling frequency"] = epoch_data_dict["sampling frequency"]

        # shape (channels, subdivisions_per_epoch, signals, subdivision_signal_length)
        eeg_signals = epoch_data_dict["EEG signals"]

        # shape (channels, signals, subdivisions_per_epoch, subdivision_signal_length)
        output_dict["EEG signals"] = eeg_signals.transpose(1, 2)

        output_dict["epoch id in recording"] = epoch_data_dict["epoch id in recording"]
        output_dict["recording id"] = epoch_data_dict["recording id"]

        return output_dict, epoch_label_id



















