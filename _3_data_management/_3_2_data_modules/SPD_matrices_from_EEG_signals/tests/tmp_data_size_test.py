import sys

import torch

from _3_data_management._3_2_data_modules.DatasetWrapper import get_dataset_from_config
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset

number_or_recordings = 62

recording_indices = list(range(number_or_recordings))
extra_epochs_on_each_side = 10
signal_preprocessing_strategy = "z_score_normalization"
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]],
                                ["bandpass_filtering", [13, 22]],
                                ["bandpass_filtering", [22, 30]],
                                ["bandpass_filtering", [30, 45]]]
covariance_estimator = "cov"
statistic_vectors_for_matrix_augmentation = ["psd"]
transfer_recording_wise_matrices = True

test_clipping = 24
recordings_per_test_set = 2
test_clipping_amount = 2 * recordings_per_test_set * max(0, (test_clipping - extra_epochs_on_each_side))

sequence_of_epochs_length = 2 * extra_epochs_on_each_side + 1

dataset = get_dataset_from_config("SPD_matrices_from_EEG_MASS_dataset_PredicAlert_signals_config.yaml")
assert type(dataset) == SPDFromEEGDataset
dataset.setup(recording_indices, extra_epochs_on_each_side, signal_preprocessing_strategy, channel_wise_transformations,
              covariance_estimator, statistic_vectors_for_matrix_augmentation, transfer_recording_wise_matrices)

length_of_dataset = len(dataset)
length_of_dataset_with_test_clipping = length_of_dataset - test_clipping_amount

data_dict, data_labels = dataset[0]
matrices = data_dict["matrices"]
statistic_matrices = data_dict["statistic matrices"]
recording_wise_matrices = data_dict["recording-wise matrices"]
recording_mean_statistic_matrices = data_dict["recording mean statistic matrices"]

print(matrices.shape)
print(statistic_matrices.shape)
print(recording_wise_matrices.shape)
print(recording_mean_statistic_matrices.shape)

matrices_weight = (torch.numel(matrices) * matrices.element_size()) // sequence_of_epochs_length
statistic_matrices_weight = (torch.numel(statistic_matrices) * statistic_matrices.element_size()) // sequence_of_epochs_length
recording_wise_matrices_weight = torch.numel(recording_wise_matrices) * recording_wise_matrices.element_size()
recording_mean_statistic_matrices_weight = torch.numel(recording_mean_statistic_matrices) * recording_mean_statistic_matrices.element_size()

per_epoch_data_weight = matrices_weight + statistic_matrices_weight
print(matrices_weight, statistic_matrices_weight, per_epoch_data_weight)

per_recording_data_weight = recording_wise_matrices_weight + recording_mean_statistic_matrices_weight
print(recording_wise_matrices_weight, recording_mean_statistic_matrices_weight, per_recording_data_weight)

total_data_weight = per_epoch_data_weight * length_of_dataset_with_test_clipping + per_recording_data_weight * number_or_recordings
print(length_of_dataset_with_test_clipping, total_data_weight)

