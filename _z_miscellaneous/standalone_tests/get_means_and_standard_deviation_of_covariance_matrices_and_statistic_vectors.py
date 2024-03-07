import json
from os.path import dirname, realpath, join

import torch
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset
from _4_models.spd_matrix_powers_and_log import matrix_pow
from _z_miscellaneous.standalone_tests.nested_dicts_and_lists_exploration import pretty_print_recursive_exploration

eeg_signals = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"]
labels = ["N3", "N2", "N1", "REM", "Awake"]
extra_epochs_on_each_side = 0
transfer_recording_wise_matrices = True
data_reader_config_file = "SPD_matrices_from_EEG_config.yaml"
preprocessed_dataset_name = "MASS_SS3_dataset_with_EUSIPCO_signals_config_backup"
clip_recordings_by_amount = None
rebalance_set_by_oversampling = False
recording_indices = list(range(62))
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]],
                                ["bandpass_filtering", [13, 22]],
                                ["bandpass_filtering", [22, 30]],
                                ["bandpass_filtering", [30, 45]]]

signal_preprocessing_strategies_list = ["raw_signals", "z_score_normalization"]
covariance_estimators_list = ["cov", "oas"]
statistic_vectors_to_return_list = [["psd"], ["mean"], ["max_minus_min"]]
use_recording_wise_simple_covariances_list = [True, False]
no_covariances_list = [True, False]


number_of_channels = len(channel_wise_transformations)
matrix_size = len(eeg_signals)
matrices_per_epoch = 30

key_vectors = "For statistic vectors"
key_means = "The average of all item scalars"
key_std = "The standard deviation of all item scalars"

output_dict = {}

for signal_preprocessing_strategy in signal_preprocessing_strategies_list:
    for covariance_estimator in covariance_estimators_list:
        for statistic_vectors_to_return in statistic_vectors_to_return_list:
            for use_recording_wise_simple_covariances in use_recording_wise_simple_covariances_list:
                for no_covariances in no_covariances_list:

                    dataset = SPDFromEEGDataset(eeg_signals, labels, data_reader_config_file)
                    dataset.setup(recording_indices, extra_epochs_on_each_side, signal_preprocessing_strategy,
                                  channel_wise_transformations, covariance_estimator, statistic_vectors_to_return,
                                  transfer_recording_wise_matrices, rebalance_set_by_oversampling,
                                  clip_recordings_by_amount,
                                  use_recording_wise_simple_covariances=use_recording_wise_simple_covariances,
                                  no_covariances=no_covariances)

                    if no_covariances:
                        key_matrices = "For variance matrices (covariances removed)"
                    else:
                        key_matrices = "For covariance matrices"

                    if signal_preprocessing_strategy == "z_score_normalization":
                        key_strategy = "With z-score normalization applied"
                    elif signal_preprocessing_strategy == "raw_signals":
                        key_strategy = "Without normalizing the EEG signals"
                    else:
                        raise ValueError

                    key_matrix_estimator = "Using the %s covariance estimator" % covariance_estimator
                    assert len(statistic_vectors_to_return) == 1
                    key_statistic_vectors = "Computed from the %s statistic" % statistic_vectors_to_return[0]

                    key_no_whitening = "Un-whitened"
                    if use_recording_wise_simple_covariances:
                        key_whitening = "Whitened using recording-wise covariance matrices"
                    else:
                        key_whitening = "Whitened using affine-invariant mean matrices"

                    accumulated_unwhitened_matrices_list = []
                    accumulated_whitened_matrices_list = []
                    accumulated_vectors_list = []
                    for i in range(len(dataset)):
                        data, _ = dataset[i]
                        matrices = data["matrices"]
                        vectors = data["statistic matrices"]
                        whitening_matrices = data["recording-wise matrices"]

                        assert matrices.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, matrix_size)
                        assert vectors.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, 1)
                        assert whitening_matrices.shape == (number_of_channels, matrix_size, matrix_size)

                        accumulated_vectors_list.append(vectors.squeeze(1).squeeze(-1))
                        matrices = matrices.squeeze(1)
                        accumulated_unwhitened_matrices_list.append(matrices)

                        whitening_matrices = matrix_pow(whitening_matrices, -.5)
                        whitening_matrices = whitening_matrices.unsqueeze(-3)
                        whitened_matrices = torch.matmul(whitening_matrices, torch.matmul(matrices, whitening_matrices))

                        assert len(whitening_matrices.shape) == len(matrices.shape) == len(whitened_matrices.shape) == 4
                        accumulated_whitened_matrices_list.append(whitened_matrices)

                    accumulated_unwhitened_matrices = torch.cat(accumulated_unwhitened_matrices_list, dim=1)
                    accumulated_whitened_matrices = torch.cat(accumulated_whitened_matrices_list, dim=1)
                    accumulated_vectors = torch.cat(accumulated_vectors_list, dim=1)

                    if key_strategy not in output_dict.keys():
                        output_dict[key_strategy] = {}

                    for channel_id in range(number_of_channels):

                        key_channel = "Along the channel of ID %d" % channel_id
                        if key_channel not in output_dict[key_strategy].keys():
                            output_dict[key_strategy][key_channel] = {}

                        if key_vectors not in output_dict[key_strategy][key_channel].keys():
                            output_dict[key_strategy][key_channel][key_vectors] = {}

                        if key_statistic_vectors not in output_dict[key_strategy][key_channel][key_vectors].keys():
                            output_dict[key_strategy][key_channel][key_vectors][key_statistic_vectors] = {
                                key_means: torch.mean(accumulated_vectors[channel_id, ...]).item(),
                                key_std: torch.std(accumulated_vectors[channel_id, ...]).item()
                            }

                        if key_matrix_estimator not in output_dict[key_strategy][key_channel].keys():
                            output_dict[key_strategy][key_channel][key_matrix_estimator] = {}

                        if key_matrices not in output_dict[key_strategy][key_channel][key_matrix_estimator].keys():
                            output_dict[key_strategy][key_channel][key_matrix_estimator][key_matrices] = {}

                        if key_no_whitening not in output_dict[key_strategy][key_channel][key_matrix_estimator][key_matrices].keys():
                            output_dict[key_strategy][key_channel][key_matrix_estimator][key_matrices][key_no_whitening] = {
                                key_means: torch.mean(accumulated_unwhitened_matrices[channel_id, ...]).item(),
                                key_std: torch.std(accumulated_unwhitened_matrices[channel_id, ...]).item()
                            }

                        if key_whitening not in output_dict[key_strategy][key_channel][key_matrix_estimator][key_matrices].keys():
                            output_dict[key_strategy][key_channel][key_matrix_estimator][key_matrices][key_whitening] = {
                                key_means: torch.mean(accumulated_whitened_matrices[channel_id, ...]).item(),
                                key_std: torch.std(accumulated_whitened_matrices[channel_id, ...]).item()
                            }

                    pretty_print_recursive_exploration(output_dict, indent=6)

current_script_directory = dirname(realpath(__file__))
output_file = join(current_script_directory, "means_and_standard_deviation_of_covariance_matrices_and_statistic_vectors.json")
with open(output_file, "w") as f:
    json.dump(output_dict, f, indent=6)




