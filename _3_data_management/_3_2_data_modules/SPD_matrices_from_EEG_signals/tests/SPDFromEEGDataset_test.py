import torch
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals.utils import \
    batch_remove_non_diagonal_elements_from_matrices
from _3_data_management._3_2_data_modules.DatasetWrapper import get_dataset_from_config
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.SPDFromEEGDataset import SPDFromEEGDataset
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.tests.utils_for_testing import \
    get_epochs_for_recordings_in_preprocessed_dataset

recording_indices_and_rebalance_set_by_oversampling_list = [(list(range(62)), False), (list(range(62)), True), ([10, 4], True)]
eeg_signals = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"]
labels = ["N3", "N2", "N1", "REM", "Awake"]
extra_epochs_on_each_side_list = [0, 5, 10]
signal_preprocessing_strategy = "z_score_normalization"
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [30, 45]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]]]
covariance_estimator = "cov"
statistic_vectors_to_return_list = [["mean", "psd"], []]
transfer_recording_wise_matrices = True
data_reader_config_file = "SPD_matrices_from_EEG_config.yaml"
preprocessed_dataset_name = "MASS_SS3_dataset_with_PredicAlert_signals_config"
clip_recordings_by_amount_list = [None, 24]

number_of_signals = len(eeg_signals)
number_of_channels = len(channel_wise_transformations)
subdivisions_per_epoch = 30

for recording_indices, rebalance_set_by_oversampling in recording_indices_and_rebalance_set_by_oversampling_list:
    for extra_epochs_on_each_side in extra_epochs_on_each_side_list:
        for statistic_vectors_to_return in statistic_vectors_to_return_list:
            for clip_recordings_by_amount in clip_recordings_by_amount_list:
                dataset = SPDFromEEGDataset(eeg_signals, labels, data_reader_config_file)
                dataset.setup(recording_indices, extra_epochs_on_each_side, signal_preprocessing_strategy,
                              channel_wise_transformations, covariance_estimator, statistic_vectors_to_return,
                              transfer_recording_wise_matrices, rebalance_set_by_oversampling,
                              clip_recordings_by_amount)

                clipping = extra_epochs_on_each_side
                if clip_recordings_by_amount is not None:
                    clipping = clip_recordings_by_amount
                expected_data_distribution_dict = get_epochs_for_recordings_in_preprocessed_dataset(
                    preprocessed_dataset_name, recording_indices, clipping)
                if not rebalance_set_by_oversampling:
                    expected_dataset_length = expected_data_distribution_dict["total"]
                else:
                    expected_dataset_length = expected_data_distribution_dict["N2"] * 5
                assert len(dataset) == expected_dataset_length

                sequences_of_epochs_length = 1 + 2*extra_epochs_on_each_side
                data_dict, data_labels = dataset[0]

                assert isinstance(data_labels, torch.Tensor)
                assert data_labels.shape == (sequences_of_epochs_length,)
                assert isinstance(data_dict["matrices"], torch.Tensor)
                assert data_dict["matrices"].shape == (number_of_channels, sequences_of_epochs_length, subdivisions_per_epoch, number_of_signals, number_of_signals)
                assert isinstance(data_dict["recording-wise matrices"], torch.Tensor)
                assert data_dict["recording-wise matrices"].shape == (number_of_channels, number_of_signals, number_of_signals)

                if not statistic_vectors_to_return:
                    assert "statistic matrices" not in data_dict.keys()
                    assert "recording mean statistic matrices" not in data_dict.keys()

                else:
                    assert isinstance(data_dict["statistic matrices"], torch.Tensor)
                    assert data_dict["statistic matrices"].shape == (number_of_channels, sequences_of_epochs_length, subdivisions_per_epoch, number_of_signals, len(statistic_vectors_to_return))
                    assert isinstance(data_dict["recording mean statistic matrices"], torch.Tensor)
                    assert data_dict["recording mean statistic matrices"].shape == (number_of_channels, number_of_signals, len(statistic_vectors_to_return))


dataset = get_dataset_from_config("SPD_matrices_from_EEG_MASS_dataset_PredicAlert_signals_config.yaml")
assert type(dataset) == SPDFromEEGDataset

dataset.setup([10, 4], 10, signal_preprocessing_strategy, channel_wise_transformations, covariance_estimator, [],
              transfer_recording_wise_matrices, False, None, no_covariances=True)
data_dict, data_labels = dataset[0]
matrices = data_dict["matrices"].numpy()
assert (matrices == batch_remove_non_diagonal_elements_from_matrices(matrices)).all()
recording_matrices = data_dict["recording-wise matrices"].numpy()
assert (recording_matrices == batch_remove_non_diagonal_elements_from_matrices(recording_matrices)).all()
