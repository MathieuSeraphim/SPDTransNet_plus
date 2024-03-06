from os.path import realpath, dirname, join, isfile
import torch
from jsonargparse import ArgumentParser
from _3_data_management._3_1_data_readers.DataReaderWrapper import DataReaderWrapper

current_script_directory = dirname(realpath(__file__))
root_directory = dirname(dirname(dirname(current_script_directory)))
configs_directory = join(root_directory, "_1_configs")
data_reading_configs_directory = join(configs_directory, "_1_2_data_reading")
data_reading_file = join(data_reading_configs_directory, "SPD_matrices_from_EEG_config.yaml")
assert isfile(data_reading_file)

parser = ArgumentParser()
parser.add_class_arguments(DataReaderWrapper, "wrapper")
data_reader_config = parser.parse_path(data_reading_file)
constructed_data_reader = parser.instantiate_classes(data_reader_config).wrapper.data_reader

eeg_signals = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"]
labels = ["N3", "N2", "N1", "REM", "Awake"]
signal_preprocessing_strategies = ["z_score_normalization"]
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [30, 45]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]]]
covariance_estimator = "cov"
statistic_vectors_to_return = ["mean", "psd"]
return_epoch_eeg_signals = False
return_recording_wise_matrices_list = ["affine_invariant_mean", "simple_covariance"]
return_recording_eeg_signals = False

number_of_signals = len(eeg_signals)
number_of_channels = len(channel_wise_transformations)
subdivisions_per_epoch = 30

for signal_preprocessing_strategy in signal_preprocessing_strategies:
    for return_recording_wise_matrices in return_recording_wise_matrices_list:
        all_epochs_labels_list, recording_wise_data_list = constructed_data_reader.setup(
            eeg_signals, labels, signal_preprocessing_strategy, channel_wise_transformations, covariance_estimator,
            statistic_vectors_to_return, return_epoch_eeg_signals, return_recording_wise_matrices,
            return_recording_eeg_signals)

        assert constructed_data_reader.output_signal_preprocessing_strategy == signal_preprocessing_strategy
        assert constructed_data_reader.output_list_of_channel_transformation_indices == [0, 1, 6, 2, 3]
        assert constructed_data_reader.output_covariance_estimator == covariance_estimator
        assert constructed_data_reader.output_statistic_vectors
        assert constructed_data_reader.output_statistic_vectors_list == statistic_vectors_to_return
        assert not constructed_data_reader.output_epoch_eeg_signals
        assert constructed_data_reader.setup_output_recording_mean_matrices ^  constructed_data_reader.setup_output_recording_covariance_matrices
        assert constructed_data_reader.setup_output_recording_mean_statistic_vectors
        assert not constructed_data_reader.setup_output_recording_eeg_signals

        epochs_counter = 0
        for recording_wise_data_dict in recording_wise_data_list:
            epochs_counter += recording_wise_data_dict["epoch ids range"][1] - recording_wise_data_dict["epoch ids range"][0]
            assert isinstance(recording_wise_data_dict["recording-wise matrices"], torch.Tensor)
            assert recording_wise_data_dict["recording-wise matrices"].shape == (number_of_channels, number_of_signals, number_of_signals)
            assert len(recording_wise_data_dict["mean statistic vectors"].keys()) == len(statistic_vectors_to_return)
            for stat in statistic_vectors_to_return:
                assert isinstance(recording_wise_data_dict["mean statistic vectors"][stat], torch.Tensor)
                assert recording_wise_data_dict["mean statistic vectors"][stat].shape == (number_of_channels, number_of_signals)
        assert epochs_counter == len(all_epochs_labels_list)

        epoch_data_dict = constructed_data_reader.get_element_data(0)

        assert isinstance(epoch_data_dict["matrices"], torch.Tensor)
        assert epoch_data_dict["matrices"].shape == (number_of_channels, subdivisions_per_epoch, number_of_signals, number_of_signals)
        assert len(epoch_data_dict["statistic vectors"].keys()) == len(statistic_vectors_to_return)
        for stat in statistic_vectors_to_return:
            assert isinstance(epoch_data_dict["statistic vectors"][stat], torch.Tensor)
            assert epoch_data_dict["statistic vectors"][stat].shape == (number_of_channels, subdivisions_per_epoch, number_of_signals)

    
    
