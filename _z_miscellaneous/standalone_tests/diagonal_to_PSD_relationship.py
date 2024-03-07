from os import makedirs
from os.path import dirname, realpath, join, exists
from shutil import rmtree

import numpy as np
import torch
from torch import flatten
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixAugmentationLayer import \
    MatrixAugmentationLayer
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixWhiteningLayer import \
    MatrixWhiteningLayer
from matplotlib import pyplot as plt

dataset_name = "MASS_SS3"
eeg_signals = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"]
labels = ["N3", "N2", "N1", "REM", "Awake"]
extra_epochs_on_each_side = 0
data_reader_config_file = "SPD_matrices_from_EEG_config.yaml"
preprocessed_dataset_name = "MASS_SS3_dataset_with_EUSIPCO_signals_config"
clip_recordings_by_amount = None
rebalance_set_by_oversampling = False
recording_indices = list(range(62))
recording_indices_full_list = [[i] for i in recording_indices] + [recording_indices]
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]],
                                ["bandpass_filtering", [13, 22]],
                                ["bandpass_filtering", [22, 30]],
                                ["bandpass_filtering", [30, 45]]]
signal_preprocessing_strategy = "z_score_normalization"
statistic_vector_name = "psd"
statistic_vectors_to_return = [statistic_vector_name]
no_covariances = False
covariance_estimator = "cov"
extra_dimensions = 1
matrices_per_epoch = 30
number_of_channels = len(channel_wise_transformations)
matrix_size = len(eeg_signals)

whitening_states = {
    "no_whitening": {
        "transfer_recording_wise_matrices": False,
        "use_recording_wise_simple_covariances": False,
        "operate_whitening": False
    },
    "old_whitening": {
        "transfer_recording_wise_matrices": True,
        "use_recording_wise_simple_covariances": True,
        "operate_whitening": True
    },
    "new_whitening": {
        "transfer_recording_wise_matrices": True,
        "use_recording_wise_simple_covariances": False,
        "operate_whitening": True
    }
}

augmentation_factors_list = [2.49, 665.49, None]

# https://colorswall.com/palette/171299
electrode_to_color_list = [
    "#ff6961",  # Red
    "#ffb480",  # Orange
    "#f8f38d",  # Yellow
    "#42d6a4",  # Green
    "#08cad1",  # Cyan
    "#59adf6",  # Blue
    "#9d94ff",  # Purple
    "#c780e8",  # Violet
    "k"  # Black
]

x_name = "Diagonal values"
y_name = "Average PSD"

current_script_directory = dirname(realpath(__file__))

for augmentation_factor in augmentation_factors_list:

    extra_bit = ""
    if augmentation_factor is None:
        augmentation_size = 0
        augmentation_factor = 0
        extra_bit = "_no_augmentation"
    else:
        augmentation_size = 1
        extra_bit = "_augmentation_factor_of_%f" % augmentation_factor

    output_directory = join(current_script_directory, "diagonal_to_PSD_relationship_in_" + dataset_name + extra_bit)
    if not exists(output_directory):
        makedirs(output_directory)

    augmentation_layer = MatrixAugmentationLayer()
    augmentation_flag, augmented_matrix_size = augmentation_layer.setup(matrix_size, augmentation_size, augmentation_factor)

    for whitening_state in whitening_states.keys():
        whitening_state_output_folder = join(output_directory, whitening_state)
        if not exists(whitening_state_output_folder):
            makedirs(whitening_state_output_folder)

        last_channel_folder = join(whitening_state_output_folder, "channel_%d" % (number_of_channels - 1))
        last_channel_combined_signals_file = join(last_channel_folder, "combined_signals.png")
        if exists(last_channel_combined_signals_file):
            continue

        operate_whitening = whitening_states[whitening_state]["operate_whitening"]
        transfer_recording_wise_matrices = whitening_states[whitening_state]["transfer_recording_wise_matrices"]
        use_recording_wise_simple_covariances = whitening_states[whitening_state]["use_recording_wise_simple_covariances"]

        dataset = SPDFromEEGDataset(eeg_signals, labels, data_reader_config_file)
        dataset.setup(recording_indices, extra_epochs_on_each_side, signal_preprocessing_strategy,
                      channel_wise_transformations, covariance_estimator, statistic_vectors_to_return,
                      transfer_recording_wise_matrices, rebalance_set_by_oversampling,
                      clip_recordings_by_amount,
                      use_recording_wise_simple_covariances=use_recording_wise_simple_covariances,
                      no_covariances=no_covariances)

        whitening_layer = MatrixWhiteningLayer()
        whitening_layer.setup(augmented_matrix_size, operate_whitening, extra_dimensions)

        accumulated_matrices_list = []
        accumulated_statistic_vectors_list = []
        for i in range(len(dataset)):
            data, _ = dataset[i]
            matrices = data["matrices"]
            assert matrices.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, matrix_size)
            matrices = matrices.squeeze(1)

            statistic_vectors = data["statistic matrices"]
            assert statistic_vectors.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, 1)
            statistic_vectors = statistic_vectors.squeeze(1)

            whitening_matrices = None
            whitening_statistic_vectors = None
            if operate_whitening:
                whitening_matrices = data["recording-wise matrices"]
                whitening_statistic_vectors = data["recording mean statistic matrices"]
                assert whitening_matrices.shape == (number_of_channels, matrix_size, matrix_size)
                assert whitening_statistic_vectors.shape == (number_of_channels, matrix_size, 1)

                whitening_matrices = augmentation_layer(whitening_matrices, whitening_statistic_vectors)

            matrices = augmentation_layer(matrices, statistic_vectors)

            matrices = whitening_layer(matrices, whitening_matrices)
            accumulated_matrices_list.append(matrices)

            statistic_vectors = statistic_vectors.squeeze(-1)
            if augmentation_flag:
                final_ones = torch.ones(statistic_vectors.shape[:-1] + (1,))
                statistic_vectors = torch.cat((statistic_vectors, final_ones), dim=-1)
                del final_ones

            accumulated_statistic_vectors_list.append(statistic_vectors)

            del data

        del dataset, whitening_layer

        accumulated_statistic_vectors = torch.stack(accumulated_statistic_vectors_list, dim=1)
        accumulated_matrices = torch.stack(accumulated_matrices_list, dim=1)

        matrix_diagonals = accumulated_matrices.diagonal(dim1=-2, dim2=-1)
        assert matrix_diagonals.shape == accumulated_statistic_vectors.shape

        del accumulated_statistic_vectors_list, accumulated_matrices_list, accumulated_matrices

        for channel_id in range(number_of_channels):

            channel_folder = join(whitening_state_output_folder, "channel_%d" % channel_id)
            if not exists(channel_folder):
                makedirs(channel_folder)

            combined_signals_file = join(channel_folder, "combined_signals.png")
            if exists(combined_signals_file):
                continue

            plt.close('all')
            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot()

            for signal_id in range(augmented_matrix_size):
                signal_file = join(channel_folder, "signal_%d.png" % signal_id)

                channel_and_signal_accumulated_statistic_vectors = flatten(accumulated_statistic_vectors[channel_id, ..., signal_id])
                channel_and_signal_matrix_diagonals = flatten(matrix_diagonals[channel_id, ..., signal_id])

                assert len(channel_and_signal_accumulated_statistic_vectors.shape) == len(channel_and_signal_matrix_diagonals.shape) == 1
                assert len(channel_and_signal_accumulated_statistic_vectors) == len(channel_and_signal_matrix_diagonals)

                x = channel_and_signal_matrix_diagonals.numpy()
                y = channel_and_signal_accumulated_statistic_vectors.numpy()
                color = electrode_to_color_list[signal_id]

                ax.scatter(x, y, s=60, alpha=0.7, c=color)
                b, a = np.polyfit(x, y, deg=1)
                xseq = np.linspace(x.min(), x.max(), num=len(x))
                label = "Signal %d - - PSD(i) = %.5E * C(i, i) + %.5E" % (signal_id, b, a)
                ax.plot(xseq, a + b * xseq, color=color, lw=2.5, label=label)

                if exists(signal_file):
                    del x, y, xseq, channel_and_signal_accumulated_statistic_vectors, channel_and_signal_matrix_diagonals
                    continue

                fig1 = plt.figure(figsize=(9, 9))
                ax1 = fig1.add_subplot()
                ax1.scatter(x, y, s=60, alpha=0.7, c=color)
                ax1.plot(xseq, a + b * xseq, color=color, lw=2.5, label=label)
                fig1.suptitle("Channel %d - Signal %d" % (channel_id, signal_id))
                ax1.set_xlabel(x_name)
                ax1.set_ylabel(y_name)
                ax1.legend(loc="upper left")
                fig1.savefig(signal_file)

                del fig1, ax1, x, y, xseq, channel_and_signal_accumulated_statistic_vectors, channel_and_signal_matrix_diagonals

            fig.suptitle("Channel %d - All signals" % channel_id)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.legend(loc="upper left")
            fig.savefig(combined_signals_file)

            del fig, ax

        del matrix_diagonals, accumulated_statistic_vectors

    del augmentation_layer











