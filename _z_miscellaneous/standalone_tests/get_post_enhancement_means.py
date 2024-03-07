from os import mkdir, listdir
from os.path import dirname, realpath, join, isdir
import seaborn as seaborn
import torch
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals.utils import \
    batch_spd_matrices_affine_invariant_mean
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset
from _3_data_management._3_2_data_modules.utils import get_cross_validation_recording_indices
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixAugmentationLayer import \
    MatrixAugmentationLayer
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixWhiteningLayer import \
    MatrixWhiteningLayer
from matplotlib import pyplot as plt

from _5_execution.run_model import get_model_dict_and_checkpoint_file_for_transfer_learning
from _5_execution.utils import get_model_from_checkpoint_file

current_script_directory = dirname(realpath(__file__))
root_directory = dirname(dirname(current_script_directory))

eeg_signals = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"]
labels = ["N3", "N2", "N1", "REM", "Awake"]
extra_epochs_on_each_side = 0
transfer_recording_wise_matrices = True
dataset_name = "MASS_SS3"
data_reader_config_file = "EEG_epochs_and_corresponding_SPD_matrices_config.yaml"
preprocessed_dataset_name = "MASS_SS3_dataset_with_EUSIPCO_signals_config"
clip_recordings_by_amount = None
rebalance_set_by_oversampling = False
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]],
                                ["bandpass_filtering", [13, 22]],
                                ["bandpass_filtering", [22, 30]],
                                ["bandpass_filtering", [30, 45]]]
signal_preprocessing_strategy = "z_score_normalization"
covariance_estimator = "cov"
statistic_vectors_to_return = ["psd"]
correct_non_spd_matrices = True
impose_minimal_eigenvalue = None
get_epoch_eeg_signals = True

number_of_recordings = 62
recording_indices_list = list(range(number_of_recordings))
handcrafted_features_DAW_augmentation_factor = 17.63068728
handcrafted_features_MAW_augmentation_factor = 2.490444574183345

number_of_folds = 31
recording_index_to_test_set_fold_index_dict = {}
for fold_index in range(number_of_folds):
    fold_dict = get_cross_validation_recording_indices(dataset_name, fold_index)
    for test_recording_index in fold_dict["test"]:
        recording_index_to_test_set_fold_index_dict[test_recording_index] = fold_index
assert sorted(list(recording_index_to_test_set_fold_index_dict.keys())) == recording_indices_list

learned_features_lightning_logs_folder = "lightning_logs_all_folds_final_WPA_learned_augmentation"
run_to_fold_index_offset = 200
learned_features_lightning_logs_folder_path = join(root_directory, learned_features_lightning_logs_folder)
learned_features_lightning_logs_subfolders = [subfolder_name
                                              for subfolder_name in listdir(learned_features_lightning_logs_folder_path)
                                              if subfolder_name[:8] == "version_"]
learned_features_lightning_logs_subfolders_dict = {}
for learned_features_lightning_logs_subfolder in learned_features_lightning_logs_subfolders:
    run_id = int(learned_features_lightning_logs_subfolder.split("_")[-1])
    fold_index = run_id - run_to_fold_index_offset
    learned_features_lightning_logs_subfolders_dict[fold_index] = learned_features_lightning_logs_subfolder
assert sorted(list(learned_features_lightning_logs_subfolders_dict.keys())) == list(range(number_of_folds))


number_of_channels = len(channel_wise_transformations)
matrix_size = len(eeg_signals)
non_WPA_augmentation_size = 1
non_WPA_augmented_matrix_size = matrix_size + non_WPA_augmentation_size
WPA_augmentation_size = 3
WPA_augmented_matrix_size = matrix_size + WPA_augmentation_size
matrices_per_epoch = 30
whitening_extra_dimensions = 1

handcrafted_features_DAW_augmentation_layer = MatrixAugmentationLayer()
handcrafted_features_DAW_augmentation_layer.setup(matrix_size=matrix_size, augmentation_size=non_WPA_augmentation_size,
                                                  initial_augmentation_factor=handcrafted_features_DAW_augmentation_factor,
                                                  augmentation_factor_learnable=False)

handcrafted_features_MAW_augmentation_layer = MatrixAugmentationLayer()
handcrafted_features_MAW_augmentation_layer.setup(matrix_size=matrix_size, augmentation_size=non_WPA_augmentation_size,
                                                  initial_augmentation_factor=handcrafted_features_MAW_augmentation_factor,
                                                  augmentation_factor_learnable=False)

whitening_layer_1 = MatrixWhiteningLayer()
whitening_layer_1.setup(matrix_size=matrix_size, extra_dimensions=whitening_extra_dimensions)

whitening_layer_2 = MatrixWhiteningLayer()
whitening_layer_2.setup(matrix_size=non_WPA_augmented_matrix_size, extra_dimensions=whitening_extra_dimensions)

generic_filename = "mean_matrix_for_channel_%d_with_%s_enrichment_%s_augmentation_features_and_alpha_at_%.2E.png"
generic_title = "Recording-wise affine-invariant mean matrix, for channel %d, %s enrichment (using %s augmentation features) and alpha = %.2E"

for recording_index in recording_indices_list:
    dataset = SPDFromEEGDataset(eeg_signals, labels, data_reader_config_file)
    dataset.setup([recording_index], extra_epochs_on_each_side, signal_preprocessing_strategy,
                  channel_wise_transformations, covariance_estimator, statistic_vectors_to_return,
                  transfer_recording_wise_matrices, rebalance_set_by_oversampling,
                  clip_recordings_by_amount, get_epoch_eeg_signals=get_epoch_eeg_signals)
    number_of_epochs = len(dataset)
    total_matrices_per_channel = number_of_epochs * matrices_per_epoch

    folder_name = join(current_script_directory, "recording_%d" % recording_index)
    if not isdir(folder_name):
        mkdir(folder_name)
        
    cleaned_version_folder_name = join(folder_name, "clean_version")
    if not isdir(cleaned_version_folder_name):
        mkdir(cleaned_version_folder_name)

    recording_fold_index = recording_index_to_test_set_fold_index_dict[recording_index]
    model_from_checkpoint_dict = {
        "pretrained_logs_folder_name": learned_features_lightning_logs_folder,
        "pretrained_logs_subfolder_name": learned_features_lightning_logs_subfolders_dict[recording_fold_index],
        "pretrained_logs_folder_absolute_location": root_directory
    }
    model_config_dict, model_checkpoint_file\
        = get_model_dict_and_checkpoint_file_for_transfer_learning(model_from_checkpoint_dict, {})
    learned_augmentation_model = get_model_from_checkpoint_file(model_config_dict, model_checkpoint_file)
    learned_augmentation_model.freeze()
    learned_features_WPA_augmentation_layer = learned_augmentation_model.data_formatting_block.augmentation_layer
    learned_features_WPA_augmentation_factor = learned_features_WPA_augmentation_layer.augmentation_factor

    augmented_unwhitened_matrices_for_DAW_list = []
    MAW_matrices_list = []
    WPA_matrices_list = []

    for i in range(len(dataset)):
        data, _ = dataset[i]
        matrices = data["matrices"]
        vectors = data["statistic matrices"]
        whitening_matrices = data["recording-wise matrices"]
        mean_vectors = data["recording mean statistic matrices"]
        eeg_signals = data["EEG signals"]

        assert matrices.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, matrix_size)
        assert vectors.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, non_WPA_augmentation_size)
        assert eeg_signals.shape[:-1] == (number_of_channels, 1, matrices_per_epoch, matrix_size)
        assert whitening_matrices.shape == (number_of_channels, matrix_size, matrix_size)
        assert mean_vectors.shape == (number_of_channels, matrix_size, non_WPA_augmentation_size)

        matrices = matrices.squeeze(1)
        vectors = vectors.squeeze(1)
        eeg_signals = eeg_signals.squeeze(1)

        unaugmented_whitened_matrices = whitening_layer_1.forward(matrices, whitening_matrices)
        WPA_matrices = learned_features_WPA_augmentation_layer.forward(eeg_signals, unaugmented_whitened_matrices)

        augmented_unwhitened_matrices_for_DAW = handcrafted_features_DAW_augmentation_layer.forward(matrices, vectors)

        augmented_unwhitened_matrices_for_MAW = handcrafted_features_MAW_augmentation_layer.forward(matrices, vectors)
        augmented_whitening_matrices_for_MAW = handcrafted_features_MAW_augmentation_layer.forward(whitening_matrices, mean_vectors)
        MAW_matrices = whitening_layer_2.forward(augmented_unwhitened_matrices_for_MAW, augmented_whitening_matrices_for_MAW)

        assert augmented_unwhitened_matrices_for_DAW.shape == MAW_matrices.shape\
               == (number_of_channels, matrices_per_epoch, non_WPA_augmented_matrix_size, non_WPA_augmented_matrix_size)

        augmented_unwhitened_matrices_for_DAW_list.append(augmented_unwhitened_matrices_for_DAW)
        MAW_matrices_list.append(MAW_matrices)
        WPA_matrices_list.append(WPA_matrices)

    # shape (number_of_channels, total_matrices_per_channel, augmented_matrix_size, augmented_matrix_size)
    all_augmented_unwhitened_matrices_for_DAW = torch.cat(augmented_unwhitened_matrices_for_DAW_list, dim=1)
    all_MAW_matrices = torch.cat(MAW_matrices_list, dim=1)
    all_WPA_matrices = torch.cat(WPA_matrices_list, dim=1)
    assert all_augmented_unwhitened_matrices_for_DAW.shape == all_MAW_matrices.shape\
        == (number_of_channels, total_matrices_per_channel, non_WPA_augmented_matrix_size, non_WPA_augmented_matrix_size)

    assert all_WPA_matrices.shape == (number_of_channels, total_matrices_per_channel, WPA_augmented_matrix_size,
                                      WPA_augmented_matrix_size)

    for channel_id in range(len(channel_wise_transformations)):
        channel_augmented_unwhitened_matrices_for_DAW = all_augmented_unwhitened_matrices_for_DAW[channel_id, ...]
        channel_MAW_matrices = all_MAW_matrices[channel_id, ...]
        channel_WPA_matrices = all_WPA_matrices[channel_id, ...]

        channel_DAW_whitening_matrix = batch_spd_matrices_affine_invariant_mean(
            channel_augmented_unwhitened_matrices_for_DAW.numpy(),
            correct_non_spd_matrices=correct_non_spd_matrices,
            impose_minimal_eigenvalue=impose_minimal_eigenvalue)
        channel_MAW_mean_matrix = batch_spd_matrices_affine_invariant_mean(
            channel_MAW_matrices.numpy(),
            correct_non_spd_matrices=correct_non_spd_matrices,
            impose_minimal_eigenvalue=impose_minimal_eigenvalue)
        channel_WPA_mean_matrix = batch_spd_matrices_affine_invariant_mean(
            channel_WPA_matrices.numpy(),
            correct_non_spd_matrices=correct_non_spd_matrices,
            impose_minimal_eigenvalue=impose_minimal_eigenvalue)

        channel_DAW_matrices = whitening_layer_2.forward(channel_augmented_unwhitened_matrices_for_DAW,
                                                         torch.tensor(channel_DAW_whitening_matrix,
                                                                      dtype=channel_augmented_unwhitened_matrices_for_DAW.dtype))
        channel_DAW_mean_matrix = batch_spd_matrices_affine_invariant_mean(
            channel_DAW_matrices.numpy(),
            correct_non_spd_matrices=correct_non_spd_matrices,
            impose_minimal_eigenvalue=impose_minimal_eigenvalue)
        assert channel_DAW_mean_matrix.shape == (non_WPA_augmented_matrix_size, non_WPA_augmented_matrix_size)

        channel_DAW_filename = generic_filename % (channel_id, "DAW", "handcrafted", handcrafted_features_DAW_augmentation_factor)
        channel_DAW_cleaned_version_filename = join(cleaned_version_folder_name, channel_DAW_filename)
        channel_DAW_filename = join(folder_name, channel_DAW_filename)
        channel_DAW_title = generic_title % (channel_id, "DAW", "handcrafted", handcrafted_features_DAW_augmentation_factor)

        channel_MAW_filename = generic_filename % (channel_id, "MAW", "handcrafted", handcrafted_features_MAW_augmentation_factor)
        channel_MAW_cleaned_version_filename = join(cleaned_version_folder_name, channel_MAW_filename)
        channel_MAW_filename = join(folder_name, channel_MAW_filename)
        channel_MAW_title = generic_title % (channel_id, "MAW", "handcrafted", handcrafted_features_MAW_augmentation_factor)

        channel_WPA_filename = generic_filename % (channel_id, "WPA", "learned", learned_features_WPA_augmentation_factor)
        channel_WPA_cleaned_version_filename = join(cleaned_version_folder_name, channel_WPA_filename)
        channel_WPA_filename = join(folder_name, channel_WPA_filename)
        channel_WPA_title = generic_title % (channel_id, "WPA", "learned", learned_features_WPA_augmentation_factor)

        plt.figure(figsize=(10, 9))
        figure = seaborn.heatmap(channel_DAW_mean_matrix, annot=True, square=True, fmt=".1g").get_figure()
        figure.suptitle(channel_DAW_title)
        figure.savefig(channel_DAW_filename)
        plt.close("all")  # Might be useful, or not

        plt.figure(figsize=(10, 9))
        figure = seaborn.heatmap(channel_MAW_mean_matrix, annot=True, square=True, fmt=".1g").get_figure()
        figure.suptitle(channel_MAW_title)
        figure.savefig(channel_MAW_filename)
        plt.close("all")  # Might be useful, or not

        plt.figure(figsize=(12, 11))
        figure = seaborn.heatmap(channel_WPA_mean_matrix, annot=True, square=True, fmt=".1g").get_figure()
        figure.suptitle(channel_WPA_title)
        figure.savefig(channel_WPA_filename)
        plt.close("all")  # Might be useful, or not

        plt.figure(figsize=(9, 9))
        figure = seaborn.heatmap(channel_DAW_mean_matrix, annot=True, cbar=False, square=True, xticklabels=False,
                                 yticklabels=False, fmt=".1g").get_figure()
        figure.savefig(channel_DAW_cleaned_version_filename, bbox_inches="tight", pad_inches=0.01)
        plt.close("all")  # Might be useful, or not

        plt.figure(figsize=(9, 9))
        figure = seaborn.heatmap(channel_MAW_mean_matrix, annot=True, cbar=False, square=True, xticklabels=False,
                                 yticklabels=False, fmt=".1g").get_figure()
        figure.savefig(channel_MAW_cleaned_version_filename, bbox_inches="tight", pad_inches=0.01)
        plt.close("all")  # Might be useful, or not

        plt.figure(figsize=(11, 11))
        figure = seaborn.heatmap(channel_WPA_mean_matrix, annot=True, cbar=False, square=True, xticklabels=False,
                                 yticklabels=False, fmt=".1g").get_figure()
        figure.savefig(channel_WPA_cleaned_version_filename, bbox_inches="tight", pad_inches=0.01)
        plt.close("all")  # Might be useful, or not







