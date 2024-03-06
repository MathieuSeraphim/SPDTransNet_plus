import warnings
from typing import Optional

import torch
from torch.nn import Module
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixAugmentationLayer import \
    MatrixAugmentationLayer
from _4_models._4_2_signal_feature_extractor_models.extractor_model.BaseExtractorModel import BaseExtractorModel


class MatrixAugmentationThroughComputedFeaturesLayer(Module):

    def __init__(self):
        super(MatrixAugmentationThroughComputedFeaturesLayer, self).__init__()
        self.__setup_done_flag = False

        self.augmentation_size = None
        self.matrix_size = None
        self.augmentation_factor = None  # Unused within the class, defined purely for getting this value for logging

        self.feature_extractor_model = None
        self.base_augmentation_layer = MatrixAugmentationLayer()

    def setup(self, matrix_size: int, feature_extractor_model: Optional[BaseExtractorModel],
              initial_augmentation_factor: float = 1., augmentation_factor_learnable: bool = False,
              number_of_augmentation_features_optional_parameter: Optional[int] = None):
        assert not self.__setup_done_flag

        if number_of_augmentation_features_optional_parameter is not None:
            assert number_of_augmentation_features_optional_parameter == int(number_of_augmentation_features_optional_parameter)
            assert number_of_augmentation_features_optional_parameter >= 0

        if feature_extractor_model is None:
            warning_flag = False
            if initial_augmentation_factor != 0:
                warning_flag = True
            if augmentation_factor_learnable:
                warning_flag = True
            initial_augmentation_factor = 0.
            augmentation_factor_learnable = False
            if warning_flag:
                warnings.warn("feature_extractor_model set to None - augmentation factor fixed at 0.")
            if number_of_augmentation_features_optional_parameter is not None:
                self.augmentation_size = number_of_augmentation_features_optional_parameter
            else:
                self.augmentation_size = 0
        else:
            self.augmentation_size = feature_extractor_model.single_output_feature_length
            if number_of_augmentation_features_optional_parameter is not None:
                assert self.augmentation_size == number_of_augmentation_features_optional_parameter

        self.feature_extractor_model = feature_extractor_model

        self.matrix_size = matrix_size
        self.augmentation_factor = initial_augmentation_factor  # Unused within the class

        active_flag, augmented_matrix_size = self.base_augmentation_layer.setup(self.matrix_size,
                                                                                self.augmentation_size,
                                                                                initial_augmentation_factor,
                                                                                augmentation_factor_learnable)

        assert augmented_matrix_size == self.matrix_size + self.augmentation_size

        if not active_flag:
            assert augmented_matrix_size == self.matrix_size
            del self.feature_extractor_model
            self.feature_extractor_model = None

        self.__setup_done_flag = True
        return active_flag, augmented_matrix_size

    # eeg_signals of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_subdivisions_per_epoch, number_of_signals, subdivision_signal_length)
    # OR (number_of_channels, number_of_subdivisions_per_epoch, number_of_signals, subdivision_signal_length)
    # spd_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, matrix_size, matrix_size)
    # OR (number_of_channels, number_of_matrices_per_epoch, matrix_size, matrix_size)
    # output of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, augmented_matrix_size, augmented_matrix_size)
    # OR (number_of_channels, number_of_matrices_per_epoch, augmented_matrix_size, augmented_matrix_size)
    # Here, number_of_signals == matrix_size
    def forward(self, eeg_signals: torch.Tensor, spd_matrices: torch.Tensor):
        assert self.__setup_done_flag

        eeg_signals_shape = eeg_signals.shape
        spd_matrices_shape = spd_matrices.shape
        assert eeg_signals_shape[:-2] == spd_matrices_shape[:-2]
        assert eeg_signals_shape[-2] == spd_matrices_shape[-2] == spd_matrices_shape[-1] == self.matrix_size
        number_of_matrices_per_epoch = spd_matrices_shape[-3]

        assert len(eeg_signals_shape) in (6, 4)  # Cases 1 and 2 below

        if self.feature_extractor_model is not None:
            self.augmentation_factor = self.base_augmentation_layer.augmentation_factor  # In case it's learnable

            # The extractor takes an input of shape
            # (batch_size, number_of_independent_channels, number_of_parallel_channels, signal_length)
            # Here, the parallel channels are the different signals, the independent channels are the actual channels
            # (as they are of a different nature, they should be handled through different weights)
            # and signal_length refers to the entire signal over every subdivision, i.e.
            # number_of_independent_channels -> number_of_channels
            # number_of_parallel_channels -> number_of_signals
            # signal_length -> number_of_subdivisions_per_epoch * subdivision_signal_length = full_signal_length

            # Shape (..., number_of_signals, number_of_subdivisions_per_epoch * subdivision_signal_length = full_signal_length)
            eeg_signals = eeg_signals.transpose(-3, -2).contiguous()
            eeg_signals_shape = eeg_signals.shape
            eeg_signals_shape = list(eeg_signals_shape[:-2]) + [eeg_signals_shape[-2] * eeg_signals_shape[-1]]
            eeg_signals = eeg_signals.view(eeg_signals_shape)

            # Case 1
            # Shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_signals, full_signal_length)
            # becomes (new_batch_size = batch_size * sequences_of_epochs_length, number_of_channels, number_of_signals, full_signal_length)
            if len(eeg_signals_shape) == 5:
                eeg_signals = eeg_signals.transpose(1, 2).contiguous()

                eeg_signals_shape = eeg_signals.shape  # Post transpose, prior to fusing batch size
                eeg_signals_old_batch_size_shape = list(eeg_signals_shape[:2])

                eeg_signals_shape = [eeg_signals_shape[0] * eeg_signals_shape[1]] + list(eeg_signals_shape[2:])
                eeg_signals = eeg_signals.view(eeg_signals_shape)

            # Case 2
            # Shape (number_of_channels, number_of_signals, full_signal_length)
            # becomes (new_batch_size = 1, number_of_channels, number_of_signals, full_signal_length)
            elif len(eeg_signals_shape) == 3:
                eeg_signals = eeg_signals.unsqueeze(0)
                eeg_signals_shape = list(eeg_signals.shape)
                eeg_signals_old_batch_size_shape = []

            else:
                raise ValueError("Dimensions wrong, should be caught by prior assert")

            # Shape (new_batch_size, number_of_channels, number_of_signals = matrix_size, full_signal_length)
            augmentation_features = self.feature_extractor_model(eeg_signals)
            augmentation_features_shape = augmentation_features.shape
            assert augmentation_features_shape[:-1] == tuple(eeg_signals_shape[:-1])
            assert augmentation_features_shape[-1] == number_of_matrices_per_epoch * self.augmentation_size

            # Shape (batch_size, sequences_of_epochs_length, number_of_channels, matrix_size, number_of_matrices_per_epoch * matrix_augmentation_size)
            # OR (number_of_channels, matrix_size, number_of_matrices_per_epoch * matrix_augmentation_size)
            augmentation_features_shape = eeg_signals_old_batch_size_shape + list(augmentation_features_shape)[1:]
            augmentation_features = augmentation_features.view(augmentation_features_shape)

            # Case 1
            # Shape (batch_size, sequences_of_epochs_length, number_of_channels, matrix_size, number_of_matrices_per_epoch * matrix_augmentation_size)
            # becomes (batch_size, number_of_channels, sequences_of_epochs_length, matrix_size, number_of_matrices_per_epoch * matrix_augmentation_size)
            if len(augmentation_features_shape) == 5:
                augmentation_features = augmentation_features.transpose(1, 2).contiguous()
                augmentation_features_shape = list(augmentation_features.shape)

            #case 2
            # Shape remains (number_of_channels, matrix_size, number_of_matrices_per_epoch * matrix_augmentation_size
            else:
                assert len(augmentation_features_shape) == 3

            # shape (..., matrix_size, number_of_matrices_per_epoch, matrix_augmentation_size)
            augmentation_features_shape = augmentation_features_shape[:-1] + [number_of_matrices_per_epoch, self.augmentation_size]
            augmentation_features = augmentation_features.view(augmentation_features_shape)

            # shape (..., number_of_matrices_per_epoch, matrix_size, matrix_augmentation_size)
            augmentation_matrices = augmentation_features.transpose(-3, -2).contiguous()
            augmentation_matrices_shape = augmentation_matrices.shape

            assert augmentation_matrices_shape[:-1] == spd_matrices_shape[:-1]
            assert augmentation_matrices_shape[-1] == self.augmentation_size

        else:  # augmentation_factor set to 0, or no feature_extractor_model
            augmentation_matrices_shape = spd_matrices_shape[:-1] + (self.augmentation_size,)
            augmentation_matrices = torch.zeros(size=augmentation_matrices_shape, dtype=spd_matrices.dtype,
                                                device=spd_matrices.device)

        # shape (..., number_of_matrices_per_epoch, augmented_matrix_size, augmented_matrix_size)
        final_matrices = self.base_augmentation_layer(spd_matrices, augmentation_matrices)
        return final_matrices

