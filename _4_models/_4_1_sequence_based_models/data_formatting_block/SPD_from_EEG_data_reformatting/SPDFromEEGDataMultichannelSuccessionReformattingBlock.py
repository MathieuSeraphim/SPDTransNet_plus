from typing import Union
import torch
from torch.nn import Linear
from _4_models._4_1_sequence_based_models.data_formatting_block.BaseDataFormattingBlock import BaseDataFormattingBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixAugmentationLayer import \
    MatrixAugmentationLayer
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixVectorizationLayer import \
    MatrixVectorizationLayer
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixWhiteningLayer import \
    MatrixWhiteningLayer
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.VectorChannelWiseCombinationBySuccessionLayer import \
    VectorChannelWiseCombinationBySuccessionLayer


class SPDFromEEGDataMultichannelSuccessionReformattingBlock(BaseDataFormattingBlock):

    EXTRA_DIMENSIONS_COMPARED_TO_SINGLE_MATRICES = 2
    INDEX_OF_CHANNEL_DIMENSION = 1

    def __init__(self):
        super(SPDFromEEGDataMultichannelSuccessionReformattingBlock, self).__init__()
        self.__setup_done_flag = False

        self.augmentation_layer = MatrixAugmentationLayer()
        self.whitening_layer = MatrixWhiteningLayer()
        self.vectorization_layer = MatrixVectorizationLayer()
        self.combination_layer = VectorChannelWiseCombinationBySuccessionLayer()
        self.final_linear_projection_layer = None

        self.operate_matrix_augmentation = False
        self.operate_whitening = False

        self.number_of_vectors_per_epoch_post_combination = None
        self.number_of_channels = None
        self.vector_size = None
        self.final_vector_size = None

    def setup(self, original_matrix_size: int, initial_number_of_matrices_per_epoch: int,  number_of_channels: int,
              augmentation_size: int, initial_augmentation_factor: float = 1.,
              augmentation_factor_learnable: bool = False, operate_whitening: bool = False,
              matrix_multiplication_factor: float = 1., singular_or_eigen_value_minimum: Union[float, None] = None,
              final_linear_projection_to_given_vector_size: Union[int, None] = None,
              decomposition_operator: str = "svd"):
        assert not self.__setup_done_flag

        self.number_of_channels = number_of_channels

        self.operate_matrix_augmentation, augmented_matrix_size = self.augmentation_layer.setup(
            original_matrix_size, augmentation_size, initial_augmentation_factor, augmentation_factor_learnable)

        self.whitening_layer.setup(augmented_matrix_size, operate_whitening,
                                   extra_dimensions=self.EXTRA_DIMENSIONS_COMPARED_TO_SINGLE_MATRICES,
                                   matrix_multiplication_factor=matrix_multiplication_factor,
                                   decomposition_operator=decomposition_operator)
        self.operate_whitening = operate_whitening

        self.vector_size = self.vectorization_layer.setup(augmented_matrix_size, singular_or_eigen_value_minimum,
                                                          decomposition_operator)

        self.number_of_vectors_per_epoch_post_combination = self.combination_layer.setup(
            initial_number_of_matrices_per_epoch, self.vector_size, self.number_of_channels,
            self.INDEX_OF_CHANNEL_DIMENSION)

        if final_linear_projection_to_given_vector_size is None:
            self.final_vector_size = self.vector_size
        else:
            self.final_vector_size = final_linear_projection_to_given_vector_size
            self.final_linear_projection_layer = Linear(self.vector_size, self.final_vector_size)

        self.__setup_done_flag = True
        return self.final_vector_size, self.number_of_vectors_per_epoch_post_combination

    # sequence_of_spd_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, matrix_size, matrix_size)
    # sequence_of_augmentation_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_vectors_per_epoch, matrix_size, matrix_augmentation_size)
    # mean_recording_spd_matrices of shape (batch_size, number_of_channels, matrix_size, matrix_size)
    # mean_recording_augmentation_matrices of shape (batch_size, number_of_channels, matrix_size, matrix_augmentation_size)
    # output of shape (batch_size, sequences_of_epochs_length, number_of_matrices_per_epoch * number_of_channels, vector_size)
    def forward(self, sequence_of_spd_matrices: torch.Tensor,
                sequence_of_augmentation_matrices: Union[torch.Tensor, None] = None,
                mean_recording_spd_matrices: Union[torch.Tensor, None] = None,
                mean_recording_augmentation_matrices: Union[torch.Tensor, None] = None):
        assert self.__setup_done_flag

        input_shape = sequence_of_spd_matrices.shape
        assert input_shape[self.INDEX_OF_CHANNEL_DIMENSION] == self.number_of_channels

        if self.operate_matrix_augmentation:
            assert sequence_of_augmentation_matrices is not None and not torch.isnan(sequence_of_augmentation_matrices).any()
            sequence_of_spd_matrices = self.augmentation_layer(sequence_of_spd_matrices,
                                                               sequence_of_augmentation_matrices)
            if self.operate_whitening:
                assert mean_recording_spd_matrices is not None and not torch.isnan(mean_recording_spd_matrices).any()
                assert mean_recording_augmentation_matrices is not None and not torch.isnan(mean_recording_augmentation_matrices).any()
                mean_recording_spd_matrices = self.augmentation_layer(mean_recording_spd_matrices,
                                                                      mean_recording_augmentation_matrices)

        if self.operate_whitening:
            assert mean_recording_spd_matrices is not None and not torch.isnan(mean_recording_spd_matrices).any()
            sequence_of_spd_matrices = self.whitening_layer(sequence_of_spd_matrices, mean_recording_spd_matrices)

        # (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, vector_size)
        sequence_of_vectorized_matrices = self.vectorization_layer(sequence_of_spd_matrices)
        assert sequence_of_vectorized_matrices.shape[self.INDEX_OF_CHANNEL_DIMENSION] == self.number_of_channels
        assert input_shape[:-2] == sequence_of_vectorized_matrices.shape[:-1]
        assert sequence_of_vectorized_matrices.shape[-1] == self.vector_size

        # # (batch_size, sequences_of_epochs_length, number_of_matrices_per_epoch * number_of_channels, vector_size)
        sequence_of_vectorized_matrices_with_channels_reorganized_in_succession = self.combination_layer(
            sequence_of_vectorized_matrices)
        assert len(sequence_of_vectorized_matrices_with_channels_reorganized_in_succession.shape)\
               == len(sequence_of_vectorized_matrices.shape) - 1
        assert sequence_of_vectorized_matrices_with_channels_reorganized_in_succession.shape[-1] == self.vector_size
        assert sequence_of_vectorized_matrices_with_channels_reorganized_in_succession.shape[-2]\
               == self.number_of_vectors_per_epoch_post_combination

        output_tensor = sequence_of_vectorized_matrices_with_channels_reorganized_in_succession
        if self.final_linear_projection_layer is not None:
            output_tensor = self.final_linear_projection_layer(output_tensor)
            assert output_tensor.shape[-1] == self.final_vector_size

        return output_tensor



