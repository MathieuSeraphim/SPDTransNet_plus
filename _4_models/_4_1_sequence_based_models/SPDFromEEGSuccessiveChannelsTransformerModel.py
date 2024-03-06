import warnings
from typing import List, Union, Dict, Any
import torch
from _4_models._4_1_sequence_based_models.SequenceToClassificationBaseModel import SequenceToClassificationBaseModel
from _4_models._4_1_sequence_based_models.classification_block.BaseClassificationBlock import BaseClassificationBlock
from _4_models._4_1_sequence_based_models.classification_block.classification_from_sequence_central_feature.CentralGroupOfFeaturesInSequenceClassificationBlock import \
    CentralGroupOfFeaturesInSequenceClassificationBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.BaseDataFormattingBlock import BaseDataFormattingBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.SPDFromEEGDataMultichannelSuccessionReformattingBlock import \
    SPDFromEEGDataMultichannelSuccessionReformattingBlock
from _4_models._4_1_sequence_based_models.inter_element_block.BaseInterElementBlock import BaseInterElementBlock
from _4_models._4_1_sequence_based_models.inter_element_block.Transformer_based_feature_comparison.TransformerBasedLearnablePositionalEncodingSequenceToSequenceInterElementBlock import \
    TransformerBasedLearnablePositionalEncodingSequenceToSequenceInterElementBlock
from _4_models._4_1_sequence_based_models.inter_element_block.Transformer_based_feature_comparison.TransformerBasedSPDLearnablePositionalEncodingSequenceToSequenceInterElementBlock import \
    TransformerBasedSPDLearnablePositionalEncodingSequenceToSequenceInterElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.BaseIntraElementBlock import BaseIntraElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock import \
    TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock import \
    TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock


class SPDFromEEGSuccessiveChannelsTransformerModel(SequenceToClassificationBaseModel):

    COMPATIBLE_DATA_FORMATTING_BLOCKS = (SPDFromEEGDataMultichannelSuccessionReformattingBlock,)
    COMPATIBLE_INTRA_ELEMENT_BLOCKS = (TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock,
                                       TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock)
    COMPATIBLE_INTER_ELEMENT_BLOCKS = (TransformerBasedLearnablePositionalEncodingSequenceToSequenceInterElementBlock,
                                       TransformerBasedSPDLearnablePositionalEncodingSequenceToSequenceInterElementBlock)
    COMPATIBLE_CLASSIFICATION_BLOCKS = (CentralGroupOfFeaturesInSequenceClassificationBlock,)

    def __init__(
            self, loss_function_config_dict: Dict[str, Any], class_labels_list: List[str],
            data_formatting_block: BaseDataFormattingBlock, intra_element_block: BaseIntraElementBlock,
            inter_element_block: BaseInterElementBlock, classification_block: BaseClassificationBlock,
            number_of_eeg_signals: int, number_of_channels: int, extra_epochs_on_each_side: int,
            number_of_subdivisions_per_epoch: int, matrix_augmentation_size: int, augmentation_factor: float,
            augmentation_factor_learnable: bool, operate_whitening: bool,
            final_linear_projection_to_given_vector_size: Union[int, None], number_of_intra_epoch_encoder_heads: int,
            intra_epoch_encoder_feedforward_dimension: int, intra_epoch_encoder_dropout_rate: float,
            number_of_intra_epoch_encoder_layers: int, number_of_inter_epoch_encoder_heads: int,
            inter_epoch_encoder_feedforward_dimension: int, inter_epoch_encoder_dropout_rate: float,
            number_of_inter_epoch_encoder_layers: int, fully_connected_intermediary_dimension: int,
            fully_connected_dropout_rate: float, learning_rate: float,
            matrix_multiplication_factor: float = 1., singular_or_eigen_value_minimum: Union[float, None] = None,
            number_of_epoch_wise_feature_vectors: Union[int, None] = None,
            optimisation_config_dict: Union[Dict[str, Any], None] = None,
            decomposition_operator: str = "svd",
            *,
            svd_singular_value_minimum: Union[float, None] = None
    ):
        self.save_hyperparameters(logger=False,
                                  ignore=["data_formatting_block", "intra_element_block", "inter_element_block",
                                          "classification_block", "svd_singular_value_minimum"])

        # Deprecating svd_singular_value_minimum in favor of singular_or_eigen_value_minimum
        if svd_singular_value_minimum is not None:
            warnings.warn("svd_singular_value_minimum is deprecated - please use singular_or_eigen_value_minimum")
            if singular_or_eigen_value_minimum is None:
                singular_or_eigen_value_minimum = svd_singular_value_minimum
                self.hparams.update({"singular_or_eigen_value_minimum": singular_or_eigen_value_minimum})



        self.original_matrix_size = number_of_eeg_signals
        self.original_number_of_channels = number_of_channels
        self.central_epoch_index = extra_epochs_on_each_side
        self.sequences_of_epochs_length = 2*extra_epochs_on_each_side + 1
        self.number_of_subdivisions_per_epoch = number_of_subdivisions_per_epoch
        self.matrix_augmentation_size = matrix_augmentation_size

        self.operate_augmentation_flag = self.matrix_augmentation_size != 0
        self.track_augmentation_factor_flag = self.operate_augmentation_flag and augmentation_factor_learnable
        self.operate_whitening_flag = operate_whitening

        super(SPDFromEEGSuccessiveChannelsTransformerModel, self).__init__(loss_function_config_dict, class_labels_list,
                                                                           data_formatting_block, intra_element_block,
                                                                           inter_element_block, classification_block,
                                                                           learning_rate, optimisation_config_dict)

        self.loss_strategies = ["central_epoch", "central_epochs_transition_penalty",
                                "cross_entropy_with_label_smoothing", "focal_loss"]
        if loss_function_config_dict["name"] in ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]:
            self.loss_strategy = "central_epoch"
        elif loss_function_config_dict["name"] == "cross_entropy_with_transition_penalty":
            assert loss_function_config_dict["args"]["number_of_classes"] == self.number_of_classes
            assert self.sequences_of_epochs_length > 1
            self.loss_strategy = "central_epochs_transition_penalty"
        else:
            raise NotImplementedError

        if number_of_epoch_wise_feature_vectors is None:
            number_of_epoch_wise_feature_vectors = self.original_number_of_channels
        else:
            number_of_vectors_pooled_into_feature_vector = (self.number_of_subdivisions_per_epoch * self.original_number_of_channels) / number_of_epoch_wise_feature_vectors
            assert int(number_of_vectors_pooled_into_feature_vector) == number_of_vectors_pooled_into_feature_vector

        extended_central_epoch_range_start = self.central_epoch_index * number_of_epoch_wise_feature_vectors
        extended_central_epoch_range_end = extended_central_epoch_range_start + number_of_epoch_wise_feature_vectors
        extended_central_epoch_indices = range(extended_central_epoch_range_start, extended_central_epoch_range_end)

        self.data_formatting_setup_kwargs = {
            "original_matrix_size": self.original_matrix_size,
            "initial_number_of_matrices_per_epoch": self.number_of_subdivisions_per_epoch,
            "number_of_channels": self.original_number_of_channels,
            "augmentation_size": self.matrix_augmentation_size,
            "initial_augmentation_factor": augmentation_factor,
            "augmentation_factor_learnable": augmentation_factor_learnable,
            "operate_whitening": self.operate_whitening_flag,
            "matrix_multiplication_factor": matrix_multiplication_factor,
            "singular_or_eigen_value_minimum": singular_or_eigen_value_minimum,
            "final_linear_projection_to_given_vector_size": final_linear_projection_to_given_vector_size,
            "decomposition_operator": decomposition_operator
        }

        self.intra_element_setup_kwargs = {
            "sequence_length": self.sequences_of_epochs_length,
            "number_of_output_features": number_of_epoch_wise_feature_vectors,
            "number_of_encoder_heads": number_of_intra_epoch_encoder_heads,
            "encoder_feedforward_dimension": intra_epoch_encoder_feedforward_dimension,
            "encoder_dropout_rate": intra_epoch_encoder_dropout_rate,
            "number_of_encoder_layers": number_of_intra_epoch_encoder_layers
        }

        self.inter_element_setup_kwargs = {
            "number_of_encoder_heads": number_of_inter_epoch_encoder_heads,
            "encoder_feedforward_dimension": inter_epoch_encoder_feedforward_dimension,
            "encoder_dropout_rate": inter_epoch_encoder_dropout_rate,
            "number_of_encoder_layers": number_of_inter_epoch_encoder_layers
        }

        self.classification_setup_kwargs = {
            "number_of_classes": self.number_of_classes,
            "relevant_feature_indices": extended_central_epoch_indices,
            "fully_connected_intermediary_dimension": fully_connected_intermediary_dimension,
            "fully_connected_dropout_rate": fully_connected_dropout_rate
        }

    def send_hparams_to_logger(self):
        hparams_dict = dict(self.hparams)

        hparams_dict["data_formatting_block"] = self.get_block_dict(self.data_formatting_block)
        hparams_dict["intra_element_block"] = self.get_block_dict(self.intra_element_block)
        hparams_dict["inter_element_block"] = self.get_block_dict(self.inter_element_block)
        hparams_dict["classification_block"] = self.get_block_dict(self.classification_block)

        model_dict = self.get_block_dict(self)
        model_dict["init_args"] = hparams_dict

        output_dict = {"model": model_dict}
        self.hparams.clear()
        self.hparams.update(output_dict)
        self.save_hyperparameters()

    def setup(self, stage: str, no_logging: bool = False):
        self.obtain_example_input_array()
        if not self._SequenceToClassificationBaseModel__setup_done_flag:
            if not no_logging:
                self.send_hparams_to_logger()

            vector_size, extended_sequences_of_epochs_length = self.data_formatting_block.setup(
                **self.data_formatting_setup_kwargs)

            extended_sequence_length = self.intra_element_block.setup(
                vector_size=vector_size, number_of_vectors_per_element=extended_sequences_of_epochs_length,
                **self.intra_element_setup_kwargs)

            self.inter_element_block.setup(sequence_length=extended_sequence_length, vector_size=vector_size,
                                           **self.inter_element_setup_kwargs)

            self.classification_block.setup(sequence_length=extended_sequence_length, vector_size=vector_size,
                                            **self.classification_setup_kwargs)

            self._SequenceToClassificationBaseModel__setup_done_flag = True

    # sequence_of_spd_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, matrix_size, matrix_size)
    # sequence_of_augmentation_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_vectors_per_epoch, matrix_size, matrix_augmentation_size)
    # mean_recording_spd_matrices of shape (batch_size, number_of_channels, matrix_size, matrix_size)
    # mean_recording_augmentation_matrices of shape (batch_size, number_of_channels, matrix_size, matrix_augmentation_size)
    # output of shape (batch_size, number_of_classes)
    def forward(self, sequence_of_spd_matrices: torch.Tensor,
                sequence_of_augmentation_matrices: Union[torch.Tensor, None] = None,
                mean_recording_spd_matrices: Union[torch.Tensor, None] = None,
                mean_recording_augmentation_matrices: Union[torch.Tensor, None] = None):

        return super(SPDFromEEGSuccessiveChannelsTransformerModel, self).forward(
            sequence_of_spd_matrices=sequence_of_spd_matrices,
            sequence_of_augmentation_matrices=sequence_of_augmentation_matrices,
            mean_recording_spd_matrices=mean_recording_spd_matrices,
            mean_recording_augmentation_matrices=mean_recording_augmentation_matrices
        )

    def preprocess_input(self, input, set_name):
        model_input_dict = {
            "sequence_of_spd_matrices": None,
            "sequence_of_augmentation_matrices": None,
            "mean_recording_spd_matrices": None,
            "mean_recording_augmentation_matrices": None
        }

        sequence_of_spd_matrices = input["matrices"]
        batch_size = sequence_of_spd_matrices.shape[0]
        assert sequence_of_spd_matrices.shape[1:] == (self.original_number_of_channels, self.sequences_of_epochs_length, self.number_of_subdivisions_per_epoch, self.original_matrix_size, self.original_matrix_size)
        model_input_dict["sequence_of_spd_matrices"] = sequence_of_spd_matrices

        if self.operate_whitening_flag:
            mean_recording_spd_matrices = input["recording-wise matrices"]
            assert mean_recording_spd_matrices.shape == (batch_size, self.original_number_of_channels, self.original_matrix_size, self.original_matrix_size)
            model_input_dict["mean_recording_spd_matrices"] = mean_recording_spd_matrices

        if self.operate_augmentation_flag:
            sequence_of_augmentation_matrices = input["statistic matrices"]
            assert sequence_of_augmentation_matrices.shape == (batch_size, self.original_number_of_channels, self.sequences_of_epochs_length, self.number_of_subdivisions_per_epoch, self.original_matrix_size, self.matrix_augmentation_size)
            model_input_dict["sequence_of_augmentation_matrices"] = sequence_of_augmentation_matrices
            
            if self.operate_whitening_flag:
                mean_recording_augmentation_matrices = input["recording mean statistic matrices"]
                assert mean_recording_augmentation_matrices.shape == (batch_size, self.original_number_of_channels, self.original_matrix_size, self.matrix_augmentation_size)
                model_input_dict["mean_recording_augmentation_matrices"] = mean_recording_augmentation_matrices

        return model_input_dict

    def process_loss_function_inputs(self, predictions, groundtruth, set_name):
        assert self.loss_strategy in self.loss_strategies

        assert len(predictions.shape) == 2 and len(groundtruth.shape) == 2
        assert predictions.shape[0] == groundtruth.shape[0]
        assert groundtruth.shape[1] == self.sequences_of_epochs_length
        assert predictions.shape[1] == self.number_of_classes

        if self.loss_strategy == "central_epoch":
            return {"input": predictions, "target": groundtruth[:, self.central_epoch_index].long()}
        if self.loss_strategy == "central_epochs_transition_penalty":
            return {"input": predictions,
                    "target": groundtruth[:, self.central_epoch_index].long(),
                    "previous_epoch_target": groundtruth[:, self.central_epoch_index - 1].long()}

    def training_step(self, batch, batch_idx):
        if self.track_augmentation_factor_flag:
            augmentation_factor = self.data_formatting_block.augmentation_layer.augmentation_factor
            self.log('on_step_augmentation_factor/%s' % self.TRAINING_SET_NAME, augmentation_factor.detach().cpu(),
                     on_step=True, on_epoch=False)
        return super(SPDFromEEGSuccessiveChannelsTransformerModel, self).training_step(batch, batch_idx)

    def epoch_level_log_metrics(self, outputs, set_name):
        if self.track_augmentation_factor_flag:
            augmentation_factor = self.data_formatting_block.augmentation_layer.augmentation_factor
            self.log('augmentation_factor/%s' % set_name, augmentation_factor.detach().cpu(), sync_dist=True)
        super(SPDFromEEGSuccessiveChannelsTransformerModel, self).epoch_level_log_metrics(outputs, set_name)

    def obtain_example_input_array(self):
        placeholder_batch_size = 1
        sequence_of_augmentation_matrices = torch.tensor((float("nan"),), dtype=self.dtype)
        mean_recording_spd_matrices = torch.tensor((float("nan"),), dtype=self.dtype)
        mean_recording_augmentation_matrices = torch.tensor((float("nan"),), dtype=self.dtype)

        sequence_of_spd_matrices = torch.rand(placeholder_batch_size, self.original_number_of_channels, self.sequences_of_epochs_length, self.number_of_subdivisions_per_epoch, self.original_matrix_size, self.original_matrix_size, dtype=self.dtype)
        if self.operate_augmentation_flag:
            sequence_of_augmentation_matrices = torch.rand(placeholder_batch_size, self.original_number_of_channels, self.sequences_of_epochs_length, self.number_of_subdivisions_per_epoch, self.original_matrix_size, self.matrix_augmentation_size, dtype=self.dtype)
        if self.operate_whitening_flag:
            mean_recording_spd_matrices = torch.rand(placeholder_batch_size, self.original_number_of_channels, self.original_matrix_size, self.original_matrix_size, dtype=self.dtype)
            if self.operate_augmentation_flag:
                mean_recording_augmentation_matrices = torch.rand(placeholder_batch_size, self.original_number_of_channels, self.original_matrix_size, self.matrix_augmentation_size, dtype=self.dtype)

        self.example_input_array = (sequence_of_spd_matrices, sequence_of_augmentation_matrices, mean_recording_spd_matrices, mean_recording_augmentation_matrices)


