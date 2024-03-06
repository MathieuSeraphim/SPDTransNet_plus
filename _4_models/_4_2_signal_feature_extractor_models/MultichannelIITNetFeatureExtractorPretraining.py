import torch
from typing import List, Dict, Any, Union
from _4_models._4_2_signal_feature_extractor_models.FeatureExtractorPretrainingBaseModel import \
    FeatureExtractorPretrainingBaseModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.BaseExtractorModel import BaseExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.MultichannelIITNetFeatureExtractorModel import \
    MultichannelIITNetFeatureExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.MultichannelIITNetFeatureExtractorModel_v2 import \
    MultichannelIITNetFeatureExtractorModel_v2
from _4_models._4_2_signal_feature_extractor_models.features_to_classification_block.BaseFeaturesToClassificationBlock import \
    BaseFeaturesToClassificationBlock
from _4_models._4_2_signal_feature_extractor_models.features_to_classification_block.pretraining_feature_to_classification_block.MultichannelFeatureExtractorPretrainingBlock import \
    MultichannelFeatureExtractorPretrainingBlock


class MultichannelIITNetFeatureExtractorPretraining(FeatureExtractorPretrainingBaseModel):

    COMPATIBLE_EXTRACTOR_MODELS = (MultichannelIITNetFeatureExtractorModel, MultichannelIITNetFeatureExtractorModel_v2)
    COMPATIBLE_FEATURES_TO_CLASSIFICATION_BLOCKS = (MultichannelFeatureExtractorPretrainingBlock,)

    def __init__(self, loss_function_config_dict: Dict[str, Any], class_labels_list: List[str],
                 extractor_model: BaseExtractorModel,
                 features_to_classification_block: BaseFeaturesToClassificationBlock,
                 number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
                 resnet_num_layers: int, dropout_rate: float, number_of_output_features: int,
                 output_feature_length: int, learning_rate: float,
                 optimisation_config_dict: Union[Dict[str, Any], None] = None):
        assert loss_function_config_dict["name"] in ("cross_entropy", "cross_entropy_with_label_smoothing")

        self.save_hyperparameters(logger=False,
                                  ignore=["extractor_model", "features_to_classification_block"])

        self.number_of_parallel_channels = number_of_parallel_channels
        self.number_of_independent_channels = number_of_independent_channels
        self.signal_length = signal_length

        super(MultichannelIITNetFeatureExtractorPretraining, self).__init__(loss_function_config_dict,
                                                                            class_labels_list, extractor_model,
                                                                            features_to_classification_block,
                                                                            learning_rate,
                                                                            optimisation_config_dict)

        self.extractor_setup_kwargs = {
            "number_of_independent_channels": number_of_independent_channels,
            "number_of_parallel_channels": number_of_parallel_channels,
            "signal_length": signal_length,
            "resnet_num_layers": resnet_num_layers,
            "dropout_rate": dropout_rate,
            "number_of_output_features": number_of_output_features,
            "single_output_feature_length": output_feature_length
        }

        self.features_to_classification_setup_kwargs = {
            "first_feature_dimension": number_of_independent_channels,
            "second_feature_dimension": number_of_parallel_channels,
            "number_of_classes": self.number_of_classes
        }

    def send_hparams_to_logger(self):
        hparams_dict = dict(self.hparams)

        hparams_dict["extractor_model"] = self.get_block_dict(self.extractor_model)
        hparams_dict["features_to_classification_block"] = self.get_block_dict(self.features_to_classification_block)

        model_dict = self.get_block_dict(self)
        model_dict["init_args"] = hparams_dict

        output_dict = {"model": model_dict}
        self.hparams.clear()
        self.hparams.update(output_dict)
        self.save_hyperparameters()

    def setup(self, stage: str, no_logging: bool = False):
        self.obtain_example_input_array()
        if not self._FeatureExtractorPretrainingBaseModel__setup_done_flag:
            if not no_logging:
                self.send_hparams_to_logger()
            output_features_length = self.extractor_model.setup(**self.extractor_setup_kwargs)
            self.features_to_classification_block.setup(third_feature_dimension=output_features_length,
                                                        **self.features_to_classification_setup_kwargs)
            self._FeatureExtractorPretrainingBaseModel__setup_done_flag = True

    def forward(self, input_signal):
        return super(MultichannelIITNetFeatureExtractorPretraining, self).forward(input_signal=input_signal)

    def preprocess_input(self, input, set_name):

        # shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_subwindows, signal_length / number_of_subwindows)
        eeg_epoch_signals = input["EEG signals"]

        assert len(eeg_epoch_signals.shape) == 5
        original_shape = eeg_epoch_signals.shape
        batch_size = original_shape[0]

        assert (original_shape[1], original_shape[2]) == (self.number_of_independent_channels, self.number_of_parallel_channels)
        assert original_shape[3] * original_shape[4] == self.signal_length

        signal_shape = (batch_size, self.number_of_independent_channels, self.number_of_parallel_channels, self.signal_length)
        return {"input_signal": eeg_epoch_signals.view(signal_shape)}

    def obtain_example_input_array(self):
        placeholder_batch_size = 1
        input_signal = torch.rand(placeholder_batch_size, self.number_of_independent_channels, self.number_of_parallel_channels, self.signal_length, dtype=self.dtype)
        self.example_input_array = input_signal


