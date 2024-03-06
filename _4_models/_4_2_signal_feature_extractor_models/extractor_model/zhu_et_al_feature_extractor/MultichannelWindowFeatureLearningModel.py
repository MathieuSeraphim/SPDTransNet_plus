from typing import Optional, Dict, Any
import torch
from torchsummary import summary
from _4_models._4_2_signal_feature_extractor_models.extractor_model.BaseExtractorModel import BaseExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelLinearLayerWithIndependentAndParallelChannels import \
    MultichannelLinearLayerWithIndependentAndParallelChannels
from _4_models._4_2_signal_feature_extractor_models.extractor_model.zhu_et_al_feature_extractor.layers.MultichannelEpochToFeaturesNetworkWithIndependentAndParallelChannels import \
    MultichannelEpochToFeaturesNetworkWithIndependentAndParallelChannels


class MultichannelWindowFeatureLearningModel(BaseExtractorModel):

    # The signal has two types of channels: independent (will pass through a different set of weights) ans parallel
    # (same weights, no mixing)
    #
    # For EEG signals, parallel channels correspond to signals originating from different electrodes but that underwent
    # the same preprocessing
    # independent channels correspond to different preprocessing pipelines, and therefore a need for different weights
    def __init__(self):
        super(MultichannelWindowFeatureLearningModel, self).__init__()
        self.__setup_done_flag = False

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.number_of_output_features = None
        self.output_features_length = None
        self.signal_length = None

        self.resizing_layer_input_length = None
        self.resizing_layer_output_length = None

        self.main_model = MultichannelEpochToFeaturesNetworkWithIndependentAndParallelChannels()
        self.resizing_layer = MultichannelLinearLayerWithIndependentAndParallelChannels()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              number_of_output_features: int, single_output_feature_length: int,
              non_default_inputs_kwargs: Optional[Dict[str, Any]] = None):
        assert not self.__setup_done_flag

        super(MultichannelWindowFeatureLearningModel, self).setup(
            single_output_feature_length=single_output_feature_length
        )

        assert number_of_independent_channels > 0
        assert number_of_parallel_channels > 0

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels
        self.signal_length = signal_length

        self.number_of_output_features = number_of_output_features
        self.output_features_length = number_of_output_features * self.single_output_feature_length

        if non_default_inputs_kwargs is None:
            non_default_inputs_kwargs = {}

        number_of_output_features, pre_resizing_output_feature_length = self.main_model.setup(
            number_of_independent_channels=number_of_independent_channels,
            number_of_parallel_channels=number_of_parallel_channels,
            input_signal_length=self.signal_length,
            number_of_windows_in_signal=self.number_of_output_features,
            **non_default_inputs_kwargs
        )
        assert number_of_output_features == self.number_of_output_features
        assert pre_resizing_output_feature_length > 0

        self.resizing_layer_input_length = self.number_of_output_features * pre_resizing_output_feature_length

        self.resizing_layer.setup(
            number_of_independent_channels=number_of_independent_channels,
            number_of_parallel_channels=number_of_parallel_channels,
            number_of_regular_channels=1,
            input_length=self.resizing_layer_input_length,
            output_length=self.output_features_length,
            number_of_shared_weight_groups=self.number_of_output_features,
            regular_channels_combined_with_independent_channels=False
        )

        print()
        print("Set up a feature encoder detived from Zhu et al.'s approach, with:")
        print("- An input size of %d," % self.signal_length)
        print("- An output of %d vectors of size %d for the window feature learning CNN," % (self.number_of_output_features, pre_resizing_output_feature_length))
        print("- A final output of length %d, that is %d features of length %d." % (self.output_features_length, self.number_of_output_features, self.single_output_feature_length))
        print()

        self.__setup_done_flag = True
        return self.output_features_length

    # input_signal of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, signal_length)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, output_features_length)
    def forward(self, input_signal):
        assert self.__setup_done_flag

        batch_size = input_signal.shape[0]

        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_output_features, pre_resizing_output_feature_length)
        cnn_output = self.main_model(input_signal)
        del input_signal

        # (batch_size, number_of_independent_channels, number_of_parallel_channels, 1, resizing_layer_input_length = number_of_output_features * pre_resizing_output_feature_length)
        resizing_layer_input = cnn_output.view((
            batch_size,
            self.number_of_independent_channels,
            self.number_of_parallel_channels,
            1,
            self.resizing_layer_input_length
        ))

        # (batch_size, number_of_independent_channels, number_of_parallel_channels, 1, output_features_length)
        # with output_features_length = number_of_output_features * single_output_feature_length
        final_resizing_layer_output = self.resizing_layer(resizing_layer_input)
        assert final_resizing_layer_output.shape == (
            batch_size,
            self.number_of_independent_channels,
            self.number_of_parallel_channels,
            1,
            self.output_features_length
        )

        output = final_resizing_layer_output.view(
            batch_size,
            self.number_of_independent_channels,
            self.number_of_parallel_channels,
            self.output_features_length
        )
        return output


if __name__ == "__main__":

    number_of_independent_channels = 7
    number_of_parallel_channels = 8
    epoch_length_in_seconds = 30
    sampling_frequency = 256
    dropout_rate = 0
    single_output_feature_length = 3

    input_signal_length = epoch_length_in_seconds * sampling_frequency
    input_shape = (number_of_independent_channels, number_of_parallel_channels, input_signal_length)
    batched_input_shape = (1,) + input_shape

    model = MultichannelWindowFeatureLearningModel()
    model.setup(number_of_independent_channels=number_of_independent_channels,
                number_of_parallel_channels=number_of_parallel_channels,
                signal_length=input_signal_length,
                number_of_output_features=epoch_length_in_seconds,
                single_output_feature_length=single_output_feature_length)

    summary(model, input_shape, device="cpu")

    input = torch.rand(*batched_input_shape)
    output = model(input)
    print(output.shape)


