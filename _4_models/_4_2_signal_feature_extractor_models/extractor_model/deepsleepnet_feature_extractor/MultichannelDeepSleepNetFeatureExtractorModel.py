import torch
from torch.nn import ModuleList, Linear
from _4_models._4_2_signal_feature_extractor_models.extractor_model.BaseExtractorModel import BaseExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.deepsleepnet_feature_extractor.layers.MultichannelDeepSleepNetFeatureExtractionFilterBlock import \
    MultichannelDeepSleepNetFeatureExtractionSmallFiltersBlock, \
    MultichannelDeepSleepNetFeatureExtractionLargeFiltersBlock


class MultichannelDeepSleepNetFeatureExtractorModel(BaseExtractorModel):

    # The signal has two types of channels: independent (will pass through a different set of weights) ans parallel
    # (same weights, no mixing)
    #
    # For EEG signals, parallel channels correspond to signals originating from different electrodes but that underwent
    # the same preprocessing
    # independent channels correspond to different preprocessing pipelines, and therefore a need for different weights
    #
    # Standard DeepSleepNet network: first_layers_shape_argument = sampling_frequency,
    # signal_length = 30 * sampling_frequency (30s EEG epochs)
    def __init__(self):
        super(MultichannelDeepSleepNetFeatureExtractorModel, self).__init__()
        self.__setup_done_flag = False

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.output_features_length = None
        self.combined_filters_output_signal_length = None
        self.filters_output_channels = None
        self.combined_filters_total_output_size = None
        self.signal_length = None

        self.small_filters_block = MultichannelDeepSleepNetFeatureExtractionSmallFiltersBlock()
        self.large_filters_block = MultichannelDeepSleepNetFeatureExtractionLargeFiltersBlock()
        self.resizing_layers = ModuleList()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              first_layers_shape_argument: int, number_of_output_features: int, single_output_feature_length: int):
        assert not self.__setup_done_flag

        super(MultichannelDeepSleepNetFeatureExtractorModel, self).setup(
            single_output_feature_length=single_output_feature_length
        )

        assert number_of_independent_channels > 0
        assert number_of_parallel_channels > 0
        assert signal_length >= 4 * first_layers_shape_argument > 0

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels

        self.output_features_length = number_of_output_features * self.single_output_feature_length

        small_filters_output_signal_length = self.small_filters_block.setup(
            number_of_independent_channels, number_of_parallel_channels, signal_length, first_layers_shape_argument
        )
        large_filters_output_signal_length = self.large_filters_block.setup(
            number_of_independent_channels, number_of_parallel_channels, signal_length, first_layers_shape_argument
        )
        self.combined_filters_output_signal_length = small_filters_output_signal_length + large_filters_output_signal_length

        self.filters_output_channels = self.small_filters_block.SUBSEQUENT_CONV_OUTPUT_CHANNELS
        assert self.filters_output_channels == self.large_filters_block.SUBSEQUENT_CONV_OUTPUT_CHANNELS

        self.combined_filters_total_output_size = self.combined_filters_output_signal_length * self.filters_output_channels

        for channel in range(self.number_of_independent_channels):
            self.resizing_layers.append(Linear(self.combined_filters_total_output_size, self.output_features_length))

        self.signal_length = signal_length

        print()
        print("Set up a DeepSleepNet-derived feature encoder, with:")
        print("- An input size of %d," % signal_length)
        print("- An output of size %d over %d channels for the small filters block," % (small_filters_output_signal_length, self.filters_output_channels))
        print("- An output of size %d over %d channels for the large filters block," % (large_filters_output_signal_length, self.filters_output_channels))
        print("- A final output of length %d, that is %d features of length %d." % (self.output_features_length, number_of_output_features, single_output_feature_length))
        print()

        self.__setup_done_flag = True
        return self.output_features_length

    # input_signal of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, signal_length)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, output_features_length)
    def forward(self, input_signal):
        assert self.__setup_done_flag

        batch_size = input_signal.shape[0]

        small_filters_output = self.small_filters_block(input_signal)
        large_filters_output = self.large_filters_block(input_signal)
        combined_filters_output = torch.cat((small_filters_output, large_filters_output), dim=-1)

        assert combined_filters_output.shape == (batch_size,
                                                 self.number_of_independent_channels * self.filters_output_channels,
                                                 self.number_of_parallel_channels,
                                                 self.combined_filters_output_signal_length)
        combined_filters_output = combined_filters_output.view(batch_size,
                                                               self.number_of_independent_channels,
                                                               self.filters_output_channels,
                                                               self.number_of_parallel_channels,
                                                               self.combined_filters_output_signal_length)

        combined_filters_output = combined_filters_output.transpose(2, 3)
        combined_filters_output = combined_filters_output.contiguous().view(batch_size,
                                                                            self.number_of_independent_channels,
                                                                            self.number_of_parallel_channels,
                                                                            self.combined_filters_total_output_size)

        # Tuple of number_of_independent_channels tensors of shape (batch_size, number_of_parallel_channels, combined_filters_total_output_size)
        independent_channels_wise_combined_filters_output = torch.unbind(combined_filters_output, dim=1)

        assert len(independent_channels_wise_combined_filters_output) == len(self.resizing_layers) == self.number_of_independent_channels
        output_tensors_list = []
        for i in range(self.number_of_independent_channels):
            output_tensors_list.append(self.resizing_layers[i](independent_channels_wise_combined_filters_output[i]))

        output = torch.stack(output_tensors_list, dim=1)
        assert output.shape == (batch_size, self.number_of_independent_channels, self.number_of_parallel_channels, self.output_features_length)

        return output



