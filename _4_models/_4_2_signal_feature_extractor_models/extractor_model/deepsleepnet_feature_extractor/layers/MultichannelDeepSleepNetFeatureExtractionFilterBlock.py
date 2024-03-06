from torch.nn import Module, Dropout, MaxPool2d
from _4_models._4_2_signal_feature_extractor_models.extractor_model.deepsleepnet_feature_extractor.layers.MultichannelDeepSleepNetConv1dLayer import \
    MultichannelDeepSleepNetConv1dLayer
from _4_models._4_2_signal_feature_extractor_models.extractor_model.deepsleepnet_feature_extractor.layers.MultichannelDeepSleepNetMaxPool1dLayer import \
    MultichannelDeepSleepNetMaxPool1dLayer


class MultichannelDeepSleepNetFeatureExtractionFilterBlock(Module):

    # DeepSleepNet constants
    FIRST_CONV_INPUT_CHANNELS = 1
    SUBSEQUENT_CONV_STRIDE = 1
    DROPOUT = .5

    def __init__(self):
        super(MultichannelDeepSleepNetFeatureExtractionFilterBlock, self).__init__()
        self.__setup_done_flag = False

        self.number_of_parallel_channels = None
        self.input_second_dimension = None
        self.input_signal_length = None
        self.output_second_dimension = None
        self.output_signal_length = None

        self.first_conv = MultichannelDeepSleepNetConv1dLayer()
        self.first_max_pool = MultichannelDeepSleepNetMaxPool1dLayer()
        self.dropout = Dropout(self.DROPOUT)
        self.second_conv = MultichannelDeepSleepNetConv1dLayer()
        self.third_conv = MultichannelDeepSleepNetConv1dLayer()
        self.fourth_conv = MultichannelDeepSleepNetConv1dLayer()
        self.second_max_pool = MultichannelDeepSleepNetMaxPool1dLayer()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int,
              initial_signal_length: int, first_conv_kernel_size: int, first_conv_output_channels: int,
              first_conv_stride: int, subsequent_conv_kernel_size: int, subsequent_conv_output_channels: int,
              first_max_pool_size_and_stride: int, second_max_pool_size_and_stride: int):
        assert not self.__setup_done_flag

        self.number_of_parallel_channels = number_of_parallel_channels

        self.input_second_dimension = number_of_independent_channels
        self.input_signal_length = initial_signal_length

        first_max_pool_input_signal_length = self.first_conv.setup(
            number_of_independent_channels, number_of_parallel_channels, self.input_signal_length,
            self.FIRST_CONV_INPUT_CHANNELS, first_conv_output_channels, first_conv_kernel_size, first_conv_stride)

        second_conv_input_signal_length = self.first_max_pool.setup(
            number_of_independent_channels, number_of_parallel_channels, first_conv_output_channels,
            first_max_pool_input_signal_length, first_max_pool_size_and_stride, first_max_pool_size_and_stride
        )

        third_conv_input_signal_length = self.second_conv.setup(
            number_of_independent_channels, number_of_parallel_channels, second_conv_input_signal_length,
            first_conv_output_channels, subsequent_conv_output_channels, subsequent_conv_kernel_size,
            self.SUBSEQUENT_CONV_STRIDE
        )
        self.output_second_dimension = subsequent_conv_output_channels * number_of_independent_channels

        fourth_conv_input_signal_length = self.third_conv.setup(
            number_of_independent_channels, number_of_parallel_channels, third_conv_input_signal_length,
            subsequent_conv_output_channels, subsequent_conv_output_channels, subsequent_conv_kernel_size,
            self.SUBSEQUENT_CONV_STRIDE
        )

        second_max_pool_input_signal_length = self.fourth_conv.setup(
            number_of_independent_channels, number_of_parallel_channels, fourth_conv_input_signal_length,
            subsequent_conv_output_channels, subsequent_conv_output_channels, subsequent_conv_kernel_size,
            self.SUBSEQUENT_CONV_STRIDE
        )
        
        self.output_signal_length = self.second_max_pool.setup(
            number_of_independent_channels, number_of_parallel_channels, subsequent_conv_output_channels,
            second_max_pool_input_signal_length, second_max_pool_size_and_stride, second_max_pool_size_and_stride
        )

        self.__setup_done_flag = True
        return self.output_signal_length

    # input_signal of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, signal_length)
    def forward(self, input_signal):
        assert self.__setup_done_flag

        assert len(input_signal.shape) == 4
        assert input_signal.shape[1:] == (self.input_second_dimension, self.number_of_parallel_channels, self.input_signal_length)
        batch_size = input_signal.shape[0]

        x = self.first_conv(input_signal)
        x = self.first_max_pool(x)
        x = self.dropout(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.fourth_conv(x)
        output = self.second_max_pool(x)

        assert output.shape == (batch_size, self.output_second_dimension, self.number_of_parallel_channels, self.output_signal_length)
        return output


class MultichannelDeepSleepNetFeatureExtractionSmallFiltersBlock(MultichannelDeepSleepNetFeatureExtractionFilterBlock):

    FIRST_CONV_KERNEL_DIVIDER = 2
    FIRST_CONV_STRIDE_DIVIDER = 16
    FIRST_CONV_OUTPUT_CHANNELS = 64
    SUBSEQUENT_CONV_OUTPUT_CHANNELS = 128
    SUBSEQUENT_CONV_KERNEL_SIZE = 8
    FIRST_MAX_POOL_SIZE_AND_STRIDE = 8
    SECOND_MAX_POOL_SIZE_AND_STRIDE = 4

    def __init__(self):
        super(MultichannelDeepSleepNetFeatureExtractionSmallFiltersBlock, self).__init__()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              first_layers_shape_argument: int, **ignored_kwargs):
        first_conv_kernel_size = int(first_layers_shape_argument / self.FIRST_CONV_KERNEL_DIVIDER)
        first_conv_stride = int(first_layers_shape_argument / self.FIRST_CONV_STRIDE_DIVIDER)
        return super(MultichannelDeepSleepNetFeatureExtractionSmallFiltersBlock, self).setup(
            number_of_independent_channels, number_of_parallel_channels, signal_length, first_conv_kernel_size,
            self.FIRST_CONV_OUTPUT_CHANNELS, first_conv_stride, self.SUBSEQUENT_CONV_KERNEL_SIZE,
            self.SUBSEQUENT_CONV_OUTPUT_CHANNELS, self.FIRST_MAX_POOL_SIZE_AND_STRIDE,
            self.SECOND_MAX_POOL_SIZE_AND_STRIDE
        )


class MultichannelDeepSleepNetFeatureExtractionLargeFiltersBlock(MultichannelDeepSleepNetFeatureExtractionFilterBlock):
    FIRST_CONV_KERNEL_MULTIPLIER = 4
    FIRST_CONV_STRIDE_DIVIDER = 2
    FIRST_CONV_OUTPUT_CHANNELS = 64
    SUBSEQUENT_CONV_OUTPUT_CHANNELS = 128
    SUBSEQUENT_CONV_KERNEL_SIZE = 6
    FIRST_MAX_POOL_SIZE_AND_STRIDE = 4
    SECOND_MAX_POOL_SIZE_AND_STRIDE = 2

    def __init__(self):
        super(MultichannelDeepSleepNetFeatureExtractionLargeFiltersBlock, self).__init__()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              first_layers_shape_argument: int, **ignored_kwargs):
        first_conv_kernel_size = first_layers_shape_argument * self.FIRST_CONV_KERNEL_MULTIPLIER
        first_conv_stride = int(first_layers_shape_argument / self.FIRST_CONV_STRIDE_DIVIDER)
        return super(MultichannelDeepSleepNetFeatureExtractionLargeFiltersBlock, self).setup(
            number_of_independent_channels, number_of_parallel_channels, signal_length, first_conv_kernel_size,
            self.FIRST_CONV_OUTPUT_CHANNELS, first_conv_stride, self.SUBSEQUENT_CONV_KERNEL_SIZE,
            self.SUBSEQUENT_CONV_OUTPUT_CHANNELS, self.FIRST_MAX_POOL_SIZE_AND_STRIDE,
            self.SECOND_MAX_POOL_SIZE_AND_STRIDE
        )
