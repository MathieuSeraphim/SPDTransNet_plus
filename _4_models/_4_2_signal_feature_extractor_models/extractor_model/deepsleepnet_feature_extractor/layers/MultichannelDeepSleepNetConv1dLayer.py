from torch.nn import Module, Conv2d, ReLU
from _4_models._4_2_signal_feature_extractor_models.extractor_model.deepsleepnet_feature_extractor.layers.MultichannelDeepSleepNetBatchNorm1dLayer import \
    MultichannelDeepSleepNetBatchNorm1dLayer
from _4_models._4_2_signal_feature_extractor_models.extractor_model.utils import \
    expected_padding_values_along_dimension_assuming_SAME_padding, expected_convolution_output_size_along_dimension, \
    expected_output_size_along_dimension_assuming_SAME_padding


class MultichannelDeepSleepNetConv1dLayer(Module):

    # DeepSleepNet constants
    DILATION = 1
    BIAS = None

    def __init__(self):
        super(MultichannelDeepSleepNetConv1dLayer, self).__init__()
        self.__setup_done_flag = False

        self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer = None
        self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer = None
        self.height_when_seen_as_a_2d_conv_layer = None
        self.input_width_when_seen_as_a_2d_conv_layer = None
        self.output_width_when_seen_as_a_2d_conv_layer = None

        self.multichannel_conv1d = None
        self.batch_norm = MultichannelDeepSleepNetBatchNorm1dLayer()
        self.relu = ReLU()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              conv1d_input_channels: int, conv1d_output_channels: int, conv1d_kernel_size: int, conv1d_stride: int):
        assert not self.__setup_done_flag

        self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer = number_of_independent_channels * conv1d_input_channels
        self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer = number_of_independent_channels * conv1d_output_channels
        self.height_when_seen_as_a_2d_conv_layer = number_of_parallel_channels
        self.input_width_when_seen_as_a_2d_conv_layer = signal_length

        number_of_groups_when_seen_as_a_2d_conv_layer = number_of_independent_channels
        kernel_size_when_seen_as_a_2d_conv_layer = (1, conv1d_kernel_size)
        stride_when_seen_as_a_2d_conv_layer = (1, conv1d_stride)

        if conv1d_stride == 1:
            padding_when_seen_as_a_2d_conv_layer = "same"
            self.output_width_when_seen_as_a_2d_conv_layer = self.input_width_when_seen_as_a_2d_conv_layer

        else:
            # Original Tensorflow code: padding = SAME even with stride > 1
            padding_left, padding_right = expected_padding_values_along_dimension_assuming_SAME_padding(
                signal_length, conv1d_kernel_size, conv1d_stride)
            padding = padding_left
            padding_when_seen_as_a_2d_conv_layer = (0, padding)
            self.output_width_when_seen_as_a_2d_conv_layer = expected_convolution_output_size_along_dimension(
                signal_length, conv1d_kernel_size, conv1d_stride, padding, self.DILATION)

            if padding_left == padding_right:
                assert self.output_width_when_seen_as_a_2d_conv_layer ==\
                       expected_output_size_along_dimension_assuming_SAME_padding(signal_length, conv1d_stride)
            else:  # Different values of left / right padding unsupported by PyTorch
                assert padding_right == padding_left + 1
                assert self.output_width_when_seen_as_a_2d_conv_layer ==\
                       expected_output_size_along_dimension_assuming_SAME_padding(signal_length, conv1d_stride) - 1

        self.multichannel_conv1d = Conv2d(
            in_channels=self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer,
            out_channels=self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer,
            kernel_size=kernel_size_when_seen_as_a_2d_conv_layer,
            stride=stride_when_seen_as_a_2d_conv_layer,
            padding=padding_when_seen_as_a_2d_conv_layer,
            dilation=self.DILATION,
            groups=number_of_groups_when_seen_as_a_2d_conv_layer,
            bias=self.BIAS
        )

        self.batch_norm.setup(number_of_independent_channels, number_of_parallel_channels, conv1d_output_channels)

        self.__setup_done_flag = True
        return self.output_width_when_seen_as_a_2d_conv_layer

    # input_signal of shape (batch_size, number_of_independent_channels * conv1d_input_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, output_signal_length)
    def forward(self, input_signal):
        assert self.__setup_done_flag

        input_shape = input_signal.shape
        assert len(input_shape) == 4
        assert input_shape[1:] == (self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer,
                                   self.height_when_seen_as_a_2d_conv_layer,
                                   self.input_width_when_seen_as_a_2d_conv_layer)

        output_shape = [s for s in input_shape]
        output_shape[1] = self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer
        output_shape[-1] = self.output_width_when_seen_as_a_2d_conv_layer
        output_shape = tuple(output_shape)

        transformed_signal = self.multichannel_conv1d(input_signal)
        transformed_signal = self.batch_norm(transformed_signal)
        transformed_signal = self.relu(transformed_signal)

        assert transformed_signal.shape == output_shape
        return transformed_signal

