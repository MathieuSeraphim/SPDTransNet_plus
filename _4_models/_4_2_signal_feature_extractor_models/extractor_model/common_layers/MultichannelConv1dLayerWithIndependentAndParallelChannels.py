from _4_models._4_2_signal_feature_extractor_models.extractor_model.utils import \
    expected_convolution_output_size_along_dimension
from torch.nn import Module, Conv2d


# This was built after the MultichannelDeepSleepNetConv1dLayer class, hence the duplication.


class MultichannelConv1dLayerWithIndependentAndParallelChannels(Module):

    # Normal Conv1d:
    #     input of shape (batch_size, conv1d_input_channels, input_length)
    #     output of shape (batch_size, conv1d_output_channels, output_length)
    # Conv1d with parallel channels (should go through the same weights, but shouldn't mix) - implemented with Conv2d
    # with kernel height at 1, and:
    #     input of shape (batch_size, conv1d_input_channels, number_of_parallel_channels, input_length)
    #     output of shape (batch_size, conv1d_output_channels, number_of_parallel_channels, output_length)
    # Now, adding independent channels (should have their own weights, but still included in the same tensor for
    # optimized parallel computations) as groups (cf Conv1d/2d documentation), we have:
    #     input of shape (batch_size, number_of_independent_channels * conv1d_input_channels, number_of_parallel_channels, input_length)
    #     output of shape (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, output_length)
    # Independent and parallel channels are fixed for the entire CNN.
    def __init__(self):
        super(MultichannelConv1dLayerWithIndependentAndParallelChannels, self).__init__()
        self.__setup_done_flag = False

        self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer = None
        self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer = None
        self.height_when_seen_as_a_2d_conv_layer = None
        self.input_width_when_seen_as_a_2d_conv_layer = None
        self.output_width_when_seen_as_a_2d_conv_layer = None

        self.multichannel_conv1d = None

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              conv1d_input_channels: int, conv1d_output_channels: int, conv1d_kernel_size: int, conv1d_stride: int = 1,
              conv1d_padding: int = 0, conv1d_dilation: int = 1, bias: bool = True):
        assert not self.__setup_done_flag

        self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer = number_of_independent_channels * conv1d_input_channels
        self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer = number_of_independent_channels * conv1d_output_channels
        self.height_when_seen_as_a_2d_conv_layer = number_of_parallel_channels
        self.input_width_when_seen_as_a_2d_conv_layer = signal_length

        number_of_groups_when_seen_as_a_2d_conv_layer = number_of_independent_channels
        kernel_size_when_seen_as_a_2d_conv_layer = (1, conv1d_kernel_size)
        stride_when_seen_as_a_2d_conv_layer = (1, conv1d_stride)
        padding_when_seen_as_a_2d_conv_layer = (0, conv1d_padding)
        dilation_when_seen_as_a_2d_conv_layer = (1, conv1d_dilation)

        self.output_width_when_seen_as_a_2d_conv_layer = expected_convolution_output_size_along_dimension(
            signal_length, conv1d_kernel_size, conv1d_stride, conv1d_padding, conv1d_dilation)

        self.multichannel_conv1d = Conv2d(
            in_channels=self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer,
            out_channels=self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer,
            kernel_size=kernel_size_when_seen_as_a_2d_conv_layer,
            stride=stride_when_seen_as_a_2d_conv_layer,
            padding=padding_when_seen_as_a_2d_conv_layer,
            dilation=dilation_when_seen_as_a_2d_conv_layer,
            groups=number_of_groups_when_seen_as_a_2d_conv_layer,
            bias=bias
        )

        self.__setup_done_flag = True
        return self.output_width_when_seen_as_a_2d_conv_layer

    # input of shape (batch_size, number_of_independent_channels * conv1d_input_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, output_signal_length)
    def forward(self, input):
        assert self.__setup_done_flag

        input_shape = input.shape
        assert len(input_shape) == 4
        assert input_shape[1:] == (self.total_number_of_input_channels_when_seen_as_a_2d_conv_layer,
                                   self.height_when_seen_as_a_2d_conv_layer,
                                   self.input_width_when_seen_as_a_2d_conv_layer)

        output_shape = [s for s in input_shape]
        output_shape[1] = self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer
        output_shape[-1] = self.output_width_when_seen_as_a_2d_conv_layer
        output_shape = tuple(output_shape)

        transformed_signal = self.multichannel_conv1d(input)

        assert transformed_signal.shape == output_shape
        return transformed_signal

