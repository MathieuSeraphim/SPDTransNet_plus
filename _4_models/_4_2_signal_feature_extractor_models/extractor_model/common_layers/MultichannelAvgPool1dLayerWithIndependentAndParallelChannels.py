from _4_models._4_2_signal_feature_extractor_models.extractor_model.utils import \
    expected_convolution_output_size_along_dimension
from torch.nn import Module, AvgPool2d


class MultichannelAvgPool1dLayerWithIndependentAndParallelChannels(Module):
    avg_pool1d_dilation = 1  # Average pooling doesn't have a dilation argument

    # Normal AvgPool1d:
    #     input of shape (batch_size, avg_pool1d_channels, input_length)
    #     output of shape (batch_size, avg_pool1d_channels, output_length)
    # AvgPool1d with parallel channels (should go through the same weights, but shouldn't mix) - implemented with AvgPool2d
    # with kernel height at 1, and:
    #     input of shape (batch_size, avg_pool1d_channels, number_of_parallel_channels, input_length)
    #     output of shape (batch_size, avg_pool1d_channels, number_of_parallel_channels, output_length)
    # Now, adding independent channels (should have their own weights, but still included in the same tensor for
    # optimized parallel computations) as groups (cf AvgPool1d/2d documentation), we have:
    #     input of shape (batch_size, number_of_independent_channels * avg_pool1d_channels, number_of_parallel_channels, input_length)
    #     output of shape (batch_size, number_of_independent_channels * avg_pool1d_channels, number_of_parallel_channels, output_length)
    # Independent and parallel channels are fixed for the entire CNN.
    def __init__(self):
        super(MultichannelAvgPool1dLayerWithIndependentAndParallelChannels, self).__init__()
        self.__setup_done_flag = False

        self.total_number_of_channels = None
        self.height_when_seen_as_a_2d_avg_pool_layer = None
        self.input_width_when_seen_as_a_2d_avg_pool_layer = None
        self.output_width_when_seen_as_a_2d_avg_pool_layer = None

        self.multichannel_max_pool1d = None

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              regular_channels: int, avg_pool1d_kernel_size: int, avg_pool1d_stride: int = 1,
              avg_pool1d_padding: int = 0):
        assert not self.__setup_done_flag

        self.total_number_of_channels = number_of_independent_channels * regular_channels
        self.height_when_seen_as_a_2d_avg_pool_layer = number_of_parallel_channels
        self.input_width_when_seen_as_a_2d_avg_pool_layer = signal_length

        kernel_size_when_seen_as_a_2d_avg_pool_layer = (1, avg_pool1d_kernel_size)
        stride_when_seen_as_a_2d_avg_pool_layer = (1, avg_pool1d_stride)
        padding_when_seen_as_a_2d_avg_pool_layer = (0, avg_pool1d_padding)

        self.output_width_when_seen_as_a_2d_avg_pool_layer = expected_convolution_output_size_along_dimension(
            signal_length, avg_pool1d_kernel_size, avg_pool1d_stride, avg_pool1d_padding, self.avg_pool1d_dilation)

        self.multichannel_avg_pool1d = AvgPool2d(
            kernel_size=kernel_size_when_seen_as_a_2d_avg_pool_layer,
            stride=stride_when_seen_as_a_2d_avg_pool_layer,
            padding=padding_when_seen_as_a_2d_avg_pool_layer
        )

        self.__setup_done_flag = True
        return self.output_width_when_seen_as_a_2d_avg_pool_layer

    # input of shape (batch_size, number_of_independent_channels * regular_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels * regular_channels, number_of_parallel_channels, output_signal_length)
    def forward(self, input):
        assert self.__setup_done_flag

        input_shape = input.shape
        assert len(input_shape) == 4
        assert input_shape[1:] == (self.total_number_of_channels,
                                   self.height_when_seen_as_a_2d_avg_pool_layer,
                                   self.input_width_when_seen_as_a_2d_avg_pool_layer)

        output_shape = [s for s in input_shape]
        output_shape[1] = self.total_number_of_channels
        output_shape[-1] = self.output_width_when_seen_as_a_2d_avg_pool_layer
        output_shape = tuple(output_shape)

        transformed_signal = self.multichannel_avg_pool1d(input)

        assert transformed_signal.shape == output_shape
        return transformed_signal

