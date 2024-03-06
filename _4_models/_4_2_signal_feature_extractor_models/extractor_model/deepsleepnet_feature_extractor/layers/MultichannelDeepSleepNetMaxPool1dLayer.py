from torch.nn import Module, MaxPool2d
from _4_models._4_2_signal_feature_extractor_models.extractor_model.utils import \
    expected_padding_values_along_dimension_assuming_SAME_padding, expected_convolution_output_size_along_dimension, \
    expected_output_size_along_dimension_assuming_SAME_padding


class MultichannelDeepSleepNetMaxPool1dLayer(Module):

    # DeepSleepNet constants
    DILATION = 1

    def __init__(self):
        super(MultichannelDeepSleepNetMaxPool1dLayer, self).__init__()
        self.__setup_done_flag = False

        self.total_number_of_channels_when_seen_as_a_2d_max_pool_layer = None

        self.height_when_seen_as_a_2d_max_pool_layer = None
        self.input_width_when_seen_as_a_2d_max_pool_layer = None
        self.output_width_when_seen_as_a_2d_max_pool_layer = None

        self.max_pool = None

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, extra_channels: int,
              signal_length: int, kernel_size_1d: int, stride_1d: int):
        assert not self.__setup_done_flag

        self.total_number_of_channels_when_seen_as_a_2d_max_pool_layer = number_of_independent_channels * extra_channels
        self.height_when_seen_as_a_2d_max_pool_layer = number_of_parallel_channels
        self.input_width_when_seen_as_a_2d_max_pool_layer = signal_length

        kernel_size_when_seen_as_a_2d_max_pool_layer = (1, kernel_size_1d)
        stride_when_seen_as_a_2d_max_pool_layer = (1, stride_1d)

        # Original Tensorflow code: padding = SAME
        padding_left, padding_right = expected_padding_values_along_dimension_assuming_SAME_padding(
            signal_length, kernel_size_1d, stride_1d)
        padding = padding_left
        padding_when_seen_as_a_2d_max_pool_layer = (0, padding)
        self.output_width_when_seen_as_a_2d_max_pool_layer = expected_convolution_output_size_along_dimension(
            signal_length, kernel_size_1d, stride_1d, padding, self.DILATION)
        
        if padding_left == padding_right:
            assert self.output_width_when_seen_as_a_2d_max_pool_layer ==\
                   expected_output_size_along_dimension_assuming_SAME_padding(signal_length, stride_1d)
        else:  # Different values of left / right padding unsupported by PyTorch
            assert padding_right == padding_left + 1
            assert self.output_width_when_seen_as_a_2d_max_pool_layer ==\
                   expected_output_size_along_dimension_assuming_SAME_padding(signal_length, stride_1d) - 1

        self.max_pool = MaxPool2d(kernel_size=kernel_size_when_seen_as_a_2d_max_pool_layer,
                                  stride=stride_when_seen_as_a_2d_max_pool_layer,
                                  padding=padding_when_seen_as_a_2d_max_pool_layer,
                                  dilation=self.DILATION)

        self.__setup_done_flag = True
        return self.output_width_when_seen_as_a_2d_max_pool_layer

    # input_signal of shape (batch_size, number_of_independent_channels * extra_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels * extra_channels, number_of_parallel_channels, output_signal_length)
    def forward(self, input_signal):
        assert self.__setup_done_flag

        input_shape = input_signal.shape
        assert len(input_shape) == 4
        assert input_shape[1:] == (self.total_number_of_channels_when_seen_as_a_2d_max_pool_layer,
                                   self.height_when_seen_as_a_2d_max_pool_layer,
                                   self.input_width_when_seen_as_a_2d_max_pool_layer)

        output_shape = [s for s in input_shape]
        output_shape[-1] = self.output_width_when_seen_as_a_2d_max_pool_layer
        output_shape = tuple(output_shape)

        transformed_signal = self.max_pool(input_signal)

        assert transformed_signal.shape == output_shape
        return transformed_signal


