from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelConv1dLayerWithIndependentAndParallelChannels import \
    MultichannelConv1dLayerWithIndependentAndParallelChannels
from torch.nn import BatchNorm1d, ReLU
from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelMaxPool1dLayerWithIndependentAndParallelChannels import \
    MultichannelMaxPool1dLayerWithIndependentAndParallelChannels


class MultichannelIITNetInitialAndDownsampleLayer(MultichannelConv1dLayerWithIndependentAndParallelChannels):

    def __init__(self):
        super(MultichannelIITNetInitialAndDownsampleLayer, self).__init__()
        self.batch_norm = None
        self.only_batch_norm = False
        self.final_signal_length = None

        self.relu = ReLU()
        self.max_pool = MultichannelMaxPool1dLayerWithIndependentAndParallelChannels()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              conv1d_input_channels: int, conv1d_output_channels: int, conv1d_kernel_size: int,
              max_pool1d_kernel_size: int = None, conv1d_stride: int = 1, max_pool1d_stride: int = 1,
              conv1d_padding: int = 0, max_pool1d_padding: int = 0, conv1d_dilation: int = 1,
              max_pool1d_dilation: int = 1, bias: bool = True, only_batch_norm: int = False):
        current_signal_length = super(MultichannelIITNetInitialAndDownsampleLayer, self).setup(
            number_of_independent_channels, number_of_parallel_channels, signal_length, conv1d_input_channels,
            conv1d_output_channels, conv1d_kernel_size, conv1d_stride, conv1d_padding, conv1d_dilation, bias)

        combined_channels = self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer * number_of_parallel_channels
        self.batch_norm = BatchNorm1d(combined_channels)
        self.only_batch_norm = only_batch_norm

        if not self.only_batch_norm:
            assert max_pool1d_kernel_size is not None
            current_signal_length = self.max_pool.setup(number_of_independent_channels,
                                                        number_of_parallel_channels,
                                                        current_signal_length,
                                                        conv1d_output_channels,
                                                        max_pool1d_kernel_size,
                                                        max_pool1d_stride,
                                                        max_pool1d_padding,
                                                        max_pool1d_dilation)

        self.final_signal_length = current_signal_length
        return current_signal_length


    # input of shape (batch_size, number_of_independent_channels * conv1d_input_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, output_signal_length)
    def forward(self, input):
        assert self._MultichannelConv1dLayerWithIndependentAndParallelChannels__setup_done_flag
        
        # (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, conv_output_signal_length)
        conv_output = super(MultichannelIITNetInitialAndDownsampleLayer, self).forward(input)
        conv_output_shape = conv_output.shape

        # (batch_size, number_of_independent_channels * conv1d_output_channels * number_of_parallel_channels, conv_output_signal_length)
        bn_input = conv_output.view((conv_output_shape[0]), conv_output_shape[1] * conv_output_shape[2], conv_output_shape[3])
        bn_output = self.batch_norm(bn_input)

        # (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, conv_output_signal_length)
        bn_output = bn_output.view(conv_output_shape)

        if self.only_batch_norm:
            return bn_output

        # output of shape (batch_size, number_of_independent_channels * conv1d_output_channels, number_of_parallel_channels, max_pool_output_signal_length)
        relu_output = self.relu(bn_output)
        max_pool_output = self.max_pool(relu_output)

        return max_pool_output
        