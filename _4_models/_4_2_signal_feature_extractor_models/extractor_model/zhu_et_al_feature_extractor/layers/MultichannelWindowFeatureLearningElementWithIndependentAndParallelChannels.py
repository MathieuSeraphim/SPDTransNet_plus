from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelConv1dLayerWithIndependentAndParallelChannels import \
    MultichannelConv1dLayerWithIndependentAndParallelChannels
from torch.nn import BatchNorm1d, ReLU


class MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels(MultichannelConv1dLayerWithIndependentAndParallelChannels):

    def __init__(self):
        super(MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels, self).__init__()
        self.batch_norm = None
        self.relu = ReLU()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              conv1d_input_channels: int, conv1d_output_channels: int, conv1d_kernel_size: int, conv1d_stride: int = 1,
              conv1d_padding: int = 0, conv1d_dilation: int = 1, bias: bool = True):
        output_signal_length = super(MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels, self).setup(
            number_of_independent_channels, number_of_parallel_channels, signal_length, conv1d_input_channels,
            conv1d_output_channels, conv1d_kernel_size, conv1d_stride, conv1d_padding, conv1d_dilation, bias)

        combined_channels = self.total_number_of_output_channels_when_seen_as_a_2d_conv_layer * number_of_parallel_channels
        self.batch_norm = BatchNorm1d(combined_channels)

        return output_signal_length

    # input of shape (batch_size, number_of_independent_channels * input_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels * output_channels, number_of_parallel_channels, output_signal_length)
    def forward(self, input):
        assert self._MultichannelConv1dLayerWithIndependentAndParallelChannels__setup_done_flag

        # (batch_size, number_of_independent_channels * output_channels, number_of_parallel_channels, output_signal_length)
        conv_output = super(MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels, self).forward(input)
        conv_output_shape = conv_output.shape

        # (batch_size, number_of_independent_channels * output_channels * number_of_parallel_channels, output_signal_length)
        bn_input = conv_output.view((conv_output_shape[0]), conv_output_shape[1] * conv_output_shape[2],
                                    conv_output_shape[3])
        bn_output = self.batch_norm(bn_input)

        # (batch_size, number_of_independent_channels * output_channels, number_of_parallel_channels, output_signal_length)
        bn_output = bn_output.view(conv_output_shape)
        final_output = self.relu(bn_output)

        return final_output



