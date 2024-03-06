from typing import List, Optional
import torch
from torch.nn import Module, Sequential
from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelAvgPool1dLayerWithIndependentAndParallelChannels import \
    MultichannelAvgPool1dLayerWithIndependentAndParallelChannels
from _4_models._4_2_signal_feature_extractor_models.extractor_model.zhu_et_al_feature_extractor.layers.MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels import \
    MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels


# Adapted from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7313068/


class MultichannelWindowFeatureLearningBlockWithIndependentAndParallelChannels(Module):

    default_conv1d_padding = 0
    default_conv1d_dilation = 1

    # Constants (we're doing global average pooling)
    avg_pool1d_stride = 1
    avg_pool1d_padding = 0
    final_output_signal_length = 1

    def __init__(self):
        super(MultichannelWindowFeatureLearningBlockWithIndependentAndParallelChannels, self).__init__()
        self.__setup_done_flag = False

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.input_signal_length = None
        self.first_input_channels = None
        self.final_output_channels = None

        self.input_expanded_number_of_independent_channels = None
        self.output_expanded_number_of_independent_channels = None
        self.last_conv_block_output_signal_length = None

        self.conv_blocks_sequence = None
        self.global_avg_pool = MultichannelAvgPool1dLayerWithIndependentAndParallelChannels()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, input_signal_length: int,
              number_of_conv_blocks: int, first_input_channels: int, output_channels_list: List[int],
              conv1d_kernel_sizes_list: List[int], conv1d_strides_list: List[int],
              conv1d_padding_list: Optional[List[int]] = None, conv1d_dilation_list: Optional[List[int]] = None,
              bias: bool = True):
        assert not self.__setup_done_flag

        assert number_of_conv_blocks >= 1
        assert first_input_channels >= 1
        assert len(output_channels_list) == len(conv1d_kernel_sizes_list) == len(conv1d_strides_list) == number_of_conv_blocks

        if conv1d_padding_list is not None:
            assert len(conv1d_padding_list) == number_of_conv_blocks
        else:
            conv1d_padding_list = [self.default_conv1d_padding] * number_of_conv_blocks

        if conv1d_dilation_list is not None:
            assert len(conv1d_dilation_list) == number_of_conv_blocks
        else:
            conv1d_dilation_list = [self.default_conv1d_dilation] * number_of_conv_blocks

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels
        self.input_signal_length = input_signal_length
        self.first_input_channels = first_input_channels
        self.final_output_channels = output_channels_list[-1]

        self.input_expanded_number_of_independent_channels = self.number_of_independent_channels * self.first_input_channels
        self.output_expanded_number_of_independent_channels = self.number_of_independent_channels * self.final_output_channels

        conv_blocks_list = []
        next_conv_block_input_signal_length = input_signal_length
        next_conv_block_input_channels = self.first_input_channels
        for conv_block_id in range(number_of_conv_blocks):
            conv_block = MultichannelWindowFeatureLearningElementWithIndependentAndParallelChannels()
            next_conv_block_input_signal_length = conv_block.setup(self.number_of_independent_channels,
                                                                   self.number_of_parallel_channels,
                                                                   next_conv_block_input_signal_length,
                                                                   next_conv_block_input_channels,
                                                                   output_channels_list[conv_block_id],
                                                                   conv1d_kernel_sizes_list[conv_block_id],
                                                                   conv1d_strides_list[conv_block_id],
                                                                   conv1d_padding_list[conv_block_id],
                                                                   conv1d_dilation_list[conv_block_id],
                                                                   bias)
            conv_blocks_list.append(conv_block)
            next_conv_block_input_channels = output_channels_list[conv_block_id]
        self.conv_blocks_sequence = Sequential(*conv_blocks_list)
        self.last_conv_block_output_signal_length = next_conv_block_input_signal_length
        assert self.final_output_channels == next_conv_block_input_channels

        avg_pool1d_input_signal_length = self.last_conv_block_output_signal_length
        avg_pool1d_regular_channels = self.final_output_channels
        avg_pool1d_kernel_size = avg_pool1d_input_signal_length  # We're doing global average pooling

        avg_pool1d_output_signal_length = self.global_avg_pool.setup(self.number_of_independent_channels,
                                                                     self.number_of_parallel_channels,
                                                                     avg_pool1d_input_signal_length,
                                                                     avg_pool1d_regular_channels,
                                                                     avg_pool1d_kernel_size,
                                                                     self.avg_pool1d_stride,
                                                                     self.avg_pool1d_padding)
        assert avg_pool1d_output_signal_length == self.final_output_signal_length

        final_output_overall_feature_size = self.final_output_channels

        self.__setup_done_flag = True
        return final_output_overall_feature_size

    # input of shape (batch_size, number_of_independent_channels * first_input_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, final_output_overall_feature_size = final_output_channels)
    def forward(self, x):
        assert self.__setup_done_flag

        batch_size = x.shape[0]
        assert x.shape[1:] == (self.input_expanded_number_of_independent_channels, self.number_of_parallel_channels,
                               self.input_signal_length)

        # (batch_size, number_of_independent_channels * final_input_channels, number_of_parallel_channels, last_conv_block_output_signal_length)
        conv_output = self.conv_blocks_sequence(x)
        assert conv_output.shape[1:] == (self.output_expanded_number_of_independent_channels, self.number_of_parallel_channels, self.last_conv_block_output_signal_length)

        # (batch_size, number_of_independent_channels * final_input_channels, number_of_parallel_channels, final_output_signal_length = 1)
        avg_pool_output = self.global_avg_pool(conv_output)
        assert avg_pool_output.shape[1:] == (self.output_expanded_number_of_independent_channels, self.number_of_parallel_channels, self.final_output_signal_length)

        # (batch_size, number_of_independent_channels, final_input_channels, number_of_parallel_channels)
        penultimate_output_shape = (batch_size, self.number_of_independent_channels, self.final_output_channels, self.number_of_parallel_channels)
        penultimate_output = avg_pool_output.view(penultimate_output_shape)

        # (batch_size, number_of_independent_channels, number_of_parallel_channels, final_output_overall_feature_size = final_output_channels)
        final_output = torch.transpose(penultimate_output, -2, -1).contiguous()

        return final_output

