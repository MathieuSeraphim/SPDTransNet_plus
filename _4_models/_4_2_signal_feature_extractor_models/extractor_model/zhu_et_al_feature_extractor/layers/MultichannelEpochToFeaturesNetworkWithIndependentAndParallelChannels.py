from typing import Optional, List
from torchsummary import summary
import torch
from torch.nn import Module
from _4_models._4_2_signal_feature_extractor_models.extractor_model.zhu_et_al_feature_extractor.layers.MultichannelWindowFeatureLearningBlockWithIndependentAndParallelChannels import \
    MultichannelWindowFeatureLearningBlockWithIndependentAndParallelChannels


# Adapted from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7313068/


class MultichannelEpochToFeaturesNetworkWithIndependentAndParallelChannels(Module):

    default_number_of_conv_blocks = 5
    default_output_channels_list = [64, 64, 128, 128, 256]
    default_kernel_sizes_list = [5, 5, 3, 3, 3]
    default_strides_list = [3, 3, 2, 1, 1]

    def __init__(self):
        super(MultichannelEpochToFeaturesNetworkWithIndependentAndParallelChannels, self).__init__()
        self.__setup_done_flag = False
        self.network = MultichannelWindowFeatureLearningBlockWithIndependentAndParallelChannels()
        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.input_shape = None
        self.intermediary_shape = None
        self.conv_input_shape = None
        self.conv_output_shape = None
        self.penultimate_output_shape = None
        self.output_shape = None
        self.number_of_windows = None
        self.window_size = None

    # Keep default values for default behavior
    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, input_signal_length: int,
              number_of_windows_in_signal: int, number_of_conv_blocks: int = 5, first_input_channels: int = 1,
              output_channels_list: Optional[List[int]] = None, conv1d_kernel_sizes_list: Optional[List[int]] = None,
              conv1d_strides_list: Optional[List[int]] = None, conv1d_padding_list: Optional[List[int]] = None,
              conv1d_dilation_list: Optional[List[int]] = None, bias: bool = True):
        assert not self.__setup_done_flag

        assert number_of_conv_blocks >= 1
        assert first_input_channels >= 1

        assert input_signal_length >= number_of_windows_in_signal > 0
        assert input_signal_length % number_of_windows_in_signal == 0
        self.number_of_windows = number_of_windows_in_signal
        self.window_size = input_signal_length // number_of_windows_in_signal

        # If at least one of these lists takes its default value
        if output_channels_list is None or conv1d_kernel_sizes_list is None or conv1d_strides_list is None:
            assert number_of_conv_blocks == self.default_number_of_conv_blocks

            # Set relevant default lists
            if output_channels_list is None:
                output_channels_list = self.default_output_channels_list
            if conv1d_kernel_sizes_list is None:
                conv1d_kernel_sizes_list = self.default_kernel_sizes_list
            if conv1d_strides_list is None:
                conv1d_strides_list = self.default_strides_list

        # Checking that all non-default lists have the proper length
        list_of_input_lists = [output_channels_list, conv1d_kernel_sizes_list, conv1d_strides_list, conv1d_padding_list,
                               conv1d_dilation_list]
        for input_list in list_of_input_lists:
            if input_list is not None:
                assert len(input_list) == number_of_conv_blocks

        output_feature_size = self.network.setup(number_of_independent_channels * self.number_of_windows,
                                                 number_of_parallel_channels,
                                                 self.window_size,
                                                 number_of_conv_blocks,
                                                 first_input_channels,
                                                 output_channels_list,
                                                 conv1d_kernel_sizes_list,
                                                 conv1d_strides_list,
                                                 conv1d_padding_list,
                                                 conv1d_dilation_list,
                                                 bias)
        assert output_feature_size == output_channels_list[-1]

        self.input_shape = (number_of_independent_channels * first_input_channels,
                            number_of_parallel_channels,
                            input_signal_length)
        self.intermediary_shape = (number_of_independent_channels,
                                   first_input_channels,
                                   number_of_parallel_channels,
                                   self.number_of_windows,
                                   self.window_size)
        self.conv_input_shape = (number_of_independent_channels * self.number_of_windows * first_input_channels,
                                 number_of_parallel_channels,
                                 self.window_size)
        self.conv_output_shape = (number_of_independent_channels * self.number_of_windows,
                                  number_of_parallel_channels,
                                  output_feature_size)
        self.penultimate_output_shape = (number_of_independent_channels,
                                         self.number_of_windows,
                                         number_of_parallel_channels,
                                         output_feature_size)
        self.output_shape = (number_of_independent_channels,
                             number_of_parallel_channels,
                             self.number_of_windows,
                             output_feature_size)  # Final feature size

        self.__setup_done_flag = True
        return self.number_of_windows, output_feature_size

    # input of shape (batch_size, number_of_independent_channels * first_input_channels, number_of_parallel_channels, input_signal_length)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_windows, final_feature_size)
    def forward(self, input):
        assert self.__setup_done_flag

        batch_size = input.shape[0]
        batch_size_as_tuple = (batch_size,)
        assert input.shape[1:] == self.input_shape

        intermediary = input.view(batch_size_as_tuple + self.intermediary_shape)
        intermediary = torch.permute(intermediary, (0, 1, 4, 2, 3, 5)).contiguous()
        conv_input = intermediary.view(batch_size_as_tuple + self.conv_input_shape)

        conv_output = self.network(conv_input)
        assert conv_output.shape[1:] == self.conv_output_shape

        penultimate_output = conv_output.view(batch_size_as_tuple + self.penultimate_output_shape)
        output = torch.permute(penultimate_output, (0, 1, 3, 2, 4)).contiguous()
        assert output.shape == batch_size_as_tuple + self.output_shape

        return output


if __name__ == "__main__":

    number_of_independent_channels = 7
    number_of_parallel_channels = 8
    epoch_length_in_seconds = 30
    sampling_frequency = 256
    dropout_rate = 0

    input_signal_length = epoch_length_in_seconds * sampling_frequency
    input_shape = (number_of_independent_channels, number_of_parallel_channels, input_signal_length)
    batched_input_shape = (1,) + input_shape

    model = MultichannelEpochToFeaturesNetworkWithIndependentAndParallelChannels()
    number_of_output_feature_vectors, output_feature_vectors_length = model.setup(
        number_of_independent_channels=number_of_independent_channels,
        number_of_parallel_channels=number_of_parallel_channels,
        input_signal_length=input_signal_length,
        number_of_windows_in_signal=epoch_length_in_seconds
    )
    model.to("cuda:0")

    print(number_of_output_feature_vectors, output_feature_vectors_length)
    summary(model, input_shape)

    input = torch.rand(*batched_input_shape).to("cuda:0")
    output = model(input)
    print(output.shape)



