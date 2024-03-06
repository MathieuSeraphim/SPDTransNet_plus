import copy
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Parameter
from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelConv1dLayerWithIndependentAndParallelChannels import \
    MultichannelConv1dLayerWithIndependentAndParallelChannels


class MultichannelSequentialLinearLayerWithIndependentAndParallelChannels(Module):

    def __init__(self):
        super(MultichannelSequentialLinearLayerWithIndependentAndParallelChannels, self).__init__()
        self.__setup_done_flag = False

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.number_of_regular_channels = None
        self.input_length_per_group = None
        self.output_length_per_group = None
        self.number_of_groups = None
        self.regular_channels_combined_with_independent_channels_flag = None

        self.layers_list = ModuleList()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, input_length: int,
              output_length: int, number_of_regular_channels: int, number_of_shared_weight_groups: int = 1,
              bias: bool = True, regular_channels_combined_with_independent_channels: bool = True):
        assert not self.__setup_done_flag

        assert number_of_shared_weight_groups >= 1
        assert input_length % number_of_shared_weight_groups == 0
        assert output_length % number_of_shared_weight_groups == 0

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels
        self.number_of_regular_channels = number_of_regular_channels
        self.input_length_per_group = input_length // number_of_shared_weight_groups
        self.output_length_per_group = output_length // number_of_shared_weight_groups
        self.number_of_groups = number_of_shared_weight_groups
        self.regular_channels_combined_with_independent_channels_flag = regular_channels_combined_with_independent_channels

        for channel in range(self.number_of_independent_channels):
            self.layers_list.append(Linear(self.input_length_per_group, self.output_length_per_group, bias=bias))

        self.__setup_done_flag = True

    # input of shape (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, input_length = number_of_shared_weight_groups * input_length_per_group)
    # output of shape (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, output_length = number_of_shared_weight_groups * output_length_per_group)
    # OR
    # input of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, input_length = number_of_shared_weight_groups * input_length_per_group)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, output_length = number_of_shared_weight_groups * output_length_per_group)
    def forward(self, input: Tensor):
        assert self.__setup_done_flag

        input_shape = input.shape

        if self.regular_channels_combined_with_independent_channels_flag:
            assert len(input_shape) == 4
            assert input_shape[1:] == (self.number_of_independent_channels * self.number_of_regular_channels,
                                       self.number_of_parallel_channels,
                                       self.input_length_per_group * self.number_of_groups)

            intermediary_shape = (
                input_shape[0],
                self.number_of_independent_channels,
                self.number_of_regular_channels,
                self.number_of_parallel_channels,
                input_shape[-1]
            )
            # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups * input_length_per_group)
            intermediary = input.view(intermediary_shape)

        else:
            assert len(input_shape) == 5
            assert input_shape[1:] == (self.number_of_independent_channels,
                                       self.number_of_parallel_channels,
                                       self.number_of_regular_channels,
                                       self.input_length_per_group * self.number_of_groups)

            intermediary = input

        # number_of_shared_weight_groups tensors of shape
        # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, input_length_per_group)
        # OR
        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, input_length_per_group)
        split_intermediary = torch.tensor_split(intermediary, self.number_of_groups, dim=-1)

        # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups, input_length_per_group)
        # OR
        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups, input_length_per_group)
        combined_linear_inputs = torch.stack(split_intermediary, dim=-2)
        
        # number_of_independent_channels tensors of shape        
        # (batch_size, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups, input_length_per_group)
        # OR
        # (batch_size, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups, input_length_per_group)
        linear_inputs_list = torch.unbind(combined_linear_inputs, dim=1)
        assert len(linear_inputs_list) == self.number_of_independent_channels

        # number_of_independent_channels tensors of shape
        # (batch_size, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups, output_length_per_group)
        # OR
        # (batch_size, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups, output_length_per_group)
        linear_outputs_list = []
        for i in range(self.number_of_independent_channels):
            linear_outputs_list.append(self.layers_list[i](linear_inputs_list[i]))

        # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups, output_length_per_group)
        # OR
        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups, output_length_per_group)
        combined_linear_outputs = torch.stack(linear_outputs_list, dim=1)

        if self.regular_channels_combined_with_independent_channels_flag:
            assert combined_linear_outputs.shape == (
                input_shape[0],
                self.number_of_independent_channels,
                self.number_of_regular_channels,
                self.number_of_parallel_channels,
                self.number_of_groups,
                self.output_length_per_group
            )

            # (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups * output_length_per_group)
            output = combined_linear_outputs.view(
                input_shape[0],
                self.number_of_independent_channels * self.number_of_regular_channels,
                self.number_of_parallel_channels,
                self.number_of_groups * self.output_length_per_group
            )

        else:
            assert combined_linear_outputs.shape == (
                input_shape[0],
                self.number_of_independent_channels,
                self.number_of_parallel_channels,
                self.number_of_regular_channels,
                self.number_of_groups,
                self.output_length_per_group
            )

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups * output_length_per_group)
            output = combined_linear_outputs.view(
                input_shape[0],
                self.number_of_independent_channels,
                self.number_of_parallel_channels,
                self.number_of_regular_channels,
                self.number_of_groups * self.output_length_per_group
            )

        return output


class MultichannelLinearLayerWithIndependentAndParallelChannels(MultichannelConv1dLayerWithIndependentAndParallelChannels):

    conv_1d_kernel_size = 1

    def __init__(self):
        super(MultichannelLinearLayerWithIndependentAndParallelChannels, self).__init__()

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.number_of_regular_channels = None
        self.input_length_per_group = None
        self.output_length_per_group = None
        self.number_of_groups = None
        self.regular_channels_combined_with_independent_channels_flag = None

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, input_length: int,
              output_length: int, number_of_regular_channels: int, number_of_shared_weight_groups: int = 1,
              bias: bool = True, regular_channels_combined_with_independent_channels: bool = True):
        assert number_of_shared_weight_groups >= 1
        assert input_length % number_of_shared_weight_groups == 0
        assert output_length % number_of_shared_weight_groups == 0

        conv1d_input_channels = input_length // number_of_shared_weight_groups
        conv1d_output_channels = output_length // number_of_shared_weight_groups
        conv1d_signal_length = number_of_regular_channels * number_of_shared_weight_groups

        conv1d_output_length = super(MultichannelLinearLayerWithIndependentAndParallelChannels, self).setup(
            number_of_independent_channels=number_of_independent_channels,
            number_of_parallel_channels=number_of_parallel_channels,
            signal_length=conv1d_signal_length,
            conv1d_input_channels=conv1d_input_channels,
            conv1d_output_channels=conv1d_output_channels,
            conv1d_kernel_size=self.conv_1d_kernel_size,
            bias=bias
        )  # Asserts False if setup was previously done
        assert conv1d_output_length == conv1d_signal_length

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels  # Same as self.height_when_seen_as_a_2d_conv_layer
        self.number_of_regular_channels = number_of_regular_channels
        self.input_length_per_group = conv1d_input_channels
        self.output_length_per_group = conv1d_output_channels
        self.number_of_groups = number_of_shared_weight_groups
        self.regular_channels_combined_with_independent_channels_flag = regular_channels_combined_with_independent_channels

    # input of shape (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, input_length = number_of_shared_weight_groups * input_length_per_group)
    # output of shape (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, output_length = number_of_shared_weight_groups * output_length_per_group)
    # OR
    # input of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, input_length = number_of_shared_weight_groups * input_length_per_group)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, output_length = number_of_shared_weight_groups * output_length_per_group)
    def forward(self, input: Tensor):
        assert self._MultichannelConv1dLayerWithIndependentAndParallelChannels__setup_done_flag

        input_shape = input.shape
        intermediary_shape = (
            input_shape[0],
            self.number_of_independent_channels,
            self.number_of_regular_channels,
            self.number_of_parallel_channels,
            input_shape[-1]
        )

        if self.regular_channels_combined_with_independent_channels_flag:
            assert len(input_shape) == 4
            assert input_shape[1:] == (self.number_of_independent_channels * self.number_of_regular_channels,
                                       self.number_of_parallel_channels,
                                       self.input_length_per_group * self.number_of_groups)

            # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups * input_length_per_group)
            intermediary = input.view(intermediary_shape)

        else:
            assert len(input_shape) == 5
            assert input_shape[1:] == (self.number_of_independent_channels,
                                       self.number_of_parallel_channels,
                                       self.number_of_regular_channels,
                                       self.input_length_per_group * self.number_of_groups)

            # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups * input_length_per_group)
            intermediary = torch.transpose(input, 2, 3).contiguous()
            assert intermediary.shape == intermediary_shape

        # number_of_shared_weight_groups tensors of shape
        # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, input_length_per_group)
        split_intermediary = torch.tensor_split(intermediary, self.number_of_groups, dim=-1)

        # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups, input_length_per_group)
        recombined_intermediary = torch.stack(split_intermediary, dim=-2)

        # (batch_size, number_of_independent_channels, input_length_per_group, number_of_parallel_channels, number_of_shared_weight_groups, number_of_regular_channels)
        final_intermediary = torch.transpose(recombined_intermediary, 2, -1).contiguous()
        assert final_intermediary.shape == (
            input_shape[0],
            self.number_of_independent_channels,
            self.input_length_per_group,
            self.number_of_parallel_channels,
            self.number_of_groups,
            self.number_of_regular_channels
        )

        # (batch_size, number_of_independent_channels * input_length_per_group, number_of_parallel_channels, number_of_shared_weight_groups * number_of_regular_channels)
        conv_input_shape = (
            input_shape[0],
            self.number_of_independent_channels * self.input_length_per_group,
            self.number_of_parallel_channels,
            self.number_of_groups * self.number_of_regular_channels
        )
        conv_input = final_intermediary.view(conv_input_shape)

        # (batch_size, number_of_independent_channels * output_length_per_group, number_of_parallel_channels, number_of_shared_weight_groups * number_of_regular_channels)
        conv_output = super(MultichannelLinearLayerWithIndependentAndParallelChannels, self).forward(conv_input)
        assert conv_output.shape == (
            input_shape[0],
            self.number_of_independent_channels * self.output_length_per_group,
            self.number_of_parallel_channels,
            self.number_of_groups * self.number_of_regular_channels
        )

        conv_output_reshaped_shape = (
            input_shape[0],
            self.number_of_independent_channels,
            self.output_length_per_group,
            self.number_of_parallel_channels,
            self.number_of_groups,
            self.number_of_regular_channels
        )
        conv_output_reshaped = conv_output.view(conv_output_reshaped_shape)

        # (batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups, output_length_per_group)
        penultimate_output = torch.transpose(conv_output_reshaped, 2, -1).contiguous()
        assert penultimate_output.shape == (
            input_shape[0],
            self.number_of_independent_channels,
            self.number_of_regular_channels,
            self.number_of_parallel_channels,
            self.number_of_groups,
            self.output_length_per_group
        )

        if not self.regular_channels_combined_with_independent_channels_flag:

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups * output_length_per_group)
            output = torch.transpose(penultimate_output, 2, 3).contiguous()
            output = output.view(
                input_shape[0],
                self.number_of_independent_channels,
                self.number_of_parallel_channels,
                self.number_of_regular_channels,
                self.number_of_groups * self.output_length_per_group
            )

            return output

        # (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups * output_length_per_group)
        output = penultimate_output.view(
            input_shape[0],
            self.number_of_independent_channels * self.number_of_regular_channels,
            self.number_of_parallel_channels,
            self.number_of_groups * self.output_length_per_group
        )

        return output


if __name__ == "__main__":

    batch_size = 32
    number_of_independent_channels = 7
    number_of_parallel_channels = 8
    number_of_regular_channels = 12
    number_of_shared_weight_groups = 30
    input_length_per_group = 4 * 128
    output_length_per_group = 8 * 3
    input_length = number_of_shared_weight_groups * input_length_per_group
    output_length = number_of_shared_weight_groups * output_length_per_group

    model_kwargs = {
        "number_of_independent_channels": number_of_independent_channels,
        "number_of_parallel_channels": number_of_parallel_channels,
        "input_length": input_length,
        "output_length": output_length,
        "number_of_regular_channels": number_of_regular_channels,
        "number_of_shared_weight_groups": number_of_shared_weight_groups
    }

    test_tensor_1 = torch.rand(batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, number_of_shared_weight_groups * input_length_per_group)

    test_tensor_2 = copy.deepcopy(test_tensor_1)
    test_tensor_2 = torch.transpose(test_tensor_2, 2, 3).contiguous()
    test_tensor_2 = test_tensor_2.view((batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, number_of_shared_weight_groups * input_length_per_group))

    output_shape_1 = (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_regular_channels, output_length)
    output_shape_2 = (batch_size, number_of_independent_channels * number_of_regular_channels, number_of_parallel_channels, output_length)

    sequential_model_1 = MultichannelSequentialLinearLayerWithIndependentAndParallelChannels()
    sequential_model_1.setup(**model_kwargs, regular_channels_combined_with_independent_channels=False)

    conv_based_model_1 = MultichannelLinearLayerWithIndependentAndParallelChannels()
    conv_based_model_1.setup(**model_kwargs, regular_channels_combined_with_independent_channels=False)

    sequential_model_2 = MultichannelSequentialLinearLayerWithIndependentAndParallelChannels()
    sequential_model_2.setup(**model_kwargs, regular_channels_combined_with_independent_channels=True)

    conv_based_model_2 = MultichannelLinearLayerWithIndependentAndParallelChannels()
    conv_based_model_2.setup(**model_kwargs, regular_channels_combined_with_independent_channels=True)

    common_weights_tensor = copy.deepcopy(conv_based_model_1.multichannel_conv1d.weight)
    common_bias_tensor = copy.deepcopy(conv_based_model_1.multichannel_conv1d.bias)

    assert conv_based_model_2.multichannel_conv1d.weight.shape == common_weights_tensor.shape
    conv_based_model_2.multichannel_conv1d.weight = copy.deepcopy(common_weights_tensor)
    assert conv_based_model_2.multichannel_conv1d.bias.shape == common_bias_tensor.shape
    conv_based_model_2.multichannel_conv1d.bias = copy.deepcopy(common_bias_tensor)

    common_weights_tensor = common_weights_tensor.squeeze()
    subdivided_weights_tensor_list = torch.tensor_split(common_weights_tensor, 7)
    subdivided_bias_tensor_list = torch.tensor_split(common_bias_tensor, 7)

    for i in range(number_of_independent_channels):
        subdivided_weights_tensor = Parameter(subdivided_weights_tensor_list[i])
        subdivided_bias_tensor = Parameter(subdivided_bias_tensor_list[i])

        assert sequential_model_1.layers_list[i].weight.shape == subdivided_weights_tensor.shape
        sequential_model_1.layers_list[i].weight = copy.deepcopy(subdivided_weights_tensor)
        assert sequential_model_1.layers_list[i].bias.shape == subdivided_bias_tensor.shape
        sequential_model_1.layers_list[i].bias = copy.deepcopy(subdivided_bias_tensor)

        assert sequential_model_2.layers_list[i].weight.shape == subdivided_weights_tensor.shape
        sequential_model_2.layers_list[i].weight = copy.deepcopy(subdivided_weights_tensor)
        assert sequential_model_2.layers_list[i].bias.shape == subdivided_bias_tensor.shape
        sequential_model_2.layers_list[i].bias = copy.deepcopy(subdivided_bias_tensor)

    sequential_model_1_output = sequential_model_1(test_tensor_1)
    conv_based_model_1_output = conv_based_model_1(test_tensor_1)
    sequential_model_2_output = sequential_model_2(test_tensor_2)
    conv_based_model_2_output = conv_based_model_2(test_tensor_2)

    assert sequential_model_1_output.shape == output_shape_1
    assert conv_based_model_1_output.shape == output_shape_1
    assert sequential_model_2_output.shape == output_shape_2
    assert conv_based_model_2_output.shape == output_shape_2

    sequential_model_2_output_reshaped = sequential_model_2_output.view((batch_size, number_of_independent_channels, number_of_regular_channels,  number_of_parallel_channels, output_length))
    sequential_model_2_output_reshaped = sequential_model_2_output_reshaped.transpose(2, 3).contiguous()
    assert sequential_model_2_output_reshaped.shape == sequential_model_1_output.shape

    conv_based_model_2_output_reshaped = conv_based_model_2_output.view((batch_size, number_of_independent_channels, number_of_regular_channels, number_of_parallel_channels, output_length))
    conv_based_model_2_output_reshaped = conv_based_model_2_output_reshaped.transpose(2, 3).contiguous()
    assert conv_based_model_2_output_reshaped.shape == conv_based_model_1_output.shape == sequential_model_1_output.shape

    assert (sequential_model_2_output_reshaped == sequential_model_1_output).all()
    assert (conv_based_model_2_output_reshaped == conv_based_model_1_output).all()
    assert torch.allclose(sequential_model_1_output, conv_based_model_1_output, atol=1e-5, rtol=0)  # Differ from at most 10^-5


