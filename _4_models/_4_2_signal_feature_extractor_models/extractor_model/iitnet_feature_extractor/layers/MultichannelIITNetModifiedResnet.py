from typing import Type
import torch
from torch import nn
from torch.nn import Module, Sequential, Conv2d, BatchNorm1d, Dropout
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.layers.MultichannelIITNetBlockLayers import \
    BasicBlock, Bottleneck, IITNetBlock
from torchsummary import summary
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.layers.MultichannelIITNetInitialAndDownsampleLayer import \
    MultichannelIITNetInitialAndDownsampleLayer


# Adapted from the ResNetFeature class of https://github.com/gist-ailab/IITNet-official/blob/main/models/resnet.py
# I do not pretend to understand the original notation, I only adapted it to suit our purposes
from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelMaxPool1dLayerWithIndependentAndParallelChannels import \
    MultichannelMaxPool1dLayerWithIndependentAndParallelChannels


class MultichannelIITNetModifiedResnet(Module):

    layer_config_dict = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    inplanes = 16

    initial_layer_input_channels = 1
    initial_layer_output_channels = 16
    initial_layer_conv1d_kernel_size = 7
    initial_layer_conv1d_stride = 2
    initial_layer_conv1d_padding = 3
    initial_layer_max_pool1d_kernel_size = 3
    initial_layer_max_pool1d_stride = 2
    initial_layer_max_pool1d_padding = 1
    initial_layer_bias = False

    downsample_kernel_size = 1
    downsample_bias = False

    layer_1_output_channel_parameter = 16
    layer_2_output_channel_parameter = 16
    layer_3_output_channel_parameter = 32
    layer_4_output_channel_parameter = 32
    layer_1_stride = 1
    layer_2_stride = 2
    layer_3_stride = 2
    layer_4_stride = 2

    final_max_pool1d_kernel_size = 3
    final_max_pool1d_stride = 2
    final_max_pool1d_padding = 1

    def __init__(self):
        super(MultichannelIITNetModifiedResnet, self).__init__()
        self.__setup_done_flag = False

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.layers = None
        self.input_signal_length = None
        self.final_signal_length = None
        self.final_regular_channels = None

        self.initial_layer = MultichannelIITNetInitialAndDownsampleLayer()
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.maxpool = None
        self.dropout = None

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, input_signal_length: int,
              dropout_rate: float, resnet_num_layers: int = 50):
        assert not self.__setup_done_flag

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels
        self.input_signal_length = input_signal_length

        assert resnet_num_layers in self.layer_config_dict.keys()
        self.layers = self.layer_config_dict[resnet_num_layers]
        if resnet_num_layers == 18 or resnet_num_layers == 34:
            block = BasicBlock
        elif resnet_num_layers == 50 or resnet_num_layers == 101 or resnet_num_layers == 152:
            block = Bottleneck
        else:
            raise NotImplementedError("Shouldn't trigger.")

        current_signal_length = input_signal_length
        current_signal_length = self.initial_layer.setup(number_of_independent_channels=number_of_independent_channels,
                                                         number_of_parallel_channels=number_of_parallel_channels,
                                                         signal_length=current_signal_length,
                                                         conv1d_input_channels=self.initial_layer_input_channels,
                                                         conv1d_output_channels=self.initial_layer_output_channels,
                                                         conv1d_kernel_size=self.initial_layer_conv1d_kernel_size,
                                                         max_pool1d_kernel_size=self.initial_layer_max_pool1d_kernel_size,
                                                         conv1d_stride=self.initial_layer_conv1d_stride,
                                                         max_pool1d_stride=self.initial_layer_max_pool1d_stride,
                                                         conv1d_padding=self.initial_layer_conv1d_padding,
                                                         max_pool1d_padding=self.initial_layer_max_pool1d_padding,
                                                         bias=self.initial_layer_bias)

        self.layer1, current_signal_length, _ = self._make_layer(current_signal_length, block,
                                                                 self.layer_1_output_channel_parameter, self.layers[0],
                                                                 stride=self.layer_1_stride, first=True)
        self.layer2, current_signal_length, current_regular_channels = self._make_layer(current_signal_length, block,
                                                                                        self.layer_2_output_channel_parameter,
                                                                                        self.layers[1],
                                                                                        stride=self.layer_2_stride)

        self.maxpool = MultichannelMaxPool1dLayerWithIndependentAndParallelChannels()
        current_signal_length = self.maxpool.setup(number_of_independent_channels=number_of_independent_channels,
                                                   number_of_parallel_channels=number_of_parallel_channels,
                                                   signal_length=current_signal_length,
                                                   regular_channels=current_regular_channels,
                                                   max_pool1d_kernel_size=self.final_max_pool1d_kernel_size,
                                                   max_pool1d_stride=self.final_max_pool1d_stride,
                                                   max_pool1d_padding=self.final_max_pool1d_padding)

        self.layer3, current_signal_length, _ = self._make_layer(current_signal_length, block,
                                                                 self.layer_3_output_channel_parameter, self.layers[2],
                                                                 stride=self.layer_3_stride)
        self.layer4, self.final_signal_length, self.final_regular_channels = self._make_layer(
            current_signal_length, block, self.layer_4_output_channel_parameter, self.layers[3], stride=self.layer_4_stride)

        self.dropout = Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        number_of_output_feature_vectors = self.final_signal_length
        output_feature_vectors_length = self.final_regular_channels

        self.__setup_done_flag = True
        return number_of_output_feature_vectors, output_feature_vectors_length

    def _make_layer(self, input_signal_length: int, block: Type[IITNetBlock], planes: int, blocks: int, stride: int = 1,
                    first: bool = False):
        downsample = None

        if (stride != 1 and first is False) or self.inplanes != planes * block.expansion:
            downsample_stride = stride
            if block == BasicBlock:  # Botch to make BasicBlock work
                downsample_stride *= 2

            downsample = MultichannelIITNetInitialAndDownsampleLayer()
            downsample.setup(number_of_independent_channels=self.number_of_independent_channels,
                             number_of_parallel_channels=self.number_of_parallel_channels,
                             signal_length=input_signal_length,
                             conv1d_input_channels=self.inplanes,
                             conv1d_output_channels=planes * block.expansion,
                             conv1d_kernel_size=self.downsample_kernel_size,
                             conv1d_stride=downsample_stride,
                             bias=self.downsample_bias,
                             only_batch_norm=True)

        layers = []
        current_signal_length = input_signal_length
        first_block = block(self.inplanes, planes, self.number_of_independent_channels,
                            self.number_of_parallel_channels, current_signal_length, stride, downsample)
        current_signal_length = first_block.get_output_length()
        current_output_channels = first_block.get_output_channels()
        layers.append(first_block)

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            current_block = block(self.inplanes, planes, self.number_of_independent_channels,
                                  self.number_of_parallel_channels, current_signal_length)
            current_signal_length = current_block.get_output_length()
            current_output_channels = first_block.get_output_channels()
            layers.append(current_block)

        return Sequential(*layers), current_signal_length, current_output_channels

    # shape (batch_size, number_of_independent_channels * 1, number_of_parallel_channels, epoch_length)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_output_feature_vectors, output_feature_vectors_length)
    # For resnet_num_layers = 50, we should always have output_feature_vectors_length = 128 (i.e. 128 channels created
    # throughout the network, cf. the final permutation)
    def forward(self, x):
        assert self.__setup_done_flag

        batch_size = x.shape[0]
        assert x.shape[1:] == (self.number_of_independent_channels, self.number_of_parallel_channels, self.input_signal_length)

        f = self.initial_layer(x)
        del x
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.maxpool(f)
        f = self.layer3(f)
        f = self.layer4(f)

        # (batch_size, number_of_independent_channels * final_regular_channels, number_of_parallel_channels, final_signal_length)
        # Here, final_regular_channels = output_feature_vectors_length and final_signal_length = number_of_output_feature_vectors
        f = self.dropout(f)
        assert f.shape == (batch_size, self.number_of_independent_channels * self.final_regular_channels,
                           self.number_of_parallel_channels, self.final_signal_length)

        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_output_feature_vectors = final_signal_length, output_feature_vectors_length = final_regular_channels)
        # As stated before, the final output length will be the above number of channels (yes, it's weird)
        f = f.view(batch_size, self.number_of_independent_channels, self.final_regular_channels,
                   self.number_of_parallel_channels, self.final_signal_length)
        f = f.permute(0, 1, 3, 4, 2)

        return f


if __name__ == "__main__":

    number_of_independent_channels = 7
    number_of_parallel_channels = 8
    epoch_length_in_seconds = 30
    sampling_frequency = 256
    dropout_rate = 0
    resnet_num_layers = 50

    input_signal_length = epoch_length_in_seconds * sampling_frequency
    input_shape = (number_of_independent_channels, number_of_parallel_channels, input_signal_length)
    batched_input_shape = (1,) + input_shape

    model = MultichannelIITNetModifiedResnet()
    number_of_output_feature_vectors, output_feature_vectors_length = model.setup(
        number_of_independent_channels, number_of_parallel_channels, input_signal_length, dropout_rate,
        resnet_num_layers)
    model.to("cuda:0")

    print(number_of_output_feature_vectors, output_feature_vectors_length)
    summary(model, input_shape)

    input = torch.rand(*batched_input_shape).to("cuda:0")
    output = model(input)
    print(output.shape)
