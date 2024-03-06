from torch.nn import Module, BatchNorm1d, ReLU
from _4_models._4_2_signal_feature_extractor_models.extractor_model.common_layers.MultichannelConv1dLayerWithIndependentAndParallelChannels import \
    MultichannelConv1dLayerWithIndependentAndParallelChannels


# input of shape (batch_size, number_of_independent_channels * in_planes, number_of_parallel_channels, input_length)
# output of shape (batch_size, number_of_independent_channels * out_planes, number_of_parallel_channels, output_length)
def conv3(in_planes, out_planes, number_of_independent_channels, number_of_parallel_channels, input_length, stride=1):
    kernel_size = 3
    padding = 1
    bias = False

    conv_layer = MultichannelConv1dLayerWithIndependentAndParallelChannels()
    output_length = conv_layer.setup(number_of_independent_channels=number_of_independent_channels,
                                     number_of_parallel_channels=number_of_parallel_channels,
                                     signal_length=input_length,
                                     conv1d_input_channels=in_planes,
                                     conv1d_output_channels=out_planes,
                                     conv1d_kernel_size=kernel_size,
                                     conv1d_stride=stride,
                                     conv1d_padding=padding,
                                     bias=bias)

    return conv_layer, output_length


class IITNetBlock(Module):
    expansion = None
    expected_output_shape = None
    output_channels = None

    def __init__(self, inplanes, planes, number_of_independent_channels, number_of_parallel_channels, input_length,
                 stride=1, downsample=None):
        super(IITNetBlock, self).__init__()

    def get_output_length(self):
        assert type(self) != IITNetBlock
        assert len(self.expected_output_shape) == 3
        return self.expected_output_shape[2]

    def get_output_channels(self):
        assert self.output_channels is not None
        return self.output_channels


# WARNING: this block evidently wasn't tested in the final version of IITNet, as it included errors.
# I corrected one error in the code (see conv2 & bn2 in __init__()), and modified the stride passed on to downsample to
# make it work (see MultichannelIITNetModifiedResnet._make_layer()), but I have no idea if this is in the spirit of the
# original.
class BasicBlock(IITNetBlock):
    expansion = 1

    def __init__(self, inplanes, planes, number_of_independent_channels, number_of_parallel_channels, input_length,
                 stride=1, downsample=None):
        super(BasicBlock, self).__init__(inplanes, planes, number_of_independent_channels, number_of_parallel_channels,
                                         input_length, stride, downsample)

        current_length = input_length
        combined_channels = number_of_independent_channels * planes * number_of_parallel_channels

        self.conv1, current_length = conv3(inplanes, planes, number_of_independent_channels,
                                           number_of_parallel_channels, current_length, stride)
        self.bn1 = BatchNorm1d(combined_channels)
        self.relu = ReLU(inplace=True)

        # The original implementation defines conv2 and bn2, but never use them
        # I'm choosing to believe that this was a mistake, as it may lead to errors down the mine
        self.conv2, current_length = conv3(planes, planes, number_of_independent_channels,
                                           number_of_parallel_channels, current_length, stride)
        self.bn2 = BatchNorm1d(combined_channels)

        self.downsample = downsample

        self.stride = stride
        self.expected_output_shape = (number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        self.output_channels = planes

    # shape (batch_size, number_of_independent_channels * inplanes, number_of_parallel_channels, input_length)
    # output of shape shape (batch_size, number_of_independent_channels * out_planes, number_of_parallel_channels, output_length)
    # out_planes = planes and output_length = final current_length (see above) if downsample is chosen correctly
    def forward(self, x):
        assert len(x.shape) == 4
        batch_size = x.shape[0]

        residual = x

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = self.conv1(x)
        out_shape_1 = out.shape

        # (batch_size, number_of_independent_channels * planes * number_of_parallel_channels, current_length)
        out = out.view((out_shape_1[0]), out_shape_1[1] * out_shape_1[2], out_shape_1[3])
        out = self.bn1(out)
        out = self.relu(out)

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = out.view(out_shape_1)

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = self.conv2(out)
        out_shape_2 = out.shape

        # (batch_size, number_of_independent_channels * planes * number_of_parallel_channels, current_length)
        out = out.view((out_shape_2[0]), out_shape_2[1] * out_shape_2[2], out_shape_2[3])
        out = self.bn2(out)

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = out.view(out_shape_2)

        # Shape unknown (see thereafter)
        if self.downsample is not None:
            residual = self.downsample(x)

        print(x.shape, out.shape, residual.shape)

        # residual should have the same shape as out_shape_2, or at least be broadcastable onto the same shape
        out += residual
        out = self.relu(out)

        expected_output_shape = (batch_size,) + self.expected_output_shape
        assert out.shape == out_shape_2 == expected_output_shape

        return out


class Bottleneck(IITNetBlock):
    expansion = 4

    def __init__(self, inplanes, planes, number_of_independent_channels, number_of_parallel_channels, input_length,
                 stride=1, downsample=None):
        super(Bottleneck, self).__init__(inplanes, planes, number_of_independent_channels, number_of_parallel_channels,
                                         input_length, stride, downsample)

        current_length = input_length
        self.conv1 = MultichannelConv1dLayerWithIndependentAndParallelChannels()
        current_length = self.conv1.setup(number_of_independent_channels, number_of_parallel_channels, current_length,
                                          inplanes, planes, conv1d_kernel_size=1, bias=False)

        combined_channels = number_of_independent_channels * planes * number_of_parallel_channels
        self.bn1 = BatchNorm1d(combined_channels)

        self.conv2 = MultichannelConv1dLayerWithIndependentAndParallelChannels()
        current_length = self.conv2.setup(number_of_independent_channels, number_of_parallel_channels, current_length,
                                          planes, planes, conv1d_kernel_size=3, conv1d_stride=stride, conv1d_padding=1,
                                          bias=False)
        self.bn2 = BatchNorm1d(combined_channels)  # combined_channels unchanged

        out_planes = planes * self.expansion
        self.conv3 = MultichannelConv1dLayerWithIndependentAndParallelChannels()
        current_length = self.conv3.setup(number_of_independent_channels, number_of_parallel_channels, current_length,
                                          planes, out_planes, conv1d_kernel_size=1, bias=False)

        combined_channels = number_of_independent_channels * out_planes * number_of_parallel_channels
        self.bn3 = BatchNorm1d(combined_channels)
        self.relu = ReLU(inplace=True)

        self.downsample = downsample
        if downsample is not None:
            current_length = downsample.final_signal_length
        
        self.stride = stride
        self.expected_output_shape = (number_of_independent_channels * out_planes, number_of_parallel_channels, current_length)
        self.output_channels = out_planes

    # shape (batch_size, number_of_independent_channels * inplanes, number_of_parallel_channels, input_length)
    # output of shape shape (batch_size, number_of_independent_channels * out_planes, number_of_parallel_channels, output_length)
    # out_planes = planes * expansion and output_length = final current_length (see above) if downsample is chosen correctly
    def forward(self, x):
        assert len(x.shape) == 4
        batch_size = x.shape[0]

        residual = x

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = self.conv1(x)
        out_shape_1 = out.shape

        # (batch_size, number_of_independent_channels * planes * number_of_parallel_channels, current_length)
        out = out.view((out_shape_1[0]), out_shape_1[1] * out_shape_1[2], out_shape_1[3])
        out = self.bn1(out)
        out = self.relu(out)

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = out.view(out_shape_1)

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = self.conv2(out)
        out_shape_2 = out.shape

        # (batch_size, number_of_independent_channels * planes * number_of_parallel_channels, current_length)
        out = out.view((out_shape_2[0]), out_shape_2[1] * out_shape_2[2], out_shape_2[3])
        out = self.bn2(out)
        out = self.relu(out)

        # (batch_size, number_of_independent_channels * planes, number_of_parallel_channels, current_length)
        out = out.view(out_shape_2)

        # (batch_size, number_of_independent_channels * out_planes, number_of_parallel_channels, current_length)
        out = self.conv3(out)
        out_shape_3 = out.shape

        # (batch_size, number_of_independent_channels * out_planes * number_of_parallel_channels, current_length)
        out = out.view((out_shape_3[0]), out_shape_3[1] * out_shape_3[2], out_shape_3[3])
        out = self.bn3(out)

        # (batch_size, number_of_independent_channels * out_planes, number_of_parallel_channels, current_length)
        out = out.view(out_shape_3)

        # residual should have the same shape as out_shape_3, or at least be broadcastable onto the same shape
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        expected_output_shape = (batch_size,) + self.expected_output_shape
        assert out.shape == out_shape_3 == expected_output_shape

        return out

