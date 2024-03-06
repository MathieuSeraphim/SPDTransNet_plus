from typing import Union
from torch.nn import Module, BatchNorm1d


class MultichannelDeepSleepNetBatchNorm1dLayer(Module):

    def __init__(self):
        super(MultichannelDeepSleepNetBatchNorm1dLayer, self).__init__()
        self.__setup_done_flag = False

        self.first_channel_dimension = None
        self.second_channel_dimension = None
        self.combined_channel_dimension = None

        self.batch_norm = None

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int,
              number_of_conv1d_channels: int, decay: Union[float, None] = 0.999, epsilon: float = 1.e-5):
        assert not self.__setup_done_flag
        assert number_of_independent_channels > 0
        assert number_of_parallel_channels > 0
        assert number_of_conv1d_channels > 0
        assert 0 <= decay <= 1
        assert 0 <= epsilon

        self.first_channel_dimension = number_of_independent_channels * number_of_conv1d_channels
        self.second_channel_dimension = number_of_parallel_channels
        self.combined_channel_dimension = self.first_channel_dimension * self.second_channel_dimension

        if decay is None:
            momentum = None
        else:
            momentum = 1 - decay
        self.batch_norm = BatchNorm1d(self.combined_channel_dimension, epsilon, momentum)

        self.__setup_done_flag = True

    # input_signal of shape (batch_size, number_of_independent_channels * conv1d_channels, number_of_parallel_channels, signal_length)
    # output of the same shape
    def forward(self, input_signal):
        assert self.__setup_done_flag

        input_shape = input_signal.shape
        assert len(input_shape) == 4
        assert input_shape[1:-1] == (self.first_channel_dimension, self.second_channel_dimension)

        transformed_signal = input_signal.view(input_shape[0], self.combined_channel_dimension, input_shape[-1])
        transformed_signal = self.batch_norm(transformed_signal)

        return transformed_signal.view(input_shape)

