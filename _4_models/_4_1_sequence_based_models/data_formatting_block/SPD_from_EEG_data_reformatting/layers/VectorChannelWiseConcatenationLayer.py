import torch
from torch.nn import Module


class VectorChannelWiseConcatenationLayer(Module):

    def __init__(self):
        super(VectorChannelWiseConcatenationLayer, self).__init__()
        self.__setup_done_flag = False
        self.__just_remove_channel_flag = True

        self.vector_size = None
        self.number_of_channels = None
        self.index_of_channel_dimension = None
        self.concatenated_vector_size = None

    def setup(self, vector_size: int, number_of_channels: int, index_of_channel_dimension: int):
        assert not self.__setup_done_flag
        self.vector_size = vector_size
        self.number_of_channels = number_of_channels
        self.index_of_channel_dimension = index_of_channel_dimension
        assert self.vector_size > 0
        assert number_of_channels >= 1
        self.concatenated_vector_size = self.vector_size * self.number_of_channels

        self.__just_remove_channel_flag = number_of_channels == 1
        self.__setup_done_flag = True
        return self.concatenated_vector_size

    # spd_matrices_to_vectorize of shape (<prior dimensions>, channels, <later dimensions>, vector_size)
    # output of shape (<prior dimensions>, <later dimensions>, vector_size * channels)
    def forward(self, vectors_to_concatenate: torch.Tensor):
        assert self.__setup_done_flag

        pre_concatenation_shape = vectors_to_concatenate.shape
        assert len(pre_concatenation_shape) >= 2 and self.index_of_channel_dimension != len(pre_concatenation_shape) - 1
        assert pre_concatenation_shape[self.index_of_channel_dimension] == self.number_of_channels
        assert pre_concatenation_shape[-1] == self.vector_size

        if self.__just_remove_channel_flag:
            output_vector = vectors_to_concatenate.unsqueeze(dim=self.index_of_channel_dimension)
            assert len(output_vector.shape) == len(pre_concatenation_shape) - 1
            return output_vector

        vectors_in_channel_wise_list = torch.unbind(vectors_to_concatenate, dim=self.index_of_channel_dimension)
        concatenated_vectors = torch.cat(vectors_in_channel_wise_list, dim=-1)

        post_concatenation_shape = concatenated_vectors.shape
        assert post_concatenation_shape == tuple([pre_concatenation_shape[i]
                                                  for i in range(len(pre_concatenation_shape)-1)
                                                  if i != self.index_of_channel_dimension]
                                                 + [self.concatenated_vector_size])

        return concatenated_vectors







