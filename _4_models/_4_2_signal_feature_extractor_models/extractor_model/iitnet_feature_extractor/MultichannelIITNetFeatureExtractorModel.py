import torch
from torchsummary import summary
from torch.nn import Linear, ModuleList
from _4_models._4_2_signal_feature_extractor_models.extractor_model.BaseExtractorModel import BaseExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.layers.MultichannelIITNetModifiedResnet import \
    MultichannelIITNetModifiedResnet


class MultichannelIITNetFeatureExtractorModel(BaseExtractorModel):

    # The signal has two types of channels: independent (will pass through a different set of weights) ans parallel
    # (same weights, no mixing)
    #
    # For EEG signals, parallel channels correspond to signals originating from different electrodes but that underwent
    # the same preprocessing
    # independent channels correspond to different preprocessing pipelines, and therefore a need for different weights
    def __init__(self):
        super(MultichannelIITNetFeatureExtractorModel, self).__init__()
        self.__setup_done_flag = False

        self.number_of_independent_channels = None
        self.number_of_parallel_channels = None
        self.number_of_output_features = None
        self.output_features_length = None
        self.signal_length = None
        self.uneven_split_flag = None

        self.number_of_larger_splits = None
        self.vectors_per_larger_splits = None
        self.number_of_smaller_splits = None
        self.vectors_per_smaller_splits = None

        self.main_resnet_model = None
        self.larger_splits_resizing_layer = None
        self.resizing_layers = ModuleList()

    def setup(self, number_of_independent_channels: int, number_of_parallel_channels: int, signal_length: int,
              resnet_num_layers: int, dropout_rate: float, number_of_output_features: int,
              single_output_feature_length: int):
        assert not self.__setup_done_flag

        super(MultichannelIITNetFeatureExtractorModel, self).setup(
            single_output_feature_length=single_output_feature_length
        )

        assert number_of_independent_channels > 0
        assert number_of_parallel_channels > 0

        self.number_of_independent_channels = number_of_independent_channels
        self.number_of_parallel_channels = number_of_parallel_channels
        self.signal_length = signal_length

        self.number_of_output_features = number_of_output_features
        self.output_features_length = number_of_output_features * self.single_output_feature_length

        self.main_resnet_model = MultichannelIITNetModifiedResnet()
        resnet_number_of_output_feature_vectors, resnet_output_feature_vectors_length = self.main_resnet_model.setup(
            number_of_independent_channels=number_of_independent_channels,
            number_of_parallel_channels=number_of_parallel_channels,
            input_signal_length=signal_length,
            dropout_rate=dropout_rate,
            resnet_num_layers=resnet_num_layers
        )
        assert resnet_number_of_output_feature_vectors > number_of_output_features

        # We split the array feature vectors into number_of_output_features groups, using the same logic as NumPy's
        # array_split() function:
        #   For an array of length l that should be split into n sections, it returns l % n sub-arrays of size l//n + 1
        #   and the rest of size l//n.
        self.uneven_split_flag = (resnet_number_of_output_feature_vectors % number_of_output_features) != 0
        if self.uneven_split_flag:
            self.number_of_larger_splits = resnet_number_of_output_feature_vectors % number_of_output_features
            self.vectors_per_larger_splits = resnet_number_of_output_feature_vectors // number_of_output_features + 1
            self.number_of_smaller_splits = number_of_output_features - self.number_of_larger_splits
            self.vectors_per_smaller_splits = resnet_number_of_output_feature_vectors // number_of_output_features
        else:
            self.number_of_larger_splits = 0
            self.vectors_per_larger_splits = 0
            self.number_of_smaller_splits = number_of_output_features
            self.vectors_per_smaller_splits = resnet_number_of_output_feature_vectors // number_of_output_features
        assert self.number_of_larger_splits + self.number_of_smaller_splits == number_of_output_features
        assert self.vectors_per_larger_splits * self.number_of_larger_splits + self.vectors_per_smaller_splits * self.number_of_smaller_splits == resnet_number_of_output_feature_vectors

        # If the splits are even, we go from shape
        # (batch_size, number_of_independent_channels, number_of_parallel_channels, resnet_number_of_output_feature_vectors, resnet_output_feature_vectors_length)
        # to shape
        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_splits, vectors_per_split * resnet_output_feature_vectors_length)
        # We then linearly map that (with one Linear layer per independant channel) onto shape
        # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_splits, single_output_feature_length)
        # and then (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_splits * single_output_feature_length = output_features_length)
        # Here, number_of_splits = number_of_output_features
        #
        # If they are uneven, we first resize the larger splits, so that when recombined, we have shape
        # (batch_size, number_of_independent_channels * number_of_splits, number_of_parallel_channels, vectors_per_smaller_splits * output_feature_vectors_length)
        # prior to the aforementioned linear mapping

        if self.uneven_split_flag:
            self.larger_splits_resizing_layer = Linear(self.vectors_per_larger_splits * resnet_output_feature_vectors_length, self.vectors_per_smaller_splits * resnet_output_feature_vectors_length)

        for channel in range(self.number_of_independent_channels):
            self.resizing_layers.append(Linear(self.vectors_per_smaller_splits * resnet_output_feature_vectors_length, self.single_output_feature_length))

        print()
        print("Set up an IITNet-derived feature encoder, with:")
        print("- An input size of %d," % signal_length)
        print("- An output of %d vectors of size %d for the ResNet-derived CNN," % (resnet_number_of_output_feature_vectors, resnet_output_feature_vectors_length))
        print("- A final output of length %d, that is %d features of length %d." % (self.output_features_length, number_of_output_features, single_output_feature_length))
        if self.uneven_split_flag:
            print("As we can't evenly divide %d vectors into %d feature groups, some of the final features will be"
                  "computed from more data." % (resnet_number_of_output_feature_vectors, number_of_output_features))
        print()

        self.__setup_done_flag = True
        return self.output_features_length

    # input_signal of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, signal_length)
    # output of shape (batch_size, number_of_independent_channels, number_of_parallel_channels, output_features_length)
    def forward(self, input_signal):
        assert self.__setup_done_flag

        batch_size = input_signal.shape[0]

        # (batch_size, number_of_independent_channels, number_of_parallel_channels, resnet_number_of_output_feature_vectors, resnet_output_feature_vectors_length)
        resnet_output = self.main_resnet_model(input_signal)
        del input_signal

        resnet_output_shape = resnet_output.shape
        assert resnet_output_shape[-2] == self.vectors_per_larger_splits * self.number_of_larger_splits + self.vectors_per_smaller_splits * self.number_of_smaller_splits

        split_resnet_output = torch.tensor_split(resnet_output, self.number_of_output_features, dim=-2)
        if self.uneven_split_flag:  # see long comment in setup method
            
            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_larger_splits, vectors_per_larger_splits, resnet_output_feature_vectors_length)
            resnet_output_larger_splits = torch.stack(split_resnet_output[:self.number_of_larger_splits], dim=-3)
            assert resnet_output_larger_splits.shape == resnet_output_shape[:3] + (self.number_of_larger_splits, self.vectors_per_larger_splits, resnet_output_shape[-1])

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_smaller_splits, vectors_per_smaller_splits, resnet_output_feature_vectors_length)
            resnet_output_smaller_splits = torch.stack(split_resnet_output[self.number_of_larger_splits:], dim=-3)
            assert resnet_output_smaller_splits.shape == resnet_output_shape[:3] + (self.number_of_smaller_splits, self.vectors_per_smaller_splits, resnet_output_shape[-1])

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_[larger/smaller]_splits, vectors_per_[larger/smaller]_splits * resnet_output_feature_vectors_length)
            resnet_output_larger_splits_new_shape = resnet_output_larger_splits.shape[:-2] + (resnet_output_larger_splits.shape[-2] * resnet_output_larger_splits.shape[-1],)
            resnet_output_larger_splits = resnet_output_larger_splits.view(resnet_output_larger_splits_new_shape)
            resnet_output_smaller_splits_new_shape = resnet_output_smaller_splits.shape[:-2] + (resnet_output_smaller_splits.shape[-2] * resnet_output_smaller_splits.shape[-1],)
            resnet_output_smaller_splits = resnet_output_smaller_splits.view(resnet_output_smaller_splits_new_shape)

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_larger_splits, vectors_per_smaller_splits * resnet_output_feature_vectors_length)
            resnet_output_larger_splits_resized = self.larger_splits_resizing_layer(resnet_output_larger_splits)

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_output_features, vectors_per_smaller_splits * resnet_output_feature_vectors_length)
            resnet_output_as_grouped_splits = torch.cat((resnet_output_larger_splits_resized, resnet_output_smaller_splits), dim=-2)
            assert resnet_output_as_grouped_splits.shape == resnet_output_shape[:3] + (self.number_of_output_features, resnet_output_smaller_splits_new_shape[-1])

        else:
            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_output_features, vectors_per_smaller_splits, resnet_output_feature_vectors_length)
            resnet_output_as_grouped_splits = torch.stack(split_resnet_output, dim=-3)

            # (batch_size, number_of_independent_channels, number_of_parallel_channels, number_of_output_features, vectors_per_smaller_splits * resnet_output_feature_vectors_length)
            resnet_output_as_grouped_splits_new_shape = resnet_output_as_grouped_splits.shape[:-2] + (resnet_output_as_grouped_splits.shape[-2] * resnet_output_as_grouped_splits.shape[-1],)
            resnet_output_as_grouped_splits = resnet_output_as_grouped_splits.view(resnet_output_as_grouped_splits_new_shape)

        # Tuple of number_of_independent_channels tensors of shape (batch_size, number_of_parallel_channels, number_of_output_features, vectors_per_smaller_splits * resnet_output_feature_vectors_length)
        independent_channels_wise_resnet_output = torch.unbind(resnet_output_as_grouped_splits, dim=1)

        assert len(independent_channels_wise_resnet_output) == len(self.resizing_layers) == self.number_of_independent_channels
        output_tensors_list = []
        for i in range(self.number_of_independent_channels):
            output_tensors_list.append(self.resizing_layers[i](independent_channels_wise_resnet_output[i]))

        output = torch.stack(output_tensors_list, dim=1)
        assert output.shape == (batch_size, self.number_of_independent_channels, self.number_of_parallel_channels, self.number_of_output_features, self.single_output_feature_length)

        return output.view(batch_size, self.number_of_independent_channels, self.number_of_parallel_channels, self.output_features_length)


if __name__ == "__main__":

    number_of_independent_channels = 7
    number_of_parallel_channels = 8
    epoch_length_in_seconds = 30
    sampling_frequency = 256
    dropout_rate = 0
    resnet_num_layers = 50
    single_output_feature_length = 3

    input_signal_length = epoch_length_in_seconds * sampling_frequency
    input_shape = (number_of_independent_channels, number_of_parallel_channels, input_signal_length)
    batched_input_shape = (1,) + input_shape

    model = MultichannelIITNetFeatureExtractorModel()
    model.setup(number_of_independent_channels=number_of_independent_channels,
                number_of_parallel_channels=number_of_parallel_channels,
                signal_length=input_signal_length,
                resnet_num_layers=resnet_num_layers,
                dropout_rate=dropout_rate,
                number_of_output_features=epoch_length_in_seconds,
                single_output_feature_length=single_output_feature_length)

    summary(model, input_shape, device="cpu")

    input = torch.rand(*batched_input_shape)
    output = model(input)
    print(output.shape)

    print("-" * 100)
    print("-" * 100)

    epoch_length_in_seconds = 30
    sampling_frequency = 1024  # To ensure resnet_number_of_output_feature_vectors > number_of_output_features

    model = MultichannelIITNetFeatureExtractorModel()
    model.setup(number_of_independent_channels=number_of_independent_channels,
                number_of_parallel_channels=number_of_parallel_channels,
                signal_length=input_signal_length,
                resnet_num_layers=resnet_num_layers,
                dropout_rate=dropout_rate,
                number_of_output_features=epoch_length_in_seconds,
                single_output_feature_length=single_output_feature_length)

    summary(model, input_shape, device="cpu")

    input = torch.rand(*batched_input_shape)
    output = model(input)
    print(output.shape)


