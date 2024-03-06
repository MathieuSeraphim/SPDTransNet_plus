from torch import Tensor, unsqueeze
from torch.nn import Module, Sequential, MaxPool1d


seconds_per_epoch = 30

class IITNetResNet50WithOnlyMaxPoolings(Module):

    layer1_num_modules = 3
    layer2_num_modules = 4
    layer3_num_modules = 6
    layer4_num_modules = 3

    layer1_middle_stride = 1
    layer2_middle_stride = 2
    layer3_middle_stride = 2
    layer4_middle_stride = 2

    @staticmethod
    def make_layer(num_modules, middle_stride):
        layer_modules = []
        current_middle_stride = middle_stride  # Only applied to the first block in the layer
        for i in range(num_modules):
            layer_modules.append(
                MaxPool1d(kernel_size=1, stride=1)
            )
            layer_modules.append(
                MaxPool1d(kernel_size=3, stride=current_middle_stride, padding=1)
            )
            layer_modules.append(
                MaxPool1d(kernel_size=1, stride=1)
            )
            current_middle_stride = 1
        return Sequential(*layer_modules)

    def __init__(self):
        super(IITNetResNet50WithOnlyMaxPoolings, self).__init__()

        self.initial_layer = Sequential(
            MaxPool1d(kernel_size=7, stride=2, padding=3),
            MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(self.layer1_num_modules, self.layer1_middle_stride)
        self.layer2 = self.make_layer(self.layer2_num_modules, self.layer2_middle_stride)
        self.max_pool = MaxPool1d(kernel_size=3, stride=2)
        self.layer3 = self.make_layer(self.layer3_num_modules, self.layer3_middle_stride)
        self.layer4 = self.make_layer(self.layer4_num_modules, self.layer4_middle_stride)

    def forward(self, x):

        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.max_pool(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def get_testing_tensor(sampling_rate):
    tensor_length = seconds_per_epoch * sampling_rate
    tensor = Tensor(range(tensor_length))
    tensor = unsqueeze(tensor, dim=0)
    return tensor


if __name__ == "__main__":
    model = IITNetResNet50WithOnlyMaxPoolings()

    sampling_rate = 256
    expected_output_length = 120
    positive_input = get_testing_tensor(sampling_rate)
    negative_input = -positive_input
    upper_sample_index_per_receptive_field = model(positive_input)
    lower_sample_index_per_receptive_field = -model(negative_input)
    assert upper_sample_index_per_receptive_field.shape == lower_sample_index_per_receptive_field.shape == (1, expected_output_length)
    assert expected_output_length > 1
    upper_sample_index_per_receptive_field = upper_sample_index_per_receptive_field.squeeze()
    lower_sample_index_per_receptive_field = lower_sample_index_per_receptive_field.squeeze()

    receptive_fields_string = "Feature index,Receptive field lower step,Receptive field higher step\n"
    for i in range(expected_output_length):
        lower_index = int(lower_sample_index_per_receptive_field[i])
        upper_index = int(upper_sample_index_per_receptive_field[i])
        receptive_fields_string += "%d,%d,%d\n" % (i, lower_index, upper_index)

    output_file = "IITNet_modified_Resnet50_outputs_receptive_fields.csv"
    with open(output_file, "w") as f:
        f.write(receptive_fields_string)




    
