from math import ceil, floor

# Code to make sense of the Tensorflow 1.x padding="SAME" output, and to apply it to our Pytorch code
# Source: https://stackoverflow.com/a/44242277
import random
from typing import Union, Tuple


def expected_output_size_along_dimension_assuming_SAME_padding(input_size: int, stride: int):
    return ceil(input_size / stride)


def expected_padding_values_along_dimension_assuming_SAME_padding(input_size: int, kernel_size: int, stride: int):
    expected_output_size = expected_output_size_along_dimension_assuming_SAME_padding(input_size, stride)
    total_padding_value = max((expected_output_size - 1) * stride + kernel_size - input_size, 0)
    expected_padding_value_at_the_beginning = total_padding_value // 2
    expected_padding_value_at_the_end = total_padding_value - expected_padding_value_at_the_beginning
    return expected_padding_value_at_the_beginning, expected_padding_value_at_the_end

# End of adapted code


def expected_convolution_output_size_along_dimension(input_size: int, kernel_size: int, stride: int,
                                                     padding: Union[int, Tuple[int, int]], dilation: int):
    if isinstance(padding, tuple):
        assert len(padding) == 2
        total_padding = sum(padding)
    else:
        total_padding = padding * 2  # Padding applied to both sides

    numerator = input_size + total_padding - dilation * (kernel_size - 1) - 1
    return floor((numerator / stride) + 1)


def expected_convolution_output_size_along_dimension_when_passing_is_SAME_and_dimension_is_1(input_size: int,
                                                                                             kernel_size: int,
                                                                                             stride: int):
    padding = expected_padding_values_along_dimension_assuming_SAME_padding(input_size, kernel_size, stride)
    return expected_convolution_output_size_along_dimension(input_size, kernel_size, stride, padding, 1)


if __name__ == "__main__":
    for test_input_size in random.sample(range(10, 10000), 100):
        test_kernel_size = random.randint(1, test_input_size)
        assert expected_convolution_output_size_along_dimension_when_passing_is_SAME_and_dimension_is_1(test_input_size, test_kernel_size, 1)\
            == expected_output_size_along_dimension_assuming_SAME_padding(test_input_size, 1) == test_input_size

    known_input_size = 7680

    known_kernel_size_1 = 50
    known_stride_1 = 6
    known_output_size_1 = 1280

    known_kernel_size_2 = 128
    known_stride_2 = 16
    known_output_size_2 = 480

    assert expected_convolution_output_size_along_dimension_when_passing_is_SAME_and_dimension_is_1(known_input_size, known_kernel_size_1, known_stride_1) == expected_output_size_along_dimension_assuming_SAME_padding(known_input_size, known_stride_1) == known_output_size_1
    assert expected_convolution_output_size_along_dimension_when_passing_is_SAME_and_dimension_is_1(known_input_size, known_kernel_size_2, known_stride_2) == expected_output_size_along_dimension_assuming_SAME_padding(known_input_size, known_stride_2) == known_output_size_2





