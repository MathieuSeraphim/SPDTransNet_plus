from typing import Any, Union
import torch
from _4_models.taylor_decomposition_eigh import taylor_eigh, standard_eigh_with_custom_format
from _4_models.taylor_decomposition_svd import taylor_svd
from _4_models.utils import non_finite_values_check


spd_decomposition_operators_dict = {
    "svd": taylor_svd,
    "eig": taylor_eigh
}

symmetric_matrices_decomposition_operators_dict = {
    "eig": standard_eigh_with_custom_format
}

def batch_spd_matrices_operation(spd_matrices: torch.Tensor, operation_on_vectorized_diagonal: Any,
                                 epsilon: Union[float, None] = None, decomposition_operator: str = "svd"):
    assert decomposition_operator in spd_decomposition_operators_dict.keys()
    decomposition_operator_function = spd_decomposition_operators_dict[decomposition_operator]
    # non_finite_values_check(spd_matrices, batch_spd_matrices_operation)

    spd_matrices_shape = spd_matrices.shape
    spd_matrices = spd_matrices.view(-1, spd_matrices_shape[-2], spd_matrices_shape[-1])

    # negative_eigenvalues_check(spd_matrices)  # Very costly time-wise!
    eigenvectors, eigenvalues_diagonal = decomposition_operator_function(spd_matrices, epsilon)

    # non_finite_values_check(eigenvectors, batch_spd_matrices_operation)
    # non_finite_values_check(eigenvalues_diagonal, batch_spd_matrices_operation)

    transformed_eigenvalues_diagonal = operation_on_vectorized_diagonal(eigenvalues_diagonal)
    transformed_eigenvalues = transformed_eigenvalues_diagonal.diag_embed()
    transformed_output = eigenvectors @ transformed_eigenvalues @ eigenvectors.transpose(-2, -1)

    non_finite_values_check(transformed_output, batch_spd_matrices_operation)

    transformed_output = transformed_output.view(spd_matrices_shape)
    return transformed_output


def batch_symmetric_matrices_operation(symmetric_matrices: torch.Tensor, operation_on_vectorized_diagonal: Any,
                                       decomposition_operator: str = "eig", check_epsilon_after_operation: bool = False,
                                       epsilon: Union[float, None] = None):
    assert decomposition_operator in symmetric_matrices_decomposition_operators_dict.keys()
    decomposition_operator_function = symmetric_matrices_decomposition_operators_dict[decomposition_operator]
    # non_finite_values_check(symmetric_matrices, batch_symmetric_matrices_operation)

    symmetric_matrices_shape = symmetric_matrices.shape
    symmetric_matrices = symmetric_matrices.view(-1, symmetric_matrices_shape[-2], symmetric_matrices_shape[-1])

    eigenvectors, eigenvalues_diagonal = decomposition_operator_function(symmetric_matrices)

    # non_finite_values_check(eigenvectors, batch_symmetric_matrices_operation)
    # non_finite_values_check(eigenvalues_diagonal, batch_symmetric_matrices_operation)

    transformed_eigenvalues_diagonal = operation_on_vectorized_diagonal(eigenvalues_diagonal)
    if check_epsilon_after_operation:
        dtype = transformed_eigenvalues_diagonal.dtype
        if epsilon is None:
            epsilon = torch.finfo(dtype).eps
        else:
            epsilon = max(epsilon, torch.finfo(dtype).eps)
        transformed_eigenvalues_diagonal[transformed_eigenvalues_diagonal <= epsilon] = epsilon

    transformed_eigenvalues = transformed_eigenvalues_diagonal.diag_embed()
    transformed_output = eigenvectors @ transformed_eigenvalues @ eigenvectors.transpose(-2, -1)

    non_finite_values_check(transformed_output, batch_symmetric_matrices_operation)

    transformed_output = transformed_output.view(symmetric_matrices_shape)
    return transformed_output


# Used to project SPD matrices into the set of symmetric matrices
def matrix_log(input_matrices: torch.Tensor, epsilon: Union[float, None] = None, decomposition_operator: str = "svd",
               use_spd_specific_operation: bool = True):
    if use_spd_specific_operation:
        return batch_spd_matrices_operation(input_matrices, torch.log, epsilon, decomposition_operator)
    return batch_symmetric_matrices_operation(input_matrices, torch.log, decomposition_operator)


def matrix_pow(input_matrices: torch.Tensor, pow: float, epsilon: Union[float, None] = None,
               decomposition_operator: str = "svd", use_spd_specific_operation: bool = True):
    def my_pow(vectorized_matrices: torch.Tensor):
        return torch.pow(vectorized_matrices, pow)
    if use_spd_specific_operation:
        return batch_spd_matrices_operation(input_matrices, my_pow, epsilon, decomposition_operator)
    return batch_symmetric_matrices_operation(input_matrices, my_pow, decomposition_operator)


def matrix_exp(input_matrices: torch.Tensor, epsilon: Union[float, None] = None, decomposition_operator: str = "eig",
               use_spd_specific_operation: bool = False, check_epsilon_after_operation: bool = True):
    if use_spd_specific_operation:
        return batch_spd_matrices_operation(input_matrices, torch.exp, epsilon, decomposition_operator)
    return batch_symmetric_matrices_operation(input_matrices, torch.exp, decomposition_operator,
                                              check_epsilon_after_operation, epsilon)
