from typing import Optional
import torch
from torch.nn import Module
from _4_models.spd_matrix_powers_and_log import matrix_log


class MatrixVectorizationLayer(Module):

    def __init__(self):
        super(MatrixVectorizationLayer, self).__init__()
        self.__setup_done_flag = False

        self.matrix_size = None
        self.vector_size = None
        self.epsilon_if_any = None
        self.decomposition_operator = None
        self.regularization_term_added_to_diagonal = None

    def setup(self, matrix_size: int, singular_or_eigen_value_minimum: Optional[float] = None,
              decomposition_operator: str = "svd", regularization_term_added_to_diagonal: Optional[float] = 0):
        assert not self.__setup_done_flag
        self.matrix_size = matrix_size
        self.vector_size = int((matrix_size * (matrix_size + 1)) / 2)

        if singular_or_eigen_value_minimum is not None:
            assert singular_or_eigen_value_minimum > 0
        self.epsilon_if_any = singular_or_eigen_value_minimum
        self.decomposition_operator = decomposition_operator

        if regularization_term_added_to_diagonal is not None and regularization_term_added_to_diagonal != 0:
            self.regularization_term_added_to_diagonal = regularization_term_added_to_diagonal

        self.__setup_done_flag = True
        return self.vector_size

    # spd_matrices_to_vectorize of shape (..., matrix_size, matrix_size)
    # output of shape (..., matrix_size * (matrix_size + 1) / 2)
    def forward(self, spd_matrices_to_vectorize: torch.Tensor):
        assert self.__setup_done_flag

        matrices_shape = spd_matrices_to_vectorize.shape
        assert matrices_shape[-2] == matrices_shape[-1] == self.matrix_size
        output_shape = [*matrices_shape[:-2], self.vector_size]

        if self.regularization_term_added_to_diagonal is not None:
            identity = torch.eye(self.matrix_size, dtype=spd_matrices_to_vectorize.dtype, device=spd_matrices_to_vectorize.device)
            identity = identity.expand(*matrices_shape[:-2], self.matrix_size, self.matrix_size)
            spd_matrices_to_vectorize = spd_matrices_to_vectorize + self.regularization_term_added_to_diagonal * identity

        symmetric_matrices_to_vectorize = matrix_log(spd_matrices_to_vectorize, epsilon=self.epsilon_if_any,
                                                     decomposition_operator=self.decomposition_operator)
        assert symmetric_matrices_to_vectorize.shape == matrices_shape

        upper_triangular_mask = torch.triu(torch.ones(matrices_shape)) == 1
        vectorized_matrices = symmetric_matrices_to_vectorize[upper_triangular_mask].view(output_shape)

        return vectorized_matrices



