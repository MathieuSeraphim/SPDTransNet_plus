import torch
from torch.autograd import Function
from torch.linalg import LinAlgError
from _4_models.taylor_decomposition_svd import taylor_polynomial
from _4_models.utils import save_error_tensor

# !!! IMPORTANT !!!
# This doesn't improve performance. Please just ignore my failures.
# Left here because I can't be bothered to remove all references to eigh.

# Adapted the SVDWithRobustBackpropagation class of taylor_decomposition_svd.py to use the eigh function instead


# Only use with SPD matrices!
# Not intended for complex-valued matrices
class EIGHWithRobustBackpropagation(Function):

    @staticmethod
    def forward(ctx, input, epsilon=None):
        assert not torch.is_complex(input)

        try:
            eig_diag, eig_vec = torch.linalg.eigh(input, UPLO="U")  # The upper triangular is preferred in this project

        except LinAlgError as error:
            save_error_tensor(input, error, "torch.linalg.eigh", EIGHWithRobustBackpropagation.forward, EIGHWithRobustBackpropagation)
            raise error

        dtype = eig_diag.dtype
        if epsilon is None:
            epsilon = torch.finfo(dtype).eps
        else:
            epsilon = max(epsilon, torch.finfo(dtype).eps)

        eig_diag[eig_diag <= epsilon] = epsilon  # Zero-out eigenvalues smaller than epsilon
        ctx.save_for_backward(eig_vec, eig_diag)
        return eig_vec, eig_diag

    @staticmethod
    # Same as for SVDWithRobustBackpropagation, since the output should theoretically be the same
    def backward(ctx, grad_output1, grad_output2):
        eig_vec, eig_diag = ctx.saved_tensors
        eig_diag = eig_diag.diag_embed()
        eig_vec_deri, eig_diag_deri = grad_output1, grad_output2
        k = taylor_polynomial(eig_diag)

        # Gradient Overflow Check;
        k[k == float('inf')] = k[k != float('inf')].max()
        k[k == float('-inf')] = k[k != float('-inf')].min()
        k[k != k] = k.max()
        grad_input = (k.transpose(1, 2) * (eig_vec.transpose(1, 2).bmm(eig_vec_deri))) + torch.diag_embed(eig_diag_deri)

        # Gradient Overflow Check;
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        grad_input = eig_vec.bmm(grad_input).bmm(eig_vec.transpose(1, 2))

        # Gradient Overflow Check;
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()

        return grad_input, None  # Must have as many outputs as the forward() method has inputs


taylor_eigh = EIGHWithRobustBackpropagation.apply


def standard_eigh_with_custom_format(input):
    assert not torch.is_complex(input)

    try:
        eig_diag, eig_vec = torch.linalg.eigh(input, UPLO="U")  # The upper triangular is preferred in this project

    except LinAlgError as error:
        save_error_tensor(input, error, "torch.linalg.eigh", standard_eigh_with_custom_format)
        raise error

    return eig_vec, eig_diag
