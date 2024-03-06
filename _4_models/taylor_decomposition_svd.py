import torch
from torch.autograd import Function
from torch.linalg import LinAlgError
from _4_models.utils import save_error_tensor


# Adapted from: https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Taylor.py


# Taylor polynomial to approximate SVD gradients (See the derivation in the paper.)
def taylor_polynomial(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dtype = s.dtype
    I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.view(s.shape[0],s.shape[1],1).repeat(1, 1, s.shape[1])
    a1_t = a1.transpose(1,2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[1], s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(100):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


# Class Eigen_decomposition in the original code
# Not intended for complex-valued matrices
class SVDWithRobustBackpropagation(Function):

    @staticmethod
    def forward(ctx, input, epsilon=None):
        assert not torch.is_complex(input)

        try:
            # _, eig_diag, eig_vec = torch.svd(p, some=True, compute_uv=True)
            _, eig_diag, eig_vec_transposed = torch.linalg.svd(input, full_matrices=False)
            eig_vec = eig_vec_transposed.transpose(-2, -1)  # Equivalent result to the commented line

        except LinAlgError as error:
            save_error_tensor(input, error, "torch.linalg.svd", SVDWithRobustBackpropagation.forward,
                              SVDWithRobustBackpropagation)
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


taylor_svd = SVDWithRobustBackpropagation.apply


def standard_svd_with_custom_format(input):
    assert not torch.is_complex(input)

    try:
        # _, eig_diag, eig_vec = torch.svd(p, some=True, compute_uv=True)
        _, eig_diag, eig_vec_transposed = torch.linalg.svd(input, full_matrices=False)
        eig_vec = eig_vec_transposed.transpose(-2, -1)  # Equivalent result to the commented line

    except LinAlgError as error:
        save_error_tensor(input, error, "torch.linalg.svd", standard_svd_with_custom_format)
        raise error

    return eig_vec, eig_diag
