import warnings
import numpy as np
from numpy.linalg import LinAlgError
from typing import Any, List, Union, Tuple, Dict, Optional
import torch
from pyriemann.utils import mean_euclid
from pyriemann.utils.base import sqrtm, invsqrtm, logm, expm
from pyriemann.utils.utils import check_weights
from scipy.signal import butter, lfilter, periodogram
from scipy.stats import zscore
from pyriemann.utils.covariance import covariances
from pyriemann.utils.test import is_sym_pos_def, is_sym
import pyriemann.utils.test
from _4_models.utils import save_error_tensor


# signal of shape (signal_length,) or (number_of_signals, signal_length)
SIGNAL_PREPROCESSING_STRATEGIES = ["raw_signals", "z_score_normalization"]
def signal_preprocessing(signal: np.ndarray, strategy: str, **kwargs):
    assert 0 < len(signal.shape) <= 2
    assert strategy in SIGNAL_PREPROCESSING_STRATEGIES
    if strategy == "raw_signals":
        return signal
    elif strategy == "z_score_normalization":
        return signal_z_score_normalization(signal)
    else:
        raise NotImplementedError


def signal_z_score_normalization(signal: np.ndarray):
    return zscore(signal, axis=-1)


# signal of shape (signal_length,) or (number_of_signals, signal_length)
CHANNEL_TRANSFORMATIONS = ["none", "bandpass_filtering"]
def channel_wise_signal_transformation(signal: np.ndarray, channel_transformation_type: str,
                                       channel_transformation_config: Any, **kwargs):
    assert 0 < len(signal.shape) <= 2
    assert channel_transformation_type in CHANNEL_TRANSFORMATIONS
    if channel_transformation_type == "none":
        assert channel_transformation_config is None
        return signal
    elif channel_transformation_type == "bandpass_filtering":
        return signal_bandpass_filtering(signal, channel_transformation_config, **kwargs)


DEFAULT_BANDPASS_FILTER_ORDER = 4
def signal_bandpass_filtering(signal: np.ndarray, filter_config: List, sampling_frequency: float):
    assert len(filter_config) >= 2

    lowcut = filter_config[0]
    highcut = filter_config[1]

    if len(filter_config) == 2:
        order = DEFAULT_BANDPASS_FILTER_ORDER
    elif len(filter_config) == 3:
        order = filter_config[2]
    else:
        raise NotImplementedError

    Wn = np.array([lowcut, highcut]) / (sampling_frequency / 2)
    b, a = butter(order, Wn, "bandpass")
    return lfilter(b, a, signal, axis=-1)


# signal of shape (..., signal_length)
# output of shape (..., number_of_windows, window_size_in_steps)
# or (..., number_of_sequences, length_of_sequences, window_size_in_steps)
def subdivide_signal(signal: np.ndarray, window_size_in_steps: int,
                     length_of_sequences_if_any: Union[int, None] = None):
    signal_length = signal.shape[-1]
    total_number_of_windows = int(signal_length / window_size_in_steps)
    assert total_number_of_windows == signal_length / window_size_in_steps  # Making sure it's an integer

    windowed_signal = np.stack(np.split(signal, total_number_of_windows, axis=-1), axis=-2)

    if length_of_sequences_if_any is None:
        return windowed_signal

    total_number_of_sequences = int(total_number_of_windows / length_of_sequences_if_any)
    assert total_number_of_sequences == total_number_of_windows / length_of_sequences_if_any  # Making sure it's an integer

    return windowed_signal.reshape(-1, total_number_of_sequences, length_of_sequences_if_any, window_size_in_steps)


# signal of shape (number_of_signals, number_of_sequences, length_of_sequences, window_size_in_steps)
# or (number_of_signals, number_of_windows, window_size_in_steps)
# output_signal of shape (total_number_of_windows, number_of_signals, window_size_in_steps)
# number_of_windows_tuple of shape (number_of_sequences, length_of_sequences) or (number_of_windows,)
def batch_windowed_signal_reformatting(signal: np.ndarray):
    assert 3 <= len(signal.shape) <= 4
    
    if len(signal.shape) == 4:
        number_of_signals, number_of_sequences, length_of_sequences, window_size_in_steps = signal.shape
        total_number_of_windows = number_of_sequences * length_of_sequences
        number_of_windows_tuple = (number_of_sequences, length_of_sequences)
        signal = signal.reshape(number_of_signals, total_number_of_windows, window_size_in_steps)
    else:
        assert len(signal.shape) == 3
        number_of_signals, number_of_windows, window_size_in_steps = signal.shape
        total_number_of_windows = number_of_windows
        number_of_windows_tuple = (number_of_windows,)

    # (total_number_of_windows, number_of_signals, window_size_in_steps)
    output_signal = signal.transpose(1, 0, 2)
    assert output_signal.shape == (total_number_of_windows, number_of_signals, window_size_in_steps)
    
    return output_signal, number_of_windows_tuple


# matrices of shape (total_number_of_matrices, matrix_size, matrix_size)
# output of shape (number_of_sequences, length_of_sequences, matrix_size, matrix_size)
# or (number_of_matrices, matrix_size, matrix_size)
def batch_covariance_matrices_reformatting(matrices: np.ndarray, number_of_matrices_tuple: Tuple):
    assert len(matrices.shape) == 3
    total_number_of_matrices, matrix_size, matrix_size_2 = matrices.shape
    assert matrix_size == matrix_size_2
    
    assert 1 <= len(number_of_matrices_tuple) <= 2
    if len(number_of_matrices_tuple) == 1:
        assert total_number_of_matrices == number_of_matrices_tuple[0]
        return matrices  # No transformation

    number_of_sequences, length_of_sequences = number_of_matrices_tuple
    assert total_number_of_matrices == number_of_sequences * length_of_sequences
    return matrices.reshape(number_of_sequences, length_of_sequences, matrix_size, matrix_size)


# vector of shape (total_number_of_vectors, vector_size)
# output of shape (number_of_sequences, length_of_sequences, vector_size)
# or (number_of_vectors, vector_size)
def batch_statistic_vectors_reformatting(vectors: np.ndarray, number_of_vectors_tuple: Tuple):
    assert len(vectors.shape) == 2
    total_number_of_vectors, vector_size = vectors.shape

    assert 1 <= len(number_of_vectors_tuple) <= 2
    if len(number_of_vectors_tuple) == 1:
        assert total_number_of_vectors == number_of_vectors_tuple[0]
        return vectors  # No transformation

    number_of_sequences, length_of_sequences = number_of_vectors_tuple
    assert total_number_of_vectors == number_of_sequences * length_of_sequences
    return vectors.reshape(number_of_sequences, length_of_sequences, vector_size)


def is_pos_def(matrix: np.ndarray):
    try:
        answer = pyriemann.utils.test.is_pos_def(matrix)
    except LinAlgError as e:
        answer = False
    return answer


# matrices of shape (..., matrix_size, matrix_size), must be symmetric
def apply_epsilon_to_diagonal(matrices: torch.Tensor, eigenvalue_threshold: float, epsilon: float,
                              numpy_eigvals: bool = True):
    matrices_shape = matrices.shape
    matrix_size = matrices_shape[-1]
    assert matrices_shape[-2] == matrix_size
    dtype = matrices.dtype
    device = matrices.device

    if not is_sym(matrices):
        matrices = torch.triu(matrices) + torch.triu(matrices, diagonal=1).transpose(-2, -1)

    identity = torch.eye(n=matrix_size, dtype=dtype, device=device)

    # (unified_batch_size, matrix_size, matrix_size)
    matrices = matrices.view(-1, matrix_size, matrix_size)

    # shape (unified_batch_size, matrix_size)
    if numpy_eigvals:
        eigenvalues = np.linalg.eigvals(matrices.numpy())
        eigenvalues = torch.tensor(eigenvalues)
    else:
        eigenvalues = torch.linalg.eigvalsh(matrices, UPLO="U")

    # shape (unified_batch_size)
    try:
        min_eigenvalue_per_matrix, _ = eigenvalues.min(dim=-1)
    except RuntimeError as error:
        if torch.is_complex(eigenvalues):
            min_eigenvalue_real_part_per_matrix, _ = torch.real(eigenvalues).min(dim=-1)
            min_eigenvalue_imaginary_part_per_matrix, _ = torch.imag(eigenvalues).min(dim=-1)
            if not torch.all(min_eigenvalue_imaginary_part_per_matrix <= eigenvalue_threshold):
                save_error_tensor(matrices, error, "Non-negligible complex eigenvalues", apply_epsilon_to_diagonal)
                assert False  # Shouldn't trigger
            else:
                min_eigenvalue_per_matrix = min_eigenvalue_real_part_per_matrix
        else:
            save_error_tensor(matrices, error, "Problematic non-complex eigenvalues", apply_epsilon_to_diagonal)
            assert False  # Shouldn't trigger

    bad_matrix_indices = (min_eigenvalue_per_matrix <= eigenvalue_threshold).nonzero()
    assert len(bad_matrix_indices.shape) == 2
    assert bad_matrix_indices.shape[-1] == 1

    for bad_matrix_global_index in range(len(bad_matrix_indices)):
        bad_matrix_index = bad_matrix_indices[bad_matrix_global_index, 0]
        bad_matrix = matrices[bad_matrix_index, :, :]
        matrices[bad_matrix_index, :, :] = bad_matrix + epsilon * identity

    # (..., matrix_size, matrix_size)
    matrices = matrices.view(matrices_shape)

    return matrices


def apply_epsilon_to_diagonal_numpy(matrices: torch.Tensor, eigenvalue_threshold: float, epsilon: float):
    return apply_epsilon_to_diagonal(torch.tensor(matrices), eigenvalue_threshold, epsilon, numpy_eigvals=True).numpy()


# matrices of shape (..., matrix_size, matrix_size)
def spd_matrices_correction(matrices: Union[np.ndarray, torch.Tensor], min_epsilon: float = 1e-10,
                            max_epsilon: float = 5e-2, max_number_of_correction_loops_after_max_epsilon: int = 4):
    matrices = torch.tensor(matrices)
    matrix_size = matrices.shape[-1]
    assert matrices.shape[-2] == matrix_size

    dtype = matrices.dtype
    absolute_min_epsilon = torch.finfo(dtype).eps  # Smallest non-zero acceptable value
    epsilon = max(min_epsilon, absolute_min_epsilon)
    assert epsilon < max_epsilon

    corrected_sym = False
    used_epsilon = False
    max_epsilon_attained_countdown = np.inf
    while not is_sym_pos_def(matrices):

        if not is_sym(matrices):
            corrected_sym = True
            matrices = torch.triu(matrices) + torch.triu(matrices, diagonal=1).transpose(-2, -1)

        if not is_pos_def(matrices):
            used_epsilon = True

            if epsilon > max_epsilon:
                epsilon = max_epsilon
                if max_epsilon_attained_countdown > max_number_of_correction_loops_after_max_epsilon:
                    max_epsilon_attained_countdown = max_number_of_correction_loops_after_max_epsilon

            matrices = apply_epsilon_to_diagonal(matrices, absolute_min_epsilon, epsilon)
            epsilon *= 2  # Double epsilon each time

            if max_epsilon_attained_countdown <= 0 and not is_sym_pos_def(matrices):
                error = ValueError("Epsilon too high!")
                save_error_tensor(matrices, error, "Epsilon above " + str(max_epsilon),
                                  spd_matrices_correction)
                raise error

            max_epsilon_attained_countdown -= 1

    if corrected_sym and used_epsilon:
        warnings.warn("Corrected asymmetry in SPD matrices, as well as non-positive eigenvalues,"
                      " with a final epsilon of " + str(epsilon))
    elif corrected_sym:
        warnings.warn("Corrected asymmetry in SPD matrices")
    if used_epsilon:
        warnings.warn("Corrected SPD matrices with non-positive eigenvalues, with a final epsilon of " + str(epsilon))

    return matrices.numpy()


# signal of shape (total_number_of_windows, number_of_signals, window_size_in_steps)
# number_of_matrices_tuple of shape (number_of_sequences, length_of_sequences) or (number_of_windows,)
# output of shape (number_of_sequences, length_of_sequences, number_of_signals, number_of_signals)
# or (number_of_windows, number_of_signals, number_of_signals)
ESTIMATORS = ["cov", "mcd", "oas"]
def batch_windowed_signal_to_covariance_matrices(signal: np.ndarray, number_of_matrices_tuple: Tuple,
                                                 covariance_estimator: str,
                                                 estimator_extra_args: Union[Dict[str, Any], None] = None,
                                                 correct_non_spd_matrices: bool = False):
    assert covariance_estimator in ESTIMATORS
    estimator_kwargs = {}
    if covariance_estimator == "mcd":
        assert "random_state" in estimator_extra_args
        estimator_kwargs["random_state"] = estimator_extra_args["random_state"]

    assert np.isfinite(signal).all()
    assert len(signal.shape) == 3
    total_number_of_windows, number_of_signals, window_size_in_steps = signal.shape

    covariance_matrices = covariances(signal, estimator=covariance_estimator, **estimator_kwargs)
    assert np.isfinite(covariance_matrices).all()

    try:
        if not correct_non_spd_matrices:
            assert is_sym_pos_def(covariance_matrices)
        assert np.isreal(covariance_matrices).all()
        assert len(covariance_matrices.shape) == 3
    except AssertionError as error:
        save_error_tensor(torch.tensor(covariance_matrices), error, "Generated matrices not SPD",
                          batch_windowed_signal_to_covariance_matrices)
        raise error

    if correct_non_spd_matrices:
        if not is_sym_pos_def(covariance_matrices):
            try:
                covariance_matrices = spd_matrices_correction(covariance_matrices)
                assert is_sym_pos_def(covariance_matrices)
            except AssertionError as error:
                save_error_tensor(torch.tensor(covariance_matrices), error, "Corrected matrices still not SPD",
                                  batch_windowed_signal_to_covariance_matrices)
                raise error

    total_number_of_matrices, matrix_dim_1, matrix_dim_2 = covariance_matrices.shape
    assert (total_number_of_matrices, matrix_dim_1, matrix_dim_2)\
           == (total_number_of_windows, number_of_signals, number_of_signals)
    
    return batch_covariance_matrices_reformatting(covariance_matrices, number_of_matrices_tuple)


# signal of shape (number_of_signals, window_size_in_steps)
# output of shape (number_of_signals, number_of_signals)
def single_window_signal_to_covariance_matrices(signal: np.ndarray, covariance_estimator: str,
                                                estimator_extra_args: Union[Dict[str, Any], None] = None,
                                                correct_non_spd_matrices: bool = False):
    signal_shape = signal.shape
    assert len(signal_shape) == 2
    signal = np.expand_dims(signal, axis=0)
    number_of_matrices_tuple = (1,)
    single_covariance_matrix = batch_windowed_signal_to_covariance_matrices(signal, number_of_matrices_tuple,
                                                                            covariance_estimator, estimator_extra_args,
                                                                            correct_non_spd_matrices)
    assert single_covariance_matrix.shape == (1, signal_shape[0], signal_shape[0])
    return np.squeeze(single_covariance_matrix, axis=0)


# signal of shape (total_number_of_windows, number_of_signals, window_size_in_steps)
# number_of_vectors_tuple of shape (number_of_sequences, length_of_sequences) or (number_of_windows,)
# output of shape (number_of_sequences, length_of_sequences, number_of_signals)
# or (_number_of_windows, number_of_signals)
STATISTICS = ["psd", "mean", "max_minus_min"]
def batch_windowed_signal_to_statistic_vectors(signal: np.ndarray, number_of_vectors_tuple: Tuple, statistic: str,
                                               **kwargs):
    assert statistic in STATISTICS
    assert len(signal.shape) == 3
    total_number_of_windows, number_of_signals, window_size_in_steps = signal.shape

    if statistic == "mean":
        statistic_vectors = batch_windowed_signal_to_mean_vectors_computation(signal)
    elif statistic == "max_minus_min":
        statistic_vectors = batch_windowed_signal_to_amplitude_differential_vectors_computation(signal)
    elif statistic == "psd":
        statistic_vectors = batch_windowed_signal_to_power_spectral_density_vectors_computation(signal, **kwargs)
    else:
        raise NotImplementedError

    assert np.isreal(statistic_vectors).all()
    assert len(statistic_vectors.shape) == 2
    total_number_of_vectors, vector_size = statistic_vectors.shape
    assert (total_number_of_vectors, vector_size) == (total_number_of_windows, number_of_signals)

    return batch_statistic_vectors_reformatting(statistic_vectors, number_of_vectors_tuple)


# signal of shape (number_of_signals, total_number_of_windows, window_size_in_steps)
# output of shape (number_of_signals, total_number_of_windows)
def batch_windowed_signal_to_mean_vectors_computation(signal: np.ndarray):
    return signal.mean(axis=-1)


# signal of shape (number_of_signals, total_number_of_windows, window_size_in_steps)
# output of shape (number_of_signals, total_number_of_windows)
def batch_windowed_signal_to_amplitude_differential_vectors_computation(signal: np.ndarray):
    signal_max = signal.max(axis=-1)
    signal_min = signal.min(axis=-1)
    return signal_max - signal_min


# signal of shape (number_of_signals, total_number_of_windows, window_size_in_steps)
# output of shape (number_of_signals, total_number_of_windows)
def batch_windowed_signal_to_power_spectral_density_vectors_computation(signal: np.ndarray, sampling_frequency: float):
    _, signal_periodograms = periodogram(signal, sampling_frequency, axis=-1)
    signal_mean_psd = signal_periodograms.mean(axis=-1)
    return signal_mean_psd


# matrices of shape (..., matrix_size, matrix_size)
def batch_remove_non_diagonal_elements_from_matrices(matrices: np.ndarray):
    assert len(matrices.shape) >= 2
    matrix_size, matrix_size_2 = matrices.shape[-2:]

    identity = np.identity(matrix_size)
    identity_expanded = np.broadcast_to(identity, matrices.shape)
    return np.multiply(matrices, identity_expanded)


# matrices of shape (..., matrix_size, matrix_size)
# output of shape (matrix_size, matrix_size)
def batch_spd_matrices_affine_invariant_mean(matrices: np.ndarray, remove_non_diagonal_values: bool = False,
                                             correct_non_spd_matrices: bool = False,
                                             impose_minimal_eigenvalue: Optional[float] = None):
    assert len(matrices.shape) >= 2
    matrix_size, matrix_size_2 = matrices.shape[-2:]
    assert matrix_size == matrix_size_2

    matrices = matrices.reshape(-1, matrix_size, matrix_size)
    if remove_non_diagonal_values:
        matrices = batch_remove_non_diagonal_elements_from_matrices(matrices)

    try:
        if not correct_non_spd_matrices:
            assert is_sym_pos_def(matrices)
        assert np.isreal(matrices).all()
    except AssertionError as error:
        save_error_tensor(torch.tensor(matrices), error, "Input matrices not SPD",
                          batch_spd_matrices_affine_invariant_mean)
        raise error

    if correct_non_spd_matrices:
        if not is_sym_pos_def(matrices):
            try:
                matrices = spd_matrices_correction(matrices)
                assert is_sym_pos_def(matrices)
            except AssertionError as error:
                save_error_tensor(torch.tensor(matrices), error, "Corrected matrices still not SPD",
                                  batch_spd_matrices_affine_invariant_mean)
                raise error

    mean_matrix = mean_riemann(matrices, maxiter=500, correct_non_spd_matrices=correct_non_spd_matrices,
                               impose_minimal_eigenvalue=impose_minimal_eigenvalue)
    assert mean_matrix.shape == (matrix_size, matrix_size)
    assert is_sym_pos_def(mean_matrix)
    assert np.isreal(mean_matrix).all()

    return mean_matrix


# vectors of shape (..., vector_size)
# output of shape (vector_size,)
def batch_vectors_euclidean_mean(vectors: np.ndarray):
    assert len(vectors.shape) >= 1
    vector_size = vectors.shape[-1]

    vectors = vectors.reshape(-1, vector_size)
    assert np.isreal(vectors).all()

    mean_vector = vectors.mean(axis=0)
    assert mean_vector.shape == (vector_size,)
    assert np.isreal(mean_vector).all()

    return mean_vector


# Copied and modified PyRiemann's mean implementation due to instabilities in the iterative process
def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None, correct_non_spd_matrices: bool = False,
                 impose_minimal_eigenvalue: Optional[float] = None):
    r"""Mean of SPD matrices according to the Riemannian metric.

    The affine-invariant Riemannian mean minimizes the sum of squared
    affine-invariant Riemannian distances :math:`d_R` to all matrices [1]_:

    .. math::
         \arg \min_{\mathbf{C}} \sum_i w_i d_R (\mathbf{C}, \mathbf{C}_i)^2

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    tol : float, default=10e-9
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n_channels, n_channels), default=None
        A SPD matrix used to initialize the gradient descent.
        If None, the weighted Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Affine-invariant Riemannian mean.

    References
    ----------
    .. [1] `A differential geometric approach to the geometric mean of
        symmetric positive-definite matrices
        <https://epubs.siam.org/doi/10.1137/S0895479803436937>`_
        M. Moakher, SIAM Journal on Matrix Analysis and Applications.
        Volume 26, Issue 3, 2005
    """
    n_matrices, _, _ = covmats.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        C = mean_euclid(covmats, sample_weight=sample_weight)
    else:
        C = init

    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        C12, Cm12 = sqrtm(C), invsqrtm(C)

        # Injected change
        tmp_matrix = Cm12 @ covmats @ Cm12

        if impose_minimal_eigenvalue is not None:
            tmp_matrix = apply_epsilon_to_diagonal_numpy(tmp_matrix, impose_minimal_eigenvalue,
                                                         impose_minimal_eigenvalue)

        if correct_non_spd_matrices and not is_sym_pos_def(tmp_matrix):
            if impose_minimal_eigenvalue is not None:
                tmp_matrix = spd_matrices_correction(tmp_matrix, min_epsilon=impose_minimal_eigenvalue)
            else:
                tmp_matrix = spd_matrices_correction(tmp_matrix)

        # Injection end

        J = np.einsum('a,abc->bc', sample_weight, logm(tmp_matrix))
        C = C12 @ expm(nu * J) @ C12

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    else:
        warnings.warn('Convergence not reached')

    return C