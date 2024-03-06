import warnings
from os import listdir
from os.path import dirname, realpath, join, isdir, isfile
from typing import Union, Dict
import numpy as np
import torch
from torch import Tensor
from torch.linalg import LinAlgError
from pyriemann.utils.test import is_sym_pos_def, is_sym


def matrix_linalg_errors_test(tensor: Tensor, error_tensor_name: str, log_files_dict: Dict[int, str]):
    if error_tensor_name[-16:-3] != "__LinAlgError":
        return

    if error_tensor_name[:4] != "job_":
        return
    job_id = int(error_tensor_name.split("_")[1])

    assert len(tensor.shape) == 3
    assert tensor.shape[-1] == tensor.shape[-2]

    if job_id not in log_files_dict.keys():
        warnings.warn("Job %d doesn't have a corresponding log file!" % job_id)
        return

    log_filename = log_files_dict[job_id]
    assert isfile(log_filename)

    with open(log_filename) as file:
        lines_list = [line.rstrip() for line in file]

    batch_elements_list = []
    for line in lines_list:
        if line[:50] == "torch._C._LinAlgError: linalg.svd: (Batch element ":
            batch_element = int(line[50:].split(")")[0])
            batch_elements_list.append(batch_element)

    batch_elements_list = list(set(batch_elements_list))
    for element in batch_elements_list:
        print("    Problematic batch element %d:" % element)
        print(tensor[element, :, :].cpu().numpy())
        print()

        print("    Corresponding singular values:")
        print(torch.linalg.svdvals(tensor[element, :, :]).cpu().numpy())
        print()

    # SVD tests

    no_crashes_flag = True
    print("    Performing SVD tests...")
    print()

    try:
        torch.linalg.svd(tensor)
    except LinAlgError:
        print("    torch.linalg.svd on original device (%s) crashed." % str(tensor.device))
        print()
        no_crashes_flag = False

    try:
        torch.svd(tensor)
    except LinAlgError:
        print("    torch.svd on original device (%s) crashed." % str(tensor.device))
        print()
        no_crashes_flag = False

    if str(tensor.device) != "cpu":

        try:
            torch.linalg.svd(tensor.cpu())
        except LinAlgError:
            print("    torch.linalg.svd on CPU crashed.")
            print()
            no_crashes_flag = False

        try:
            torch.svd(tensor.cpu())
        except LinAlgError:
            print("    torch.svd on CPU crashed.")
            print()
            no_crashes_flag = False

    for element in batch_elements_list:

        try:
            torch.linalg.svd(tensor[element, :, :])
        except LinAlgError:
            print("    torch.linalg.svd for element %d on original device (%s) crashed." % (element, str(tensor.device)))
            print()
            no_crashes_flag = False

        try:
            torch.svd(tensor[element, :, :])
        except LinAlgError:
            print("    torch.svd for element %d on original device (%s) crashed." % (element, str(tensor.device)))
            print()
            no_crashes_flag = False

        if str(tensor.device) != "cpu":

            try:
                torch.linalg.svd(tensor[element, :, :].cpu())
            except LinAlgError:
                print("    torch.linalg.svd for element %d on CPU crashed." % element)
                print()
                no_crashes_flag = False

            try:
                torch.svd(tensor[element, :, :].cpu() % element)
            except LinAlgError:
                print("    torch.svd for element %d on CPU crashed.")
                print()
                no_crashes_flag = False

    if no_crashes_flag:
        print("    All tests successful, no crashes.")
        print()


def matrix_spd_checks(tensor: Tensor):
    assert len(tensor.shape) >= 2 and tensor.shape[-2] == tensor.shape[-1]  # Should be checked before function call
    matrix_size = tensor.shape[-1]
    tensor = tensor.view(-1, matrix_size, matrix_size)
    num_matrices = tensor.shape[0]

    print("Assuming that the tensor is supposed to be composed of SPD matrices:")
    print("  Number of matrices:", num_matrices)
    print()

    non_spd_matrices_indices = []
    corresponding_smallest_eigenvalues = []
    is_symmetric = []
    for i in range(num_matrices):
        if not is_sym_pos_def(tensor[i, :, :]):
            non_spd_matrices_indices.append(i)
            is_symmetric.append(is_sym(tensor[i, :, :]))
            eigenvalues = torch.linalg.eigvalsh(tensor[i, :, :], UPLO="U")
            assert len(eigenvalues.shape) == 1
            corresponding_smallest_eigenvalues.append(torch.min(eigenvalues))

    if len(non_spd_matrices_indices) == 0:
        print("  All matrices are SPD.")
    else:
        print("  Non-SPD batch matrices:")
        nb_non_spd_matrices = len(non_spd_matrices_indices)
        for j in range(nb_non_spd_matrices):
            print("    Index %d - symmetric: %s, lowest eigenvalue:" % (non_spd_matrices_indices[j], str(is_symmetric[j])), corresponding_smallest_eigenvalues[j])
            print("      All eigenvalues (obtained through torch.linalg.eigvalsh):", torch.linalg.eigvalsh(tensor[non_spd_matrices_indices[j], :, :], UPLO="U"))
        print()
        print("  In total: %d non-SPD matrices, or %f percent of the tensor." % (nb_non_spd_matrices, nb_non_spd_matrices/num_matrices))
    print()

    indices_of_matrices_with_complex_np_eigenvalues = []
    non_real_eigenvalues = []
    complex_eigenvalues = []
    for i in range(num_matrices):
        numpy_eigenvalues = np.linalg.eigvals(tensor[i, :, :].numpy())
        if not np.all(np.isreal(numpy_eigenvalues)):
            indices_of_matrices_with_complex_np_eigenvalues.append(i)
            non_real_eigenvalues.append(np.logical_not(np.isreal(numpy_eigenvalues)).sum())
            complex_eigenvalues.append(np.iscomplex(numpy_eigenvalues).sum())

    if len(indices_of_matrices_with_complex_np_eigenvalues) == 0:
        print("  All matrices have real eigenvalues, as seen with numpy.linalg.eigvals.")
    else:
        print("  Matrices with non-real eigenvalues, as seen with numpy.linalg.eigvals:")
        nb_bad_matrices = len(indices_of_matrices_with_complex_np_eigenvalues)
        for j in range(nb_bad_matrices):
            print("    Index %d - non-real eigenvalues: %s, complex eigenvalues: %d" % (indices_of_matrices_with_complex_np_eigenvalues[j], non_real_eigenvalues[j], complex_eigenvalues[j]))
            print("      Corresponding eigenvalues:", np.linalg.eigvals(tensor[non_spd_matrices_indices[j], :, :].numpy()))
            print("      ...with np.linalg.eigvalsh:", np.linalg.eigvalsh(tensor[non_spd_matrices_indices[j], :, :].numpy(), UPLO="U"))
            print("      ...with torch.linalg.eigvals:", torch.linalg.eigvals(tensor[non_spd_matrices_indices[j], :, :]))
            print("      ...with torch.linalg.eigvalsh:", torch.linalg.eigvalsh(tensor[non_spd_matrices_indices[j], :, :], UPLO="U"))
        print()
        print("  In total: %d matrices with non-real eigenvalues, or %f percent of the tensor." % (nb_bad_matrices, nb_bad_matrices/num_matrices))
    print()


def open_error_tensors(error_tensors_folder_name: Union[str, None] = None,
                       error_tensors_folder_parent_directory_path: Union[str, None] = None,
                       log_folder_name: Union[str, None] = None,
                       log_folder_parent_directory_path: Union[str, None] = None):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    np.set_printoptions(suppress=True, linewidth=np.nan)

    if error_tensors_folder_name is None:
        error_tensors_folder_name = "error_tensors"

    if error_tensors_folder_parent_directory_path is None:
        error_tensors_folder_parent_directory_path = root_directory

    potential_log_warning_flag = (log_folder_name is not None) or (log_folder_parent_directory_path is not None)
    if log_folder_name is None:
        log_folder_name = "log"

    if log_folder_parent_directory_path is None:
        log_folder_parent_directory_path = root_directory

    error_tensors_folder_path = join(error_tensors_folder_parent_directory_path, error_tensors_folder_name)
    if not isdir(error_tensors_folder_path):
        warnings.warn("Error tensor folder path %s doesn't exist!" % error_tensors_folder_path)
        return

    log_folder_path = join(log_folder_parent_directory_path, log_folder_name)
    no_log_folder_flag = not isdir(log_folder_path)
    if no_log_folder_flag and potential_log_warning_flag:
        warnings.warn("Log folder path %s doesn't exist - skipping log-based displays." % log_folder_path)

    log_files_dict = {}
    if not no_log_folder_flag:
        for log_file_name in listdir(log_folder_path):
            if log_file_name[-4:] == ".err":
                job_index_string = log_file_name.split(".")[0]
                assert job_index_string[:6] == "index_"
                job_index = int(job_index_string[6:])
                assert job_index not in log_files_dict.keys()
                log_files_dict[job_index] = join(log_folder_path, log_file_name)

    error_tensors_names_list = sorted(listdir(error_tensors_folder_path))
    for error_tensor_name in error_tensors_names_list:
        if error_tensor_name[-3:] != ".pt":
            continue

        error_tensor_path = join(error_tensors_folder_path, error_tensor_name)
        if not isfile(error_tensor_path):
            continue

        print("Analyzing error tensor %s..." % error_tensor_name)
        print()

        tensor = torch.load(error_tensor_path)

        print("  Tensor shape: %s" % str(tuple(tensor.shape)))
        print()

        # Non-finite elements

        is_not_finite_in_tensor = torch.logical_not(torch.isfinite(tensor))
        non_finite_indices = torch.nonzero(is_not_finite_in_tensor)
        number_of_non_finite_elements, _ = non_finite_indices.shape
        print("  Number of non-finite elements: %d" % number_of_non_finite_elements)
        print()

        for i in range(number_of_non_finite_elements):
            coordinates = tuple(non_finite_indices[i, :].cpu().numpy())
            value = str(tensor[coordinates].cpu().numpy())
            coordinates_str = str(coordinates)
            print("    At coordinates %s: value of %s" % (coordinates_str, value))
            print()

        # Other tests

        if not no_log_folder_flag:
            matrix_linalg_errors_test(tensor, error_tensor_name, log_files_dict)

        if len(tensor.shape) >= 2 and tensor.shape[-2] == tensor.shape[-1]:
            matrix_spd_checks(tensor)

        print()
        print()


if __name__ == "__main__":
    open_error_tensors()
