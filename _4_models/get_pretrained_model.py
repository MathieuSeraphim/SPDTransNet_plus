from os import listdir
from os.path import dirname, realpath, isdir, join, basename, isfile
from typing import Optional, Any, List, Union, Tuple
from _5_execution.run_model import get_model_and_datamodule_dicts_from_hparams_file
from _5_execution.utils import get_model_from_checkpoint_file


def get_pretrained_model(fold_index: int, pretrained_model_logs_folder: str,
                         pretrained_model_logs_folder_absolute_directory: Optional[str] = None,
                         list_of_valid_architectures: Optional[Union[List[Any], Tuple]] = None,
                         return_datamodule_dict: bool = False):

    if pretrained_model_logs_folder_absolute_directory is None:
        current_script_directory = dirname(realpath(__file__))
        root_directory = dirname(current_script_directory)
        pretrained_model_logs_folder_absolute_directory = root_directory
    assert isdir(pretrained_model_logs_folder_absolute_directory)

    pretrained_model_logs_folder = join(pretrained_model_logs_folder_absolute_directory, pretrained_model_logs_folder)
    assert isdir(pretrained_model_logs_folder)

    run_folders = [join(pretrained_model_logs_folder, run_folder) for run_folder in listdir(pretrained_model_logs_folder)
                   if run_folder[:8] == "version_" and isdir(join(pretrained_model_logs_folder, run_folder))]
    assert 0 <= fold_index < len(run_folders)

    # Sometimes I add a prefix to the run version (e.g. 4200 for fold 0, 4211 for fold 11...)
    run_folders = [run_folder for run_folder in run_folders if int(basename(run_folder).split("_")[-1][-2:]) == fold_index]
    assert len(run_folders) == 1
    run_folder = run_folders[0]

    hparams_file = join(run_folder, "hparams.yaml")
    assert isfile(hparams_file)

    checkpoint_folder = join(run_folder, "checkpoints")
    model_checkpoint_files = [join(checkpoint_folder, checkpoint_file)
                              for checkpoint_file in listdir(checkpoint_folder)]
    assert len(model_checkpoint_files) == 1
    model_checkpoint_file = model_checkpoint_files[0]

    model_dict, datamodule_dict = get_model_and_datamodule_dicts_from_hparams_file(hparams_file, is_full_path=True)
    pretrained_model = get_model_from_checkpoint_file(model_dict, model_checkpoint_file, no_logging=True)

    if list_of_valid_architectures is not None and len(list_of_valid_architectures) > 0:
        assert isinstance(pretrained_model, list_of_valid_architectures)

    if return_datamodule_dict:
        return pretrained_model, datamodule_dict
    return pretrained_model
