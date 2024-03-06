from os import listdir
from os.path import join, isfile, realpath, dirname, isdir
import yaml
from pytorch_lightning import Trainer, seed_everything
from typing import Dict, Any, Optional, List
from _3_data_management._3_2_data_modules.BaseDataModule import BaseDataModule
from _3_data_management._3_2_data_modules.DataModuleWrapper import modify_datamodule_config_dict,\
    get_datamodule_from_config_dict, get_datamodule_config_dict_from_file
from _4_models.BaseModel import BaseModel
from _4_models.ModelWrapper import get_model_from_config_dict, modify_model_config_dict, get_model_config_dict_from_file
from _5_execution.TrainerWrapper import get_trainer_config_dict_from_file, \
    get_trainer_from_config_dict
from _5_execution.utils import get_execution_configs_folder_path, get_model_from_checkpoint_file, \
    add_callback_to_trainer_config_dict

valid_trainer_modes = ["fit", "validate", "test", "predict"]
execution_methods = ["standalone", "from_hparams", "transfer_learning"]


def run_model(trainer_mode: str, trainer: Trainer, model: BaseModel, datamodule: BaseDataModule,
              random_seed: Optional[int] = 42):
    assert trainer_mode in valid_trainer_modes

    print()
    print("All classes initialized. Running model.")
    print("Execution mode: %s" % trainer_mode)
    print("Model object class: %s" % model.__class__.__name__)
    print("Number of model parameters: %d" % sum(p.numel() for p in model.parameters()))
    print("Data module object class: %s" % datamodule.__class__.__name__)

    if random_seed is not None:
        seed_everything(random_seed, workers=True)

    debug_mode_string = "Debug mode "
    if __debug__:
        debug_mode_string += "enabled. Asserts ran."
    else:
        assert False  # Cheeky check
        debug_mode_string += "disabled. Asserts ignored."
    print(debug_mode_string)
    print()

    if trainer_mode == "fit":
        model.setup("fit")
        return trainer.fit(model, datamodule)
    if trainer_mode == "validate":
        model.setup("validate")
        return trainer.validate(model, datamodule)
    if trainer_mode == "test":
        model.setup("test")
        return trainer.test(model, datamodule)
    if trainer_mode == "predict":
        model.setup("predict")
        return trainer.predict(model, datamodule)
    raise NotImplementedError


def get_model_and_datamodule_dicts_from_hparams_file(hparams_file: str, is_full_path: bool = False,
                                                     model_config_kwargs: Dict[str, Any] = {},
                                                     datamodule_config_kwargs: Dict[str, Any] = {},
                                                     no_datamodule: bool = False):
    if is_full_path:
        model_and_datamodule_config_file = hparams_file
    else:
        past_runs_hparams_folder = get_execution_configs_folder_path("past_runs_hyperparameters")
        model_and_datamodule_config_file = join(past_runs_hparams_folder, hparams_file)
    assert isfile(model_and_datamodule_config_file)

    config_dict = yaml.safe_load(open(model_and_datamodule_config_file, "r"))

    model_config_dict = config_dict["model"]
    model_config_dict = modify_model_config_dict(model_config_dict, **model_config_kwargs)


    if not no_datamodule:
        datamodule_config_dict = config_dict["datamodule"]
        datamodule_config_dict = modify_datamodule_config_dict(datamodule_config_dict, **datamodule_config_kwargs)

        return model_config_dict, datamodule_config_dict

    return model_config_dict


def get_model_dict_and_checkpoint_file_for_transfer_learning(transfer_learning_config: Dict[str, str],
                                                             model_config_kwargs: Dict[str, Any]):
    pretrained_logs_folder_name = transfer_learning_config["pretrained_logs_folder_name"]
    pretrained_logs_subfolder_name = transfer_learning_config["pretrained_logs_subfolder_name"]
    pretrained_logs_folder_absolute_location = transfer_learning_config["pretrained_logs_folder_absolute_location"]
    assert pretrained_logs_folder_name is not None and pretrained_logs_subfolder_name is not None

    if pretrained_logs_folder_absolute_location is None:
        root_directory = dirname(dirname(realpath(__file__)))
        pretrained_logs_folder_absolute_location = root_directory
    else:
        assert isdir(pretrained_logs_folder_absolute_location)
    pretrained_logs_folder = join(pretrained_logs_folder_absolute_location, pretrained_logs_folder_name)
    assert isdir(pretrained_logs_folder)
    pretrained_logs_subfolder = join(pretrained_logs_folder, pretrained_logs_subfolder_name)
    assert isdir(pretrained_logs_subfolder)

    hparams_config_file = join(pretrained_logs_subfolder, "hparams.yaml")
    assert isfile(hparams_config_file)
    model_config_dict, _ = get_model_and_datamodule_dicts_from_hparams_file(hparams_config_file,
                                                                            is_full_path=True,
                                                                            model_config_kwargs=model_config_kwargs)

    checkpoint_folder = join(pretrained_logs_subfolder, "checkpoints")
    assert isdir(checkpoint_folder)

    model_checkpoint_files = [join(checkpoint_folder, checkpoint_file)
                              for checkpoint_file in listdir(checkpoint_folder)]
    assert len(model_checkpoint_files) == 1
    model_checkpoint_file = model_checkpoint_files[0]

    return model_config_dict, model_checkpoint_file


def get_all_config_dicts(execution_method: str, trainer_config_file: str, trainer_config_kwargs: Dict[str, Any],
                         model_config_kwargs: Dict[str, Any], datamodule_config_kwargs: Dict[str, Any],
                         transfer_learning_config: Dict[str, str], extra_trainer_callbacks_list: List[Any] = [],
                         model_config_file: Optional[str] = None, datamodule_config_file: Optional[str] = None,
                         hparams_config_file: Optional[str] = None):
    model_checkpoint_file = None

    assert execution_method in execution_methods
    if execution_method == "standalone" or execution_method == "transfer_learning":
        assert datamodule_config_file is not None
        datamodule_config_dict = get_datamodule_config_dict_from_file(datamodule_config_file, **datamodule_config_kwargs)

        if execution_method == "standalone":
            assert model_config_file is not None
            model_config_dict = get_model_config_dict_from_file(model_config_file, **model_config_kwargs)

        else:
            model_config_dict, model_checkpoint_file = get_model_dict_and_checkpoint_file_for_transfer_learning(
                transfer_learning_config, model_config_kwargs
            )

    elif execution_method == "from_hparams":
        assert hparams_config_file is not None
        model_config_dict, datamodule_config_dict = get_model_and_datamodule_dicts_from_hparams_file(
            hparams_config_file,
            is_full_path=False,
            model_config_kwargs=model_config_kwargs,
            datamodule_config_kwargs=datamodule_config_kwargs)
    else:
        raise NotImplementedError("Shouldn't trigger...")

    trainer_config_dict = get_trainer_config_dict_from_file(trainer_config_file, **trainer_config_kwargs)
    for callback in extra_trainer_callbacks_list:
        trainer_config_dict = add_callback_to_trainer_config_dict(trainer_config_dict, callback)

    return trainer_config_dict, model_config_dict, datamodule_config_dict, model_checkpoint_file


def instantiate_classes(trainer_config_dict: Dict[str, Any], model_config_dict: Dict[str, Any],
                        datamodule_config_dict: Dict[str, Any], model_checkpoint_file: Optional[str] = None):
    trainer = get_trainer_from_config_dict(trainer_config_dict)
    datamodule = get_datamodule_from_config_dict(datamodule_config_dict)
    if model_checkpoint_file is None:
        model = get_model_from_config_dict(model_config_dict)
    else:
        model = get_model_from_checkpoint_file(model_config_dict, model_checkpoint_file)

    return trainer, model, datamodule


def run_model_from_cli(trainer_mode: str, instantiation_kwargs: Dict[str, Any], random_seed: Optional[int] = 42):
    trainer_config_dict, model_config_dict, datamodule_config_dict, model_checkpoint_file = get_all_config_dicts(
                                                                                            **instantiation_kwargs)
    trainer, model, datamodule = instantiate_classes(trainer_config_dict, model_config_dict, datamodule_config_dict,
                                                     model_checkpoint_file)
    run_model(trainer_mode, trainer, model, datamodule, random_seed)



