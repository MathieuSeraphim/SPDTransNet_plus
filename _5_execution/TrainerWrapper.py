from os.path import dirname, realpath, join, isfile
from typing import Dict, Union, Any, List
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import Trainer


forced_dtypes = ["float32", "float64"]
finetuning_class_path = "_4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.misc.LearnedAugmentationFinetuning.LearnedAugmentationFinetuning"


class TrainerWrapper:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer


def get_trainer_config_dict_from_file(config_file: str, logger_logs_location: str = ".",
                                      logger_logs_folder_name: str = "lightning_logs",
                                      logger_version: Union[int, None] = None,
                                      forced_dtype: Union[str, None] = None,
                                      override_extractor_lr_with_value: Union[float, None] = None,
                                      additional_callback_config_files_list: Union[List[str], None] = None):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(current_script_directory)
    configs_directory = join(root_directory, "_1_configs")
    trainer_configs_directory = join(configs_directory, "_1_6_trainer")
    trainer_config_file = join(trainer_configs_directory, config_file)
    assert isfile(trainer_config_file)

    trainer_config_dict = yaml.safe_load(open(trainer_config_file, "r"))
    trainer_config_dict["logger"][0]["init_args"]["save_dir"] = logger_logs_location
    trainer_config_dict["logger"][0]["init_args"]["name"] = logger_logs_folder_name
    trainer_config_dict["logger"][0]["init_args"]["version"] = logger_version

    list_of_additional_callback_dicts = []
    if additional_callback_config_files_list is not None:
        misc_configs_directory = join(configs_directory, "_1_z_miscellaneous")
        trainer_callback_configs_directory = join(misc_configs_directory, "trainer_callbacks")
        for additional_callback_config_file in additional_callback_config_files_list:
            callback_config_full_filename = join(trainer_callback_configs_directory, additional_callback_config_file)
            assert isfile(callback_config_full_filename)
            callback_config_dict = yaml.safe_load(open(callback_config_full_filename, "r"))
            list_of_additional_callback_dicts.append(callback_config_dict)

        if len(list_of_additional_callback_dicts) > 0:
            if override_extractor_lr_with_value is not None:

                for callback_dict_index in range(len(list_of_additional_callback_dicts)):
                    if list_of_additional_callback_dicts[callback_dict_index]["class_path"] == finetuning_class_path:
                        list_of_additional_callback_dicts[callback_dict_index]["init_args"][
                            "override_extractor_lr_with_value"] = override_extractor_lr_with_value

        if "callbacks" not in trainer_config_dict.keys():
            trainer_config_dict["callbacks"] = list_of_additional_callback_dicts
        else:
            trainer_config_dict["callbacks"] += list_of_additional_callback_dicts

    if forced_dtype is not None:
        assert forced_dtype in forced_dtypes
        if forced_dtype == "float32":
            precision = 32
        elif forced_dtype == "float64":
            precision = 64
        else:
            raise NotImplementedError

        trainer_config_dict["precision"] = precision

    return trainer_config_dict


def get_trainer_from_config_file(config_file: str, logger_logs_location: str = ".",
                                 logger_logs_folder_name: str = "lightning_logs",
                                 logger_version: Union[int, None] = None,
                                 forced_dtype: Union[str, None] = None,
                                 override_extractor_lr_with_value: Union[float, None] = None,
                                 additional_callback_config_files_list: Union[List[str], None] = None):
    trainer_config_dict = get_trainer_config_dict_from_file(config_file, logger_logs_location, logger_logs_folder_name,
                                                            logger_version, forced_dtype,
                                                            override_extractor_lr_with_value,
                                                            additional_callback_config_files_list)
    trainer = get_trainer_from_config_dict(trainer_config_dict)
    return trainer


def get_trainer_from_config_dict(config_dict: Dict[str, Any]):
    parser = ArgumentParser()

    wrapper_dict = {"wrapper":
        {"trainer":
            {
                "class_path": "pytorch_lightning.Trainer",
                "init_args": config_dict
            }
        }
    }

    parser.add_class_arguments(TrainerWrapper, "wrapper", fail_untyped=False)
    constructed_trainer = parser.instantiate_classes(wrapper_dict).wrapper.trainer
    return constructed_trainer

