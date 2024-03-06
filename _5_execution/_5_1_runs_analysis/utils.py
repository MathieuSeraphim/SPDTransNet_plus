from os.path import join, isfile
import yaml
from pytorch_lightning import Trainer
from _3_data_management._3_2_data_modules.BaseDataModule import BaseDataModule
from _4_models.BaseModel import BaseModel
from _4_models.utils import combine_dict_outputs
from _5_execution.utils import get_execution_configs_folder_path


def get_hparams_to_track_dict(hparams_to_track_file: str):
    results_analysis_configs_folder = get_execution_configs_folder_path("results_analysis")
    hparams_to_track_folder = join(results_analysis_configs_folder, "hparams_to_track")
    hparams_to_track_file = join(hparams_to_track_folder, hparams_to_track_file)
    assert isfile(hparams_to_track_file)
    return yaml.safe_load(open(hparams_to_track_file, "r"))


def get_best_run_stats_to_track_dict(best_run_stats_to_track_file: str):
    results_analysis_configs_folder = get_execution_configs_folder_path("results_analysis")
    stats_to_track_folder = join(results_analysis_configs_folder, "stats_to_track")
    best_run_stats_to_track_file = join(stats_to_track_folder, best_run_stats_to_track_file)
    assert isfile(best_run_stats_to_track_file)
    return yaml.safe_load(open(best_run_stats_to_track_file, "r"))


def get_predictions(model: BaseModel, datamodule: BaseDataModule, trainer: Trainer, set_name: str):
    if set_name == model.TRAINING_SET_NAME:
        dataloader = datamodule.train_dataloader()
    elif set_name == model.VALIDATION_SET_NAME:
        dataloader = datamodule.val_dataloader()
    elif set_name == model.TEST_SET_NAME:
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError("Set name unsupported.")

    outputs = trainer.predict(model=model, dataloaders=dataloader)
    combined_outputs = combine_dict_outputs(outputs)

    return combined_outputs

