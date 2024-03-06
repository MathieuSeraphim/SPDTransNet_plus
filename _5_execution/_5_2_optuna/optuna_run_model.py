import signal
import time
import optuna
from sqlalchemy.engine import Engine
from sqlalchemy import event
from jsonargparse import Namespace
from typing import Dict, Any, Union, Tuple
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback
from _5_execution._5_2_optuna.hparam_selection import optuna_suggest_hparam
from _5_execution._5_2_optuna.utils import before_class_instantiation
from _5_execution.run_model import run_model, instantiate_classes, get_all_config_dicts

optuna_valid_trainer_modes = ["fit", "validate"]
finetuning_lr_override_hparam_name = "override_extractor_lr_with_value"


class SignalException(Exception):
    pass


def interruption(signal, context):
    print('Received signal %d' % signal)
    raise SignalException


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    except:
        time.sleep(1)
        set_sqlite_pragma(dbapi_connection, connection_record)


def optuna_run_trial(trial: Trial, instantiation_kwargs: Dict[str, Any], hparam_selection_files: Namespace,
                     monitor: str = "mf1/validation", trainer_mode: str = "fit",
                     trainer_finetuning_lr_bracket: Union[Tuple[float, float], None] = None,
                     random_seed: Union[int, None] = 42):

    print("Trial ID:", trial.number)

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=monitor)
    if "extra_trainer_callbacks_list" not in instantiation_kwargs:
        instantiation_kwargs["extra_trainer_callbacks_list"] = []
    instantiation_kwargs["extra_trainer_callbacks_list"].append(pruning_callback)

    if trainer_finetuning_lr_bracket is not None:
        finetuning_lr = optuna_suggest_hparam(trial, finetuning_lr_override_hparam_name, "loguniform",
                                              trainer_finetuning_lr_bracket)
        instantiation_kwargs["trainer_config_kwargs"][finetuning_lr_override_hparam_name] = finetuning_lr

    trainer_dict, model_dict, datamodule_dict, model_checkpoint_file = get_all_config_dicts(**instantiation_kwargs)
    model_dict, datamodule_dict = before_class_instantiation(trial, model_dict, datamodule_dict, hparam_selection_files)
    trainer, model, datamodule = instantiate_classes(trainer_dict, model_dict, datamodule_dict, model_checkpoint_file)
    run_model(trainer_mode, trainer, model, datamodule, random_seed)

    return trainer.checkpoint_callback.best_model_score.item()


def optuna_run_model_from_cli(optuna_config: Namespace, instantiation_kwargs: Dict[str, Any], trainer_mode: str = "fit",
                              monitor: str = "mf1/validation", random_seed: Union[int, None] = 42):
    assert trainer_mode in optuna_valid_trainer_modes
    signal.signal(signal.SIGUSR2, interruption)

    # If True, there's no point in running a hyperparameter research
    assert not (optuna_config.hparam_selection_config.model is None and optuna_config.hparam_selection_config.datamodule is None)

    trainer_finetuning_lr_bracket = None
    if optuna_config.hparam_selection_config.trainer_finetuning_lr_minimum is not None and optuna_config.hparam_selection_config.trainer_finetuning_lr_maximum is not None:
        trainer_finetuning_lr_bracket = (optuna_config.hparam_selection_config.trainer_finetuning_lr_minimum,
                                         optuna_config.hparam_selection_config.trainer_finetuning_lr_maximum)

    try:

        storage = optuna.storages.RDBStorage(
            url=optuna_config.storage,
            engine_kwargs={"connect_args": {"timeout": 1000}},
        )

        study = optuna.load_study(
            study_name=optuna_config.study_name,
            storage=storage,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=optuna_config.pruner.n_startup_trials,  # Minimal number of trials to run before pruning
                n_warmup_steps=optuna_config.pruner.n_warmup_steps,  # Number of network epochs to wait before pruning
                interval_steps=optuna_config.pruner.interval_steps  # Number of network epochs between pruner acts
            )
        )

        study.optimize(
            lambda trial: optuna_run_trial(trial, instantiation_kwargs, optuna_config.hparam_selection_config, monitor,
                                           trainer_mode, trainer_finetuning_lr_bracket, random_seed),
            n_trials=1)

    except SignalException:
        print("Program interrupted by signal.")


    
