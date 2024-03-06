from jsonargparse import ArgumentParser, namespace_to_dict
from _5_execution._5_2_optuna.optuna_run_model import optuna_run_model_from_cli
from _5_execution.run_model import run_model_from_cli


if __name__ == "__main__":
    parser = ArgumentParser()

    # --execution_method standalone --execution_type test --global_seed 42 --trainer_config_file trainer_default_config.yaml --trainer_config.logger_version 65536 --model_config_file PredicAlert_signals_default_config.yaml --datamodule_config_file SPD_matrices_from_EEG_MASS_dataset_PredicAlert_signals_config.yaml --datamodule_config.cross_validation_fold_index 11

    parser.add_argument("--execution_method", type=str, default="standalone")  # "standalone", "from_hparams" or "transfer_learning"

    parser.add_argument("--execution_type", type=str, default="fit")
    parser.add_argument("--global_seed", type=int, default=None)

    parser.add_argument("--trainer_config_file", type=str, default=None)
    parser.add_argument("--trainer_config.logger_logs_location", type=str, default=".")
    parser.add_argument("--trainer_config.logger_logs_folder_name", type=str, default="lightning_logs")
    parser.add_argument("--trainer_config.logger_version", type=int, default=None)
    parser.add_argument("--trainer_config.forced_dtype", type=str, default=None)  # "float32" or "float64"
    parser.add_argument("--trainer_add_callback_config_file", action="append")

    parser.add_argument("--model_config_file", type=str, default=None)  # Mandatory if execution_method is "standalone"
    parser.add_argument("--model_config.cross_validation_fold_index", type=str, default=None)
    parser.add_argument("--model_config.signals_batch_size", type=str, default=None)

    parser.add_argument("--datamodule_config_file", type=str, default=None)  # Mandatory for execution_method in ["standalone", "transfer_learning"]
    parser.add_argument("--datamodule_config.batch_size", type=int, default=None)
    parser.add_argument("--datamodule_config.cross_validation_fold_index", type=int, default=None)

    parser.add_argument("--hparams_config_file", type=str, default=None)  # Mandatory if execution_method is "from_hparams"

    # The lightning log folder (containing hparams.yaml, the tensorboard logs and checkpoints folder) is located at
    # [ROOT]/pretrained_logs_folder_name/pretrained_logs_subfolder_name
    # OR
    # pretrained_logs_folder_absolute_location/pretrained_logs_folder_name/pretrained_logs_subfolder_name
    parser.add_argument("--transfer_learning.pretrained_logs_folder_name", type=str, default=None)  # Mandatory if execution_method is "transfer_learning"
    parser.add_argument("--transfer_learning.pretrained_logs_subfolder_name", type=str, default=None)  # Likewise
    parser.add_argument("--transfer_learning.pretrained_logs_folder_absolute_location", type=str, default=None)

    parser.add_argument("--optuna_flag", action="store_true")

    parser.add_argument("--optuna.study_name", type=str, default='dev')
    parser.add_argument("--optuna.storage", type=str, default='sqlite:///db/database.db')

    parser.add_argument("--optuna.pruner.n_startup_trials", type=int, default=10)  # Minimal number of trials to run before pruning
    parser.add_argument("--optuna.pruner.n_warmup_steps", type=int, default=3)  # Number of network epochs to wait before pruning
    parser.add_argument("--optuna.pruner.interval_steps", type=int, default=1)  # Number of network epochs between pruner acts

    parser.add_argument("--optuna.hparam_selection_config.model", type=str, default=None)
    parser.add_argument("--optuna.hparam_selection_config.datamodule", type=str, default=None)

    parser.add_argument("--optuna.hparam_selection_config.trainer_finetuning_lr_minimum", type=float, default=None)
    parser.add_argument("--optuna.hparam_selection_config.trainer_finetuning_lr_maximum", type=float, default=None)

    command_line_inputs = parser.parse_args()

    trainer_config_file = command_line_inputs.trainer_config_file
    model_config_file = command_line_inputs.model_config_file
    datamodule_config_file = command_line_inputs.datamodule_config_file
    hparams_config_file = command_line_inputs.hparams_config_file

    execution_type = command_line_inputs.execution_type
    global_seed = command_line_inputs.global_seed

    trainer_config_modifications_as_dict = namespace_to_dict(command_line_inputs.trainer_config)
    model_config_modifications_as_dict = namespace_to_dict(command_line_inputs.model_config)
    datamodule_config_modifications_as_dict = namespace_to_dict(command_line_inputs.datamodule_config)

    trainer_config_files_for_additional_callbacks_list = command_line_inputs.trainer_add_callback_config_file
    trainer_config_modifications_as_dict["additional_callback_config_files_list"] = trainer_config_files_for_additional_callbacks_list

    transfer_learning_config_as_dict = namespace_to_dict(command_line_inputs.transfer_learning)

    execution_method = command_line_inputs.execution_method

    run_with_optuna_flag = command_line_inputs.optuna_flag
    optuna_config = command_line_inputs.optuna

    instantiation_kwargs = {
        "execution_method": execution_method,
        "trainer_config_file": trainer_config_file,
        "trainer_config_kwargs": trainer_config_modifications_as_dict,
        "model_config_file": model_config_file,
        "model_config_kwargs": model_config_modifications_as_dict,
        "datamodule_config_file": datamodule_config_file,
        "datamodule_config_kwargs": datamodule_config_modifications_as_dict,
        "hparams_config_file": hparams_config_file,
        "transfer_learning_config": transfer_learning_config_as_dict
    }

    if run_with_optuna_flag:
        optuna_run_model_from_cli(optuna_config=optuna_config, instantiation_kwargs=instantiation_kwargs,
                                  trainer_mode=execution_type, random_seed=global_seed)
    else:
        run_model_from_cli(trainer_mode=execution_type, instantiation_kwargs=instantiation_kwargs,
                           random_seed=global_seed)


