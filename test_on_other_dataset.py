import copy
import csv
from argparse import ArgumentParser
from os import listdir
from torch.utils.data import DataLoader
from os.path import dirname, realpath, join, isdir, isfile
from _3_data_management._3_2_data_modules.DataModuleWrapper import get_datamodule_from_config_dict
from _4_models.BaseModel import BaseModel
from _5_execution.TrainerWrapper import get_trainer_config_dict_from_file, get_trainer_from_config_dict
from _5_execution._5_1_runs_analysis.get_run_results import get_stat_column_name
from _5_execution._5_1_runs_analysis.utils import get_best_run_stats_to_track_dict
from _5_execution.run_model import get_model_and_datamodule_dicts_from_hparams_file
from _5_execution.utils import get_model_from_checkpoint_file


if __name__ == "__main__":
    root_directory = dirname(realpath(__file__))

    parser = ArgumentParser()

    parser.add_argument("--lightning_logs_folder_name", type=str)
    parser.add_argument("--lightning_logs_folder_absolute_location", type=str, default=None)
    parser.add_argument("--lightning_logs_first_version_id", type=int, default=11)
    parser.add_argument("--lightning_logs_last_version_id", type=int, default=None)
    parser.add_argument("--test_dataset_config_file", type=str, default="Vectorized_SPD_matrices_from_EEG_MASS_SS1_dataset_EUSIPCO_signals_config.yaml")
    parser.add_argument("--test_dataset_number_of_recordings", type=int, default=53)
    parser.add_argument("--test_dataset_force_multifile", action="store_true")
    parser.add_argument("--stats_to_track_file", type=str, default="SPD_from_EEG_stats.yaml")
    parser.add_argument("--output_file_name_personalization", type=str, default=None)
    parser.add_argument("--logging_offset", type=int, default=0)

    command_line_inputs = parser.parse_args()

    lightning_logs_folder_name = command_line_inputs.lightning_logs_folder_name
    lightning_logs_folder_absolute_location = command_line_inputs.lightning_logs_folder_absolute_location

    if lightning_logs_folder_absolute_location is None:
        lightning_logs_folder_absolute_location = root_directory
    lightning_logs_folder = join(lightning_logs_folder_absolute_location, lightning_logs_folder_name)
    assert isdir(lightning_logs_folder)

    lightning_logs_first_version_id = command_line_inputs.lightning_logs_first_version_id
    lightning_logs_last_version_id = command_line_inputs.lightning_logs_last_version_id
    lightning_logs_version_ids_list = [lightning_logs_first_version_id, ]
    if lightning_logs_last_version_id is not None:
        assert lightning_logs_last_version_id > lightning_logs_first_version_id
        lightning_logs_version_ids_list += list(
            range(lightning_logs_first_version_id + 1, lightning_logs_last_version_id + 1))

    lightning_logs_runwise_folder_names_list = ["version_%d" % version_id for version_id in
                                                lightning_logs_version_ids_list]
    lightning_logs_runwise_folders_list = [join(lightning_logs_folder, runwise_folder_name) for runwise_folder_name in
                                           lightning_logs_runwise_folder_names_list]
    for lightning_logs_runwise_folder in lightning_logs_runwise_folders_list:
        assert isdir(lightning_logs_runwise_folder)
    num_runs = len(lightning_logs_runwise_folder_names_list)

    test_dataset_config_file = command_line_inputs.test_dataset_config_file
    test_dataset_number_of_recordings = command_line_inputs.test_dataset_number_of_recordings
    test_dataset_recording_indices = list(range(test_dataset_number_of_recordings))

    test_dataset_force_multifile = command_line_inputs.test_dataset_force_multifile

    output_filename_extra_string = command_line_inputs.output_file_name_personalization
    if output_filename_extra_string is None:
        output_filename_extra_string = ""
    else:
        output_filename_extra_string = "_" + output_filename_extra_string

    csv_output_filename = join(root_directory, "test_on_other_dataset%s.csv" % output_filename_extra_string)
    csv_output_file = open(csv_output_filename, "w", newline="")
    csv_output_file.close()
    write_first_line_flag = True

    test_set = getattr(BaseModel, "TEST_SET_NAME")
    stats_to_track_file = command_line_inputs.stats_to_track_file
    best_run_stats_to_track_dict = get_best_run_stats_to_track_dict(stats_to_track_file)
    assert test_set in best_run_stats_to_track_dict.keys()
    test_stats_to_track = best_run_stats_to_track_dict[test_set]

    logging_offset = command_line_inputs.logging_offset

    for run_id in range(num_runs):
        lightning_logs_runwise_folder = lightning_logs_runwise_folders_list[run_id]

        hparams_config_file = join(lightning_logs_runwise_folder, "hparams.yaml")
        assert isfile(hparams_config_file)

        model_dict, datamodule_dict = get_model_and_datamodule_dicts_from_hparams_file(hparams_config_file,
                                                                                       is_full_path=True)
        datamodule_dict["init_args"]["cross_validation_fold_index"] = -1
        datamodule_dict["init_args"]["dataset_config_file"] = test_dataset_config_file
        if test_dataset_force_multifile:
            datamodule_dict["init_args"]["save_in_single_file"] = False

        checkpoint_folder = join(lightning_logs_runwise_folder, "checkpoints")
        assert isdir(checkpoint_folder)

        model_checkpoint_files = [join(checkpoint_folder, checkpoint_file)
                                  for checkpoint_file in listdir(checkpoint_folder)]
        assert len(model_checkpoint_files) == 1
        model_checkpoint_file = model_checkpoint_files[0]

        run_dict = {"Run ID": run_id}
        offset_run_id = run_id + logging_offset

        trainer_config_file = "trainer_default_config_no_graph.yaml"
        trainer_config_modifications_as_dict = {
            "logger_logs_folder_name": "test_on_other_datasets_logs",
            "logger_version": offset_run_id
        }
        trainer_config_dict = get_trainer_config_dict_from_file(trainer_config_file,
                                                                **trainer_config_modifications_as_dict)

        trainer = get_trainer_from_config_dict(trainer_config_dict)
        model = get_model_from_checkpoint_file(model_dict, model_checkpoint_file)
        model.freeze()

        modified_datamodule_dict = copy.deepcopy(datamodule_dict)
        if "Vectorized" in modified_datamodule_dict["class_path"]:
            modified_datamodule_dict["init_args"]["run_identifier"] = offset_run_id

        datamodule = get_datamodule_from_config_dict(modified_datamodule_dict)
        datamodule.setup("test")

        dataset_kwargs = copy.deepcopy(datamodule.dataset_kwargs)
        if "Vectorized" in modified_datamodule_dict["class_path"]:
            dataset_kwargs["current_subset"] = "test"

        dataset = copy.deepcopy(datamodule.base_dataset)
        dataset.setup(**dataset_kwargs, recording_indices=test_dataset_recording_indices,
                      clip_recordings_by_amount=datamodule.clip_test_set_recordings_by_amount)

        dataloader = DataLoader(dataset, batch_size=datamodule.batch_size, shuffle=False,
                                num_workers=datamodule.dataloader_num_workers)

        test_results = trainer.test(model, dataloaders=dataloader)[0]
        for test_stat in test_stats_to_track:
            full_stat_name = "%s/%s" % (test_stat, test_set)
            if full_stat_name in test_results.keys():
                run_dict[get_stat_column_name(test_stat, test_set, True)] \
                    = test_results[full_stat_name]
            else:
                run_dict[get_stat_column_name(test_stat, test_set, True)] = None

        del trainer, datamodule, dataloader, dataset, model

        with open(csv_output_filename, "a", newline="") as csv_output_file:
            field_names = list(run_dict.keys())
            writer = csv.DictWriter(csv_output_file, fieldnames=field_names)
            if write_first_line_flag:
                writer.writeheader()
                write_first_line_flag = False
            writer.writerow(run_dict)



