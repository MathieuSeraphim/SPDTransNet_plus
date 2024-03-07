import copy
import json
from argparse import ArgumentParser
from os import listdir
from os.path import dirname, realpath, join, isdir
from typing import Union, List, Dict
from _3_data_management._3_2_data_modules.DataModuleWrapper import modify_datamodule_config_dict, \
    get_datamodule_from_config_dict
from _3_data_management._3_2_data_modules.utils import get_cross_validation_recording_indices

labels_list = ["N3", "N2", "N1", "REM", "Awake"]


def get_stat_dict_from_list_of_epoch_names_or_labels(epoch_names_or_labels: List):
    
    epoch_names_or_labels_by_class_dict = {}
    stat_dict = {"all": len(epoch_names_or_labels)}
    all_class_wise_epoch_names_or_labels = []  # Used for checks
    for label in labels_list:
        epoch_names_or_labels_by_class_dict[label] = [epoch_name_or_label
                                                      for epoch_name_or_label in epoch_names_or_labels
                                                      if label in epoch_name_or_label]
        all_class_wise_epoch_names_or_labels += epoch_names_or_labels_by_class_dict[label]
        stat_dict[label] = len(epoch_names_or_labels_by_class_dict[label])

    all_class_wise_epoch_names_or_labels.sort()
    assert all_class_wise_epoch_names_or_labels == epoch_names_or_labels  # Checks both length and the absence of doubles
    
    return stat_dict


def get_dataset_recordings_as_paths(dataset_name: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    preprocessed_data_directory = join(root_directory, join("_2_data_preprocessing", "_2_4_preprocessed_data"))
    dataset_dirs_list = listdir(preprocessed_data_directory)
    our_dataset_dirs = [preprocessed_dataset_name for preprocessed_dataset_name in dataset_dirs_list
                        if preprocessed_dataset_name[:len(dataset_name)] == dataset_name]
    assert len(our_dataset_dirs) == 1
    dataset_directory = join(preprocessed_data_directory, our_dataset_dirs[0])
    recording_folders = [join(dataset_directory, recording_folder) for recording_folder in listdir(dataset_directory)]
    return sorted(recording_folders)


def recording_wise_stats(dataset_name: str, recording_id: int, clip_ends_by_amount: Union[int, None] = None):
    assert recording_id >= 0

    recording_folders = get_dataset_recordings_as_paths(dataset_name)
    recording_folder = recording_folders[recording_id]
    epochs_folder = join(recording_folder, "epochs")
    epoch_filenames = listdir(epochs_folder)
    assert len(epoch_filenames) > 0

    epoch_filenames.sort()
    if clip_ends_by_amount is not None and clip_ends_by_amount != 0:
        assert clip_ends_by_amount > 0
        assert len(epoch_filenames) > 2 * clip_ends_by_amount
        epoch_filenames = epoch_filenames[clip_ends_by_amount:-clip_ends_by_amount]

    return get_stat_dict_from_list_of_epoch_names_or_labels(epoch_filenames)


def combine_list_of_stat_dicts(list_of_stat_dicts: List[Dict]):

    grouped_stat_dict = {}
    keys = ["all"] + labels_list
    for key in keys:
        grouped_stat_dict[key] = 0

    for stat_dict in list_of_stat_dicts:
        assert sorted(list(stat_dict.keys())) == sorted(keys)

        for key in keys:
            grouped_stat_dict[key] += stat_dict[key]

    class_wise_accumulator = 0
    for class_key in labels_list:
        class_wise_accumulator += grouped_stat_dict[class_key]
    assert class_wise_accumulator == grouped_stat_dict["all"]

    return grouped_stat_dict


def apply_rebalancing_through_oversampling_on_stat_dict(stat_dict: Dict):

    largest_class_size = 0
    classes_with_elements = []
    for class_key in labels_list:
        if stat_dict[class_key] > 0:
            classes_with_elements.append(class_key)
        if stat_dict[class_key] > largest_class_size:
            largest_class_size = stat_dict[class_key]
    assert largest_class_size > 0

    rebalanced_stats_dict = {"all": largest_class_size * len(classes_with_elements)}
    for class_key in labels_list:
        if class_key in classes_with_elements:
            rebalanced_stats_dict[class_key] = largest_class_size
        else:
            rebalanced_stats_dict[class_key] = 0

    return rebalanced_stats_dict


def get_fold_wise_stats(dataset_name: str, training_set_recording_ids: List[int],
                        validation_set_recording_ids: List[int], test_set_recording_ids: List[int],
                        extra_epochs_on_each_side: int = 10, rebalance_training_set_through_oversampling: bool = True,
                        clip_ends_of_test_set_recordings_by_amount: Union[int, None] = None):
    
    all_fold_recording_ids = sorted(training_set_recording_ids + validation_set_recording_ids + test_set_recording_ids)
    assert all_fold_recording_ids == list(range(len(all_fold_recording_ids)))
    
    assert extra_epochs_on_each_side >= 0
    if clip_ends_of_test_set_recordings_by_amount is not None:
        assert clip_ends_of_test_set_recordings_by_amount >= extra_epochs_on_each_side
    else:
        clip_ends_of_test_set_recordings_by_amount = extra_epochs_on_each_side

    list_of_training_set_stats_dicts = []
    for recording_id in training_set_recording_ids:
        list_of_training_set_stats_dicts.append(recording_wise_stats(dataset_name, recording_id,
                                                                     extra_epochs_on_each_side))
    training_set_stats_dict = combine_list_of_stat_dicts(list_of_training_set_stats_dicts)

    list_of_validation_set_stats_dicts = []
    for recording_id in validation_set_recording_ids:
        list_of_validation_set_stats_dicts.append(recording_wise_stats(dataset_name, recording_id,
                                                                       extra_epochs_on_each_side))
    validation_set_stats_dict = combine_list_of_stat_dicts(list_of_validation_set_stats_dicts)

    list_of_test_set_stats_dicts = []
    for recording_id in test_set_recording_ids:
        list_of_test_set_stats_dicts.append(recording_wise_stats(dataset_name, recording_id,
                                                                 clip_ends_of_test_set_recordings_by_amount))
    test_set_stats_dict = combine_list_of_stat_dicts(list_of_test_set_stats_dicts)

    fold_stats_dict = combine_list_of_stat_dicts([training_set_stats_dict,
                                                  validation_set_stats_dict,
                                                  test_set_stats_dict])

    unbalanced_training_set_stats_dict = unbalanced_fold_stats_dict = None
    if rebalance_training_set_through_oversampling:
        unbalanced_training_set_stats_dict = copy.deepcopy(training_set_stats_dict)
        unbalanced_fold_stats_dict = copy.deepcopy(fold_stats_dict)

        training_set_stats_dict = apply_rebalancing_through_oversampling_on_stat_dict(
            unbalanced_training_set_stats_dict)
        fold_stats_dict = combine_list_of_stat_dicts([training_set_stats_dict,
                                                      validation_set_stats_dict,
                                                      test_set_stats_dict])

    super_stats_dict = {
        "fold (all sets)": fold_stats_dict,
        "training set": training_set_stats_dict,
        "validation set": validation_set_stats_dict,
        "test set": test_set_stats_dict
    }
    if rebalance_training_set_through_oversampling:
        super_stats_dict["fold (all sets) - prior to training set rebalancing"] =\
            unbalanced_fold_stats_dict
        super_stats_dict["training set - prior to rebalancing"] = unbalanced_training_set_stats_dict

    return super_stats_dict


def get_fold_wise_stats_from_fold_index(dataset_name: str,  fold_index: int, extra_epochs_on_each_side: int = 10,
                                        rebalance_training_set_through_oversampling: bool = True,
                                        clip_ends_of_test_set_recordings_by_amount: Union[int, None] = None):
    fold_dict = get_cross_validation_recording_indices(dataset_name, fold_index)
    return get_fold_wise_stats(dataset_name, fold_dict["training"], fold_dict["validation"], fold_dict["test"],
                               extra_epochs_on_each_side, rebalance_training_set_through_oversampling,
                               clip_ends_of_test_set_recordings_by_amount)


def get_datamodule_based_fold_wise_stats(datamodule_config_dict: Dict, fold_index: int,
                                         classified_epoch_index: int, dataset_name: Union[str, None] = None):

    datamodule_config_dict = modify_datamodule_config_dict(config_dict=datamodule_config_dict,
                                                           cross_validation_fold_index=fold_index)
    datamodule_object = get_datamodule_from_config_dict(config_dict=datamodule_config_dict)

    if dataset_name is not None:
        assert datamodule_object.base_dataset.data_reader.dataset_name == dataset_name

    # This initializes the various sets
    datamodule_object.train_dataloader()
    datamodule_object.val_dataloader()
    datamodule_object.test_dataloader()
    
    training_dataset = datamodule_object.training_set
    validation_dataset = datamodule_object.validation_set
    test_dataset = datamodule_object.test_set

    training_set_list_of_label_ids = [int(dataset_output[1][classified_epoch_index])
                                      for dataset_output in training_dataset]
    validation_set_list_of_label_ids = [int(dataset_output[1][classified_epoch_index])
                                        for dataset_output in validation_dataset]
    test_set_list_of_label_ids = [int(dataset_output[1][classified_epoch_index]) for dataset_output in test_dataset]

    training_set_list_of_labels = [labels_list[label_id] for label_id in training_set_list_of_label_ids]
    validation_set_list_of_labels = [labels_list[label_id] for label_id in validation_set_list_of_label_ids]
    test_set_list_of_labels = [labels_list[label_id] for label_id in test_set_list_of_label_ids]

    training_set_stats_dict = get_stat_dict_from_list_of_epoch_names_or_labels(training_set_list_of_labels)
    validation_set_stats_dict = get_stat_dict_from_list_of_epoch_names_or_labels(validation_set_list_of_labels)
    test_set_stats_dict = get_stat_dict_from_list_of_epoch_names_or_labels(test_set_list_of_labels)

    fold_stats_dict = combine_list_of_stat_dicts([training_set_stats_dict,
                                                  validation_set_stats_dict,
                                                  test_set_stats_dict])

    super_stats_dict = {
        "fold (all sets)": fold_stats_dict,
        "training set": training_set_stats_dict,
        "validation set": validation_set_stats_dict,
        "test set": test_set_stats_dict
    }

    return super_stats_dict


def get_fold_wise_stats_manually_and_from_datamodule(
        dataset_name: str,  fold_index: int, datamodule_config_filename: str, extra_epochs_on_each_side: int = 10,
        rebalance_training_set_through_oversampling: bool = True,
        clip_ends_of_test_set_recordings_by_amount: Union[int, None] = None):

    manual_fold_wise_stats_dict = get_fold_wise_stats_from_fold_index(dataset_name, fold_index,
                                                                      extra_epochs_on_each_side,
                                                                      rebalance_training_set_through_oversampling,
                                                                      clip_ends_of_test_set_recordings_by_amount)
    #
    # datamodule_config_dict = get_datamodule_config_dict_from_file(datamodule_config_filename)
    # datamodule_config_dict["init_args"]["save_in_single_file"] = True
    # assert datamodule_config_dict["init_args"]["extra_epochs_on_each_side"] == extra_epochs_on_each_side
    # assert datamodule_config_dict["init_args"]["rebalance_training_set_by_oversampling"] ==\
    #        rebalance_training_set_through_oversampling
    # assert datamodule_config_dict["init_args"]["clip_test_set_recordings_by_amount"] ==\
    #        clip_ends_of_test_set_recordings_by_amount
    #
    # datamodule_based_fold_wise_stats_dict = get_datamodule_based_fold_wise_stats(datamodule_config_dict,
    #                                                                              fold_index,
    #                                                                              extra_epochs_on_each_side,
    #                                                                              dataset_name)

    overall_fold_wise_stats_dict = {
        "manual stats": manual_fold_wise_stats_dict,
        "datamodule-derived stats": {}#datamodule_based_fold_wise_stats_dict
    }
    
    return overall_fold_wise_stats_dict


def get_the_number_of_folds_for_dataset(dataset_name: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    configs_directory = join(root_directory, "_1_configs")
    miscellaneous_configs_directory = join(configs_directory, "_1_z_miscellaneous")
    cross_validation_folds_folder = join(miscellaneous_configs_directory, "cross_validation_folds")
    cross_validation_folds_for_dataset_folder = join(cross_validation_folds_folder, dataset_name)
    assert isdir(cross_validation_folds_for_dataset_folder)
    list_of_fold_filenames = listdir(cross_validation_folds_for_dataset_folder)
    list_of_fold_filenames = [filename for filename in list_of_fold_filenames
                              if filename[:5] == "fold_" and filename[-5:] == ".yaml"]
    number_of_folds = len(list_of_fold_filenames)
    assert number_of_folds > 0
    return number_of_folds


def get_stats_on_all_folds(dataset_name: str, datamodule_config_filename: str, extra_epochs_on_each_side: int = 10,
                           rebalance_training_set_through_oversampling: bool = True,
                           clip_ends_of_test_set_recordings_by_amount: Union[int, None] = None):

    number_of_folds = get_the_number_of_folds_for_dataset(dataset_name)

    all_folds_stats_dict = {}
    for fold_index in range(number_of_folds):
        all_folds_stats_dict["stats for fold %02d" % fold_index] = get_fold_wise_stats_manually_and_from_datamodule(
            dataset_name, fold_index, datamodule_config_filename, extra_epochs_on_each_side,
            rebalance_training_set_through_oversampling, clip_ends_of_test_set_recordings_by_amount
        )

    return all_folds_stats_dict


def recursively_get_percentages_from_stats_dict(stat_dict: Dict):
    keys_list = ["all"] + labels_list
    keys_list.sort()

    assert isinstance(stat_dict, dict)

    if sorted(list(stat_dict.keys())) == keys_list:
        all_items = stat_dict["all"]
        for key in keys_list:
            stat_dict[key] = (stat_dict[key] / all_items) * 100
        return stat_dict

    for key in stat_dict.keys():
        if isinstance(stat_dict[key], dict):
            stat_dict[key] = recursively_get_percentages_from_stats_dict(stat_dict[key])
    return stat_dict


def save_stats_on_all_folds_for_dataset(dataset_name: str, datamodule_config_filename: str,
                                        extra_epochs_on_each_side: int = 10,
                                        rebalance_training_set_through_oversampling: bool = True,
                                        clip_ends_of_test_set_recordings_by_amount: Union[int, None] = None):

    current_script_directory = dirname(realpath(__file__))
    output_filename = "stats_for_dataset_%s.json" % dataset_name
    output_filename = join(current_script_directory, output_filename)

    all_folds_stats_dict = get_stats_on_all_folds(dataset_name, datamodule_config_filename, extra_epochs_on_each_side,
                                                  rebalance_training_set_through_oversampling,
                                                  clip_ends_of_test_set_recordings_by_amount)

    with open(output_filename, "w") as f:
        json.dump(all_folds_stats_dict, f, indent=3, sort_keys=False)

    output_filename = "stats_for_dataset_%s_as_percentages.json" % dataset_name
    output_filename = join(current_script_directory, output_filename)

    all_folds_stats_as_percentages_dict = recursively_get_percentages_from_stats_dict(all_folds_stats_dict)
    with open(output_filename, "w") as f:
        json.dump(all_folds_stats_as_percentages_dict, f, indent=3, sort_keys=False)


if __name__ == "__main__":
    parser = ArgumentParser()

    # --dataset_name MASS_SS3 --datamodule_config_filename Vectorized_SPD_matrices_from_EEG_MASS_dataset_EUSIPCO_signals_length_21_config.yaml --extra_epochs_on_each_side 10 --rebalancing_on_train --extra_clipping_on_test 24

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datamodule_config_filename", type=str, required=True)
    parser.add_argument("--extra_epochs_on_each_side", type=int, default=0)
    parser.add_argument("--rebalancing_on_train", action="store_true")
    parser.add_argument("--extra_clipping_on_test", type=int, default=None)

    command_line_inputs = parser.parse_args()

    dataset_name = command_line_inputs.dataset_name
    datamodule_config_filename = command_line_inputs.datamodule_config_filename
    extra_epochs_on_each_side = command_line_inputs.extra_epochs_on_each_side
    rebalance_training_set_through_oversampling = command_line_inputs.rebalancing_on_train
    clip_ends_of_test_set_recordings_by_amount = command_line_inputs.extra_clipping_on_test

    save_stats_on_all_folds_for_dataset(dataset_name, datamodule_config_filename, extra_epochs_on_each_side,
                                        rebalance_training_set_through_oversampling,
                                        clip_ends_of_test_set_recordings_by_amount)
    

    
    
    




