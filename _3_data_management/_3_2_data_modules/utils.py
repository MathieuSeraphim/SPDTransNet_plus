import json
from os.path import dirname, realpath, join, isdir
import yaml


def get_cross_validation_recording_indices(dataset_name: str, cross_validation_fold_index: int):

    all_to_test = False
    if cross_validation_fold_index < 0:
        cross_validation_fold_index = 0
        all_to_test = True

    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    configs_directory = join(root_directory, "_1_configs")
    miscellaneous_configs_directory = join(configs_directory, "_1_z_miscellaneous")
    cross_validation_folds_folder = join(miscellaneous_configs_directory, "cross_validation_folds")
    cross_validation_folds_for_dataset_folder = join(cross_validation_folds_folder, dataset_name)
    assert isdir(cross_validation_folds_for_dataset_folder)
    fold_filename = join(cross_validation_folds_for_dataset_folder,
                         "fold_%s.yaml" % str(cross_validation_fold_index).zfill(2))
    with open(fold_filename, "r") as f:
        fold_dict = yaml.safe_load(f)

    if all_to_test:
        new_fold_dict = {
            "training": [],
            "validation": [],
            "test": fold_dict["training"] + fold_dict["validation"] + fold_dict["test"]
        }
        return new_fold_dict

    return fold_dict


if __name__ == "__main__":
    dataset = "Dreem_DOD-H"
    num_folds = 25
    num_recordings = 25
    cross_valid_training_set_indices = {}
    cross_valid_validation_set_indices = {}
    cross_valid_test_set_indices = {}
    cross_valid_test_set_indices_list = []

    for fold in range(num_folds):
        fold_dict = get_cross_validation_recording_indices(dataset, fold)
        cross_valid_training_set_indices[fold] = fold_dict["training"]
        cross_valid_validation_set_indices[fold] = fold_dict["validation"]
        cross_valid_test_set_indices[fold] = fold_dict["test"]
        cross_valid_test_set_indices_list += fold_dict["test"]

    print("All fold training sets:")
    print(json.dumps(cross_valid_training_set_indices, indent=4))
    print("All fold validation sets:")
    print(json.dumps(cross_valid_validation_set_indices, indent=4))
    print("All fold test sets:")
    print(json.dumps(cross_valid_test_set_indices, indent=4))

    if sorted(cross_valid_test_set_indices_list) == list(range(num_recordings)):
        print("Union of test sets corresponds to the entire dataset, without overlap.")

