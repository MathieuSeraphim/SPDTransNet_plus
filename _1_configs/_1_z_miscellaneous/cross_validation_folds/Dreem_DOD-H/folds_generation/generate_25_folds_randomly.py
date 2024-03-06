from os.path import realpath, dirname, join
from random import shuffle, seed
import yaml

NUMBER_OF_RECORDINGS_IN_DREEM_DOD_H = 25
TRAINING_SET_SIZE = 20
TEST_SET_SIZE = 1


def main(shuffle_recordings: bool = True, random_seed: int = 42):

    current_script_directory = dirname(realpath(__file__))
    folds_destination_folder = dirname(current_script_directory)

    all_fold_indices = list(range(NUMBER_OF_RECORDINGS_IN_DREEM_DOD_H))
    if shuffle_recordings:
        seed(random_seed)
        shuffle(all_fold_indices)

    all_test_sets = [[recording_id] for recording_id in all_fold_indices]

    for fold_index in range(NUMBER_OF_RECORDINGS_IN_DREEM_DOD_H):
        test_set = all_test_sets[fold_index]

        training_and_validation_sets = all_fold_indices.copy()
        for index in test_set:
            training_and_validation_sets.remove(index)

        if shuffle_recordings:
            shuffle(training_and_validation_sets)
        training_set = training_and_validation_sets[:TRAINING_SET_SIZE]
        validation_set = training_and_validation_sets[TRAINING_SET_SIZE:]

        fold_dict = {"training": training_set,
                     "validation": validation_set,
                     "test": test_set}

        fold_yaml_file_path = join(folds_destination_folder, "fold_%02d.yaml" % fold_index)
        with open(fold_yaml_file_path, "w") as f:
            yaml.dump(fold_dict, f)


if __name__ == "__main__":
    main()
