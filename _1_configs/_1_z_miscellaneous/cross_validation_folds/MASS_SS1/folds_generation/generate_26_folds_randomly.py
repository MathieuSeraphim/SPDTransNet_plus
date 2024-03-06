from os.path import realpath, dirname, join
from itertools import islice
from random import shuffle, seed
import yaml

# It is pretty much hardcoded that TEST_SET_MINIMUM_SIZE should be 2 and NUMBER_OF_RECORDINGS_IN_MASS_SS1 should be odd.
NUMBER_OF_RECORDINGS_IN_MASS_SS1 = 53
TRAINING_SET_SIZE = 42
TEST_SET_MINIMUM_SIZE = 2


# Taken from https://realpython.com/how-to-split-a-python-list-into-chunks/, who took it from Python 3.11's itertools
# documentation.
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# Generates 26 folds for MASS-SS1 with 42 recordings per training set, 9 per validation set and 2 per test set (except
# for the last fold, which will have 8 for the validation set and 3 for the test set).
def main(shuffle_recordings: bool = True, random_seed: int = 42):

    current_script_directory = dirname(realpath(__file__))
    folds_destination_folder = dirname(current_script_directory)

    all_fold_indices = list(range(NUMBER_OF_RECORDINGS_IN_MASS_SS1))
    if shuffle_recordings:
        seed(random_seed)
        shuffle(all_fold_indices)

    all_test_sets = []
    for test_set in batched(all_fold_indices, TEST_SET_MINIMUM_SIZE):
        all_test_sets.append(list(test_set))
    all_test_sets[-2] += all_test_sets[-1]
    all_test_sets = all_test_sets[:-1]

    for fold_index in range(len(all_test_sets)):
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
