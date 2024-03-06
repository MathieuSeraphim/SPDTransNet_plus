from _3_data_management._3_2_data_modules.DataModuleWrapper import get_datamodule_from_config_file
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.tests.utils_for_testing import \
    get_epochs_for_recordings_in_preprocessed_dataset


preprocessed_dataset_name = "MASS_SS3_dataset_with_PredicAlert_signals_config"
non_test_clipping = 10
test_clipping = 24

datamodule = get_datamodule_from_config_file("SPD_matrices_from_EEG_MASS_dataset_PredicAlert_signals_config.yaml", batch_size=128, cross_validation_fold_index=0)

# Fold 0
training = [0, 1, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 27,
  28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
  50, 52, 53, 54, 55, 57, 58, 59, 60, 61]
validation = [16, 9, 56, 34, 26, 2, 35, 51, 25, 18]
test = [10, 6]

expected_training_set_size = get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name, training, non_test_clipping)["N2"] * 5
expected_validation_set_size = get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name, validation, non_test_clipping)["total"]
expected_test_set_size = get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name, test, test_clipping)["total"]

assert expected_training_set_size == len(datamodule.training_set)
assert expected_validation_set_size == len(datamodule.validation_set)
assert expected_test_set_size == len(datamodule.test_set)

# Fold 11
training = [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25,
  26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 46, 47, 48, 50,
  51, 52, 53, 54, 56, 57, 58, 59, 60, 61]
validation = [21, 55, 10, 33, 3, 49, 38, 19, 7, 17]
test = [28, 45]

datamodule.reinitialize_datasets(11)

expected_training_set_size = get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name, training, non_test_clipping)["N2"] * 5
expected_validation_set_size = get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name, validation, non_test_clipping)["total"]
expected_test_set_size = get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name, test, test_clipping)["total"]

assert expected_training_set_size == len(datamodule.training_set)
assert expected_validation_set_size == len(datamodule.validation_set)
assert expected_test_set_size == len(datamodule.test_set)



