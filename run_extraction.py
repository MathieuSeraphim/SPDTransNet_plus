import sys
from os import listdir
from os.path import realpath, dirname, join, isdir
from _2_data_preprocessing._2_2_data_extraction._extraction_scripts import test_extraction, MASS_extraction, \
    Dreem_DOD_H_extraction

valid_datasets = ["MASS_SS1", "MASS_SS3", "Dreem_DOD-H", "test"]


def main(dataset_name: str):
    assert dataset_name in valid_datasets

    root_directory = dirname(realpath(__file__))
    data_preprocessing_directory = join(root_directory, "_2_data_preprocessing")
    original_datasets_directory = join(data_preprocessing_directory, "_2_1_original_datasets")
    dataset_folders_list = [folder_name for folder_name in listdir(original_datasets_directory)
                            if isdir(join(folder_name, original_datasets_directory))]

    if dataset_name not in dataset_folders_list:
        raise FileNotFoundError("The requested dataset isn't in its appropriate folder!")

    if dataset_name == "test":
        test_extraction.main()
    elif dataset_name == "MASS_SS1":
        MASS_extraction.main("SS1")
    elif dataset_name == "MASS_SS3":
        MASS_extraction.main("SS3")
    elif dataset_name == "Dreem_DOD-H":
        Dreem_DOD_H_extraction.main()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main("test")
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError("More than one argument.")
