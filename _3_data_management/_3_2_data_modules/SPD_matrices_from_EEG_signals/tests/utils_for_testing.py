from collections import Counter
from os import listdir
from os.path import dirname, realpath, join, isdir, isfile
from typing import List


def get_epochs_for_recordings_in_preprocessed_dataset(preprocessed_dataset_name: str, wanted_recordings_indices: List[int], clipping: int = 0):
    labels = ["N3", "N2", "N1", "REM", "Awake"]

    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(dirname(dirname(current_script_directory))))
    data_preprocessing_directory = join(root_directory, "_2_data_preprocessing")
    preprocessed_data_directory = join(data_preprocessing_directory, "_2_4_preprocessed_data")
    dataset_folder = join(preprocessed_data_directory, preprocessed_dataset_name)
    assert isdir(dataset_folder)

    recording_epochs_folders = [join(join(dataset_folder, recording_folder_name), "epochs")
                                for recording_folder_name in listdir(dataset_folder)]
    recording_epochs_folders.sort()

    output_dict = {"total": 0}
    for label in labels:
        output_dict[label] = 0

    for wanted_recording_index in wanted_recordings_indices:
        recording_epochs_folder = recording_epochs_folders[wanted_recording_index]
        assert isdir(recording_epochs_folder)
        epoch_files = listdir(recording_epochs_folder)
        epoch_files.sort()
        for epoch_file in epoch_files:
            assert isfile(join(recording_epochs_folder, epoch_file))
            assert epoch_file[-4:] == ".pkl"
        epoch_labels = [epoch_file[5:-4] for epoch_file in epoch_files]

        clipped_epoch_labels = epoch_labels
        if clipping > 0:
            clipped_epoch_labels = clipped_epoch_labels[clipping:-clipping]

        epoch_labels_counter = Counter(clipped_epoch_labels)
        for key in epoch_labels_counter.keys():
            assert key in labels
        assert len(clipped_epoch_labels) == sum([value for value in epoch_labels_counter.values()])

        output_dict["total"] += len(clipped_epoch_labels)
        for label in labels:
            if label in epoch_labels_counter.keys():
                output_dict[label] += epoch_labels_counter[label]

    return output_dict


if __name__ == "__main__":
    print(get_epochs_for_recordings_in_preprocessed_dataset("MASS_SS3_dataset_with_PredicAlert_signals_config", list(range(62)), 0))