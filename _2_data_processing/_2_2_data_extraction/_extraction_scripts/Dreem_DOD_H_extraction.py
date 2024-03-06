import json
import pickle
from os import mkdir, listdir
from os.path import dirname, realpath, join, isdir, exists
from shutil import rmtree
import h5py
import numpy as np

EPOCH_DURATION = 30
DREEM_HYPNOGRAM_CORRESPONDANCE_DICT = {
    -1: "NOT SCORED",
    0: "Awake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}
OUR_LABELS_ORDER = ["N3", "N2", "N1", "REM", "Awake"]


def process_dreem_h5(dreem_h5_file: str):
    recording_dict = {}
    list_of_eeg_signals = []
    sampling_frequency = None

    recording_data = h5py.File(dreem_h5_file, "r")

    recording_id = recording_data.attrs["record_id"].decode('UTF-8')
    recording_signal_descriptions_as_string = recording_data.attrs["description"].decode('UTF-8')
    recording_signal_descriptions = json.loads(recording_signal_descriptions_as_string)

    original_hypnogram = recording_data["hypnogram"][:]
    assert len(original_hypnogram.shape) == 1
    number_of_epochs_in_recording_pre_removal_of_bad_epochs = len(original_hypnogram)

    bad_indices, = np.where(original_hypnogram == -1)
    hypnogram = np.delete(original_hypnogram, bad_indices)
    number_of_epochs_in_recording_post_removal_of_bad_epochs = len(hypnogram)
    assert number_of_epochs_in_recording_post_removal_of_bad_epochs ==\
           number_of_epochs_in_recording_pre_removal_of_bad_epochs - len(bad_indices)

    for epoch_id in range(number_of_epochs_in_recording_post_removal_of_bad_epochs):
        hypnogram[epoch_id] = OUR_LABELS_ORDER.index(DREEM_HYPNOGRAM_CORRESPONDANCE_DICT[hypnogram[epoch_id]])

    for signal_dict in recording_signal_descriptions:
        if signal_dict["domain"] != "EEG":
            continue
        signal_name = signal_dict["name"]
        signal_path = signal_dict["path"]
        signal_sampling_frequency = signal_dict["fs"]
        if sampling_frequency is None:
            sampling_frequency = signal_sampling_frequency
        assert signal_sampling_frequency == sampling_frequency

        eeg_signal = recording_data[signal_path]
        assert len(eeg_signal.shape) == 1

        number_of_samples_per_epoch = EPOCH_DURATION * sampling_frequency
        assert len(eeg_signal) / number_of_samples_per_epoch == number_of_epochs_in_recording_pre_removal_of_bad_epochs

        extended_original_hypnogram = np.repeat(original_hypnogram, number_of_samples_per_epoch)
        extended_bad_indices, = np.where(extended_original_hypnogram == -1)
        eeg_signal = np.delete(eeg_signal, extended_bad_indices)
        assert len(eeg_signal) / number_of_samples_per_epoch == number_of_epochs_in_recording_post_removal_of_bad_epochs

        recording_dict[signal_name] = eeg_signal
        list_of_eeg_signals.append(signal_name)

    recording_dict["hypno"] = hypnogram
    recording_dict["Fs"] = sampling_frequency
    recording_dict["EEG Signals"] = list_of_eeg_signals

    return recording_id, recording_dict, sorted(list_of_eeg_signals)


def main():
    dataset = "Dreem_DOD-H"
    data_extraction_dir = dirname(dirname(realpath(__file__)))
    data_preprocessing_dir = dirname(data_extraction_dir)
    datasets_dir = join(data_preprocessing_dir, "_2_1_original_datasets")
    Dreem_data_dir = join(datasets_dir, dataset)
    Dreem_save_dir = join(data_extraction_dir, dataset + "_extracted")

    assert isdir(Dreem_data_dir)
    if exists(Dreem_save_dir):
        rmtree(Dreem_save_dir)
    mkdir(Dreem_save_dir)

    recordings_list = [join(Dreem_data_dir, recording_file) for recording_file in listdir(Dreem_data_dir)
                       if recording_file[-3:] == ".h5"]

    eeg_signals_list = None
    recording_keys = None
    for recording in recordings_list:
        recording_id, recording_dict, recording_eeg_signals = process_dreem_h5(recording)

        if recording_keys is None:
            recording_keys = recording_dict.keys()
        assert recording_keys == recording_dict.keys()

        if eeg_signals_list is None:
            eeg_signals_list = recording_eeg_signals
        assert eeg_signals_list == recording_eeg_signals

        save_file = join(Dreem_save_dir, recording_id + '.pkl')
        pickle.dump(recording_dict, open(save_file, 'wb'))

    keys_save_file = join(Dreem_save_dir, ".saved_keys.txt")
    with open(keys_save_file, "w") as f:
        f.write(", ".join(recording_keys))


if __name__ == '__main__':
    main()
