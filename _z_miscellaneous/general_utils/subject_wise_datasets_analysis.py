import csv
from os.path import dirname, realpath, join
from typing import List
from _z_miscellaneous.general_utils.fold_wise_datasets_analysis import get_dataset_recordings_as_paths, \
    recording_wise_stats, labels_list


def main(datasets_list: List[str]):

    current_script_directory = dirname(realpath(__file__))

    for dataset_name in datasets_list:
        output_file_path = join(current_script_directory, "%s_subject_wise_stats.csv" % dataset_name)
        number_of_recordings = len(get_dataset_recordings_as_paths(dataset_name))

        csv_header_list = ["ID", "all"] + labels_list
        write_first_line_flag = True

        for recording_id in range(number_of_recordings):
            recording_stats_dict = recording_wise_stats(dataset_name, recording_id)
            recording_stats_dict["ID"] = recording_id

            with open(output_file_path, "a", newline="") as csv_output_file:
                writer = csv.DictWriter(csv_output_file, fieldnames=csv_header_list)
                if write_first_line_flag:
                    writer.writeheader()
                    write_first_line_flag = False
                writer.writerow(recording_stats_dict)


if __name__ == "__main__":
    main(["MASS_SS3"])
