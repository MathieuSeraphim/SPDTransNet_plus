import json


def recursive_number_of_leaves_and_max_depth_in_nested_dict(nested_dict_or_leaf):
    if isinstance(nested_dict_or_leaf, dict):
        width_accumulator = 0
        max_depth = 0
        for key in nested_dict_or_leaf.keys():
            subdict_width, subdict_depth = recursive_number_of_leaves_and_max_depth_in_nested_dict(nested_dict_or_leaf[key])
            width_accumulator += subdict_width
            max_depth = max(max_depth, subdict_depth)
        final_depth = max_depth + 1 # Adding 1 for the dict key
        return width_accumulator, final_depth
    return 1, 1  # Leaf


def nested_dict_to_array(nested_dict_or_leaf):

    if not isinstance(nested_dict_or_leaf, dict):
        return [[nested_dict_or_leaf]]
    if len(nested_dict_or_leaf) == 0:
        return None

    array_width, array_depth = recursive_number_of_leaves_and_max_depth_in_nested_dict(nested_dict_or_leaf)

    output_array = [[None for i in range(array_width)] for j in range(array_depth)]
    current_column_index = 0

    for key in nested_dict_or_leaf.keys():
        subdict_width, subdict_depth = recursive_number_of_leaves_and_max_depth_in_nested_dict(nested_dict_or_leaf[key])
        subdict_array = nested_dict_to_array(nested_dict_or_leaf[key])

        if subdict_array is not None:
            output_array[0][current_column_index] = key
            for row_index in range(subdict_depth):
                row_index_in_output_array = row_index + 1
                for column_index in range(subdict_width):
                    column_index_in_output_array = column_index + current_column_index
                    output_array[row_index_in_output_array][column_index_in_output_array] = subdict_array[row_index][column_index]
        current_column_index += subdict_width

    return output_array


def output_array_to_csv_string(output_array):
    num_rows = len(output_array)
    num_columns = len(output_array[0])
    for column in output_array:
        assert len(column) == num_columns

    output_string = ""
    for row_id in range(num_rows):
        for column_id in range(num_columns):
            if output_array[row_id][column_id] is not None:
                output_string += str(output_array[row_id][column_id])
            output_string += ","
        output_string += "\n"

    return output_string


def nested_dict_to_multicolumn_csv_string(potentially_nested_dict: dict):
    dict_as_array = nested_dict_to_array(potentially_nested_dict)
    array_as_csv_string = output_array_to_csv_string(dict_as_array)
    return array_as_csv_string


def json_nested_dict_to_multicolumn_csv_file(json_filename, csv_filename):
    with open(json_filename, "r") as json_file:
        nested_dict = json.load(json_file)
    csv_string = nested_dict_to_multicolumn_csv_string(nested_dict)
    with open(csv_filename, "w") as csv_file:
        csv_file.write(csv_string)


if __name__ == "__main__":

    # test_dict = {
    #     "a": 65536,
    #     "b": {
    #         "Blibli": 43,
    #         "d": {
    #             35: 4242,
    #             "Other": 15
    #
    #         },
    #         "Oooooooo": "oooooooooo",
    #         "None": {}
    #     },
    #     "None2": {}
    # }
    # output_array = nested_dict_to_array(test_dict)
    # csv_string = output_array_to_csv_string(output_array)
    # print(csv_string)

    json_nested_dict_to_multicolumn_csv_file("stats_for_dataset_EUSIPCO.json", "stats_for_dataset_EUSIPCO.csv")
    json_nested_dict_to_multicolumn_csv_file("stats_for_dataset_EUSIPCO_as_percentages.json", "stats_for_dataset_EUSIPCO_as_percentages.csv")

