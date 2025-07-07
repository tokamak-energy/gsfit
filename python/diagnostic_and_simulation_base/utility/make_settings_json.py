import json
import typing

import numpy as np


def make_settings_json(data: dict[str, typing.Any], json_indent: int = 2) -> str:
    """
    Function to make dictionary into JSON string.
    The purpose of this function is for writing the "settings.json" files to MDSplus.
    The format of the string is:
    results = '{
        "settings_file_1.json": "{\n  \"json_content\": \"value\"}",
        "settings_file_2.json": "{\n  \"json_content\": 123.4}",
        "settings_file_3.csv": "123.4, 456.7, 789.0\n 0.1, 0.2, 0.3"
    }'

    The reason for this format is so that files with different formats (e.g. JSON, CSV, namelists) can all be stored

    :param data: the dictionary containing the settings files to be made into a JSON string.
    :param json_indent: the indentation of the JSON string.
    :return: the JSON string.
    """

    def convert_dict_with_numpy_to_dict_no_numpy(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Recursively searches through nested dictionaries and converts any numpy
        objects to fundamental python objects, whcih are JSON serializable.

        :param data: dictionary, which might contain numpy objects
        """
        if isinstance(data, dict):
            return_data = dict()
            for key, value in data.items():
                return_data[key] = convert_dict_with_numpy_to_dict_no_numpy(value)
            return return_data
        elif isinstance(data, list):
            return_data = []
            for item in data:
                return_data.append(convert_dict_with_numpy_to_dict_no_numpy(item))
            return return_data
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.generic):
            return data.item()
        else:
            return data

    dict_no_numpy = convert_dict_with_numpy_to_dict_no_numpy(data)

    if np.all([isinstance(value, dict) for value in dict_no_numpy.values()]):
        new_dict = dict()
        for key in dict_no_numpy.keys():
            json_internal_content = json.dumps(
                dict_no_numpy[key],
                indent=json_indent,
                sort_keys=True,
            )
            new_dict[key] = json_internal_content
    else:
        new_dict = dict_no_numpy

    json_content = json.dumps(new_dict, indent=json_indent, sort_keys=True)

    return json_content
