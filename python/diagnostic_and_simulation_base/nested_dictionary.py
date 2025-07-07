import typing
from collections import OrderedDict

import numpy as np


class ListOfNestedDict:
    def __init__(self, nested_dict_data: "NestedDict") -> None:
        """This passes in the entire dictionary"""
        self.nested_dict_data = nested_dict_data

    def __getitem__(self, key_in: str) -> typing.Any:
        # Store the keys, until we get to the "data level"

        # print(key_in)
        if not hasattr(self, "keys"):
            # initialising
            self.keys = [key_in]
        else:
            # Store key
            self.keys.append(key_in)
        # print('starting __getitem__')
        # print(self.keys)
        # print(' ')

        accumulated_results = [self.nested_dict_data[wild_card_key] for wild_card_key in self.nested_dict_data]
        # print('accumulated results')
        result_final_level = accumulated_results[0]
        for key_now in self.keys:
            result_final_level = result_final_level[key_now]
        # print('got keys')

        # print(type(result_final_level))
        if isinstance(result_final_level, NestedDict):
            return self
        elif isinstance(result_final_level, ListOfNestedDict):
            # keep going recursively, storing "self.keys" until we get to the data
            return self
        else:
            result_to_return = []
            for accumulated_result in accumulated_results:
                result_final_level = accumulated_result
                for key_now in self.keys:
                    result_final_level = result_final_level[key_now]
                result_to_return.append(result_final_level)

            result_to_return_np = np.array(result_to_return)
            # move the 0th axis to the end - this gives the desired shape
            new_axes = tuple(range(1, result_to_return_np.ndim)) + (0,)
            result_to_return_np = np.transpose(result_to_return_np, axes=new_axes)
            return result_to_return_np

    # def __len__(self):
    #     return len(self.nested_dict_data.keys()) # this probably won't work???


class NestedDict(OrderedDict):  # type: ignore
    """A custom nested dictionary

    Wild card accumulator: In MDSplus, and all of our code, by convention time is the 0th index.
    If there is vecor_data inside a node, then it is highly likely that it is a time-dependent quantity

    Example, showing the array ordering:
    ```python
    from diagnostics_analysis_base import NestedDict
    import numpy as np

    x = NestedDict()

    x["top_level_1"]["middle_level_1"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_1"]["middle_level_1"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_1"]["middle_level_2"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_1"]["middle_level_2"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_1"]["middle_level_3"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_1"]["middle_level_3"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_2"]["middle_level_1"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_2"]["middle_level_1"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_2"]["middle_level_2"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_2"]["middle_level_2"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_2"]["middle_level_3"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_2"]["middle_level_3"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_3"]["middle_level_1"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_3"]["middle_level_1"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_3"]["middle_level_2"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_3"]["middle_level_2"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_3"]["middle_level_3"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_3"]["middle_level_3"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_4"]["middle_level_1"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_4"]["middle_level_1"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_4"]["middle_level_2"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_4"]["middle_level_2"]["bottom_level_2"] = np.random.rand(6, 9)
    x["top_level_4"]["middle_level_3"]["bottom_level_1"] = np.random.rand(6, 9)
    x["top_level_4"]["middle_level_3"]["bottom_level_2"] = np.random.rand(6, 9)

    y = x["*"]["*"]["*"]
    assert y.shape == (6, 9, 2, 3, 4)
    ```
    """

    def __missing__(self, key: str) -> "NestedDict":
        """Add to the missing dict"""
        value = self[key] = type(self)()
        return value

    def to_dictionary(self) -> dict[str, typing.Any]:
        """Convert NestedDict to a standard python `dict`"""
        result = {}
        for key, value in self.items():
            if isinstance(value, NestedDict):
                result[key] = value.to_dictionary()
            else:
                result[key] = value
        return result

    def __setitem__(
        self,
        key: str,
        value: typing.Any,
    ) -> None:
        """Override __setitem__ to ensure all dictionaries are converted to NestedDict."""
        value = _convert_to_nested_dict(value)
        OrderedDict.__setitem__(self, key, value)

    # BUXTON: figure out why this fails mypy tests?
    def update(  # type: ignore
        self,
        dict_to_add: dict[typing.Any, typing.Any] | None = None,
    ) -> None:
        """Recursively updates this dictionary with another dictionary,
        ensuring that nested dictionaries are converted to NestedDict."""

        if dict_to_add is None:
            dict_to_add = {}

        for key, value in dict_to_add.items():
            if isinstance(value, dict):
                # If the key exists and is already a NestedDict, update recursively
                if key in self and isinstance(self[key], NestedDict):
                    self[key].update(value)
                else:
                    # Otherwise, convert the value to NestedDict and assign it
                    self[key] = NestedDict(value)
            else:
                # Set the value directly if it's not a dictionary
                self[key] = value

    # def update(
    #     self,
    #     dict_to_add: dict | None = None,
    # ) -> None:
    #     """Recursively updates this dictionary with another dictionary or kwargs,
    #     ensuring that nested dictionaries are converted to NestedDict."""

    #     if dict_to_add is None:
    #         dict_to_add = {}

    #     nested_dict_to_add = _convert_to_nested_dict(dict_to_add)

    #     for key, value in dict_to_add.items():
    #         if isinstance(value, dict):
    #             # Recursively update or set as NestedDict
    #             self[key] = _convert_to_nested_dict(value)
    #         else:
    #             self[key] = value

    def print_data(self, indent: int = 0) -> str:
        """Recursively generates JSON-like string with proper formatting."""
        spacing = "  " * indent  # 2 spaces per level of indentation
        items = []
        for key, value in self.items():
            if isinstance(value, NestedDict):
                # Recursively generate nested dicts
                items.append(f'{spacing}  "{key}": {{')
                items.append(value.print_data(indent + 1))  # Recursive call for nested dict
                items.append(f"{spacing}  }}")
            else:
                # Handle other types, using repr() for proper formatting
                items.append(f'{spacing}  "{key}": {repr(value)}')

        return "\n".join(items)  # Return the joined string without leading/trailing newlines

    def __str__(self) -> str:
        """Override __str__ to use print_data for JSON-like string representation."""
        return "{\n" + self.print_data() + "\n}"

    def __getitem__(self, key: str) -> typing.Any:
        # wild card
        if key == "*":
            # use wild_card and retrieve the data
            accumulated_results = [self[wild_card_key] for wild_card_key in self]
            # test if data is at this level or not
            if isinstance(accumulated_results[0], NestedDict):
                return ListOfNestedDict(self)
            else:
                # Stack the arrays along a new dimension (axis=0)
                stacked_results = np.stack(accumulated_results, axis=-1)  # Stack along the last axis
                return stacked_results

        # Normal behaviour
        return OrderedDict.__getitem__(self, key)

    # BUXTON: does this is broken!!!!!
    def print_keys(self, d: None = None, path: None = None):
        # import pdb; pdb.set_trace()
        if d is None:
            d = self  # Use the current instance as the dictionary

        if path is None:
            path = []

        # Check if keys_long is already initialized in self
        if not hasattr(self, "keys_long"):
            self.keys_long = []
            self.data_type_long = []

        for key, value in d.items():
            new_path = path + [f'["{key}"]']

            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                # print("BUXTON: error")
                self.print_keys(value, new_path)
            else:
                # Prepare the type information
                if isinstance(value, np.ndarray):
                    type_info = f" = np.ndarray;  shape={value.shape}"
                elif isinstance(value, list):
                    type_info = f" = list;  length={len(value)}"
                else:
                    type_info = f" = {type(value).__name__}"

                # Print the path with the type info
                self.keys_long.append("".join(new_path))  # Store key path
                self.data_type_long.append(type_info)  # Store type info

        if path == []:  # Check if we are at the top level
            # Find the length of the longest key path
            max_key_length = max(len(key) for key in self.keys_long)

            # Print with proper alignment
            for key, type_info in zip(self.keys_long, self.data_type_long):
                # Calculate required spaces for alignment
                spaces = " " * (max_key_length - len(key) + 4)  # 4 spaces for padding
                print(f"{key}{spaces}{type_info}")


# BUXTON: is "typing.Any" only option??
def _convert_to_nested_dict(value: typing.Any) -> typing.Any:
    """Helper function to recursively convert all dictionaries to NestedDict."""
    if isinstance(value, OrderedDict) and not isinstance(value, NestedDict):
        # Create a new NestedDict and populate it using a for loop
        nested = NestedDict()
        for k, v in value.items():
            nested[k] = _convert_to_nested_dict(v)  # Recursively convert
        return nested
    return value
