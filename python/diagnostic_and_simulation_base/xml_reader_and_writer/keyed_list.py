"""
A list of dataclasses which can be indexed by key, as well as by position, e.g. `workflow.tree["ELMAG"]` for the `<tree>` whose `<tree_name>` is ELMAG.

This module knows nothing about any one code's settings: the dataclass of the items names its key.
"""

import typing


class KeyedItem(typing.Protocol):
    """
    A dataclass which can be held in a `KeyedList`: one which names the field holding its key in `KEY`, e.g. `WorkflowTree.KEY = "tree_name"`.
    """

    KEY: typing.ClassVar[str]


ItemT = typing.TypeVar("ItemT", bound=KeyedItem)


class KeyedList(list[ItemT]):
    """
    A list of dataclasses which can be indexed by key, as well as by position.

    It holds a repeated element which its XML schema keeps unique by one of its child elements, the key, with an `xs:key` or `xs:unique`, e.g. in `workflow.xsd`:
        <xs:key name="workflow_tree_name_key">
            <xs:selector xpath="tree"/>
            <xs:field xpath="tree_name"/>
        </xs:key>
    The dataclass of the items names the key in its `KEY`, e.g. `WorkflowTree.KEY = "tree_name"`, and the field of the repeated element is a `KeyedList`,
    e.g. `tree: KeyedList[WorkflowTree]`. So:
        workflow.tree[0]          # the first tree, as in any list
        workflow.tree["ELMAG"]    # the tree whose `tree_name` is "ELMAG"
        "ELMAG" in workflow.tree  # whether there is one
    A key is always a string, so it is never mistaken for a position.
    Otherwise it is a list, which is iterated, changed and written as any other, e.g. `workflow.tree.append(WorkflowTree(...))`.
    To make one, pass it the items, e.g. `Workflow(tree=KeyedList([elmag_settings]))`.
    """

    @typing.overload
    def __getitem__(self, index: typing.SupportsIndex) -> ItemT: ...

    @typing.overload
    def __getitem__(self, index: slice) -> list[ItemT]: ...

    @typing.overload
    def __getitem__(self, index: str) -> ItemT: ...

    def __getitem__(self, index: typing.SupportsIndex | slice | str) -> ItemT | list[ItemT]:
        """
        :param index: A position, e.g. `0`, a slice, e.g. `1:3`, or a key, e.g. "ELMAG"
        :return: The item at the position, a list of the items in the slice, or the item whose key it is
        """

        if not isinstance(index, str):
            return super().__getitem__(index)

        keys: list[str] = []
        matching_items: list[ItemT] = []
        for item in self:
            key: str = getattr(item, type(item).KEY)
            keys.append(key)
            if key == index:
                matching_items.append(item)
        if len(matching_items) == 0:
            raise KeyError(f"No item has the key {index!r}. The keys are {keys}")
        if len(matching_items) > 1:
            # The XML schema keeps the keys unique, so only items added in Python can share one
            raise ValueError(f"{len(matching_items)} items have the key {index!r}, but each key must be unique")
        return matching_items[0]

    def __contains__(self, value: object) -> bool:
        """
        :param value: A key, e.g. "ELMAG", or an item
        :return: Whether an item has the key, or is the item
        """

        if not isinstance(value, str):
            return super().__contains__(value)

        for item in self:
            if getattr(item, type(item).KEY) == value:
                return True
        return False
