"""
Read XML settings into dataclasses, checking every XML file against its XML schema, using `xmlschema`.

This module knows nothing about any one code's settings.
The code passes in the dataclass to fill, e.g. GSFit's settings, and gets back an instance of that same dataclass, so the result is fully typed.
A matrix, written as one `<row>` per row, is read into a 2D numpy array; see `matrix_row_element`.
"""

import dataclasses
import types
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xmlschema
from xmlschema.validators import XsdAtomic
from xmlschema.validators import XsdComplexType
from xmlschema.validators import XsdElement
from xmlschema.validators import XsdGroup
from xmlschema.validators import XsdList
from xmlschema.validators import XsdSimpleType

DataclassT = typing.TypeVar("DataclassT")


def read_xml_settings_to_dataclasses(
    xml_file_path: Path,
    xsd_directory_path: Path,
    settings_class: type[DataclassT],
) -> tuple[DataclassT, dict[tuple[str, ...], Path]]:
    """
    Read XML settings into dataclasses, after checking every XML file against its XML schema.

    The XML file can link to other XML files, which can link to others in turn, e.g. GSFit's `GSFIT_code_settings.xml` holds
    `<equilibrium href="equilibrium/equilibrium.xml"/>` inside `<ids>`, and `equilibrium.xml` holds `<grid href="grid.xml"/>`.
    A link is followed when the type of its field is the dataclass of the linked file's root element, e.g. `grid: Grid`, and the linked file is read in its place.
    Otherwise the link is read like any other element, e.g. into a dataclass with an `href` field.
    A file linked from several places, e.g. `workflow.xml`, which every IDS's file links to, is read once, and every link to it holds the same dataclass.
    So a setting changed in Python through one of the links, e.g. `settings.ids.pf_active.workflow.psu2coil.run_name = "RUN02"`, is changed for all of them,
    as changing the file would.

    Every XML file is checked against the XML schema of the dataclass of its root element, which that dataclass names in its `XSD_PATH`,
    relative to `xsd_directory_path`, e.g. `Grid.XSD_PATH = "equilibrium/grid.xsd"`.
    So a dataclass of a linked file's root element is told apart from the dataclass of any other element by having an `XSD_PATH`.

    For example, with GSFit's settings:
        settings, xml_file_paths = read_xml_settings_to_dataclasses(
            xml_file_path=settings_path / "GSFIT_code_settings.xml",
            xsd_directory_path=gsfit_path / "settings_schema" / "xsd",
            settings_class=GsfitCodeSettings,
        )

    :param xml_file_path: Path to the XML file, e.g. `settings/default/GSFIT_code_settings.xml`
    :param xsd_directory_path: Directory holding the XML schemas, e.g. GSFit's `settings_schema/xsd`
    :param settings_class: Dataclass of the root element of `xml_file_path`, with a field for each of its child elements and attributes,
        and the `XSD_PATH` of its XML schema. It is passed in, rather than imported here, so that this module does not depend on any one code's settings
    :return settings: The settings, as an instance of `settings_class`
    :return xml_file_paths: The path of each XML file read, by where its settings are in `settings`: the names of the fields leading to them, e.g.
        `("ids", "pf_active", "workflow")` for `settings.ids.pf_active.workflow`, read from `.../workflow.xml`, and `()` for `xml_file_path` itself.
        The dataclasses hold no file names, so this is what `make_settings_from_xml_xsd` writes the settings back to, as files of the same names
    """

    # The `xsi:noNamespaceSchemaLocation` written in the XML file is only a hint for editors, so we load the XML schema ourselves
    xsd_relative_path: object = getattr(settings_class, "XSD_PATH", None)
    if not isinstance(xsd_relative_path, str):
        raise TypeError(f"`{settings_class.__name__}` has no `XSD_PATH`, so the XML schema to check {xml_file_path} against is not known")
    xsd_file_path: Path = xsd_directory_path / xsd_relative_path
    xml_schema: xmlschema.XMLSchema = xmlschema.XMLSchema(xsd_file_path)

    # Raises `xmlschema.XMLSchemaValidationError`, giving the reason and the path, e.g. `/grid/r_min`, if the file does not match the schema.
    # Otherwise returns the root element as a `dict`, with every value decoded into the type given by the schema, e.g. `xs:double` -> `float`.
    # `keep_empty=True` decodes an empty element, e.g. `<comment/>`, as "" rather than None.
    # The validation is strict, the default, so the result is never the `(data, errors)` tuple returned by lax validation
    try:
        root_element_data: dict[str, typing.Any] | None = typing.cast(dict[str, typing.Any] | None, xml_schema.to_dict(xml_file_path, keep_empty=True))
    except xmlschema.XMLSchemaValidationError as error:
        error.add_note(f"In {xml_file_path}, checked against {xsd_file_path}")  # the error does not say which file
        raise

    xml_file_paths: dict[tuple[str, ...], Path] = {(): xml_file_path.resolve()}
    settings: DataclassT = _decoded_xml_to_dataclass(
        dataclass_type=settings_class,
        element_data=root_element_data,
        element_names=[],
        xml_file_path=xml_file_path,
        xsd_directory_path=xsd_directory_path,
        file_location=(),
        xml_file_paths=xml_file_paths,
        linked_files_read={},
    )

    return settings, xml_file_paths


def _decoded_xml_to_dataclass(
    dataclass_type: type[DataclassT],
    element_data: dict[str, typing.Any] | None,
    element_names: list[str],
    xml_file_path: Path,
    xsd_directory_path: Path,
    file_location: tuple[str, ...],
    xml_file_paths: dict[tuple[str, ...], Path],
    linked_files_read: dict[tuple[Path, type], tuple[object, tuple[str, ...]]],
) -> DataclassT:
    """
    Pass an XML element, as decoded by `xmlschema`, and everything inside it, into a dataclass.

    `xmlschema` has already checked the element against its XML schema, and decoded every value into its type, e.g. `xs:double` -> `float`.
    Each field takes the value of the child element, or attribute, with the same name, after checking that it has the type of the field:
    * `bool`, `float`, `int`, `str`, or a `typing.Literal` for an enumeration
    * `X | None` for an optional element, which is None if the element is not given, or is nil, e.g. `<pulseNo xsi:nil="true"/>`
    * a dataclass, for an element holding child elements or attributes
    * `list[...]`, for a repeated element, e.g. `list[FluxLoop]`, or a space-separated list, e.g. "0.1 0.2 0.3" -> `list[float]`
    * `npt.NDArray[np.float64]`, for a matrix with one `<row>` per row; see `matrix_row_element`

    :param dataclass_type: Dataclass of the element, e.g. `Grid`
    :param element_data: The element, as decoded by `xmlschema`: the name of each child element or attribute -> its value.
        An attribute's name starts with "@", e.g. `{"@href": "grid.xml"}`. None for an element with no content, e.g. `<freegs/>`
    :param element_names: Path of the element below the root element of its XML file, e.g. `["constraints"]`, or `[]` for the root element
    :param xml_file_path: Path to the XML file holding the element, which its links are relative to
    :param xsd_directory_path: Directory holding the XML schemas, for the XML files it links to
    :param file_location: Where the settings of the XML file holding the element are, in the settings being read, e.g. `("ids", "equilibrium")`,
        or `()` for the XML file being read
    :param xml_file_paths: The path of each XML file read so far, by where its settings are; see `read_xml_settings_to_dataclasses`.
        Each XML file this element links to is added
    :param linked_files_read: Each linked XML file read so far, by its path and the dataclass of its root element: its settings,
        and where they were first found. A file linked again is not read again, and the new link holds the same settings
    :return: The element, as an instance of `dataclass_type`
    """

    if element_data is None:
        element_data = {}

    field_types: dict[str, typing.Any] = typing.get_type_hints(dataclass_type)
    field_values: dict[str, typing.Any] = {}
    for key, value in element_data.items():
        if key.startswith(("@xmlns", "@xsi:")):
            continue  # a namespace declaration, or `xsi:noNamespaceSchemaLocation`, which are not settings

        # An attribute, e.g. `@href`, is a field just like a child element
        field_name: str = key.removeprefix("@")
        field_names: list[str] = [*element_names, field_name]  # e.g. `["constraints", "flux_loop"]`
        if field_name not in field_types:
            raise TypeError(f"{xml_file_path}: `{'/'.join(field_names)}` has no field in `{dataclass_type.__name__}`")
        field_type: typing.Any = field_types[field_name]

        # An optional field, `X | None`, holds an `X` when its element is given, unless the element is nil
        value_type: typing.Any = field_type
        if typing.get_origin(field_type) in (types.UnionType, typing.Union):
            for argument_type in typing.get_args(field_type):
                if argument_type is not type(None):
                    value_type = argument_type

        # `xmlschema` decodes both a repeated element and a space-separated list as a `list`, whose items are all checked in the same way.
        # Anything else is a single item
        is_list: bool = typing.get_origin(value_type) is list
        item_type: typing.Any = value_type
        items: list[typing.Any] = [value]
        if is_list:
            if not isinstance(value, list):
                raise TypeError(f"{xml_file_path}: `{'/'.join(field_names)}` is {value!r}, but its field is `{field_type}`")
            item_type = typing.get_args(value_type)[0]
            items = value

        item_values: list[typing.Any] = []
        for item in items:
            # A nil element, e.g. `<pulseNo xsi:nil="true"/>`, is decoded as `{"@xsi:nil": "true"}`
            if isinstance(item, dict) and item.get("@xsi:nil") in ("true", "1"):
                item_values.append(None)
                continue

            # A matrix, e.g. `<regularisations>` holding one `<row>` per row, whose field is `npt.NDArray[np.float64]`.
            # The origin of that type is `npt.NDArray` where numpy defines it as a type alias, as numpy 2.5 does, and `np.ndarray` otherwise
            if typing.get_origin(item_type) in (npt.NDArray, np.ndarray):
                item_values.append(_decoded_matrix_to_ndarray(element_data=item, element_names=field_names, xml_file_path=xml_file_path))
                continue

            # An element holding child elements or attributes, or a link to an XML file holding them
            if dataclasses.is_dataclass(item_type):
                linked_xsd_relative_path: object = getattr(item_type, "XSD_PATH", None)
                if isinstance(item, dict) and "@href" in item and isinstance(linked_xsd_relative_path, str):
                    # A link, e.g. `<grid href="grid.xml"/>`, which is relative to the XML file holding it, so it can point into another directory, e.g. "../default/grid.xml".
                    # The dataclasses hold no file names, so the linked file is recorded by where its settings are, e.g. `("ids", "equilibrium", "grid")`
                    linked_xml_file_path: Path = (xml_file_path.parent / item["@href"]).resolve()
                    link_location: tuple[str, ...] = (*file_location, *field_names)
                    if link_location in xml_file_paths:
                        # Only a link inside a repeated element is found twice in the same place, and could not be told apart from the others
                        raise NotImplementedError(f"{xml_file_path}: `{'/'.join(field_names)}` is a link inside a repeated element")
                    xml_file_paths[link_location] = linked_xml_file_path

                    # A file linked again, e.g. `workflow.xml`, which every IDS's file links to: this link holds the settings already read.
                    # The files that file links to are recorded for this link too, as they were for the link it was first read through
                    if (linked_xml_file_path, item_type) in linked_files_read:
                        linked_file_read: tuple[object, tuple[str, ...]] = linked_files_read[(linked_xml_file_path, item_type)]
                        first_link_location: tuple[str, ...] = linked_file_read[1]
                        n_first_link_location: int = len(first_link_location)
                        for location, location_xml_file_path in list(xml_file_paths.items()):
                            if len(location) > n_first_link_location and location[:n_first_link_location] == first_link_location:
                                xml_file_paths[(*link_location, *location[n_first_link_location:])] = location_xml_file_path
                        item_values.append(linked_file_read[0])
                        continue

                    # The linked file is checked against the XML schema of its root element's dataclass, e.g. `Grid.XSD_PATH = "equilibrium/grid.xsd"`
                    linked_xsd_file_path: Path = xsd_directory_path / linked_xsd_relative_path
                    linked_xml_schema: xmlschema.XMLSchema = xmlschema.XMLSchema(linked_xsd_file_path)
                    try:
                        linked_root_element_data: dict[str, typing.Any] | None = typing.cast(
                            dict[str, typing.Any] | None, linked_xml_schema.to_dict(linked_xml_file_path, keep_empty=True)
                        )
                    except xmlschema.XMLSchemaValidationError as error:
                        error.add_note(f"In {linked_xml_file_path}, checked against {linked_xsd_file_path}")  # the error does not say which file
                        raise
                    linked_settings: object = _decoded_xml_to_dataclass(
                        dataclass_type=item_type,
                        element_data=linked_root_element_data,
                        element_names=[],
                        xml_file_path=linked_xml_file_path,
                        xsd_directory_path=xsd_directory_path,
                        file_location=link_location,
                        xml_file_paths=xml_file_paths,
                        linked_files_read=linked_files_read,
                    )
                    linked_files_read[(linked_xml_file_path, item_type)] = (linked_settings, link_location)
                    item_values.append(linked_settings)
                else:
                    item_values.append(
                        _decoded_xml_to_dataclass(
                            dataclass_type=item_type,
                            element_data=item,
                            element_names=field_names,
                            xml_file_path=xml_file_path,
                            xsd_directory_path=xsd_directory_path,
                            file_location=file_location,
                            xml_file_paths=xml_file_paths,
                            linked_files_read=linked_files_read,
                        )
                    )
                continue

            # Plain values: a single value, or a space-separated list of them, e.g. "0.1 0.2 0.3" -> `list[float]`
            plain_type: typing.Any = item_type
            plain_values: list[typing.Any] = [item]
            if typing.get_origin(item_type) is list:
                plain_type = typing.get_args(item_type)[0]
                plain_values = item
            for plain_value in plain_values:
                if typing.get_origin(plain_type) is typing.Literal:
                    # An enumeration, e.g. a "method"
                    if plain_value not in typing.get_args(plain_type):
                        raise TypeError(f"{xml_file_path}: `{'/'.join(field_names)}` is {plain_value!r}, which is not one of {typing.get_args(plain_type)}")
                elif type(plain_value) is not plain_type:
                    # `bool`, `float`, `int` or `str`, which `xmlschema` has already decoded, so a different type means the dataclass and the XML schema disagree
                    raise TypeError(f"{xml_file_path}: `{'/'.join(field_names)}` is {plain_value!r}, which is not a `{plain_type.__name__}`")
            item_values.append(item)

        if is_list:
            field_values[field_name] = item_values
        else:
            field_values[field_name] = item_values[0]

    # Raises `TypeError` if a field with no default is not given
    return dataclass_type(**field_values)


def _decoded_matrix_to_ndarray(element_data: dict[str, typing.Any] | None, element_names: list[str], xml_file_path: Path) -> npt.NDArray[np.float64]:
    """
    Pass a matrix, as decoded by `xmlschema`, into a 2D numpy array of shape [n_rows, n_columns]; see `matrix_row_element`.

    :param element_data: The matrix, as decoded by `xmlschema`, e.g. `{"row": [[1.0, 0.0], [0.0, 1.0]]}`, or None for no rows, e.g. `<regularisations/>`
    :param element_names: Path of the matrix below the root element of its XML file, e.g. `["passive", "regularisations"]`, for the messages
    :param xml_file_path: Path to the XML file holding the matrix, for the messages
    :return: The matrix. Shape (0, 0) when there are no rows, as the XML then gives no number of columns
    """

    element_path: str = "/".join(element_names)
    rows: list[typing.Any] = []
    if element_data is not None:
        if not isinstance(element_data, dict) or list(element_data) != ["row"] or not isinstance(element_data["row"], list):
            raise TypeError(f"{xml_file_path}: `{element_path}` is {element_data!r}, but its field is a matrix, which holds only `<row>` elements")
        rows = element_data["row"]

    n_rows: int = len(rows)
    n_columns: int = len(rows[0]) if n_rows > 0 else 0
    for i_row in range(n_rows):
        # An empty `<row></row>` is a row with no values, rather than no row
        if len(rows[i_row]) == 0:
            raise ValueError(
                f"{xml_file_path}: `{element_path}`: `<row>` number {i_row + 1} is empty. "
                f"A matrix with no rows has no `<row>` elements, e.g. `<{element_names[-1]}/>`"
            )
        # A jagged matrix, whose rows have different numbers of values
        if len(rows[i_row]) != n_columns:
            raise ValueError(
                f"{xml_file_path}: `{element_path}` is jagged: the first `<row>` has {n_columns} values, "
                f"but `<row>` number {i_row + 1} has {len(rows[i_row])}. Every row must have the same number of values"
            )
        for value in rows[i_row]:
            # `xmlschema` has already decoded each value, so a different type means the dataclass and the XML schema disagree
            if type(value) is not float:
                raise TypeError(f"{xml_file_path}: `{element_path}`: `<row>` number {i_row + 1} holds {value!r}, which is not a `float`")

    matrix: npt.NDArray[np.float64] = np.array(rows, dtype=np.float64).reshape(n_rows, n_columns)
    return matrix


def matrix_row_element(xsd_type: XsdComplexType | XsdSimpleType) -> XsdElement | None:
    """
    The `<row>` element of a matrix's XML schema type, or None for any other type.

    A matrix is an element whose only child is a repeated `<row>`, each a space-separated list of numbers, e.g.
        <regularisations>
            <row>1.0 0.0</row>
            <row>0.0 1.0</row>
        </regularisations>
    Its field is a 2D numpy array, `npt.NDArray[np.float64]`, of shape [n_rows, n_columns], which `read_xml_settings_to_dataclasses` reads it into:
    * no `<row>` is shape (0, 0), as the XML then gives no number of columns, e.g. `<regularisations/>`
    * an empty `<row></row>`, or rows with different numbers of values, are an error
    The generator of the dataclasses, and `serialise_python_dataclass_to_imas_parameters_input`, use this to tell a matrix from any other element.

    :param xsd_type: Type of an element, as read by `xmlschema`
    :return: The `<row>` element, or None if the type is not a matrix's
    """

    if not isinstance(xsd_type, XsdComplexType) or not isinstance(xsd_type.content, XsdGroup):
        return None
    particles: list[typing.Any] = list(xsd_type.content)
    if len(particles) != 1 or not isinstance(particles[0], XsdElement) or particles[0].local_name != "row":
        return None

    # `<row>` is kept for the rows of a matrix, so any other use of it is a mistake in the XML schema
    row_element: XsdElement = particles[0]
    row_type: XsdComplexType | XsdSimpleType = row_element.type
    row_item_type: XsdSimpleType | None = row_type.item_type if isinstance(row_type, XsdList) else None
    if (
        row_element.max_occurs == 1
        or len(xsd_type.attributes) > 0
        or not isinstance(row_item_type, XsdAtomic)
        or row_item_type.primitive_type.local_name != "double"
    ):
        raise ValueError(
            f"{row_element.schema.url}: `<row>` is kept for the rows of a matrix, so must be repeated, and a list of `xs:double`, in a type with no attributes"
        )
    return row_element
