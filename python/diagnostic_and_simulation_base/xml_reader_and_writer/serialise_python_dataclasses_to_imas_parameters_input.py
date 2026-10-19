"""
Serialise XML settings, held in dataclasses, to an IMAS `parameters_input`: the settings as XML, and the XML schema they follow.

This is the reverse of `read_to_python_dataclasses`, and is how the settings are recorded with what they made:
IMAS keeps the XML in the `code/parameters` of the IDS the settings made, e.g. `wall/code/parameters`.
The XML is written from the dataclass, rather than copied from the XML file it was read from, so that a setting changed in Python is recorded too.
A link to another XML file, e.g. `<workflow href="../workflow.xml"/>`, is written as the linked file, in the link's place, so that the record holds
every setting, and needs no other file. The XML schema recorded has the linked file's XML schema merged in to match; see `_merge_linked_xml_schemas`.

This module knows nothing about any one code's settings: the XML schema says how each field is written.
The `ParametersInput` made is `imas_rs`'s, from `gsfit_rs.dd_support`, which is what the Rust code takes.
"""

import copy
import dataclasses
import typing
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import xmlschema
from gsfit_rs.dd_support import ParametersInput
from xmlschema.validators import XsdAttribute
from xmlschema.validators import XsdComplexType
from xmlschema.validators import XsdElement
from xmlschema.validators import XsdGroup
from xmlschema.validators import XsdSimpleType

from .read_to_python_dataclasses import matrix_row_element

XS_NAMESPACE: str = "http://www.w3.org/2001/XMLSchema"
XS: str = "{" + XS_NAMESPACE + "}"
XSI_NAMESPACE: str = "http://www.w3.org/2001/XMLSchema-instance"

# So that a nil element is written `<pulseNo xsi:nil="true"/>`, and a merged XML schema `<xs:element>`, rather than with made-up prefixes
ElementTree.register_namespace("xs", XS_NAMESPACE)
ElementTree.register_namespace("xsi", XSI_NAMESPACE)


def serialise_python_dataclass_to_imas_parameters_input(settings: object, xsd_directory_path: Path) -> ParametersInput:
    """
    Serialise the dataclass of an XML file's root element to an IMAS `parameters_input`.

    The dataclass must be that of an XML file's root element, which names its XML schema in `XSD_PATH`, e.g. `WallSettings`.
    So the XML written is a whole XML file, which can be checked against that XML schema.
    A link to another XML file, whose field holds the dataclass of the linked file's root element, e.g. `workflow: Workflow`, is written as the linked file,
    in the link's place, e.g. `<workflow>` holding the codes of `workflow.xml`. So the XML schema it is checked against, and recorded with, is that XML schema
    with the linked file's XML schema merged in; see `_merge_linked_xml_schemas`.
    To record only the settings which were used, pass a copy holding only those, e.g. the wall settings of a database reader which reads only the ELMAG run:
        serialise_python_dataclass_to_imas_parameters_input(
            settings=WallSettings(workflow=Workflow(tree=KeyedList([wall_settings.workflow.tree["ELMAG"]]))),
            xsd_directory_path=XSD_DIRECTORY_PATH,
        )

    :param settings: Dataclass of an XML file's root element, e.g. `WallSettings`, as made by `read_xml_settings_to_dataclasses`
    :param xsd_directory_path: Directory holding the XML schemas, which `XSD_PATH` is relative to, e.g. GSFit's `settings_schema/xsd`
    :return: The `parameters_input`:
        * `parameters_value`: the settings, as XML, with each linked file in its link's place, checked against their XML schema
        * `parameters_default`: "", the IMAS value of an unset string, as there are no default settings
        * `schema`: the XML schema, as text, with the XML schema of each linked file merged in
    """

    xsd_relative_path: object = getattr(type(settings), "XSD_PATH", None)
    if not dataclasses.is_dataclass(settings) or isinstance(settings, type) or not isinstance(xsd_relative_path, str):
        raise TypeError(f"`{type(settings).__name__}` is not the dataclass of an XML file's root element, which has an `XSD_PATH`, e.g. `WallSettings`")
    xsd_file_path: Path = xsd_directory_path / xsd_relative_path
    xsd_text: str = _merge_linked_xml_schemas(xsd_file_path)
    xml_schema: xmlschema.XMLSchema = xmlschema.XMLSchema(xsd_text)
    xsd_root_elements: list[XsdElement] = list(xml_schema.elements.values())
    if len(xsd_root_elements) != 1:
        raise NotImplementedError(f"{xsd_file_path}: {len(xsd_root_elements)} root elements, rather than one")

    root_element: ElementTree.Element = dataclass_to_element(
        settings=settings,
        xsd_element=xsd_root_elements[0],
        element_path=f"/{xsd_root_elements[0].local_name}",
    )

    # Check the XML against its XML schema, as `read_xml_settings_to_dataclasses` does, so that what is recorded can always be read back.
    # Raises `xmlschema.XMLSchemaValidationError` if it does not match, e.g. if a required element is missing
    try:
        xml_schema.validate(root_element)
    except xmlschema.XMLSchemaValidationError as error:
        # The error does not say which file
        error.add_note(
            f"In the XML written from `{type(settings).__name__}`, checked against {xsd_file_path}, with the XML schema of each file it links to merged in"
        )
        raise

    ElementTree.indent(root_element, space="    ")
    parameters_value: str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ElementTree.tostring(root_element, encoding="unicode") + "\n"

    return ParametersInput(
        parameters_value=parameters_value,
        parameters_default="",
        schema=xsd_text,
    )


def _merge_linked_xml_schemas(xsd_file_path: Path) -> str:
    """
    An XML schema, with the XML schema of each file it links to merged in, in place of the link.

    A link is an element which names the XML schema of the file it links to in `<xs:appinfo><schema>`, relative to the XML schema holding the link,
    e.g. `<xs:element name="workflow" type="Link">` holding `<schema>../workflow.xsd</schema>`, in `pf_active/pf_active.xsd`.
    It is merged by giving it the type of the linked XML schema's root element in place of `Link`, e.g. `<workflow>` holding the codes,
    and adding the linked XML schema's named types to this one's, e.g. `WorkflowTree`. The link keeps its own name and documentation.
    Each linked XML schema has its own links merged in first, so the links are merged however deep they go.
    So the result is the XML schema of the XML file with each linked file written in its link's place, as `serialise_python_dataclass_to_imas_parameters_input`
    writes the settings.

    A named type which both XML schemas have, as a type used by several XML schemas is repeated in each, is kept once.
    So it must be the same in each, otherwise there would be two types of one name, and a `ValueError` is raised.

    :param xsd_file_path: Path to the XML schema, e.g. `settings_schema/xsd/pf_active/pf_active.xsd`
    :return: The merged XML schema, as text. An XML schema without links is returned as it is.
        A merged XML schema is laid out afresh, as its elements have moved, and has a second header comment saying what was merged
    """

    # The comments inside the root element are kept, e.g. the section headings
    schema_root: ElementTree.Element = ElementTree.parse(
        xsd_file_path, parser=ElementTree.XMLParser(target=ElementTree.TreeBuilder(insert_comments=True))
    ).getroot()

    # The links, all found before any is merged, as merging adds elements
    link_elements: list[ElementTree.Element] = []
    for element in schema_root.iter(f"{XS}element"):
        if element.find(f"{XS}annotation/{XS}appinfo/schema") is not None:
            link_elements.append(element)
    if len(link_elements) == 0:
        return xsd_file_path.read_text(encoding="utf-8")

    merged_links: list[str] = []
    for link_element in link_elements:
        # The linked XML schema, with its own links merged in
        annotation_element: ElementTree.Element = typing.cast(ElementTree.Element, link_element.find(f"{XS}annotation"))
        appinfo_element: ElementTree.Element = typing.cast(ElementTree.Element, annotation_element.find(f"{XS}appinfo"))
        linked_schema_element: ElementTree.Element = typing.cast(ElementTree.Element, appinfo_element.find("schema"))
        linked_xsd_relative_path: str = "".join(linked_schema_element.itertext()).strip()
        linked_xsd_file_path: Path = xsd_file_path.parent / linked_xsd_relative_path
        linked_schema_root: ElementTree.Element = ElementTree.fromstring(
            _merge_linked_xml_schemas(linked_xsd_file_path),
            parser=ElementTree.XMLParser(target=ElementTree.TreeBuilder(insert_comments=True)),
        )
        linked_root_elements: list[ElementTree.Element] = linked_schema_root.findall(f"{XS}element")
        if len(linked_root_elements) != 1:
            raise NotImplementedError(f"{linked_xsd_file_path}: {len(linked_root_elements)} root elements, rather than one")
        linked_root_element: ElementTree.Element = linked_root_elements[0]

        # The link takes the type of the linked root element, in place of `Link`, and its identity constraints, e.g. `xs:unique`.
        # It is no longer a link, so no longer names the linked XML schema
        link_element.attrib.pop("type", None)
        linked_root_type_name: str | None = linked_root_element.get("type")
        if linked_root_type_name is not None:
            link_element.set("type", linked_root_type_name)
        for linked_root_child in linked_root_element:
            if linked_root_child.tag != f"{XS}annotation":
                link_element.append(copy.deepcopy(linked_root_child))
        appinfo_element.remove(linked_schema_element)
        if len(appinfo_element) == 0:
            annotation_element.remove(appinfo_element)

        # The linked XML schema's named types join this one's, after its own, e.g. `WorkflowTree`.
        # Complex and simple types share one set of names, so a type is looked for by its name alone
        n_components: int = len(schema_root)
        i_insert: int = 0
        for i_component in range(n_components):
            if schema_root[i_component].tag in (f"{XS}complexType", f"{XS}simpleType"):
                i_insert = i_component + 1
        for linked_component in linked_schema_root:
            linked_type_name: str | None = linked_component.get("name")
            if linked_component.tag not in (f"{XS}complexType", f"{XS}simpleType") or linked_type_name is None:
                continue
            existing_component: ElementTree.Element | None = None
            for component in schema_root:
                if component.tag in (f"{XS}complexType", f"{XS}simpleType") and component.get("name") == linked_type_name:
                    existing_component = component
            if existing_component is None:
                schema_root.insert(i_insert, copy.deepcopy(linked_component))
                i_insert += 1
            elif ElementTree.canonicalize(ElementTree.tostring(existing_component, encoding="unicode"), strip_text=True) != ElementTree.canonicalize(
                ElementTree.tostring(linked_component, encoding="unicode"), strip_text=True
            ):
                raise ValueError(
                    f"{xsd_file_path} and {linked_xsd_file_path} both have a type named `{linked_type_name}`, which are not the same, so the XML schemas "
                    "cannot be merged. A type used by several XML schemas must be the same in each"
                )

        merged_links.append(f"`{link_element.get('name')}`, from {linked_xsd_relative_path}")

    # `ElementTree` keeps no comment outside the root element, so the header comments, above `<xs:schema>`, are read separately
    header_comments: list[str] = []
    with open(xsd_file_path, "rb") as xsd_file:
        for event, node in ElementTree.iterparse(xsd_file, events=("start", "comment")):
            if event == "start":
                break
            header_comments.append(node.text if node.text is not None else "")

    ElementTree.indent(schema_root, space="    ")
    xsd_text: str = '<?xml version="1.0" encoding="UTF-8"?>\n'
    for header_comment in header_comments:
        xsd_text += f"<!--{header_comment}-->\n"
    xsd_text += (
        "<!--\n"
        f"    {xsd_file_path.name}, with the XML schema of each file it links to merged in, in place of the link: {'; '.join(merged_links)}.\n"
        "    So it is the XML schema of the settings with each linked file written in its link's place, as they are recorded.\n"
        "-->\n"
    )
    xsd_text += ElementTree.tostring(schema_root, encoding="unicode") + "\n"
    return xsd_text


def dataclass_to_element(
    settings: object,
    xsd_element: XsdElement,
    element_path: str,
    write_linked_file: typing.Callable[[object, str], str] | None = None,
) -> ElementTree.Element:
    """
    Write a dataclass, and everything inside it, as an XML element, in the order and the format its XML schema gives.

    Each child element, or attribute, is written from the field with the same name:
    * `None` is written as a nil element, e.g. `<pulseNo xsi:nil="true"/>`, if the element is nillable, and is left out if the element is optional
    * a list is written as one element per item, for a repeated element, e.g. one `<flux_loop>` per flux loop
    * a dataclass is written as an element holding its own fields
    * a link to another XML file, whose field holds the dataclass of the linked file's root element, which has an `XSD_PATH`, e.g. `grid: Grid`, is written:
      * without `write_linked_file`, as the linked file, in the link's place, as `serialise_python_dataclass_to_imas_parameters_input` writes it.
        `xsd_element` is then from the XML schema with the linked file's XML schema merged in; see `_merge_linked_xml_schemas`
      * with `write_linked_file`, as the link, e.g. `<grid href="grid.xml"/>`, once `write_linked_file` has written the linked file,
        as `make_settings_from_xml_xsd` writes it. `xsd_element` is then from the XML schema of this XML file alone
    * a collapsed link, to a file whose root element holds only one element, named as the link, whose field holds that element, e.g.
      `b_field_pol_probe: KeyedList[BFieldPolProbe]`, is put back into the dataclass of the linked file's root element, which its field names in its
      `linked_file_root` metadata, e.g. `ConstraintsBFieldPolProbe`, and then written as any other link
    * a 2D numpy array is written as a matrix, one `<row>` per row; see `matrix_row_element`
    * any other value is written as text, by the XML schema type of its element, e.g. `False` -> "false" or `[0.1, 0.2]` -> "0.1 0.2",
      which also checks it against that type, e.g. that a `FinitePositiveDouble` is greater than 0

    :param settings: Dataclass of the element, e.g. `WorkflowTree`
    :param xsd_element: The element, as declared in the XML schema
    :param element_path: Path of the element, e.g. "/workflow/tree", for the messages
    :param write_linked_file: Writes a linked file, given the dataclass of its root element and the path of the link's element, e.g. "/equilibrium_settings/grid",
        and returns the linked file's path relative to the XML file being written, for the link's `href`, e.g. "grid.xml".
        None to write each linked file in its link's place
    :return: The element
    """

    if not dataclasses.is_dataclass(settings) or isinstance(settings, type):
        raise TypeError(f"{element_path} holds child elements or attributes, but its field is {settings!r}, not a dataclass")
    xsd_type: XsdComplexType | XsdSimpleType = xsd_element.type
    if xsd_element.local_name is None or not isinstance(xsd_type, XsdComplexType) or not isinstance(xsd_type.content, XsdGroup):
        raise NotImplementedError(f"{element_path}: not a named element holding child elements or attributes")
    element: ElementTree.Element = ElementTree.Element(xsd_element.local_name)
    names_written: list[str] = []

    # The dataclass of the linked file's root element, for each collapsed link, by the name of its field, e.g. "b_field_pol_probe" -> `ConstraintsBFieldPolProbe`
    linked_file_roots: dict[str, typing.Any] = {}
    for dataclass_field in dataclasses.fields(settings):
        if "linked_file_root" in dataclass_field.metadata:
            linked_file_roots[dataclass_field.name] = dataclass_field.metadata["linked_file_root"]

    # The child elements, in the order the XML schema gives
    for particle in xsd_type.content:
        if not isinstance(particle, XsdElement) or particle.local_name is None:
            raise NotImplementedError(f"{element_path}: a group inside a group")
        child_name: str = particle.local_name
        child_path: str = f"{element_path}/{child_name}"
        value: typing.Any = getattr(settings, child_name)
        names_written.append(child_name)

        # A collapsed link, whose field holds the one element of the linked file's root element, e.g. `b_field_pol_probe: KeyedList[BFieldPolProbe]`,
        # is put back into the dataclass of that root element, e.g. `ConstraintsBFieldPolProbe`, which is written as any other link, below
        if child_name in linked_file_roots:
            value = linked_file_roots[child_name](**{child_name: value})

        # `max_occurs` is None for `maxOccurs="unbounded"`
        items: list[typing.Any] = [value]
        if particle.max_occurs != 1:
            if not isinstance(value, list):
                raise TypeError(f"{child_path} is a repeated element, but its field is {value!r}, not a list")
            items = value

        for item in items:
            if item is None:
                if particle.nillable:
                    ElementTree.SubElement(element, child_name, {f"{{{XSI_NAMESPACE}}}nil": "true"})
                elif particle.min_occurs > 0:
                    raise ValueError(f"{child_path} is required, but its field is None")
                continue  # an optional element, which is left out

            # A matrix, written as one `<row>` per row, as `read_xml_settings_to_dataclasses` reads it
            if isinstance(item, np.ndarray):
                row_element: XsdElement | None = matrix_row_element(particle.type)
                if row_element is None:
                    raise TypeError(f"{child_path} is not a matrix in the XML schema, but its field is a numpy array")
                if item.ndim != 2:
                    raise ValueError(f"{child_path} is a matrix, but its field has {item.ndim} dimension(s), rather than 2")
                n_rows: int = item.shape[0]
                n_columns: int = item.shape[1]
                if n_rows > 0 and n_columns == 0:
                    raise ValueError(
                        f"{child_path} has {n_rows} row(s) with no values, which would be written as empty `<row>` elements, which are not allowed"
                    )
                matrix_element: ElementTree.Element = ElementTree.SubElement(element, child_name)
                for i_row in range(n_rows):
                    row_text: typing.Any = row_element.type.encode(item[i_row, :].tolist())
                    if not isinstance(row_text, str):
                        raise TypeError(f"{child_path}: row {i_row} is written as {row_text!r}, not text")
                    ElementTree.SubElement(matrix_element, "row").text = row_text
                continue

            # An element holding child elements or attributes. A link to another XML file, whose dataclass has an `XSD_PATH`, is written as the link,
            # with `write_linked_file`, and otherwise as the linked file, in the link's place
            if dataclasses.is_dataclass(item):
                if write_linked_file is not None and isinstance(getattr(type(item), "XSD_PATH", None), str):
                    ElementTree.SubElement(element, child_name, {"href": write_linked_file(item, child_path)})
                else:
                    element.append(dataclass_to_element(settings=item, xsd_element=particle, element_path=child_path, write_linked_file=write_linked_file))
                continue

            if not isinstance(particle.type, XsdSimpleType):
                raise TypeError(f"{child_path} holds child elements, but its field is {item!r}, not a dataclass")
            text: typing.Any = particle.type.encode(item)
            if not isinstance(text, str):
                raise TypeError(f"{child_path}: {item!r} is written as {text!r}, not text")
            ElementTree.SubElement(element, child_name).text = text

    # The attributes, e.g. `href`
    for attribute_name, xsd_attribute in xsd_type.attributes.items():
        if not isinstance(attribute_name, str) or not isinstance(xsd_attribute, XsdAttribute):
            raise NotImplementedError(f"{element_path}: an attribute wildcard")
        value = getattr(settings, attribute_name)
        names_written.append(attribute_name)
        if value is None:
            if xsd_attribute.use == "required":
                raise ValueError(f"{element_path}/@{attribute_name} is required, but its field is None")
            continue
        attribute_text: typing.Any = xsd_attribute.type.encode(value)
        if not isinstance(attribute_text, str):
            raise TypeError(f"{element_path}/@{attribute_name}: {value!r} is written as {attribute_text!r}, not text")
        element.set(attribute_name, attribute_text)

    # Every field must have been written, otherwise the dataclass is not one of this XML schema
    for dataclass_field in dataclasses.fields(settings):
        if dataclass_field.name not in names_written:
            raise TypeError(f"{element_path}: `{type(settings).__name__}.{dataclass_field.name}` is not in the XML schema")

    return element
