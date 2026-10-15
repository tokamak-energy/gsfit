import os
import posixpath
from pathlib import Path
from xml.etree import ElementTree

import xmlschema
from xmlschema.validators import XsdElement

from ..xml_reader_and_writer.serialise_python_dataclasses_to_imas_parameters_input import XSI_NAMESPACE
from ..xml_reader_and_writer.serialise_python_dataclasses_to_imas_parameters_input import dataclass_to_element
from .make_settings_json import make_settings_json


def make_settings_from_xml_xsd(
    settings_xml: object,
    settings_xsd: dict[str, str],
    settings_xml_file_paths: dict[tuple[str, ...], Path],
) -> str:
    """
    Make the XML settings, and the XML schemas they follow, into one JSON string, ready to be written to MDSplus, as `make_settings_json` does for the JSON settings.
    The string holds one entry per file, its path -> its contents:
    ```
    {
      "settings/st40_P2p3_calibrations_new_xml_schema/GSFIT_code_settings.xml": "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>\\n<gsfit_code_settings ...",
      "settings/st40_P2p3_calibrations_new_xml_schema/equilibrium/equilibrium.xml": "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>\\n<equilibrium_settings ...",
      ...
      "settings_schema/xsd/GSFIT_code_settings.xsd": "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>\\n<!--\\n    Schema for GSFIT_code_settings.xml ...",
      ...
    }
    ```

    The XML is written from `settings_xml`, rather than copied from the files it was read from, so that a setting changed in Python is recorded too.
    Each XML file is written on its own, with its links, e.g. `<grid href="grid.xml"/>`, as it was read: unlike the IDSs' `code/parameters`, a linked file is not
    written in its link's place. A file linked from several places, e.g. `workflow.xml`, which every IDS's file links to, is one file, so is written once.

    The files are named by relative paths, laid out as the code's package is, so that the relative paths inside them still lead to each other:
    * an XML file by its path relative to the directory holding the settings directory, below `settings/`, e.g.
      `settings/st40_P2p3_calibrations_new_xml_schema/wall/wall.xml`, or `settings/default/wall/wall.xml` for `href="../default/wall/wall.xml"`
    * an XML schema by its `XSD_PATH`, below `settings_schema/xsd/`, e.g. `settings_schema/xsd/wall/wall.xsd`. Only those of the XML files written are kept
    * each XML file's links, and the path of its XML schema in `xsi:noNamespaceSchemaLocation`, are relative to it, as in a settings directory, e.g.
      `xsi:noNamespaceSchemaLocation="../../../settings_schema/xsd/wall/wall.xsd"` in `settings/st40_P2p3_calibrations_new_xml_schema/wall/wall.xml`
    So the files can be written out under their paths, and the settings read back from the root XML file,
    e.g. `settings/st40_P2p3_calibrations_new_xml_schema/GSFIT_code_settings.xml`.

    Each XML file is checked against its XML schema before it is kept, so that what is recorded can always be read back.

    :param settings_xml: The settings, e.g. `GsfitCodeSettings`, as `read_xml_settings_to_dataclasses` read them, and then as changed in Python
    :param settings_xsd: The XML schemas, as text, by their path relative to the directory holding them, e.g. "wall/wall.xsd", as `XSD_PATH` gives it
    :param settings_xml_file_paths: The path of each XML file read, by where its settings are in `settings_xml`, as `read_xml_settings_to_dataclasses` returns it,
        which names the files. A linked file added in Python, which was never read, e.g. `rtgsfit_code_settings`, is named as a settings directory names it:
        its `XSD_PATH`, with ".xml" for ".xsd", in the directory of the root XML file, e.g. `rtgsfit_code_settings.xml`
    :return: The files, as a JSON string, sorted by path
    :raises ValueError: if a file linked from several places holds different settings through different links, e.g. after one of the links to `workflow.xml`
        was replaced in Python. Changing a setting through one of the links changes it for all of them, as they hold the same dataclass
    :raises xmlschema.XMLSchemaValidationError: if an XML file does not match its XML schema, e.g. after a setting was given a value of the wrong type in Python
    """

    root_xml_file_path: Path = settings_xml_file_paths[()]
    # e.g. `python/gsfit/settings/`, holding the settings directory `st40_P2p3_calibrations_new_xml_schema/`
    settings_directories_path: Path = root_xml_file_path.parent.parent

    files: dict[str, str] = {}  # path -> contents
    xml_file_locations: dict[str, tuple[str, ...]] = {}  # path of each XML file kept -> where its settings were first found in `settings_xml`, for the messages
    xml_schemas: dict[str, xmlschema.XMLSchema] = {}  # `XSD_PATH` -> XML schema

    def write_xml_file(settings: object, location: tuple[str, ...]) -> str:
        """
        Write the XML file holding `settings`, after the files it links to, into `files`, with its XML schema.

        :param settings: The dataclass of the XML file's root element, e.g. `WallSettings`
        :param location: Where `settings` are in `settings_xml`, e.g. `("ids", "wall")`, or `()` for `settings_xml` itself
        :return: The XML file's path, e.g. "settings/st40_P2p3_calibrations_new_xml_schema/wall/wall.xml"
        """

        xsd_relative_path: object = getattr(type(settings), "XSD_PATH", None)
        if not isinstance(xsd_relative_path, str):
            raise TypeError(f"`{'/'.join(location)}` is a link to another XML file, but holds `{type(settings).__name__}`, which has no `XSD_PATH`")
        xsd_path: str = f"settings_schema/xsd/{xsd_relative_path}"

        # The XML file's path, below `settings/`, relative to the directory holding the settings directory
        xml_file_path: Path | None = settings_xml_file_paths.get(location)
        if xml_file_path is None:
            xml_file_path = root_xml_file_path.parent / Path(xsd_relative_path).with_suffix(".xml")
        xml_path: str = posixpath.normpath(posixpath.join("settings", Path(os.path.relpath(xml_file_path, settings_directories_path)).as_posix()))
        if xml_path.startswith("../"):
            raise ValueError(
                f"{xml_file_path}, linked from `{'/'.join(location)}`, is outside {settings_directories_path.parent}, so cannot be named relative to it"
            )

        if xsd_relative_path not in xml_schemas:
            xml_schemas[xsd_relative_path] = xmlschema.XMLSchema(settings_xsd[xsd_relative_path])
        xml_schema: xmlschema.XMLSchema = xml_schemas[xsd_relative_path]
        xsd_root_elements: list[XsdElement] = list(xml_schema.elements.values())
        if len(xsd_root_elements) != 1:
            raise NotImplementedError(f"{xsd_path}: {len(xsd_root_elements)} root elements, rather than one")

        def write_linked_file(linked_settings: object, link_element_path: str) -> str:
            """
            Write a file this XML file links to, and give its path relative to this XML file, for the link's `href`.

            :param linked_settings: The dataclass of the linked file's root element, e.g. `Grid`
            :param link_element_path: Path of the link's element, from this XML file's root element, e.g. "/equilibrium_settings/grid"
            :return: e.g. "grid.xml", or "../workflow.xml"
            """
            linked_location: tuple[str, ...] = (*location, *link_element_path.split("/")[2:])
            linked_xml_path: str = write_xml_file(settings=linked_settings, location=linked_location)
            return posixpath.relpath(linked_xml_path, posixpath.dirname(xml_path))

        root_element: ElementTree.Element = dataclass_to_element(
            settings=settings,
            xsd_element=xsd_root_elements[0],
            element_path=f"/{xsd_root_elements[0].local_name}",
            write_linked_file=write_linked_file,
        )
        try:
            xml_schema.validate(root_element)
        except xmlschema.XMLSchemaValidationError as error:
            # The error does not say which file
            error.add_note(f"In {xml_path}, written from `{type(settings).__name__}`, at `{'/'.join(location)}`, checked against {xsd_path}")
            raise

        # Where the XML schema is, relative to the XML file, for editors, as in a settings directory; the code reads it from `XSD_PATH`
        root_element.set(f"{{{XSI_NAMESPACE}}}noNamespaceSchemaLocation", posixpath.relpath(xsd_path, posixpath.dirname(xml_path)))
        ElementTree.indent(root_element, space="    ")
        xml_text: str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ElementTree.tostring(root_element, encoding="unicode") + "\n"

        # A file linked from several places is one file, with one set of settings
        if xml_path in files and files[xml_path] != xml_text:
            raise ValueError(
                f"{xml_path} is linked from `{'/'.join(xml_file_locations[xml_path])}` and from `{'/'.join(location)}`, which hold different settings. "
                "It is one file, so must hold the same settings through every link"
            )
        files[xml_path] = xml_text
        xml_file_locations.setdefault(xml_path, location)
        files[xsd_path] = settings_xsd[xsd_relative_path]

        return xml_path

    write_xml_file(settings=settings_xml, location=())

    return make_settings_json(data=files)
