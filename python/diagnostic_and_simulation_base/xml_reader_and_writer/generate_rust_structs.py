"""
Generate Rust structs from an XML schema, and the code reading XML into them, so that the settings Rust is passed as XML are read by name and with their types.

This module knows nothing about any one code's settings.
Each code calls `generate_rust_structs` from a script of its own, giving the XML schema, the Rust file to write, and the names to use, e.g. GSFit's
`python/gsfit/settings_schema/_generate_rust_schema.py`, which generates `rust/gsfit_rs/src/settings/settings_tf.rs` from `python/gsfit/settings_schema/xsd/tf/tf.xsd`.
The examples below are GSFit's.
The file is committed and is not generated at build time, so `cargo build` needs no Python: run the script after changing the XML schema.

The Rust code is passed the settings as one XML document, as `serialise_python_dataclass_to_imas_parameters_input` writes it: the XML file, with each file it
links to written in its link's place, e.g. `<tf_settings><vacuum_toroidal_field_reference_radius>0.4</vacuum_toroidal_field_reference_radius><workflow><tree>...`.
So a link, an element naming the XML schema of the file it links to in `<xs:appinfo><schema>`, e.g. `<workflow href="../workflow.xml"/>`, is followed into the
root element of that XML schema, as if the link held it.
A link to a file whose root element holds only one element, named as the link, is collapsed into that element, as GSFit's Python dataclasses collapse it: the
link's field is that element's, read from inside the linked file's root element. E.g. `<b_field_pol_probe href="constraints_b_field_pol_probe.xml"/>`, whose
root element holds only the repeated `<b_field_pol_probe>`, is `b_field_pol_probe: Vec<...>`, read from `<b_field_pol_probe><b_field_pol_probe>...`.
A link to an XML schema which several settings files link to, e.g. GSFit's `workflow.xsd`, which every IDS's settings link to, is not followed: the XML schema is
generated into a Rust module of its own, a `SharedModule`, e.g. `rust/gsfit_rs/src/settings/settings_workflow.rs`, by its own call to `generate_rust_structs`.
The link's field is that module's root struct, e.g. `workflow: Option<SettingsWorkflow>`, which, with the function reading it, is imported from there, so that
every settings file linking to it has the same struct.

The file holds:
* a struct for each XML element holding other elements, with a field for each element inside it, named as the element, in the XML schema's order.
  A leaf has the Rust type of its XML schema type: `f64` for `xs:double`, `usize` for `xs:positiveInteger` and `xs:nonNegativeInteger`, `i64` for `xs:integer`,
  `bool` for `xs:boolean` and `String` for `xs:string`, each with the types restricting it, e.g. `FiniteDouble`, and `Array1<f64>` for a space-separated list of numbers.
  A nillable element is an `Option`, `None` when it is nil, and so is an optional element, `minOccurs="0"`, `None` when it is left out.
  A repeated element holding other elements, `minOccurs="0" maxOccurs="unbounded"`, is a `Vec`, empty when it is left out.
  A matrix, an element holding only a repeated `row`, each a space-separated list of numbers, e.g. GSFit's `Regularisations`, is an `Array2<f64>` with a row for
  each `row`, rather than a struct holding a `Vec` of its rows.
  The `<xs:documentation>` becomes the doc comments, with the units in `<xs:appinfo>` appended, e.g. " [metre]"
* `from_xml` on the root element's struct, e.g. `SettingsTf::from_xml`, and a function reading each struct from its XML element, which read the leaves with the
  hand-written functions in the crate's leaf readers, e.g. `rust/gsfit_rs/src/settings/settings_leaf_readers.rs`

The struct of an element of a named XML schema type is named after the type, following `named_type_struct_prefix`, e.g. `SettingsCoreProfilesThomsonScattering` for
`ThomsonScattering`, and is one struct wherever the type is used.
The struct of an element of an anonymous type, including a link, whose linked file's root element has an anonymous type, is named after the struct holding it
and the element, e.g. `SettingsEquilibriumNumerics` for `numerics`, inside `SettingsEquilibrium`.
A field keeps its element's name, as the XML schema is the source of truth, even when that is not snake_case, e.g. `pulseNo`.
An `xs:key` or `xs:unique` only constrains the values, as the XML was checked against its XML schema before it was passed in, so makes no difference to the Rust types.
The generator stops on any XML schema construct it does not know how to translate, rather than guess.
The XML schemas are read with `xmlschema`, which also checks that each is itself a valid XML schema, and the file is formatted with `rustfmt`.
"""

import re
import subprocess
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from xml.etree import ElementTree

import xmlschema
from xmlschema.validators import XsdAtomicBuiltin
from xmlschema.validators import XsdAtomicRestriction
from xmlschema.validators import XsdComplexType
from xmlschema.validators import XsdComponent
from xmlschema.validators import XsdElement
from xmlschema.validators import XsdGroup
from xmlschema.validators import XsdList
from xmlschema.validators import XsdSimpleType

XS_NAMESPACE: str = "http://www.w3.org/2001/XMLSchema"
XS: str = "{" + XS_NAMESPACE + "}"

# XML schema built-in type -> (Rust type, the function in the leaf readers reading it)
BUILT_IN_TYPES: dict[str, tuple[str, str]] = {
    "boolean": ("bool", "read_boolean"),
    "double": ("f64", "read_double"),
    "integer": ("i64", "read_integer"),
    "nonNegativeInteger": ("usize", "read_unsigned_integer"),
    "positiveInteger": ("usize", "read_unsigned_integer"),
    "string": ("String", "read_string"),
}

# XML schema tag -> the attributes this generator understands.
# Any other tag, or attribute, stops the generator, e.g. `xs:choice`, `xs:union`, a `default` value or a `whiteSpace` facet,
# which would change how a string is read
SUPPORTED_XS_ATTRIBUTES: dict[str, set[str]] = {
    "all": set(),
    "annotation": set(),
    "appinfo": set(),
    "attribute": {"name", "type", "use"},  # only on a link, as its `href`
    "complexType": {"name"},
    "documentation": set(),
    "element": {"maxOccurs", "minOccurs", "name", "nillable", "type"},
    "enumeration": {"value"},
    "list": {"itemType"},
    "restriction": {"base"},
    "schema": {"elementFormDefault"},
    "sequence": set(),
    "simpleType": {"name"},
    # Identity constraints only constrain the values, so make no difference to the Rust types
    "field": {"xpath"},
    "key": {"name"},
    "selector": {"xpath"},
    "unique": {"name"},
    # These only constrain the values, so make no difference to the Rust types
    "maxExclusive": {"value"},
    "maxInclusive": {"value"},
    "minExclusive": {"value"},
    "minInclusive": {"value"},
    "pattern": {"value"},
}

# A field's name is its element's, which must be a Rust identifier
RUST_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile("[A-Za-z][A-Za-z0-9_]*")
# A name Rust's `non_snake_case` lint accepts. A field with any other name, e.g. `pulseNo`, is marked as an exception to the lint
SNAKE_CASE_PATTERN: re.Pattern[str] = re.compile("[a-z][a-z0-9]*(_[a-z0-9]+)*")
# A struct's name is made from named types' names and elements' names, in PascalCase
PASCAL_CASE_PATTERN: re.Pattern[str] = re.compile("[A-Z][A-Za-z0-9]*")
# Rust's strict and reserved keywords, which cannot be a field's name
RUST_KEYWORDS: set[str] = {
    "abstract",
    "as",
    "async",
    "await",
    "become",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "do",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "final",
    "fn",
    "for",
    "gen",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "macro",
    "match",
    "mod",
    "move",
    "mut",
    "override",
    "priv",
    "pub",
    "ref",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "try",
    "type",
    "typeof",
    "unsafe",
    "unsized",
    "use",
    "virtual",
    "where",
    "while",
    "yield",
}
# The variables of the generated readers, which a field's name cannot be, as each field is read into a variable of its own name
READER_VARIABLE_NAMES: set[str] = {"child", "child_name", "child_path", "item", "item_name", "item_path", "item_read", "items", "node", "path"}


@dataclass
class SharedModule:
    """
    An XML schema which several settings files link to, generated into a Rust module of its own, whose root struct each settings file linking to it imports.
    """

    xsd_path: Path  # The XML schema, e.g. GSFit's `python/gsfit/settings_schema/xsd/workflow.xsd`
    rust_module: str  # The Rust module it is generated into, e.g. "crate::settings::settings_workflow"
    root_struct_name: str  # The struct of its root element, e.g. "SettingsWorkflow"


@dataclass
class FieldDefinition:
    name: str  # The element's name, e.g. "boundary_r"
    rust_type: str  # The field's type, e.g. "Option<f64>"
    value_type: str  # The type one element is read as, e.g. "f64", or a struct
    reader_name: str  # The function reading one element, e.g. "read_double", or the generated function reading a struct
    is_optional: bool  # `minOccurs="0"`: `None` when the element is left out
    is_nillable: bool  # `nillable="true"`: `None` when the element is nil
    is_repeated: bool  # `maxOccurs="unbounded"`: a `Vec`, with one item for each element
    documentation_lines: list[str]
    # A collapsed link: the field is the one element the linked file's root element holds, read from inside the link's element, which is required
    is_collapsed_link: bool = False


@dataclass
class StructDefinition:
    name: str  # e.g. "SettingsCoreProfilesThomsonScattering"
    documentation: str
    xsd_type_name: str  # The named XML schema type the struct is read from, e.g. "WorkflowTree", or "" for an anonymous type
    xsd_path: Path  # The XML schema declaring the type
    element_paths: list[str]  # Each element the struct is read from, e.g. "equilibrium_settings/numerics/greens"
    field_definitions: list[FieldDefinition] = field(default_factory=list)


class StructBuilder:
    """
    Builds the struct of each XML element holding other elements, from the root element of an XML schema down, following the links.

    A link, an element naming the XML schema of the file it links to in `<xs:appinfo><schema>`, relative to the XML schema holding the link, e.g.
    `<numerics href="numerics.xml"/>`, holds only an `href` attribute. In the XML the Rust code reads, the linked file's root element is written in the link's place,
    and named as the link, so the link's field is read as that root element, e.g. `numerics: SettingsEquilibriumNumerics`.
    A link to the XML schema of a `SharedModule` is not followed, as its structs are in that module: its field is the module's root struct, e.g.
    `workflow: Option<SettingsWorkflow>`.
    """

    def __init__(self, named_type_struct_prefix: str, shared_modules: list[SharedModule]) -> None:
        # What the struct of each named XML schema type is named with, followed by the type's name, e.g. "SettingsCoreProfiles"
        self.named_type_struct_prefix: str = named_type_struct_prefix
        # The XML schemas generated into modules of their own, by their path, resolved
        self.shared_modules_by_xsd_path: dict[Path, SharedModule] = {}
        for shared_module in shared_modules:
            self.shared_modules_by_xsd_path[shared_module.xsd_path.resolve()] = shared_module
        # The shared modules linked to, whose root struct, and the function reading it, are imported
        self.linked_shared_modules: list[SharedModule] = []
        # In the XML's order, the root first, as each struct is added before the structs inside it
        self.struct_definitions: list[StructDefinition] = []
        self.xml_schemas_by_path: dict[Path, xmlschema.XMLSchema] = {}
        # Each named type of the XML schemas read so far, as canonical XML, with the XML schema it was first found in, to check they agree on it
        self.canonical_types_by_name: dict[str, tuple[Path, str]] = {}
        # The structs whose fields are being added, to stop on a type holding an element of its own type
        self.struct_names_in_progress: list[str] = []
        # The functions in the leaf readers reading the leaves, which the generated file imports
        self.leaf_reader_names: set[str] = set()

    def load_xml_schema(self, xsd_path: Path) -> xmlschema.XMLSchema:
        """
        Read an XML schema, once, and check that this generator understands everything in it.

        :param xsd_path: Path to the XML schema, resolved
        :return: The XML schema
        """

        if xsd_path in self.xml_schemas_by_path:
            return self.xml_schemas_by_path[xsd_path]

        # Raises `xmlschema.XMLSchemaParseError` if the XML schema is itself not valid
        xml_schema: xmlschema.XMLSchema = xmlschema.XMLSchema(xsd_path)
        check_supported(xml_schema.root, xsd_path)

        # A type used by several XML schemas is repeated in each, and must be the same in each, as `_merge_linked_xml_schemas` needs to merge them into the
        # XML schema the settings are checked against before they reach the Rust code
        for type_name, xsd_type in xml_schema.types.items():
            canonical_type: str = ElementTree.canonicalize(ElementTree.tostring(xsd_type.elem, encoding="unicode"), strip_text=True)
            if type_name not in self.canonical_types_by_name:
                self.canonical_types_by_name[type_name] = (xsd_path, canonical_type)
            elif self.canonical_types_by_name[type_name][1] != canonical_type:
                raise ValueError(
                    f"{self.canonical_types_by_name[type_name][0]} and {xsd_path} both have a type named `{type_name}`, which are not the same. "
                    "A type used by several XML schemas must be the same in each"
                )

        self.xml_schemas_by_path[xsd_path] = xml_schema
        return xml_schema

    def add_struct(
        self,
        complex_type: XsdComplexType,
        struct_name: str,
        documentation: str,
        xsd_path: Path,
        element_path: str,
        xsd_type_name: str,
        linked_xsd_paths: list[Path],
    ) -> None:
        """
        Add the struct read from an element of `complex_type`, and the structs of the elements inside it.

        :param complex_type: The type of the element, which holds the struct's elements
        :param struct_name: The struct's name
        :param documentation: The struct's documentation
        :param xsd_path: The XML schema declaring `complex_type`, which the types of its elements are looked up in
        :param element_path: Path of the element, e.g. "equilibrium_settings/numerics", for the messages and the documentation
        :param xsd_type_name: The name of `complex_type`, if it is a named type, e.g. "WorkflowTree", which has one struct wherever it is used; otherwise ""
        :param linked_xsd_paths: The XML schemas of the links followed to reach the element, the root's first, to stop on a link back to one of them
        """

        # A named type has one struct, wherever it is used
        for struct_definition in self.struct_definitions:
            if struct_definition.name != struct_name:
                continue
            if xsd_type_name == "" or struct_definition.xsd_type_name != xsd_type_name:
                raise ValueError(
                    f"{xsd_path.name}: `{element_path}` would be the struct `{struct_name}`, "
                    f"which is already the struct of `{struct_definition.element_paths[0]}`, of a different type"
                )
            if struct_name in self.struct_names_in_progress:
                raise NotImplementedError(
                    f"{xsd_path.name}: `{element_path}` is of the type `{xsd_type_name}`, inside an element of the same type, which is not handled"
                )
            struct_definition.element_paths.append(element_path)
            return

        if len(complex_type.attributes) > 0:
            raise NotImplementedError(
                f"{xsd_path.name}: `{element_path}` has attributes, which are handled only on a link, an element naming the XML schema of the file it links to "
                "in `<xs:appinfo><schema>`"
            )
        content: XsdGroup | XsdSimpleType = complex_type.content
        if not isinstance(content, XsdGroup) or complex_type.mixed:
            raise NotImplementedError(f"{xsd_path.name}: `{element_path}` holds text, but only an element holding elements, or a leaf holding text, is handled")
        # The order of an `xs:sequence` makes no difference to the Rust types, as the XML was checked against its XML schema before it was passed in
        if content.model not in ("all", "sequence"):
            raise NotImplementedError(
                f"{xsd_path.name}: `{element_path}` holds an `xs:{content.model}`, but only an `xs:all` or `xs:sequence` of elements is handled"
            )
        if len(content) == 0:
            raise NotImplementedError(f"{xsd_path.name}: `{element_path}` holds no elements, which would be an empty struct")
        if documentation == "":
            raise ValueError(f"{xsd_path.name}: `{element_path}` has no `<xs:documentation>`, for its struct's documentation")

        # Added before the structs inside it, so the structs are in the XML's order, the root first
        new_struct_definition: StructDefinition = StructDefinition(
            name=struct_name,
            documentation=documentation,
            xsd_type_name=xsd_type_name,
            xsd_path=xsd_path,
            element_paths=[element_path],
        )
        self.struct_definitions.append(new_struct_definition)
        self.struct_names_in_progress.append(struct_name)
        for particle in content:
            if not isinstance(particle, XsdElement):
                raise NotImplementedError(f"{xsd_path.name}: `{element_path}` holds a group inside its `xs:{content.model}`, which is not handled")
            field_definition: FieldDefinition = self.element_field(
                element=particle,
                parent_struct_name=struct_name,
                xsd_path=xsd_path,
                parent_path=element_path,
                linked_xsd_paths=linked_xsd_paths,
            )
            for other_field_definition in new_struct_definition.field_definitions:
                if other_field_definition.name == field_definition.name:
                    raise ValueError(f"{xsd_path.name}: `{element_path}` holds two elements named `{field_definition.name}`")
            new_struct_definition.field_definitions.append(field_definition)
        self.struct_names_in_progress.remove(struct_name)

    def element_field(
        self,
        element: XsdElement,
        parent_struct_name: str,
        xsd_path: Path,
        parent_path: str,
        linked_xsd_paths: list[Path],
    ) -> FieldDefinition:
        """
        The field read from an element, adding the struct of its type if it holds other elements.

        :param element: An element inside a struct's element
        :param parent_struct_name: The struct holding the field
        :param xsd_path: The XML schema declaring the element
        :param parent_path: Path of the element holding it, e.g. "equilibrium_settings/numerics"
        :param linked_xsd_paths: The XML schemas of the links followed to reach the element, the root's first
        :return: The field
        """

        name: str = check_field_name(element.local_name, xsd_path, parent_path)
        element_path: str = f"{parent_path}/{name}"

        # `max_occurs` is None for `maxOccurs="unbounded"`
        is_repeated: bool = element.max_occurs is None
        if not is_repeated and element.max_occurs != 1:
            raise NotImplementedError(f"{xsd_path.name}: `{element_path}` has maxOccurs {element.max_occurs}, but only 1 and unbounded are handled")
        if element.min_occurs not in (0, 1):
            raise NotImplementedError(f"{xsd_path.name}: `{element_path}` has minOccurs {element.min_occurs}, but only 0 and 1 are handled")
        is_nillable: bool = element.nillable
        if is_repeated and (element.min_occurs != 0 or is_nillable):
            raise NotImplementedError(
                f'{xsd_path.name}: `{element_path}` is repeated, which is handled only with minOccurs="0", and not when it is nillable, as a `Vec` of `Option`s'
            )
        # A repeated element is a `Vec`, empty when it is left out, so is not an `Option` too
        is_optional: bool = element.min_occurs == 0 and not is_repeated
        if is_optional and is_nillable:
            raise NotImplementedError(
                f'{xsd_path.name}: `{element_path}` is both optional, minOccurs="0", and nillable, so `None` could mean either, which is not handled'
            )
        documentation: str = read_documentation(element)
        if documentation == "":
            raise ValueError(f"{xsd_path.name}: `{element_path}` has no `<xs:documentation>`, for its field's documentation")
        units: str = read_appinfo(element, "units")

        # The type the element is read as, and the XML schema declaring it: the linked file's root element's, for a link
        element_type: XsdComplexType | XsdSimpleType = element.type
        element_xsd_path: Path = xsd_path
        element_linked_xsd_paths: list[Path] = linked_xsd_paths
        # The struct's documentation, for an anonymous type: the element's, or, for a link, that of the linked file's root element, which describes the file
        struct_documentation: str = documentation
        # For a link to the XML schema of a shared module, the module, whose root struct the field is
        linked_shared_module: SharedModule | None = None

        linked_schema_text: str = read_appinfo(element, "schema")
        if linked_schema_text != "":
            # A link: in the XML the Rust code reads, the linked file's root element is written in the link's place, named as the link
            is_link_type: bool = (
                isinstance(element_type, XsdComplexType)
                and isinstance(element_type.content, XsdGroup)
                and len(element_type.content) == 0
                and list(element_type.attributes) == ["href"]
            )
            if not is_link_type:
                raise NotImplementedError(
                    f"{xsd_path.name}: `{element_path}` names an XML schema in `<xs:appinfo><schema>`, so is a link, but its type is not one holding only an "
                    "`href` attribute"
                )
            if is_nillable or is_repeated:
                raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is a nillable or repeated link, which is not handled")
            linked_xsd_path: Path = (xsd_path.parent / linked_schema_text).resolve()
            if linked_xsd_path in linked_xsd_paths:
                raise ValueError(f"{xsd_path.name}: `{element_path}` links to {linked_xsd_path}, which links back to it")
            linked_xml_schema: xmlschema.XMLSchema = self.load_xml_schema(linked_xsd_path)
            linked_root_elements: list[XsdElement] = list(linked_xml_schema.elements.values())
            if len(linked_root_elements) != 1:
                raise NotImplementedError(f"{linked_xsd_path.name}: {len(linked_root_elements)} root elements, rather than one")
            linked_root_element: XsdElement = linked_root_elements[0]
            linked_root_type: XsdComplexType | XsdSimpleType = linked_root_element.type
            if not isinstance(linked_root_type, XsdComplexType) or not isinstance(linked_root_type.content, XsdGroup):
                raise NotImplementedError(f"{linked_xsd_path.name}: the root element, linked to by `{element_path}`, does not hold elements")

            # A link to a file whose root element holds only one element, named as the link, is collapsed into that element
            linked_root_children: list[XsdElement] = []
            for particle in linked_root_type.content:
                if isinstance(particle, XsdElement):
                    linked_root_children.append(particle)
            is_collapsed_link: bool = len(linked_root_children) == 1 and linked_root_children[0].local_name == name

            if linked_xsd_path in self.shared_modules_by_xsd_path:
                # The linked XML schema is generated into a module of its own, so its structs are not added here
                if is_collapsed_link:
                    raise NotImplementedError(
                        f"{xsd_path.name}: `{element_path}` links to {linked_xsd_path.name}, a shared module, whose root element holds only `{name}`, "
                        "but a collapsed link to a shared module is not handled"
                    )
                linked_shared_module = self.shared_modules_by_xsd_path[linked_xsd_path]
            elif is_collapsed_link:
                return self.collapsed_link_field(
                    inner_element=linked_root_children[0],
                    name=name,
                    documentation=documentation,
                    units=units,
                    is_optional_link=is_optional,
                    parent_struct_name=parent_struct_name,
                    linked_xsd_path=linked_xsd_path,
                    element_path=element_path,
                    linked_xsd_paths=linked_xsd_paths + [linked_xsd_path],
                )

            element_type = linked_root_type
            element_xsd_path = linked_xsd_path
            element_linked_xsd_paths = linked_xsd_paths + [linked_xsd_path]
            linked_root_documentation: str = read_documentation(linked_root_element)
            if linked_root_documentation != "":
                struct_documentation = linked_root_documentation
        elif isinstance(element_type, XsdComplexType) and len(element_type.attributes) > 0:
            raise NotImplementedError(
                f"{xsd_path.name}: `{element_path}` has attributes, which are handled only on a link, an element naming the XML schema of the file it links to "
                "in `<xs:appinfo><schema>`"
            )

        value_type: str
        reader_name: str
        enumeration_values: list[str] = []
        if linked_shared_module is not None:
            # The shared module's root struct, and the function reading it, imported from the module
            value_type = linked_shared_module.root_struct_name
            reader_name = reader_function_name(value_type)
            if linked_shared_module not in self.linked_shared_modules:
                self.linked_shared_modules.append(linked_shared_module)
        elif isinstance(element_type, XsdComplexType) and matrix_row_element(element_type) is not None:
            # A matrix: a 2D array with a row for each `row`, rather than a struct holding a `Vec` of its rows
            if is_nillable:
                raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is a nillable matrix, which is not handled")
            row_element: XsdElement = matrix_row_element(element_type)
            row_rust_type, _ = simple_type_to_rust(row_element.type, element_xsd_path, f"{element_path}/row")
            if row_rust_type != "Array1<f64>":
                raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is a matrix of `{row_rust_type}` rows, but only rows of numbers are handled")
            value_type = "Array2<f64>"
            reader_name = "read_double_matrix"
            self.leaf_reader_names.add(reader_name)
        elif isinstance(element_type, XsdComplexType):
            if is_nillable:
                raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is a nillable element holding other elements, which is not handled")
            xsd_type_name: str = ""
            if element_type.local_name is not None:
                # A named type: one struct wherever it is used, documented by the type's own documentation, as the element's is about where it is used
                xsd_type_name = element_type.local_name
                value_type = self.named_type_struct_prefix + check_pascal_case(pascal_case(xsd_type_name), element_xsd_path, element_path)
                type_documentation: str = read_documentation(element_type)
                if type_documentation != "":
                    struct_documentation = type_documentation
            else:
                value_type = parent_struct_name + check_pascal_case(pascal_case(name), element_xsd_path, element_path)
            self.add_struct(
                complex_type=element_type,
                struct_name=value_type,
                documentation=struct_documentation,
                xsd_path=element_xsd_path,
                element_path=element_path,
                xsd_type_name=xsd_type_name,
                linked_xsd_paths=element_linked_xsd_paths,
            )
            reader_name = reader_function_name(value_type)
        else:
            if is_repeated:
                # e.g. a matrix's `row`, which is not a `Vec` of its rows but a 2D array
                raise NotImplementedError(
                    f"{xsd_path.name}: `{element_path}` is a repeated leaf, but only a repeated element holding other elements is handled"
                )
            value_type, reader_name = simple_type_to_rust(element_type, xsd_path, element_path)
            self.leaf_reader_names.add(reader_name)
            enumeration_values = read_enumeration(element_type)

        # The documentation, with its units appended, e.g. " [metre]", what an enumeration allows, and what `None` means
        documentation_lines: list[str] = [documentation]
        if units != "":
            documentation_lines[0] += f" [{units}]"
        if len(enumeration_values) > 0:
            quoted_enumeration_values: list[str] = []
            for enumeration_value in enumeration_values:
                quoted_enumeration_values.append(f"`{enumeration_value}`")
            documentation_lines.append("One of: " + ", ".join(quoted_enumeration_values) + ".")
        if is_optional:
            documentation_lines.append("`None` when the element is left out.")
        if is_nillable:
            documentation_lines.append("`None` when the element is nil.")
        if is_repeated:
            documentation_lines.append("One item for each element, empty when the element is left out.")

        rust_type: str = value_type
        if is_optional or is_nillable:
            rust_type = f"Option<{value_type}>"
        if is_repeated:
            rust_type = f"Vec<{value_type}>"

        return FieldDefinition(
            name=name,
            rust_type=rust_type,
            value_type=value_type,
            reader_name=reader_name,
            is_optional=is_optional,
            is_nillable=is_nillable,
            is_repeated=is_repeated,
            documentation_lines=documentation_lines,
        )

    def collapsed_link_field(
        self,
        inner_element: XsdElement,
        name: str,
        documentation: str,
        units: str,
        is_optional_link: bool,
        parent_struct_name: str,
        linked_xsd_path: Path,
        element_path: str,
        linked_xsd_paths: list[Path],
    ) -> FieldDefinition:
        """
        The field of a collapsed link: a link to a file whose root element holds only one element, named as the link.

        The field is that element's, e.g. a `Vec` for a repeated element, read from inside the link's element, which the linked file's root element is
        written as. It is documented by the link, which says what the field is for, followed by what its type is.

        :param inner_element: The one element the linked file's root element holds
        :param name: The name of the link, and of the element
        :param documentation: The link's documentation
        :param units: The link's units, or ""
        :param is_optional_link: Whether the link is optional, `minOccurs="0"`
        :param parent_struct_name: The struct holding the field
        :param linked_xsd_path: The linked file's XML schema, which declares `inner_element`
        :param element_path: Path of the link's element, e.g. "equilibrium_settings/constraints/b_field_pol_probe"
        :param linked_xsd_paths: The XML schemas of the links followed to reach the element, the root's first, and the linked file's last
        :return: The field
        """

        inner_field: FieldDefinition = self.element_field(
            element=inner_element,
            parent_struct_name=parent_struct_name,
            xsd_path=linked_xsd_path,
            parent_path=element_path,
            linked_xsd_paths=linked_xsd_paths,
        )
        if inner_field.is_nillable:
            raise NotImplementedError(f"{linked_xsd_path.name}: `{element_path}/{name}` is nillable, inside a collapsed link, which is not handled")
        if is_optional_link:
            raise NotImplementedError(
                f'{linked_xsd_path.name}: `{element_path}` is an optional link, minOccurs="0", to a file whose root element holds only `{name}`, which is not '
                "handled, as left out could mean either"
            )

        documentation_lines: list[str] = [documentation]
        if units != "":
            documentation_lines[0] += f" [{units}]"
        documentation_lines += inner_field.documentation_lines[1:]

        return FieldDefinition(
            name=name,
            rust_type=inner_field.rust_type,
            value_type=inner_field.value_type,
            reader_name=inner_field.reader_name,
            is_optional=inner_field.is_optional,
            is_nillable=False,
            is_repeated=inner_field.is_repeated,
            documentation_lines=documentation_lines,
            is_collapsed_link=True,
        )


def matrix_row_element(complex_type: XsdComplexType) -> XsdElement | None:
    """
    :param complex_type: The type of an element holding other elements
    :return: The `row` of a matrix, a type holding only a repeated `row`, `minOccurs="0" maxOccurs="unbounded"`, of a list type, e.g. GSFit's
        `Regularisations`; None for any other type
    """

    content: XsdGroup | XsdSimpleType = complex_type.content
    if not isinstance(content, XsdGroup) or content.model != "sequence" or len(content) != 1 or len(complex_type.attributes) > 0:
        return None
    row_element: object = content[0]
    if not isinstance(row_element, XsdElement) or row_element.local_name != "row":
        return None
    if row_element.max_occurs is not None or row_element.min_occurs != 0 or row_element.nillable or not isinstance(row_element.type, XsdList):
        return None

    return row_element


def generate_rust_structs(
    xsd_path: Path,
    output_path: Path,
    root_struct_name: str,
    named_type_struct_prefix: str,
    shared_modules: list[SharedModule],
    title: str,
    leaf_readers_module: str,
    rustfmt_config_path: Path,
    rust_edition: str,
    generator_path: Path,
    repository_path: Path,
) -> None:
    """
    Generate a Rust file of the structs holding an XML file's settings, and the code reading its XML into them, from its XML schema and those of the files it links to.

    For example, GSFit's `python/gsfit/settings_schema/_generate_rust_schema.py`, whose path is `generator_path`, calls:
        generate_rust_structs(
            xsd_path=generator_path.parent / "xsd" / "tf" / "tf.xsd",
            output_path=repository_path / "rust" / "gsfit_rs" / "src" / "settings" / "settings_tf.rs",
            root_struct_name="SettingsTf",
            named_type_struct_prefix="SettingsTf",
            shared_modules=[workflow_module],
            title="The settings the toroidal field is built with, read from their XML into structs",
            leaf_readers_module="crate::settings::settings_leaf_readers",
            rustfmt_config_path=repository_path / "rust" / "rustfmt.toml",
            rust_edition="2024",
            generator_path=generator_path,
            repository_path=repository_path,
        )
    where `workflow_module` is the `SharedModule` of `workflow.xsd`, which `tf.xsd` links to.

    A shared module is generated by a call of its own, with its `xsd_path`, `root_struct_name`, and `shared_modules` holding it. Its root struct's reader is then
    public to the crate, so that the settings linking to it can read it.

    :param xsd_path: The XML schema of the XML file, e.g. GSFit's `python/gsfit/settings_schema/xsd/tf/tf.xsd`
    :param output_path: The Rust file to write, e.g. `rust/gsfit_rs/src/settings/settings_tf.rs`
    :param root_struct_name: The struct of the root element, which the struct of each element of an anonymous type is named after, e.g. "SettingsTf"
    :param named_type_struct_prefix: What the struct of each named XML schema type is named with, followed by the type's name, e.g. "SettingsCoreProfiles" for
        `SettingsCoreProfilesThomsonScattering`, from `ThomsonScattering`, or "Settings" for `SettingsWorkflowTree`, from `WorkflowTree`
    :param shared_modules: The XML schemas generated into modules of their own, which are not followed when linked to, but imported from their module
    :param title: The first line of the Rust file's documentation
    :param leaf_readers_module: The Rust module holding the hand-written functions reading the leaves, e.g. "crate::settings::settings_leaf_readers"
    :param rustfmt_config_path: The `rustfmt` configuration the Rust file is formatted with, e.g. `rust/rustfmt.toml`
    :param rust_edition: The Rust edition of the crate, which `rustfmt` needs to sort the imports as `cargo fmt` does, e.g. "2024"
    :param generator_path: Path to the script calling this, which the Rust file says it is autogenerated by,
        e.g. GSFit's `python/gsfit/settings_schema/_generate_rust_schema.py`
    :param repository_path: Root of the repository, which the paths written in the Rust file are from. All of the paths above must be inside it
    """

    # Resolved, as the XML schema a link names is found by resolving its path, which must then be inside `repository_path`
    xsd_path = xsd_path.resolve()
    repository_path = repository_path.resolve()

    # When this is a shared module, the other settings import its root struct by the name `shared_modules` gives it
    is_shared_module: bool = False
    for shared_module in shared_modules:
        if shared_module.xsd_path.resolve() != xsd_path:
            continue
        if shared_module.root_struct_name != root_struct_name:
            raise ValueError(
                f"{xsd_path.name} is a shared module, whose root struct is `{shared_module.root_struct_name}`, but it is generated as `{root_struct_name}`"
            )
        is_shared_module = True

    struct_builder: StructBuilder = StructBuilder(named_type_struct_prefix=named_type_struct_prefix, shared_modules=shared_modules)
    xml_schema: xmlschema.XMLSchema = struct_builder.load_xml_schema(xsd_path)
    root_elements: list[XsdElement] = list(xml_schema.elements.values())
    if len(root_elements) != 1:
        raise NotImplementedError(f"{xsd_path.name}: {len(root_elements)} root elements, rather than one")
    root_element: XsdElement = root_elements[0]
    root_element_name: str | None = root_element.local_name
    root_type: XsdComplexType | XsdSimpleType = root_element.type
    if root_element_name is None or not isinstance(root_type, XsdComplexType) or root_element.nillable:
        raise NotImplementedError(f"{xsd_path.name}: the root element is not a named element holding other elements")

    struct_builder.add_struct(
        complex_type=root_type,
        struct_name=root_struct_name,
        documentation=read_documentation(root_element),
        xsd_path=xsd_path,
        element_path=root_element_name,
        xsd_type_name="",
        linked_xsd_paths=[xsd_path],
    )

    output_path.write_text(
        render_rust_file(
            struct_definitions=struct_builder.struct_definitions,
            root_struct_name=root_struct_name,
            root_element_name=root_element_name,
            root_xsd_path=xsd_path,
            is_shared_module=is_shared_module,
            linked_shared_modules=struct_builder.linked_shared_modules,
            leaf_reader_names=struct_builder.leaf_reader_names,
            title=title,
            leaf_readers_module=leaf_readers_module,
            generator_path_from_repository=generator_path.resolve().relative_to(repository_path).as_posix(),
            repository_path=repository_path,
        ),
        encoding="utf-8",
    )
    # Formatted, so that the file is as `cargo fmt` would leave it
    subprocess.run(
        ["rustfmt", "--edition", rust_edition, "--config-path", str(rustfmt_config_path), str(output_path)],
        check=True,
    )

    print(f"Wrote {len(struct_builder.struct_definitions)} structs to {output_path.resolve().relative_to(repository_path).as_posix()}:")
    for struct_definition in struct_builder.struct_definitions:
        print(f"    {struct_definition.name}")


def render_rust_file(
    struct_definitions: list[StructDefinition],
    root_struct_name: str,
    root_element_name: str,
    root_xsd_path: Path,
    is_shared_module: bool,
    linked_shared_modules: list[SharedModule],
    leaf_reader_names: set[str],
    title: str,
    leaf_readers_module: str,
    generator_path_from_repository: str,
    repository_path: Path,
) -> str:
    """
    :param struct_definitions: The structs, the root's first
    :param root_struct_name: The struct of the root element, e.g. "SettingsTf"
    :param root_element_name: The name of the root element, e.g. "tf_settings"
    :param root_xsd_path: The XML schema of the root element, resolved
    :param is_shared_module: Whether this is a shared module, whose root struct's reader is public to the crate, for the settings linking to it
    :param linked_shared_modules: The shared modules linked to, whose root struct, and the function reading it, are imported
    :param leaf_reader_names: The functions in the leaf readers reading the leaves
    :param title: The first line of the file's documentation
    :param leaf_readers_module: The Rust module holding the leaf readers, e.g. "crate::settings::settings_leaf_readers"
    :param generator_path_from_repository: The script calling this, e.g. "python/gsfit/settings_schema/_generate_rust_schema.py"
    :param repository_path: Root of the repository, resolved, which the paths written in the file are from
    :return: The Rust file, before `rustfmt`
    """

    # The functions the generated readers call, from the leaf readers
    imported_names: set[str] = {"child_elements", "unexpected_element"}
    imported_names.update(leaf_reader_names)
    ndarray_names: set[str] = set()
    has_matrix: bool = False
    has_collapsed_link: bool = False
    for struct_definition in struct_definitions:
        for field_definition in struct_definition.field_definitions:
            if field_definition.is_nillable:
                imported_names.add("is_nil")
            if field_definition.is_collapsed_link:
                # The link's element is there once, and is required
                has_collapsed_link = True
                imported_names.add("repeated_element")
                imported_names.add("missing_element")
            elif not field_definition.is_repeated:
                imported_names.add("repeated_element")
                if not field_definition.is_optional:
                    imported_names.add("missing_element")
            if field_definition.value_type == "Array1<f64>":
                ndarray_names.add("Array1")
            if field_definition.value_type == "Array2<f64>":
                ndarray_names.add("Array2")
                has_matrix = True

    root_xsd_relative_path: str = root_xsd_path.relative_to(repository_path).as_posix()
    lines: list[str] = [
        f"//! {title}",
        "//!",
        f"//! **WARNING** This file is autogenerated by `{generator_path_from_repository}`.",
        "//! Any changes will be overwritten.",
        "//!",
        f"//! Generated from the XML schema `{root_xsd_relative_path}`, and the XML schemas of the files it links to.",
    ]
    if is_shared_module:
        lines += [
            "//! Several settings files link to this XML schema, and read their link into its root struct, with its reader, which is public to the crate,",
            "//! so that they all have the same struct.",
        ]
    for linked_shared_module in linked_shared_modules:
        linked_xsd_relative_path: str = linked_shared_module.xsd_path.resolve().relative_to(repository_path).as_posix()
        lines.append(
            f"//! The link to `{linked_xsd_relative_path}` is read into `{linked_shared_module.rust_module}::{linked_shared_module.root_struct_name}`, "
            "which is shared by the settings linking to it."
        )
    lines += [
        "//!",
        "//! Each XML element holding other elements is read into a struct, with a field for each element inside it, named as the element.",
        f"//! A leaf is read, by the hand-written functions in `{leaf_readers_module}`, as the Rust type of its XML schema type:",
        "//! * `xs:double`, or a type restricting it, e.g. `FiniteDouble`, is an `f64`",
        "//! * `xs:positiveInteger` or `xs:nonNegativeInteger`, or a type restricting either, is a `usize`",
        "//! * `xs:integer`, or a type restricting it, is an `i64`",
        "//! * `xs:boolean`, or a type restricting it, e.g. `TrueOrFalse`, is a `bool`",
        "//! * `xs:string`, or a type restricting it, e.g. an enumeration, is a `String`",
        "//! * a space-separated list of numbers is an `Array1<f64>`",
        '//! * a nillable element is an `Option`, `None` when it is nil, `xsi:nil="true"`, and so is an optional element, `None` when it is left out',
        "//! * a repeated element holding other elements is a `Vec`, empty when it is left out",
    ]
    if has_matrix:
        lines.append(
            "//! * a matrix, an element holding only `row`s, each a space-separated list of numbers, is an `Array2<f64>`, of shape (0, 0) with no `row`"
        )
    if has_collapsed_link:
        lines += [
            "//!",
            "//! A link to a file whose root element holds only one element, named as the link, is collapsed into that element: the link's field is that element's,",
            "//! read from inside the link's element, e.g. `<b_field_pol_probe><b_field_pol_probe>...`.",
        ]
    lines += [
        "//!",
        "//! The XML is not checked against its XML schema here, as it was checked before it was passed in.",
        "//! Only what filling the structs needs is checked: that each element is in the XML schema, is there at most once unless it is repeated, and is there if it",
        "//! is required, and that each leaf can be read as its Rust type.",
        "",
    ]
    sorted_imported_names: list[str] = sorted(imported_names)
    lines.append(f"use {leaf_readers_module}::{{{', '.join(sorted_imported_names)}}};")
    # The root struct of each shared module linked to, and the function reading it
    for linked_shared_module in linked_shared_modules:
        lines.append(
            f"use {linked_shared_module.rust_module}::{{{linked_shared_module.root_struct_name}, {reader_function_name(linked_shared_module.root_struct_name)}}};"
        )
    sorted_ndarray_names: list[str] = sorted(ndarray_names)
    if len(sorted_ndarray_names) == 1:
        lines.append(f"use ndarray::{sorted_ndarray_names[0]};")
    elif len(sorted_ndarray_names) > 1:
        lines.append(f"use ndarray::{{{', '.join(sorted_ndarray_names)}}};")
    lines.append("use roxmltree::{Document, Node};")
    lines += [
        "",
        "/// The script generating this file, for the error saying to re-run it",
        f'const GENERATOR_PATH: &str = "{generator_path_from_repository}";',
        "",
    ]

    # The structs
    for struct_definition in struct_definitions:
        lines += render_struct(struct_definition, repository_path)
        lines.append("")

    # Reading the root struct from the XML
    root_xml_name: str = root_xsd_path.with_suffix(".xml").name
    lines += [
        f"impl {root_struct_name} {{",
        f"    /// Read the settings from their XML: that of `{root_xml_name}`, with each file it links to written in its link's place, as",
        "    /// `serialise_python_dataclass_to_imas_parameters_input` writes it.",
        "    ///",
        "    /// Each element is read into the field of the same name. The XML is not checked against its XML schema here, as it was checked before it",
        "    /// was passed in.",
        "    ///",
        "    /// # Errors",
        f"    /// The XML is not well-formed, its root element is not `<{root_element_name}>`, an element is not in the XML schema this was generated",
        "    /// from, is there more than once unless it is repeated, or is required but missing, or a leaf cannot be read as its Rust type",
        f"    pub fn from_xml(xml: &str) -> Result<{root_struct_name}, String> {{",
        '        let document: Document = Document::parse(xml).map_err(|error| format!("the settings are not well-formed XML: {error}"))?;',
        "        let root: Node = document.root_element();",
        f'        if root.tag_name().name() != "{root_element_name}" {{',
        "            return Err(format!(",
        f'                "the root element of the settings is <{{}}>, rather than <{root_element_name}>",',
        "                root.tag_name().name()",
        "            ));",
        "        }",
        "",
        f'        {reader_function_name(root_struct_name)}(root, "{root_element_name}")',
        "    }",
        "}",
        "",
    ]

    # Reading each struct from its XML element. A shared module's root struct is read by the settings linking to it too
    for struct_definition in struct_definitions:
        is_public_to_crate: bool = is_shared_module and struct_definition.name == root_struct_name
        lines += render_reader(struct_definition=struct_definition, is_public_to_crate=is_public_to_crate)
        lines.append("")

    return "\n".join(lines)


def render_struct(struct_definition: StructDefinition, repository_path: Path) -> list[str]:
    """
    :param struct_definition: A struct
    :param repository_path: Root of the repository, resolved, which the paths written in the file are from
    :return: Its definition's lines
    """

    quoted_element_paths: list[str] = []
    for element_path in struct_definition.element_paths:
        quoted_element_paths.append(f"`{element_path}`")
    element_paths_text: str = ", ".join(quoted_element_paths)
    elements_text: str = "the element" if len(quoted_element_paths) == 1 else "the elements"
    xsd_relative_path: str = struct_definition.xsd_path.relative_to(repository_path).as_posix()
    source: str = f"Read from {elements_text} {element_paths_text}, declared in `{xsd_relative_path}`."
    if struct_definition.xsd_type_name != "":
        source = (
            f"Read from {elements_text} {element_paths_text}, of the XML schema type `{struct_definition.xsd_type_name}`, declared in `{xsd_relative_path}`."
        )

    lines: list[str] = [
        f"/// {struct_definition.documentation}",
        "///",
        f"/// {source}",
        "#[derive(Clone, Debug, PartialEq)]",
        f"pub struct {struct_definition.name} {{",
    ]
    for field_definition in struct_definition.field_definitions:
        for documentation_line in field_definition.documentation_lines:
            lines.append(f"    /// {documentation_line}")
        if SNAKE_CASE_PATTERN.fullmatch(field_definition.name) is None:
            lines.append('    #[expect(non_snake_case, reason = "named as its XML element, as the XML schema is the source of truth")]')
        lines.append(f"    pub {field_definition.name}: {field_definition.rust_type},")
    lines.append("}")

    return lines


def render_reader(struct_definition: StructDefinition, is_public_to_crate: bool) -> list[str]:
    """
    The function reading a struct from its XML element.

    Each field is read into a variable of its own name, an `Option`, `None` until its element is read, so that a repeated element can be found,
    and then a missing one. For a nillable element it is an `Option<Option<...>>`, `Some(None)` once the element is read, if it is nil.
    A repeated element is pushed onto a `Vec`, as it can be there any number of times.

    :param struct_definition: A struct
    :param is_public_to_crate: Whether the function is `pub(crate)`, for the root struct of a shared module, which the settings linking to it read
    :return: The function's lines
    """

    lines: list[str] = [f"/// Read `{struct_definition.name}` from its XML element, whose path is `path`"]
    # Each field is read into a variable of its own name, which is not snake_case if the element's name is not
    for field_definition in struct_definition.field_definitions:
        if SNAKE_CASE_PATTERN.fullmatch(field_definition.name) is None:
            lines.append('#[expect(non_snake_case, reason = "each field is read into a variable of its own name, which is named as its XML element")]')
            break
    visibility: str = "pub(crate) " if is_public_to_crate else ""
    lines.append(
        f"{visibility}fn {reader_function_name(struct_definition.name)}(node: Node<'_, '_>, path: &str) -> Result<{struct_definition.name}, String> {{"
    )
    for field_definition in struct_definition.field_definitions:
        if field_definition.is_collapsed_link and field_definition.is_repeated:
            lines.append(f"    let mut {field_definition.name}: Option<{field_definition.rust_type}> = None; // `Some` once the link's element is read")
        elif field_definition.is_collapsed_link:
            lines.append(
                f"    let mut {field_definition.name}: Option<Option<{field_definition.value_type}>> = None; // `Some` once the link's element is read"
            )
        elif field_definition.is_repeated:
            lines.append(f"    let mut {field_definition.name}: {field_definition.rust_type} = Vec::new();")
        elif field_definition.is_nillable:
            lines.append(f"    let mut {field_definition.name}: Option<{field_definition.rust_type}> = None; // `Some(None)` once read, if the element is nil")
        else:
            lines.append(f"    let mut {field_definition.name}: Option<{field_definition.value_type}> = None;")
    lines += [
        "    for child in child_elements(node, path)? {",
        "        let child_name: &str = child.tag_name().name();",
        '        let child_path: String = format!("{path}/{child_name}");',
        "        match child_name {",
    ]
    for field_definition in struct_definition.field_definitions:
        if field_definition.is_collapsed_link:
            lines += render_collapsed_link_reader(field_definition)
            continue
        if field_definition.is_repeated:
            # The path of each item is that of the element, with its index, e.g. "tf_settings/workflow/tree(0)"
            lines.append(
                f'            "{field_definition.name}" => {field_definition.name}.push({field_definition.reader_name}(child, &format!("{{child_path}}({{}})", {field_definition.name}.len()))?),'
            )
            continue
        read_expression: str = f"{field_definition.reader_name}(child, &child_path)?"
        lines += [
            f'            "{field_definition.name}" => {{',
            f"                if {field_definition.name}.is_some() {{",
            "                    return Err(repeated_element(&child_path));",
            "                }",
        ]
        if field_definition.is_nillable:
            lines.append(f"                {field_definition.name} = Some(if is_nil(child) {{ None }} else {{ Some({read_expression}) }});")
        else:
            lines.append(f"                {field_definition.name} = Some({read_expression});")
        lines.append("            }")
    lines += [
        "            _ => return Err(unexpected_element(&child_path, GENERATOR_PATH)),",
        "        }",
        "    }",
        "",
        f"    Ok({struct_definition.name} {{",
    ]
    for field_definition in struct_definition.field_definitions:
        missing_link: str = f'missing_element(&format!("{{path}}/{field_definition.name}"))'
        if field_definition.is_collapsed_link and (field_definition.is_repeated or field_definition.is_optional):
            lines.append(f"        {field_definition.name}: {field_definition.name}.ok_or_else(|| {missing_link})?,")
        elif field_definition.is_collapsed_link:
            # The element inside the link's element is required too
            missing_inner: str = f'missing_element(&format!("{{path}}/{field_definition.name}/{field_definition.name}"))'
            lines.append(f"        {field_definition.name}: {field_definition.name}.ok_or_else(|| {missing_link})?.ok_or_else(|| {missing_inner})?,")
        elif field_definition.is_optional or field_definition.is_repeated:
            lines.append(f"        {field_definition.name},")
        else:
            lines.append(f"        {field_definition.name}: {field_definition.name}.ok_or_else(|| {missing_link})?,")
    lines += [
        "    })",
        "}",
    ]

    return lines


def render_collapsed_link_reader(field_definition: FieldDefinition) -> list[str]:
    """
    The match arm reading a collapsed link: the elements named as the link, inside the link's element, which is there once.

    :param field_definition: The field of a collapsed link
    :return: The match arm's lines
    """

    name: str = field_definition.name
    lines: list[str] = [
        f'            "{name}" => {{',
        f"                if {name}.is_some() {{",
        "                    return Err(repeated_element(&child_path));",
        "                }",
    ]
    if field_definition.is_repeated:
        lines += [
            f"                let mut items: {field_definition.rust_type} = Vec::new();",
            "                for item in child_elements(child, &child_path)? {",
            "                    let item_name: &str = item.tag_name().name();",
            '                    let item_path: String = format!("{child_path}/{item_name}");',
            f'                    if item_name != "{name}" {{',
            "                        return Err(unexpected_element(&item_path, GENERATOR_PATH));",
            "                    }",
            # The path of each item is that of the element, with its index, e.g. "equilibrium_settings/constraints/b_field_pol_probe/b_field_pol_probe(0)"
            f'                    items.push({field_definition.reader_name}(item, &format!("{{item_path}}({{}})", items.len()))?);',
            "                }",
            f"                {name} = Some(items);",
        ]
    else:
        lines += [
            f"                let mut item_read: Option<{field_definition.value_type}> = None;",
            "                for item in child_elements(child, &child_path)? {",
            "                    let item_name: &str = item.tag_name().name();",
            '                    let item_path: String = format!("{child_path}/{item_name}");',
            f'                    if item_name != "{name}" {{',
            "                        return Err(unexpected_element(&item_path, GENERATOR_PATH));",
            "                    }",
            "                    if item_read.is_some() {",
            "                        return Err(repeated_element(&item_path));",
            "                    }",
            f"                    item_read = Some({field_definition.reader_name}(item, &item_path)?);",
            "                }",
            f"                {name} = Some(item_read);",
        ]
    lines.append("            }")

    return lines


def simple_type_to_rust(simple_type: XsdSimpleType, xsd_path: Path, element_path: str) -> tuple[str, str]:
    """
    :param simple_type: The simple type of a leaf, named or anonymous, e.g. `xs:double`, a restriction of a type, or a list
    :param xsd_path: The XML schema declaring the leaf, for the messages
    :param element_path: Path of the leaf, for the messages
    :return: (the Rust type, the function in the leaf readers reading it), e.g. ("f64", "read_double")
    """

    if isinstance(simple_type, XsdAtomicBuiltin):
        built_in_name: str = str(simple_type.local_name)
        if built_in_name not in BUILT_IN_TYPES:
            raise NotImplementedError(
                f"{xsd_path.name}: `{element_path}` is of the built-in type xs:{built_in_name}, but only xs:{', xs:'.join(sorted(BUILT_IN_TYPES))} are handled"
            )
        return BUILT_IN_TYPES[built_in_name]

    if isinstance(simple_type, XsdList):
        item_rust_type, _ = simple_type_to_rust(simple_type.item_type, xsd_path, element_path)
        if item_rust_type != "f64":
            raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is a list of `{item_rust_type}`, but only lists of numbers are handled")
        return "Array1<f64>", "read_double_list"

    if isinstance(simple_type, XsdAtomicRestriction):
        # A restriction only constrains the values, e.g. by `minExclusive` or `enumeration` (see `SUPPORTED_XS_ATTRIBUTES`),
        # so has the Rust type of its base type
        base_type: object = simple_type.base_type
        if not isinstance(base_type, XsdSimpleType):
            raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is of a simple type restricting a complex type, which is not handled")
        return simple_type_to_rust(base_type, xsd_path, element_path)

    raise NotImplementedError(f"{xsd_path.name}: `{element_path}` is of the simple type {type(simple_type).__name__}, e.g. a union, which is not handled")


def read_enumeration(simple_type: XsdSimpleType) -> list[str]:
    """
    :param simple_type: The simple type of a leaf
    :return: The values an enumeration allows, from the type, or the nearest type it restricts which has them; [] for a type which is not an enumeration
    """

    if not isinstance(simple_type, XsdAtomicRestriction):
        return []
    if simple_type.enumeration is not None:
        enumeration_values: list[str] = []
        for enumeration_value in simple_type.enumeration:
            enumeration_values.append(str(enumeration_value))
        return enumeration_values
    base_type: object = simple_type.base_type
    if isinstance(base_type, XsdSimpleType):
        return read_enumeration(base_type)
    return []


def check_supported(schema_root: ElementTree.Element, xsd_path: Path) -> None:
    """
    Stop on any XML schema tag or attribute this generator does not know how to translate.

    :param schema_root: `<xs:schema>`
    :param xsd_path: Path to the XML schema, for the message
    """

    # What is inside `<xs:appinfo>` or `<xs:documentation>` is free content, e.g. `<units>`, so is not checked
    free_content_nodes: set[int] = set()
    for node in schema_root.iter():
        if node.tag in (f"{XS}appinfo", f"{XS}documentation"):
            for descendant in node.iter():
                if descendant is not node:
                    free_content_nodes.add(id(descendant))

    for node in schema_root.iter():
        if not isinstance(node.tag, str) or id(node) in free_content_nodes:
            continue  # an XML comment, or free content
        if not node.tag.startswith(XS):
            raise NotImplementedError(f"{xsd_path.name}: <{node.tag}>")
        tag: str = node.tag[len(XS) :]
        if tag not in SUPPORTED_XS_ATTRIBUTES:
            raise NotImplementedError(f"{xsd_path.name}: <xs:{tag}>, which is not handled")
        for attribute_name in node.attrib:
            if attribute_name not in SUPPORTED_XS_ATTRIBUTES[tag]:
                raise NotImplementedError(f"{xsd_path.name}: <xs:{tag} {attribute_name}=...>, which is not handled")


def read_documentation(component: XsdComponent) -> str:
    """
    :param component: XML schema component, e.g. an element or a type
    :return: Its `<xs:documentation>`, with the whitespace normalised, so on one line; "" when it has none
    """

    words: list[str] = []
    if component.annotation is not None:
        for documentation in component.annotation.documentation:
            words += "".join(documentation.itertext()).split()

    return " ".join(words)


def read_appinfo(component: XsdComponent, tag: str) -> str:
    """
    :param component: XML schema component, e.g. an element
    :param tag: A tag inside its `<xs:appinfo>`, e.g. "units" or "schema"
    :return: The text of that tag, e.g. "metre" or "grid.xsd", or "" when it has none
    """

    text: str = ""
    if component.annotation is None:
        return text
    for appinfo in component.annotation.appinfo:
        for appinfo_element in appinfo.findall(tag):
            text = "".join(appinfo_element.itertext()).strip()

    return text


def check_field_name(name: str | None, xsd_path: Path, parent_path: str) -> str:
    """
    :param name: The name of an element
    :param xsd_path: The XML schema declaring the element, for the message
    :param parent_path: Path of the element holding it, for the message
    :return: `name`, once checked that it can be a field's name, and the name of the variable it is read into
    """

    if name is None or RUST_IDENTIFIER_PATTERN.fullmatch(name) is None or name in RUST_KEYWORDS or name in READER_VARIABLE_NAMES:
        raise NotImplementedError(
            f"{xsd_path.name}: `{parent_path}` holds an element named {name!r}, which cannot be a field's name: it must be a Rust identifier, "
            f"and neither a Rust keyword nor one of {sorted(READER_VARIABLE_NAMES)}, the generated readers' variables"
        )

    return name


def check_pascal_case(name: str, xsd_path: Path, element_path: str) -> str:
    """
    :param name: Part of a struct's name, from a type's or an element's name, e.g. "WorkflowTree"
    :param xsd_path: The XML schema it is from, for the message
    :param element_path: Path of the element, for the message
    :return: `name`, once checked that it can be part of a struct's name
    """

    if PASCAL_CASE_PATTERN.fullmatch(name) is None:
        raise NotImplementedError(f"{xsd_path.name}: `{element_path}` would give a struct a name with {name!r}, which is not PascalCase")

    return name


def pascal_case(name: str) -> str:
    """
    :param name: e.g. `workflow_tree`, or `WorkflowTree`
    :return: e.g. `WorkflowTree`, or `WorkflowTree`, as only the first letter of each word is changed
    """

    pascal_case_name: str = ""
    for word in name.split("_"):
        pascal_case_name += word[:1].upper() + word[1:]

    return pascal_case_name


def reader_function_name(struct_name: str) -> str:
    """
    :param struct_name: e.g. `SettingsCoreProfilesThomsonScattering`
    :return: The name of the generated function reading the struct from its XML element, e.g. `read_settings_core_profiles_thomson_scattering`
    """

    snake_case_name: str = ""
    for character in struct_name:
        if character.isupper() and snake_case_name != "":
            snake_case_name += "_"
        snake_case_name += character.lower()

    return f"read_{snake_case_name}"
