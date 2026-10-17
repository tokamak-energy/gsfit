"""
Generate the Python types in `python/gsfit/settings_schema/python/` from the XML schemas in `python/gsfit/settings_schema/xsd/`.

The modules are laid out as the XML schemas are, which are laid out as a settings directory is, e.g. `wall/wall.xsd` -> `wall/wall.py`.
Each module holds a dataclass for each XML element which has child elements or attributes.
The dataclasses can be changed, so that the settings can be edited in Python after they are read, e.g. `settings.ids.equilibrium.timeslices.method = "arange"`.
The dataclass of the root element has `XSD_PATH`, the path of its XML schema, which `read_xml_settings_to_dataclasses` checks the XML file against.
A link to another XML file, e.g. `<grid href="grid.xml"/>`, names the XML schema of that file in `<xs:appinfo><schema>`,
and has the type of that schema's root element, e.g. `Grid`, imported from its module.
So `GsfitCodeSettings`, in `gsfit_code_settings.py`, holds all of GSFit's settings: `GSFIT_code_settings.xml` with every link followed, however deep.
Enumerations become `typing.Literal` aliases, `<xs:documentation>` becomes docstrings, and the units in `<xs:appinfo>` become comments.
A matrix, an element holding one `<row>` per row, becomes a 2D numpy array, `npt.NDArray[np.float64]`, rather than a class; see `matrix_row_element`.
A repeated element which an `xs:key` or `xs:unique` keeps unique by one of its child elements, e.g. each `<tree>` by its `<tree_name>` in `workflow.xsd`,
becomes a `KeyedList`, which can be indexed by that key as well as by position, e.g. `workflow.tree["ELMAG"]`; see `ModuleBuilder.list_keys`.
The generator stops on any XML schema construct it does not know how to translate, rather than guess.

The XML schemas are read with `xmlschema`, which also checks that each is itself a valid XML schema.

Usage:
    python python/gsfit/settings_schema/_generate_python_schema.py
"""

import keyword
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from xml.etree.ElementTree import Element

import xmlschema
from diagnostic_and_simulation_base.xml_reader_and_writer import matrix_row_element
from xmlschema.validators import XsdAtomicBuiltin
from xmlschema.validators import XsdAtomicRestriction
from xmlschema.validators import XsdAttribute
from xmlschema.validators import XsdComplexType
from xmlschema.validators import XsdComponent
from xmlschema.validators import XsdElement
from xmlschema.validators import XsdGroup
from xmlschema.validators import XsdKey
from xmlschema.validators import XsdList
from xmlschema.validators import XsdSimpleType
from xmlschema.validators import XsdUnique

XS_NAMESPACE: str = "http://www.w3.org/2001/XMLSchema"
XS: str = "{" + XS_NAMESPACE + "}"

# This script sits in `python/gsfit/settings_schema/`, three levels below the repository root
REPOSITORY_PATH: Path = Path(__file__).resolve().parents[3]

# Paths from the repository root, as quoted in the generated files.
# The XML schemas are in `xsd/`, next to this script, and the generated modules are written to `python/`, next to this script
GENERATOR_PATH: str = Path(__file__).resolve().relative_to(REPOSITORY_PATH).as_posix()
SCHEMA_PATH: str = (Path(__file__).resolve().parent / "xsd").relative_to(REPOSITORY_PATH).as_posix()
OUTPUT_PATH: str = (Path(__file__).resolve().parent / "python").relative_to(REPOSITORY_PATH).as_posix()

# XML schema built-in type -> Python type
BUILTIN_TYPES: dict[str, str] = {
    "anyURI": "str",
    "boolean": "bool",
    "double": "float",
    "int": "int",
    "nonNegativeInteger": "int",
    "string": "str",
}

# XML schema tag -> the attributes this generator understands.
# Any other tag, or attribute, stops the generator.
SUPPORTED_XS_ATTRIBUTES: dict[str, set[str]] = {
    "all": set(),
    "annotation": set(),
    "appinfo": set(),
    "attribute": {"name", "type", "use"},
    "complexType": {"name"},
    "documentation": set(),
    "element": {"maxOccurs", "minOccurs", "name", "nillable", "type"},
    "enumeration": {"value"},
    "list": {"itemType"},
    "restriction": {"base"},
    "schema": {"elementFormDefault"},
    "sequence": set(),
    "simpleType": {"name"},
    # Identity constraints, which make a repeated element with a key a `KeyedList`, and otherwise only constrain the values; see `ModuleBuilder.list_keys`
    "field": {"xpath"},
    "key": {"name"},
    "selector": {"xpath"},
    "unique": {"name"},
    # These only constrain the values, so make no difference to the Python types
    "maxExclusive": {"value"},
    "maxInclusive": {"value"},
    "minExclusive": {"value"},
    "minInclusive": {"value"},
    "pattern": {"value"},
}

# Names a field cannot take, as the generated modules use them
RESERVED_NAMES: set[str] = {"KEY", "KeyedList", "XSD_PATH", "bool", "dataclass", "field", "float", "int", "list", "np", "npt", "str", "typing"}


@dataclass
class FieldDefinition:
    name: str
    python_type: str  # e.g. `float | None`
    default: str  # e.g. `None`, or "" for a required field
    documentation: str
    units: str


@dataclass
class ClassDefinition:
    name: str
    documentation: str
    field_definitions: list[FieldDefinition] = field(default_factory=list)
    xsd_path: str = ""  # e.g. "wall/wall.xsd" for the class of a root element, otherwise ""
    key: str = ""  # e.g. "tree_name" for the class of the items of a `KeyedList`, otherwise ""


@dataclass
class TypeAliasDefinition:
    name: str
    documentation: str
    values: list[str]


def main() -> None:
    schema_path: Path = REPOSITORY_PATH / SCHEMA_PATH
    output_path: Path = REPOSITORY_PATH / OUTPUT_PATH

    xsd_paths: list[Path] = sorted(schema_path.rglob("*.xsd"))

    # The class of each XML schema's root element, by the XML schema's path, e.g. "equilibrium/grid.xsd" -> "Grid",
    # which is the type of a link to an XML file of that schema.
    # They are all found before any module is built, so that a module can import the class of any XML file it links to
    root_class_names_by_xsd_path: dict[str, str] = {}
    for xsd_path in xsd_paths:
        root_elements: list[XsdElement] = list(xmlschema.XMLSchema(xsd_path).elements.values())
        if len(root_elements) != 1:
            raise NotImplementedError(f"{xsd_path.name}: {len(root_elements)} root elements, rather than one")
        root_type: XsdComplexType | XsdSimpleType = root_elements[0].type
        if not isinstance(root_type, XsdComplexType):
            raise NotImplementedError(f"{xsd_path.name}: the root element is of a simple type")
        # Named as `ModuleBuilder.element_type` names it: after the root element's type if that is named, otherwise after the root element
        root_name: str | None = root_type.local_name if root_type.local_name is not None else root_elements[0].local_name
        if root_name is None:
            raise NotImplementedError(f"{xsd_path.name}: the root element has no name")
        root_class_names_by_xsd_path[xsd_path.relative_to(schema_path).as_posix()] = pascal_case(root_name)

    # One module per XML schema, at the same path, e.g. `wall/wall.xsd` -> `wall/wall.py`
    written_paths: list[Path] = []
    for xsd_path in xsd_paths:
        module_builder: ModuleBuilder = ModuleBuilder(xsd_path, schema_path, root_class_names_by_xsd_path)
        module_path: Path = output_path / module_relative_path(xsd_path.relative_to(schema_path).as_posix())
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(module_builder.render(), encoding="utf-8")
        written_paths.append(module_path)

    # An `__init__.py` in each directory, which Python needs to import from it, importing the modules and directories in it
    package_paths: list[Path] = [output_path]
    for module_path in written_paths:
        relative_parts: tuple[str, ...] = module_path.parent.relative_to(output_path).parts
        for i_part in range(len(relative_parts)):
            package_path: Path = output_path.joinpath(*relative_parts[: i_part + 1])
            if package_path not in package_paths:
                package_paths.append(package_path)
    init_paths: list[Path] = []
    for package_path in package_paths:
        module_names: list[str] = []
        for module_path in written_paths:
            if module_path.parent == package_path:
                module_names.append(module_path.stem)
        subpackage_names: list[str] = []
        for other_package_path in package_paths:
            if other_package_path.parent == package_path:
                if other_package_path.name in module_names:
                    raise ValueError(f"{package_path}: `{other_package_path.name}` would be both a module and a directory")
                subpackage_names.append(other_package_path.name)
        init_path: Path = package_path / "__init__.py"
        init_path.write_text(render_init(package_path.relative_to(output_path).as_posix(), module_names + subpackage_names), encoding="utf-8")
        init_paths.append(init_path)
    written_paths += init_paths

    print(f"Wrote {len(xsd_paths)} modules, and their `__init__.py`, to {output_path}")

    # Remove the modules of XML schemas which no longer exist, e.g. after one is renamed or moved, so that they cannot still be imported.
    # Only files this generator wrote are removed
    for python_path in sorted(output_path.rglob("*.py")):
        if python_path not in written_paths and f"autogenerated by `{GENERATOR_PATH}`" in python_path.read_text(encoding="utf-8"):
            python_path.unlink()
            print(f"Removed {python_path.relative_to(output_path)}, as its XML schema no longer exists")


class ModuleBuilder:
    """
    Builds the Python module for one XML schema.

    A link to another XML file, e.g. `<grid href="grid.xml"/>`, is an element whose type holds only an `href` attribute.
    It names the XML schema of the file it links to in `<xs:appinfo><schema>`, relative to this XML schema, e.g. `grid.xsd`.
    Its field has the type of that XML schema's root element, e.g. `grid: Grid`, imported from that XML schema's module, because
    `read_xml_settings_to_dataclasses` reads the linked file in the link's place, checked against the `XSD_PATH` of that type.

    A repeated element with a key, e.g. `<tree>` keyed by its `<tree_name>`, has a `KeyedList` for its field, e.g. `tree: KeyedList[WorkflowTree]`,
    and the class of its items names the key in `KEY`, e.g. `WorkflowTree.KEY = "tree_name"`; see `list_keys`.
    """

    def __init__(self, xsd_path: Path, schema_path: Path, root_class_names_by_xsd_path: dict[str, str]) -> None:
        """
        :param xsd_path: Path to the XML schema
        :param schema_path: Directory holding every XML schema, which their paths are relative to
        :param root_class_names_by_xsd_path: Class of the root element of every XML schema, by its path, e.g. "equilibrium/grid.xsd" -> "Grid", for the links
        """

        self.xsd_path: Path = xsd_path
        self.schema_path: Path = schema_path
        self.xsd_relative_path: str = xsd_path.relative_to(schema_path).as_posix()  # e.g. "wall/wall.xsd"
        self.root_class_names_by_xsd_path: dict[str, str] = root_class_names_by_xsd_path
        # Raises `xmlschema.XMLSchemaParseError` if the XML schema is itself not valid
        self.xml_schema: xmlschema.XMLSchema = xmlschema.XMLSchema(xsd_path)
        check_supported(self.xml_schema.root, xsd_path)

        self.class_definitions: list[ClassDefinition] = []
        self.class_names_by_complex_type: dict[XsdComplexType, str] = {}
        # The keys of the repeated child elements of each complex type with a class, e.g. `{"tree": "tree_name"}`; see `list_keys`
        self.list_keys_by_complex_type: dict[XsdComplexType, dict[str, str]] = {}
        self.type_alias_definitions: list[TypeAliasDefinition] = []
        # The classes of the root elements of the linked XML files, and the modules `render` imports them from, e.g. "WallSettings" -> ".wall.wall"
        self.imported_module_names_by_class_name: dict[str, str] = {}
        # Whether a field is a matrix, a numpy array, so the module imports numpy
        self.uses_numpy: bool = False
        # Whether a field is a `KeyedList`, so the module imports it
        self.uses_keyed_list: bool = False

        # Reserve the Python names of the named types first, so that a class for an anonymous type is never given one of them.
        # Only complex types (classes) and enumerations (`typing.Literal` aliases) get a Python name
        self.taken_names: set[str] = set()
        for type_name, xsd_type in self.xml_schema.types.items():
            is_enumeration: bool = isinstance(xsd_type, XsdAtomicRestriction) and xsd_type.enumeration is not None
            if isinstance(xsd_type, XsdComplexType) or is_enumeration:
                python_name: str = pascal_case(type_name)
                if python_name in self.taken_names:
                    raise ValueError(f"{xsd_path.name}: two types are both called {python_name} in Python")
                self.taken_names.add(python_name)

        root_elements: list[XsdElement] = list(self.xml_schema.elements.values())
        if len(root_elements) != 1 or root_elements[0].local_name is None:
            raise NotImplementedError(f"{xsd_path.name}: {len(root_elements)} root elements, rather than one with a name")
        self.root_element_name: str = root_elements[0].local_name
        documentation, _ = read_annotation(root_elements[0])
        self.root_class_name: str = self.element_type(root_elements[0], parent_class_name="", documentation=documentation)
        if self.root_class_name != root_class_names_by_xsd_path.get(self.xsd_relative_path):
            raise ValueError(f"{xsd_path.name}: the root element's class, {self.root_class_name}, is not the one a link to it would import")

        # The root element's class carries the path of this XML schema, which `read_xml_settings_to_dataclasses` checks the XML file against
        for class_definition in self.class_definitions:
            if class_definition.name == self.root_class_name:
                class_definition.xsd_path = self.xsd_relative_path

    def element_type(self, element: XsdElement, parent_class_name: str, documentation: str) -> str:
        """
        :param element: An element, e.g. `<xs:element name="grid">`
        :param parent_class_name: Class holding the element, used to make a class name unique
        :param documentation: The element's documentation, used for the class of an anonymous complex type
        :return: Python type of one occurrence of the element, e.g. `float`, `Grid`, or `npt.NDArray[np.float64]` for a matrix
        """

        xsd_type: XsdComplexType | XsdSimpleType = element.type
        if not isinstance(xsd_type, XsdComplexType):
            return self.simple_type(xsd_type)

        # A matrix, holding one `<row>` per row, is a 2D numpy array rather than a class
        if matrix_row_element(xsd_type) is not None:
            self.uses_numpy = True
            return "npt.NDArray[np.float64]"

        # The keys are on the element, but they make the fields of its type's class `KeyedList`s
        list_keys: dict[str, str] = self.list_keys(element)

        # A named complex type has one class, shared by every element of that type, so their repeated child elements must have the same keys
        if xsd_type.local_name is not None:
            if xsd_type in self.class_names_by_complex_type:
                if list_keys != self.list_keys_by_complex_type[xsd_type]:
                    raise NotImplementedError(
                        f"{self.xsd_path.name}: two elements of type {xsd_type.local_name} have different keys, {self.list_keys_by_complex_type[xsd_type]} "
                        f"and {list_keys}, but share one class"
                    )
                return self.class_names_by_complex_type[xsd_type]
            type_documentation, _ = read_annotation(xsd_type)
            return self.add_class(xsd_type, pascal_case(xsd_type.local_name), type_documentation, list_keys)

        # An anonymous complex type's class is named after its element, and after the class holding the element too if that name is taken
        if element.local_name is None:
            raise NotImplementedError(f"{self.xsd_path.name}: an element with no name")
        class_name: str = pascal_case(element.local_name)
        if class_name in self.taken_names:
            class_name = parent_class_name + class_name
        if class_name in self.taken_names:
            raise ValueError(f"{self.xsd_path.name}: {class_name} is used for two different types")
        self.taken_names.add(class_name)
        return self.add_class(xsd_type, class_name, documentation, list_keys)

    def list_keys(self, element: XsdElement) -> dict[str, str]:
        """
        The keys of the repeated child elements of an element, which make their fields `KeyedList`s.

        A key is an `xs:key` or `xs:unique` on the element, which keeps a repeated child element unique by one of the child's own child elements, e.g. in `workflow.xsd`:
            <xs:key name="workflow_tree_name_key">
                <xs:selector xpath="tree"/>
                <xs:field xpath="tree_name"/>
            </xs:key>
        The key must be in every item, and be a string, so that it can select every item, and is never mistaken for a position.
        Any other identity constraint only constrains the values, e.g. `<xs:field xpath="."/>`, which keeps the values of a repeated simple element unique.

        :param element: An element of a complex type, e.g. `<xs:element name="workflow">`
        :return: The key of each repeated child element which has one, by the child element's name, e.g. `{"tree": "tree_name"}`
        """

        xsd_type: XsdComplexType | XsdSimpleType = element.type
        if not isinstance(xsd_type, XsdComplexType) or not isinstance(xsd_type.content, XsdGroup):
            return {}
        child_elements_by_name: dict[str, XsdElement] = {}
        for particle in xsd_type.content:
            if isinstance(particle, XsdElement) and particle.local_name is not None:
                child_elements_by_name[particle.local_name] = particle

        list_keys: dict[str, str] = {}
        for identity in element.identities:
            if not isinstance(identity, (XsdKey, XsdUnique)) or identity.selector is None:
                continue
            list_name: str = identity.selector.path  # e.g. "tree"
            field_paths: list[str] = []
            for identity_field in identity.fields:
                field_paths.append(identity_field.path)
            if list_name not in child_elements_by_name or len(field_paths) != 1:
                continue

            # A key is a child element of the items, which are a repeated element holding child elements
            list_element: XsdElement = child_elements_by_name[list_name]
            list_type: XsdComplexType | XsdSimpleType = list_element.type
            if list_element.max_occurs == 1 or not isinstance(list_type, XsdComplexType) or not isinstance(list_type.content, XsdGroup):
                continue
            key_element: XsdElement | None = None
            for particle in list_type.content:
                if isinstance(particle, XsdElement) and particle.local_name == field_paths[0]:
                    key_element = particle
            if key_element is None:
                continue

            key_type: XsdComplexType | XsdSimpleType = key_element.type
            if (
                key_element.min_occurs == 0
                or key_element.max_occurs != 1
                or key_element.nillable
                or not isinstance(key_type, XsdSimpleType)
                or self.simple_type(key_type) != "str"
            ):
                raise NotImplementedError(
                    f"{self.xsd_path.name}: `{list_name}` is keyed by `{field_paths[0]}`, which is not a single, required, string element. "
                    "A key must be in every item, and be a string, so that it can select every item, and is never mistaken for a position"
                )
            if list_name in list_keys:
                raise NotImplementedError(f"{self.xsd_path.name}: `{list_name}` has two keys, `{list_keys[list_name]}` and `{field_paths[0]}`")
            list_keys[list_name] = field_paths[0]

        return list_keys

    def simple_type(self, simple_type: XsdSimpleType) -> str:
        """
        :param simple_type: Simple type, named or anonymous, e.g. `xs:double`, a restriction of a type, or a list
        :return: Python type: a built-in type, `list[...]`, or a `typing.Literal` alias for an enumeration
        """

        if isinstance(simple_type, XsdAtomicBuiltin):
            builtin_name: str = str(simple_type.local_name)
            if builtin_name not in BUILTIN_TYPES:
                raise NotImplementedError(f"{self.xsd_path.name}: built-in type xs:{builtin_name}")
            return BUILTIN_TYPES[builtin_name]

        if isinstance(simple_type, XsdList):
            return f"list[{self.simple_type(simple_type.item_type)}]"

        if isinstance(simple_type, XsdAtomicRestriction):
            if simple_type.enumeration is not None:
                values: list[str] = []
                for value in simple_type.enumeration:
                    if not isinstance(value, str):
                        raise NotImplementedError(f"{self.xsd_path.name}: enumeration value {value!r}, which is not a string")
                    values.append(value)
                if simple_type.local_name is None:
                    return "typing.Literal[" + ", ".join(python_string(value) for value in values) + "]"
                alias_name: str = pascal_case(simple_type.local_name)
                for type_alias_definition in self.type_alias_definitions:
                    if type_alias_definition.name == alias_name:
                        return alias_name
                documentation, _ = read_annotation(simple_type)
                self.type_alias_definitions.append(TypeAliasDefinition(name=alias_name, documentation=documentation, values=values))
                return alias_name

            # Any other restriction, e.g. `minExclusive` or `pattern`, only constrains the values, so has the Python type of its base type
            if not isinstance(simple_type.base_type, XsdSimpleType):
                raise NotImplementedError(f"{self.xsd_path.name}: simple type {simple_type.local_name} restricts a complex type")
            return self.simple_type(simple_type.base_type)

        raise NotImplementedError(f"{self.xsd_path.name}: simple type {simple_type.local_name}")

    def add_class(self, complex_type: XsdComplexType, class_name: str, documentation: str, list_keys: dict[str, str]) -> str:
        """
        Add a class for a complex type, and for the complex types inside it.

        :param complex_type: Complex type, named or anonymous
        :param class_name: Name of the new class
        :param documentation: Docstring of the new class
        :param list_keys: The key of each repeated child element which has one, by the child element's name, e.g. `{"tree": "tree_name"}`; see `list_keys`
        :return: `class_name`
        """

        class_definition: ClassDefinition = ClassDefinition(name=class_name, documentation=documentation)
        # Added before its children, so the classes are written in the same order as the XML: the root first
        self.class_definitions.append(class_definition)
        self.class_names_by_complex_type[complex_type] = class_name
        self.list_keys_by_complex_type[complex_type] = list_keys

        # The child elements, in a `<xs:sequence>` or `<xs:all>`, then the attributes
        if not isinstance(complex_type.content, XsdGroup):
            raise NotImplementedError(f"{self.xsd_path.name}: {class_name} has text content")
        for particle in complex_type.content:
            if not isinstance(particle, XsdElement):
                raise NotImplementedError(f"{self.xsd_path.name}: a group inside a group, in {class_name}")
            key_name: str = list_keys.get(str(particle.local_name), "")
            class_definition.field_definitions.append(self.element_field(particle, class_name, key_name))
        for attribute in complex_type.attributes.values():
            if not isinstance(attribute, XsdAttribute):
                raise NotImplementedError(f"{self.xsd_path.name}: an attribute wildcard in {class_name}")
            class_definition.field_definitions.append(self.attribute_field(attribute))

        return class_name

    def element_field(self, element: XsdElement, class_name: str, key_name: str) -> FieldDefinition:
        """
        :param element: An element inside a complex type
        :param class_name: Class holding the field
        :param key_name: The key of the element, if it is a repeated element with one, e.g. "tree_name", which makes its field a `KeyedList`; otherwise ""
        :return: Field for the element
        """

        name: str = check_field_name(element.local_name, self.xsd_path)
        documentation, units = read_annotation(element)

        # A link to another XML file, e.g. `<grid href="grid.xml"/>`, has the type of the root element of the XML schema it names, e.g. `Grid`
        xsd_type: XsdComplexType | XsdSimpleType = element.type
        python_type: str
        is_link: bool = (
            isinstance(xsd_type, XsdComplexType)
            and isinstance(xsd_type.content, XsdGroup)
            and len(xsd_type.content) == 0
            and list(xsd_type.attributes) == ["href"]
        )
        if is_link:
            if len(element.identities) > 0:
                raise NotImplementedError(
                    f"{self.xsd_path.name}: the link `{name}` has an identity constraint, which belongs on the linked file's root element"
                )
            linked_schema: str = read_appinfo(element, "schema")
            if linked_schema == "":
                raise NotImplementedError(
                    f"{self.xsd_path.name}: the link `{name}` does not name the XML schema of the file it links to, in `<xs:appinfo><schema>`"
                )
            # Raises `ValueError` if the XML schema is outside the directory of XML schemas
            linked_xsd_relative_path: str = (self.xsd_path.parent / linked_schema).resolve().relative_to(self.schema_path).as_posix()
            if linked_xsd_relative_path not in self.root_class_names_by_xsd_path:
                raise ValueError(f"{self.xsd_path.name}: the link `{name}` names the XML schema {linked_xsd_relative_path}, which does not exist")
            python_type = self.root_class_names_by_xsd_path[linked_xsd_relative_path]
            linked_module_name: str = relative_module_name(self.xsd_relative_path, linked_xsd_relative_path)
            if python_type not in self.imported_module_names_by_class_name:
                if python_type in self.taken_names:
                    raise ValueError(f"{self.xsd_path.name}: {python_type}, from {linked_xsd_relative_path}, has the same name as a class of this XML schema")
                self.imported_module_names_by_class_name[python_type] = linked_module_name
                self.taken_names.add(python_type)
            elif self.imported_module_names_by_class_name[python_type] != linked_module_name:
                raise ValueError(f"{self.xsd_path.name}: {python_type} is the root element of two linked XML schemas")
        else:
            python_type = self.element_type(element, parent_class_name=class_name, documentation=documentation)

        # `max_occurs` is None for `maxOccurs="unbounded"`
        default: str = ""
        if element.max_occurs != 1:
            if element.nillable:
                raise NotImplementedError(f"{self.xsd_path.name}: {name} is both repeated and nillable")
            if key_name != "":
                # A repeated element with a key, which can be indexed by it as well as by position, e.g. `tree["ELMAG"]`.
                # The class of its items names the key in `KEY`
                item_class_definitions: list[ClassDefinition] = []
                for class_definition in self.class_definitions:
                    if class_definition.name == python_type:
                        item_class_definitions.append(class_definition)
                if len(item_class_definitions) != 1:
                    raise NotImplementedError(
                        f"{self.xsd_path.name}: `{name}` is keyed by `{key_name}`, but its items, {python_type}, are not a class of this XML schema"
                    )
                if item_class_definitions[0].key not in ("", key_name):
                    raise NotImplementedError(f"{self.xsd_path.name}: {python_type} is keyed by both `{item_class_definitions[0].key}` and `{key_name}`")
                item_class_definitions[0].key = key_name
                self.uses_keyed_list = True
                python_type = f"KeyedList[{python_type}]"
                if element.min_occurs == 0:
                    default = "field(default_factory=KeyedList)"
                if documentation != "":
                    documentation = documentation.removesuffix(".") + ".\n"
                documentation += f"A `KeyedList`, which can be indexed by `{key_name}` as well as by position, e.g. `{name}[{key_name}]`"
            else:
                python_type = f"list[{python_type}]"
                if element.min_occurs == 0:
                    default = "field(default_factory=list)"
        else:
            if element.nillable or element.min_occurs == 0:
                python_type = f"{python_type} | None"
            if element.min_occurs == 0:
                default = "None"

        return FieldDefinition(name=name, python_type=python_type, default=default, documentation=documentation, units=units)

    def attribute_field(self, attribute: XsdAttribute) -> FieldDefinition:
        """
        :param attribute: An attribute of a complex type, whose type is always a simple type
        :return: Field for the attribute
        """

        name: str = check_field_name(attribute.local_name, self.xsd_path)
        documentation, units = read_annotation(attribute)
        python_type: str = self.simple_type(attribute.type)
        default: str = ""
        if attribute.use != "required":
            python_type = f"{python_type} | None"
            default = "None"

        return FieldDefinition(name=name, python_type=python_type, default=default, documentation=documentation, units=units)

    def render(self) -> str:
        """
        :return: Source code of the module
        """

        xml_relative_path: str = Path(self.xsd_relative_path).with_suffix(".xml").as_posix()
        # The linked modules, sorted as `ruff` sorts the imports: the furthest away first
        imports: list[tuple[int, str, str]] = []  # (number of leading dots, module name, class name), e.g. (1, ".wall.wall", "WallSettings")
        for class_name, module_name in self.imported_module_names_by_class_name.items():
            n_dots: int = len(module_name) - len(module_name.lstrip("."))
            imports.append((-n_dots, module_name, class_name))
        imports.sort()

        lines: list[str] = [
            '"""',
            f"GSFit settings types for `{xml_relative_path}`",
            "",
            f"**WARNING** This file is autogenerated by `{GENERATOR_PATH}`.",
            "Any changes will be overwritten.",
            "",
            f"This module defines the types of `{xml_relative_path}`, generated from the XML schema `{SCHEMA_PATH}/{self.xsd_relative_path}`.",
            f"Its root element, `<{self.root_element_name}>`, is `{self.root_class_name}`.",
        ]
        if len(imports) > 0:
            lines.append(
                f"Each link to another XML file has the type of that file's root element, e.g. `{imports[0][2]}`, from `{imports[0][1]}`, "
                "as the linked file is read in the link's place."
            )
        lines.append('"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")

        # Import only what is used. Every module has a root element, whose `XSD_PATH` is a `typing.ClassVar`
        uses_field: bool = False
        for class_definition in self.class_definitions:
            for field_definition in class_definition.field_definitions:
                if field_definition.default.startswith("field("):
                    uses_field = True
        lines.append("import typing")
        lines.append("from dataclasses import dataclass")
        if uses_field:
            lines.append("from dataclasses import field")
        # Imported, rather than only for type checking, as `read_xml_settings_to_dataclasses` reads the type of each field.
        # As `ruff` sorts them: within the third-party imports, each `import ...` before any `from ... import ...`
        third_party_lines: list[str] = []
        if self.uses_numpy:
            third_party_lines.append("import numpy as np")
            third_party_lines.append("import numpy.typing as npt")
        if self.uses_keyed_list:
            third_party_lines.append("from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList")
        if len(third_party_lines) > 0:
            lines.append("")
            lines += third_party_lines
        if len(imports) > 0:
            lines.append("")
            for _, module_name, class_name in imports:
                lines.append(f"from {module_name} import {class_name}")

        # As `ruff` wants: one blank line between the imports and an assignment, two before a class
        for type_alias_definition in self.type_alias_definitions:
            lines += [""] + render_type_alias(type_alias_definition)
        for class_definition in self.class_definitions:
            lines += ["", ""] + render_class(class_definition)

        return "\n".join(lines) + "\n"


def module_relative_path(xsd_relative_path: str) -> Path:
    """
    :param xsd_relative_path: Path of an XML schema, relative to the directory of XML schemas, e.g. "GSFIT_code_settings.xsd" or "wall/wall.xsd"
    :return: Path of its module, relative to the directory of modules, in lower case as Python modules are, e.g. `gsfit_code_settings.py` or `wall/wall.py`
    """

    return Path(xsd_relative_path).parent / f"{Path(xsd_relative_path).stem.lower()}.py"


def relative_module_name(from_xsd_relative_path: str, to_xsd_relative_path: str) -> str:
    """
    :param from_xsd_relative_path: Path of the XML schema whose module imports, e.g. "GSFIT_code_settings.xsd"
    :param to_xsd_relative_path: Path of the XML schema whose module is imported, e.g. "wall/wall.xsd"
    :return: The relative module name to import from, e.g. ".wall.wall", or ".grid" from "equilibrium/equilibrium.xsd" to "equilibrium/grid.xsd"
    """

    from_package_parts: tuple[str, ...] = module_relative_path(from_xsd_relative_path).parent.parts
    to_module_parts: tuple[str, ...] = module_relative_path(to_xsd_relative_path).with_suffix("").parts

    # The directories both modules are in
    n_common_parts: int = 0
    for i_part in range(min(len(from_package_parts), len(to_module_parts) - 1)):
        if from_package_parts[i_part] != to_module_parts[i_part]:
            break
        n_common_parts += 1

    # One dot for the importing module's own directory, and one more for each directory up from it
    return "." * (1 + len(from_package_parts) - n_common_parts) + ".".join(to_module_parts[n_common_parts:])


def render_type_alias(type_alias_definition: TypeAliasDefinition) -> list[str]:
    """
    :param type_alias_definition: Enumeration
    :return: Lines of source code, e.g. `Method: typing.TypeAlias = typing.Literal[...]`
    """

    lines: list[str] = []
    for documentation_line in type_alias_definition.documentation.splitlines():
        lines.append(f"# {documentation_line}")
    if len(type_alias_definition.values) == 1:
        # A trailing comma would make a tuple, so a single value stays on one line
        lines.append(f"{type_alias_definition.name}: typing.TypeAlias = typing.Literal[{python_string(type_alias_definition.values[0])}]")
        return lines
    lines.append(f"{type_alias_definition.name}: typing.TypeAlias = typing.Literal[")
    for value in type_alias_definition.values:
        lines.append(f"    {python_string(value)},")
    lines.append("]")
    return lines


def render_class(class_definition: ClassDefinition) -> list[str]:
    """
    :param class_definition: Class
    :return: Lines of source code
    """

    docstring_lines: list[str] = []
    if class_definition.documentation != "":
        docstring_lines += class_definition.documentation.splitlines()
    parameter_lines: list[str] = []
    for field_definition in class_definition.field_definitions:
        parameter_text: str = " ".join(field_definition.documentation.split())
        if field_definition.units != "":
            parameter_text += f" [{field_definition.units}]"
        if parameter_text.strip() != "":
            parameter_lines.append(f":param {field_definition.name}: {parameter_text.strip()}")
    if len(parameter_lines) > 0:
        if len(docstring_lines) > 0:
            docstring_lines.append("")
        docstring_lines += parameter_lines

    # The class body is up to four blocks, separated by blank lines: the docstring, `XSD_PATH` for a root element's class,
    # `KEY` for the class of the items of a `KeyedList`, and the fields
    body_blocks: list[list[str]] = []
    if len(docstring_lines) > 0:
        docstring_block: list[str] = ['    """']
        for docstring_line in docstring_lines:
            docstring_block.append(f"    {escape_docstring(docstring_line)}" if docstring_line != "" else "")
        docstring_block.append('    """')
        body_blocks.append(docstring_block)
    if class_definition.xsd_path != "":
        body_blocks.append(
            [
                f"    # The XML schema of the XML files whose root element this is, relative to `{SCHEMA_PATH}/`",
                f"    XSD_PATH: typing.ClassVar[str] = {python_string(class_definition.xsd_path)}",
            ]
        )
    if class_definition.key != "":
        body_blocks.append(
            [
                "    # The field holding the key, which selects an item of this class from its `KeyedList`, as the XML schema keeps it unique",
                f"    KEY: typing.ClassVar[str] = {python_string(class_definition.key)}",
            ]
        )
    if len(class_definition.field_definitions) > 0:
        field_block: list[str] = []
        for field_definition in class_definition.field_definitions:
            line: str = f"    {field_definition.name}: {field_definition.python_type}"
            if field_definition.default != "":
                line += f" = {field_definition.default}"
            if field_definition.units != "":
                line += f"  # [{field_definition.units}]"
            field_block.append(line)
        body_blocks.append(field_block)

    lines: list[str] = ["@dataclass(kw_only=True)", f"class {class_definition.name}:"]
    if len(body_blocks) == 0:
        lines.append("    pass")
    for i_block in range(len(body_blocks)):
        if i_block > 0:
            lines.append("")
        lines += body_blocks[i_block]

    return lines


def render_init(package_relative_path: str, names: list[str]) -> str:
    """
    :param package_relative_path: Directory of the package, relative to the directory of modules, e.g. "." or "wall"
    :param names: The modules and directories in it, e.g. `["wall"]`
    :return: Source code of its `__init__.py`
    """

    title: str = "GSFit settings types"
    schema_directory: str = f"{SCHEMA_PATH}/"
    if package_relative_path != ".":
        title += f" for `{package_relative_path}/`"
        schema_directory += f"{package_relative_path}/"
    lines: list[str] = [
        '"""',
        title,
        "",
        f"**WARNING** This file is autogenerated by `{GENERATOR_PATH}`.",
        "Any changes will be overwritten.",
        "",
        f"This package holds one module per XML schema in `{schema_directory}`, laid out as the XML schemas are, each defining the types of one settings XML file.",
        '"""',
        "",
    ]
    for name in sorted(names):
        lines.append(f"from . import {name}")
    lines += ["", "__all__ = ["]
    for name in sorted(names):
        lines.append(f'    "{name}",')
    lines.append("]")
    return "\n".join(lines) + "\n"


def check_supported(schema_root: Element, xsd_path: Path) -> None:
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
            raise NotImplementedError(f"{xsd_path.name}: <xs:{tag}>")
        for attribute_name in node.attrib:
            if attribute_name not in SUPPORTED_XS_ATTRIBUTES[tag]:
                raise NotImplementedError(f"{xsd_path.name}: <xs:{tag} {attribute_name}=...>")


def read_annotation(component: XsdComponent) -> tuple[str, str]:
    """
    :param component: XML schema component, e.g. an element, attribute or type
    :return: (documentation, units), from its `<xs:annotation>`; either is "" when missing
    """

    documentation_lines: list[str] = []
    if component.annotation is not None:
        for documentation in component.annotation.documentation:
            for documentation_line in "".join(documentation.itertext()).splitlines():
                if documentation_line.strip() != "":
                    documentation_lines.append(documentation_line.strip())

    return "\n".join(documentation_lines), read_appinfo(component, "units")


def read_appinfo(component: XsdComponent, tag: str) -> str:
    """
    :param component: XML schema component, e.g. an element
    :param tag: A tag inside its `<xs:appinfo>`, e.g. "units" or "schema"
    :return: The text of that tag, e.g. "m" or "wall/wall.xsd", or "" when missing
    """

    text: str = ""
    if component.annotation is None:
        return text
    for appinfo in component.annotation.appinfo:
        for appinfo_element in appinfo.findall(tag):
            text = "".join(appinfo_element.itertext()).strip()
    return text


def check_field_name(name: str | None, xsd_path: Path) -> str:
    """
    :param name: Name of an XML element or attribute
    :param xsd_path: Path to the XML schema, for the message
    :return: `name`, once checked that it can be used as a Python field name
    """

    if name is None or not name.isidentifier() or keyword.iskeyword(name) or name in RESERVED_NAMES:
        raise NotImplementedError(f"{xsd_path.name}: {name} cannot be used as a Python field name")
    return name


def pascal_case(name: str) -> str:
    """
    :param name: e.g. `b_field_pol_probe`, `constraints__b_field_pol_probe`, or `WorkflowTree`
    :return: e.g. `BFieldPolProbe`, `ConstraintsBFieldPolProbe`, or `WorkflowTree`
    """

    words: list[str] = name.split("_")
    python_name: str = ""
    for word in words:
        python_name += word[:1].upper() + word[1:]
    return python_name


def python_string(value: str) -> str:
    """
    :param value: Text
    :return: Python string literal, in double quotes
    """

    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def escape_docstring(text: str) -> str:
    """
    :param text: One line of a docstring
    :return: The line, with backslashes and triple quotes escaped
    """

    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


if __name__ == "__main__":
    main()
