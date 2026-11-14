"""
IMAS IDS Rust Code Generator

Generates Rust struct definitions from IMAS Data Dictionary XSD schema files.
Includes accessor, view, and accumulator types for slicing Vec<T> fields.

Every input is an XSD schema: the Data Dictionary's schema of each IDS, and GSFit's custom keys for it in
`custom_keys/custom_<ids>_keys.xsd`. GSFit's settings are not kept in an IDS: their structs are generated from
their own XML schemas by `python/gsfit/settings_schema/_generate_rust_schema.py`.

Structures of GSFit's own which are not part of any IDS, e.g. the Greens tables in `greens_storage.xsd`, are
generated as an IDS is, so that they have the same accessors, into the crate which holds them; see `build_structure`.
"""

import inspect
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

# Location of this script: `rust/imas_rs/imas_updater/`
IMAS_UPDATER_DIR: Path = Path(__file__).resolve().parent
# The `imas_rs` crate root: `rust/imas_rs/`
CRATE_DIR: Path = IMAS_UPDATER_DIR.parent
# The cargo workspace root: `rust/`, which holds `rustfmt.toml`
WORKSPACE_DIR: Path = CRATE_DIR.parent
# Local clone of the IMAS Data Dictionary; see `README.md` for the pinned version
DATA_DICTIONARY_DIR: Path = IMAS_UPDATER_DIR / "IMAS-Data-Dictionary"
# The Data Dictionary's schema of the types shared by every IDS, e.g. the base type `FLT_1D` and the structure
# `identifier_static`. Each IDS's schema includes it, and so does the schema of each IDS's custom keys
DD_SUPPORT_XSD_FILE: Path = DATA_DICTIONARY_DIR / "schemas" / "utilities" / "dd_support.xsd"
# Rust edition of the `imas_rs` crate, needed so that `rustfmt` sorts imports correctly
RUST_EDITION: str = "2024"
# The XSD schemas of GSFit's hand-written, non-IMAS keys, merged into the generated structs; see
# `add_custom_keys_to_ids`. These live here, and not in `src/ids/`, because they are an input to this
# script rather than compiled Rust
CUSTOM_KEYS_DIR: Path = IMAS_UPDATER_DIR / "custom_keys"
# The repository root: `rust/` sits directly inside it
REPO_ROOT: Path = WORKSPACE_DIR.parent
# The XSD schema of the Greens tables, a structure of GSFit's own rather than part of any IDS, and the module `build_structure` generates
# from it, in the `gsfit_rs` crate, whose `Equilibrium` holds them: their structs, their Python bindings, and their pickling
GREENS_STORAGE_XSD_FILE: Path = IMAS_UPDATER_DIR / "greens_storage.xsd"
GREENS_STORAGE_RUST_FILE: Path = WORKSPACE_DIR / "gsfit_rs" / "src" / "greens_storage.rs"
# Where the generated Python type stub goes. This is the `gsfit_rs` Python package, which
# mounts the `imas_rs` classes as the `gsfit_rs.imas` submodule.
PYTHON_STUB_FILE: Path = REPO_ROOT / "python" / "gsfit_rs" / "imas.pyi"
# Plain-text record of the Data Dictionary version the generated files came from. This is the
# single source of truth for that version: `imas_rs::IMAS_DD_VERSION` includes it, the badge in
# the top-level `README.md` reads it, and `README.md` in this directory points at it.
IMAS_DD_VERSION_FILE: Path = CRATE_DIR / "src" / "imas_dd_version.txt"


# XML Schema namespace
XS_NS = "{http://www.w3.org/2001/XMLSchema}"


# Mapping from base types to their Array1 types for accumulators
SCALAR_TO_ARRAY = {
    "FLT_0D": ("f64", "Array1<f64>"),
    "INT_0D": ("i32", "Array1<i32>"),
    "STR_0D": ("String", "Vec<String>"),  # Strings are cloned
    "CPX_0D": ("Complex64", "Array1<Complex64>"),
}

# Whether a scalar's projection has to `.clone()` the value out when it is gathered across an array
# of structures: a `String` is not `Copy`. Every scalar type gathers through the one generic
# `Accumulator`, hand-written in `dd_base_types.rs` and parameterised by a projection closure, so
# the DD field path is emitted exactly once - inside the closure - instead of being baked into a
# generated per-field struct.
SCALAR_PROJECTION_NEEDS_CLONE: dict[str, bool] = {
    "FLT_0D": False,
    "INT_0D": False,
    "STR_0D": True,
    "CPX_0D": False,
}


@dataclass
class Field:
    """Represents a field (element) within a complex type."""

    name: str
    rust_type: str
    documentation: str = ""
    units: str = ""
    is_array: bool = False  # True if maxOccurs="unbounded"
    is_optional: bool = False

    @property
    def inner_type(self) -> str:
        """Get the inner type for Vec<T> fields."""
        match = re.match(r"Vec<(.+)>", self.rust_type)
        return match.group(1) if match else self.rust_type

    @property
    def is_scalar(self) -> bool:
        """Check if this field is a 0D scalar type (FLT_0D, INT_0D, ...).

        Only 0D leaves can be gathered across an array of structures, so this drives the
        accumulator machinery. Use `is_base_type` to ask whether a field is DD data rather
        than a nested structure.
        """
        return self.rust_type in SCALAR_TO_ARRAY

    @property
    def is_base_type(self) -> bool:
        """Check if this field is any data dictionary base type (FLT_0D, FLT_2D, INT_1D, ...)."""
        return self.rust_type in BASE_TYPES

    @property
    def is_struct(self) -> bool:
        """Check if this field is a struct (not a base type or Vec)."""
        if self.is_array:
            return False
        return not self.is_base_type and not self.rust_type.startswith("Vec<")


@dataclass
class ComplexType:
    """Represents an XSD complexType that maps to a Rust struct."""

    name: str
    fields: list[Field] = field(default_factory=list)
    documentation: str = ""


def snake_to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase for Rust struct names."""
    return "".join(word.capitalize() for word in name.split("_"))


def path_to_pascal_case(path: str) -> str:
    """Convert a DD field path ('global_quantities.magnetic_axis.r') to PascalCase."""
    return "".join(snake_to_pascal_case(part) for part in path.split("."))


def view_name_for(parent_type_name: str, field_path: str) -> str:
    """Name a nested view after the full DD field path it reads.

    Views and accumulators must be keyed on the *path*, not on the leaf name or
    the DD type name. The same leaf name occurs at many paths (`r` under
    `global_quantities.magnetic_axis`, `global_quantities.current_centre` and
    `boundary.closest_wall_point`), and the same DD type is reused at many paths
    (`EquilibriumConstraints0d` under both `constraints.b_field_tor_vacuum_r` and
    `constraints.ip`). Keying on either one made de-duplication collapse distinct
    paths onto whichever was generated first, so a gather through one path
    silently returned another path's values.
    """
    return f"{parent_type_name}{path_to_pascal_case(field_path)}View"


# Base types that map directly to Rust type aliases
BASE_TYPES = {
    "FLT_0D",
    "FLT_1D",
    "FLT_2D",
    "FLT_3D",
    "FLT_4D",
    "FLT_5D",
    "FLT_6D",
    "INT_0D",
    "INT_1D",
    "INT_2D",
    "INT_3D",
    "STR_0D",
    "STR_1D",
    "CPX_0D",
    "CPX_1D",
    "CPX_2D",
}


# Data dictionary simple types: `<kind>_type` is 0D, `<kind>_<n>d_type` is nD
SIMPLE_TYPE_PATTERN = re.compile(r"^(flt|int|str|cpx)(?:_(\d)d)?_type$")


def rust_type_from_simple_type(type_attr: str) -> Optional[str]:
    """Map a data dictionary simple type ("flt_type", "flt_2d_type") onto a base type alias.

    Returns None when the name is not a simple type at all, or names a dimensionality which
    `dd_base_types.rs` does not define. The caller then treats it as a reference to a complexType,
    which surfaces as a visible stub rather than as a silently wrong mapping.
    """
    match = SIMPLE_TYPE_PATTERN.match(type_attr)
    if match is None:
        return None
    kind, dimensions = match.groups()
    rust_type: str = f"{kind.upper()}_{dimensions or 0}D"
    return rust_type if rust_type in BASE_TYPES else None


def get_rust_type_from_group_ref(group_ref: str) -> str:
    """Map XSD group references to Rust type aliases."""
    if group_ref in BASE_TYPES:
        return group_ref
    return group_ref


def extract_documentation(element: ET.Element) -> str:
    """Extract documentation string from xs:annotation/xs:documentation.

    The documentation is dedented, so that a schema can indent it to line up with its element, as
    GSFit's schemas do: each line keeps only the indentation it has relative to the others, and the
    blank lines around it are removed. A blank line inside it, between two paragraphs, is kept.
    """
    annotation = element.find(f"{XS_NS}annotation")
    if annotation is not None:
        doc = annotation.find(f"{XS_NS}documentation")
        if doc is not None and doc.text:
            lines: list[str] = []
            for line in inspect.cleandoc(doc.text).split("\n"):
                lines.append(line.rstrip())
            return "\n".join(lines)
    return ""


def extract_units(element: ET.Element) -> str:
    """Extract units string from xs:annotation/xs:appinfo/units."""
    annotation = element.find(f"{XS_NS}annotation")
    if annotation is not None:
        appinfo = annotation.find(f"{XS_NS}appinfo")
        if appinfo is not None:
            units = appinfo.find("units")
            if units is not None and units.text:
                return units.text.strip()
    return ""


def extract_lifecycle_status(element: ET.Element) -> str:
    """Extract lifecycle_status from xs:annotation/xs:appinfo/lifecycle_status."""
    annotation = element.find(f"{XS_NS}annotation")
    if annotation is not None:
        appinfo = annotation.find(f"{XS_NS}appinfo")
        if appinfo is not None:
            status = appinfo.find("lifecycle_status")
            if status is not None and status.text:
                return status.text.strip()
    return ""


def is_deprecated(element: ET.Element) -> bool:
    """Check if an element is deprecated (obsolescent lifecycle status)."""
    return extract_lifecycle_status(element) in ("obsolescent", "deprecated")


def parse_element(element: ET.Element, known_types: set[str]) -> Optional[Field]:
    """Parse an xs:element and return a Field."""
    # Skip deprecated (obsolescent) nodes
    if is_deprecated(element):
        return None

    name = element.get("name")
    if not name:
        # Check for ref attribute
        ref_attr = element.get("ref")
        if ref_attr:
            name = ref_attr
        else:
            return None

    # Check if it's an array of structures.
    #
    # The data dictionary marks these either as `unbounded` or with an explicit upper
    # bound: `wall/description_2d` is `maxOccurs="3"` and `wall/.../limiter/unit` is
    # `maxOccurs="33"`, and both are arrays. Only the default `maxOccurs="1"` means a
    # single occurrence.
    max_occurs = element.get("maxOccurs", "1")
    is_array = max_occurs == "unbounded" or (max_occurs.isdigit() and int(max_occurs) > 1)

    # Get documentation and units
    documentation = extract_documentation(element)
    units = extract_units(element)

    # Determine the Rust type
    rust_type = None

    # Check for type attribute (reference to another complexType)
    type_attr = element.get("type")
    if type_attr:
        # A simple type ("flt_type", "flt_2d_type") maps onto a base type alias; anything else is
        # a reference to another complexType
        simple_rust_type: Optional[str] = rust_type_from_simple_type(type_attr)
        if simple_rust_type is not None:
            rust_type = simple_rust_type
        else:
            rust_type = snake_to_pascal_case(type_attr)
    else:
        # Check for inline complexType with group ref
        inline_complex = element.find(f"{XS_NS}complexType")
        if inline_complex is not None:
            group = inline_complex.find(f"{XS_NS}group")
            if group is not None:
                group_ref = group.get("ref")
                if group_ref:
                    rust_type = get_rust_type_from_group_ref(group_ref)

    # Check for ref attribute (reference to global element)
    ref_attr = element.get("ref")
    if ref_attr:
        # Skip some common refs that we'll handle specially
        if ref_attr in ("ids_properties", "time"):
            return None  # Skip these for now
        rust_type = snake_to_pascal_case(ref_attr)
        name = ref_attr

    if rust_type is None:
        rust_type = "Unknown"

    # Wrap in Vec if it's an array
    if is_array:
        rust_type = f"Vec<{rust_type}>"

    return Field(
        name=name,
        rust_type=rust_type,
        documentation=documentation,
        units=units,
        is_array=is_array,
    )


def parse_complex_type(ct_element: ET.Element, known_types: set[str]) -> ComplexType:
    """Parse an xs:complexType element and return a ComplexType."""
    name = ct_element.get("name", "")
    documentation = extract_documentation(ct_element)

    fields = []

    # Find the sequence element
    sequence = ct_element.find(f"{XS_NS}sequence")
    if sequence is not None:
        for elem in sequence.findall(f"{XS_NS}element"):
            field_obj = parse_element(elem, known_types)
            if field_obj:
                fields.append(field_obj)

    return ComplexType(
        name=name,
        fields=fields,
        documentation=documentation,
    )


def parse_xsd(xsd_path: Path) -> tuple[list[ComplexType], Optional[ComplexType]]:
    """
    Parse an XSD file and return all complex types and the root element type.

    Returns:
        Tuple of (list of ComplexTypes, root element ComplexType if found)
    """
    tree = ET.parse(xsd_path)
    root = tree.getroot()

    complex_types = []
    known_types = set()
    root_element_type = None

    # First pass: collect all type names
    for ct in root.findall(f"{XS_NS}complexType"):
        name = ct.get("name")
        if name:
            known_types.add(name)

    # Second pass: parse complex types
    for ct in root.findall(f"{XS_NS}complexType"):
        complex_type = parse_complex_type(ct, known_types)
        if complex_type.name:
            complex_types.append(complex_type)

    # Find root element (e.g., <xs:element name="equilibrium">)
    for elem in root.findall(f"{XS_NS}element"):
        elem_name = elem.get("name")
        if elem_name:
            # Parse inline complexType within the element
            inline_complex = elem.find(f"{XS_NS}complexType")
            if inline_complex is not None:
                root_element = parse_complex_type(inline_complex, known_types)
                root_element.name = elem_name
                root_element.documentation = extract_documentation(elem)
                root_element_type = root_element
                break

    return complex_types, root_element_type


def sanitize_rust_identifier(name: str) -> str:
    """Ensure the name is a valid Rust identifier."""
    # Rust reserved keywords
    keywords = {
        "type",
        "match",
        "move",
        "ref",
        "self",
        "super",
        "mod",
        "use",
        "pub",
        "fn",
        "struct",
        "enum",
        "impl",
        "trait",
        "for",
        "loop",
        "while",
        "if",
        "else",
        "return",
        "break",
        "continue",
        "const",
        "static",
        "mut",
        "as",
        "in",
        "where",
        "async",
        "await",
        "dyn",
    }
    if name in keywords:
        return f"r#{name}"
    return name


def sanitize_field_path(path: str) -> str:
    """Sanitize a field path like 'boundary.type' to 'boundary.r#type'."""
    parts = path.split(".")
    return ".".join(sanitize_rust_identifier(p) for p in parts)


def collect_used_types(complex_types: list[ComplexType], root_element: Optional[ComplexType]) -> tuple[set[str], set[str]]:
    """
    Collect all types used in the struct definitions.

    Returns:
        Tuple of (base_types used, complex_types referenced)
    """
    used_base_types = set()
    referenced_types = set()

    all_types = complex_types + ([root_element] if root_element else [])
    defined_types = {snake_to_pascal_case(ct.name) for ct in complex_types}
    if root_element:
        defined_types.add(snake_to_pascal_case(root_element.name))

    for ct in all_types:
        for f in ct.fields:
            # Extract base type from Vec<T> if needed
            match = re.match(r"Vec<(.+)>", f.rust_type)
            inner_type = match.group(1) if match else f.rust_type

            if inner_type in BASE_TYPES:
                used_base_types.add(inner_type)
            elif inner_type not in defined_types and inner_type != "Unknown":
                referenced_types.add(inner_type)

    return used_base_types, referenced_types


# =============================================================================
# View, Accessor, and Accumulator Generation
# =============================================================================


def find_vec_field_types(complex_types: list[ComplexType], root_element: Optional[ComplexType]) -> dict[str, ComplexType]:
    """
    Find all types that are used in Vec<T> fields.
    Returns a dict mapping type name (PascalCase) to its ComplexType definition.
    """
    all_types = complex_types + ([root_element] if root_element else [])
    type_map = {snake_to_pascal_case(ct.name): ct for ct in all_types}

    vec_types = {}
    for ct in all_types:
        for f in ct.fields:
            if f.is_array:
                inner_type = f.inner_type
                if inner_type in type_map:
                    vec_types[inner_type] = type_map[inner_type]

    return vec_types


def scalar_accumulator_field(f: Field, field_name: str, parent_type_name: str, field_path: str) -> tuple[str, str]:
    """Return the (declaration, initialiser) lines for one gathered scalar leaf.

    The leaf is a generic `Accumulator` carrying a projection closure, so the DD
    field path appears exactly once - in the closure, where the compiler checks it
    - rather than being encoded in a generated type name.

    Args:
        f: The scalar field.
        field_name: The Rust-safe field name (e.g. "r#type").
        parent_type_name: The array-of-structures element type (e.g. "EquilibriumTimeSlice").
        field_path: Path to the field from the parent (e.g. "global_quantities.magnetic_axis.r").
    """
    projection = f"|item: &{parent_type_name}| item.{sanitize_field_path(field_path)}"
    if SCALAR_PROJECTION_NEEDS_CLONE[f.rust_type]:
        projection += ".clone()"

    decl = f"    pub {field_name}: Accumulator<'a, {parent_type_name}, {f.rust_type}, D>,"
    init = f"            {field_name}: Accumulator::new(elements.clone(), {projection}),"
    return decl, init


def element_has_name(ct: ComplexType) -> bool:
    """Whether elements of this type have a string `name`, so that an accessor can select one by it.

    The data dictionary defines `name` as a "short string identifier (unique for a given device)".
    """
    for f in ct.fields:
        if f.name == "name" and f.rust_type == "STR_0D":
            return True
    return False


def nested_slice_method(f: Field, field_name: str, parent_type_name: str, field_path: str, element_is_named: bool) -> list[str]:
    """Return the method which selects from a nested array of structures from inside a view.

    This is what makes `description_2d(..).limiter.unit(..)` possible: `limiter` is a view over every
    `description_2d`, and `unit(..)` slices the `unit` array under each of them, adding one dimension
    after the `description_2d`'s: `description_2d(..).limiter.unit(..).resistivity` is
    `(n_description_2d, n_unit)`. Where the elements have a `name`, `unit(name)` instead takes the element
    with that name under every `description_2d`, adding no dimension:
    `description_2d(..).limiter.unit(name).resistivity` is `(n_description_2d)`.
    The method takes any `Selection`; see `dd_base_types.rs`.

    Args:
        f: The array-of-structures field.
        field_name: The Rust-safe field name.
        parent_type_name: The array-of-structures element type the view gathers over.
        field_path: Path to the field from the parent (e.g. "limiter.unit").
        element_is_named: Whether the field's elements have a `name`, so can be selected by it.
    """
    inner_type: str = f.inner_type
    lines: list[str] = [
        f"    /// Select from `{field_path}` under every element. A range, e.g. `.{field_name}(..)`, adds one dimension,",
        "    /// after those of the levels sliced above it and before the leaf's own.",
    ]
    if element_is_named:
        lines.append(f"    /// A name, `.{field_name}(name)`, takes the one element with that `name` under every element, and adds none.")
    lines += [
        "    ///",
        "    /// # Panics",
        "    /// If a range does not select the same number of elements under every element.",
    ]
    if element_is_named:
        lines.append("    /// If, under any element, no element or more than one has the name.")
    lines += [
        f"    pub fn {field_name}<S: Selection<'a, {inner_type}>>(&self, selection: S) -> S::NestedOutput<D> {{",
        "        selection.select_nested(",
        f"            {rust_string_literal(field_path)},",
        "            &self.slice_elements,",
        f"            |item: &{parent_type_name}| item.{sanitize_field_path(field_path)}.as_slice(),",
        "        )",
        "    }",
    ]
    return lines


# What every view already uses for itself: the private field holding the elements it gathers
# over, and the methods of a slice view. A data dictionary node with one of these names would
# collide with it, so the generator refuses rather than emitting code which does not compile.
VIEW_RESERVED_NAMES: set[str] = {"slice_elements", "new", "len", "is_empty", "shape", "iter"}


def collect_view_members(
    ct: ComplexType,
    parent_type_name: str,
    field_path_prefix: str,
    type_map: dict[str, ComplexType],
    generated_types: set[str],
    ancestry: tuple[str, ...],
) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Collect the members of one view: (field declarations, field initialisers, methods, nested view code).

    Shared by the slice view over an array of structures and the views over the structures nested
    inside it, which differ only in their wrapper. Every member reads through the view's
    `elements`, which are the same elements whichever member is used:
    * a scalar leaf becomes an `Accumulator`;
    * a nested structure becomes a nested view, generated recursively;
    * a nested array of structures becomes a method which slices it under every element, which is
      what lets `coil(..).element(..)` gather over two levels.

    `ancestry` carries the DD types currently being expanded, so a self-referential schema stops
    instead of recursing forever.
    """
    field_defs: list[str] = []
    field_inits: list[str] = []
    methods: list[list[str]] = []
    nested_code: list[str] = []

    for f in ct.fields:
        field_name: str = sanitize_rust_identifier(f.name)
        field_path: str = f"{field_path_prefix}.{f.name}" if field_path_prefix else f.name

        if f.name in VIEW_RESERVED_NAMES:
            raise ValueError(
                f"`{parent_type_name}.{field_path}` is named `{f.name}`, which a generated view already uses "
                f"for itself; rename the view's own member in `VIEW_RESERVED_NAMES` and the code which emits it"
            )

        if f.is_scalar:
            decl, init = scalar_accumulator_field(f, field_name, parent_type_name, field_path)
            field_defs.append(decl)
            field_inits.append(init)

        elif f.is_struct and f.rust_type in type_map:
            if f.rust_type in ancestry:
                # Self-referential DD type: stop expanding rather than recurse forever.
                continue
            # Nested struct - one view per field path (not per DD type)
            nested_view_name: str = view_name_for(parent_type_name, field_path)
            field_defs.append(f"    pub {field_name}: {nested_view_name}<'a, D>,")
            field_inits.append(f"            {field_name}: {nested_view_name}::new(elements.clone()),")

            nested_struct_code: str = generate_view_for_struct(
                f.rust_type,
                type_map[f.rust_type],
                parent_type_name,
                field_path,
                type_map,
                generated_types,
                ancestry + (f.rust_type,),
            )
            if nested_struct_code:
                nested_code.append(nested_struct_code)

        elif f.is_array and f.inner_type in type_map:
            methods.append(nested_slice_method(f, field_name, parent_type_name, field_path, element_has_name(type_map[f.inner_type])))

    return field_defs, field_inits, methods, nested_code


def emit_view(
    view_name: str,
    documentation: str,
    parent_type_name: str,
    field_defs: list[str],
    field_inits: list[str],
    methods: list[list[str]],
) -> list[str]:
    """Emit a view struct and its `impl`. `D` is the rank of what the view gathers."""
    lines: list[str] = []
    lines.append(f"/// {documentation}")
    lines.append(f"pub struct {view_name}<'a, D> {{")
    lines.extend(field_defs)
    lines.append(f"    slice_elements: Elements<'a, {parent_type_name}, D>,")
    lines.append("}")
    lines.append("")
    lines.append(f"impl<'a, D: Dimension> {view_name}<'a, D> {{")
    lines.append(f"    pub fn new(elements: Elements<'a, {parent_type_name}, D>) -> Self {{")
    lines.append("        Self {")
    lines.extend(field_inits)
    # Last, so that it moves once every other member has taken its clone
    lines.append("            slice_elements: elements,")
    lines.append("        }")
    lines.append("    }")
    for method in methods:
        lines.append("")
        lines.extend(method)
    lines.append("}")
    lines.append("")
    return lines


def generate_view_for_struct(
    struct_name: str,
    struct_ct: ComplexType,
    parent_type_name: str,
    field_path_prefix: str,
    type_map: dict[str, ComplexType],
    generated_types: set[str],
    ancestry: tuple[str, ...] = (),
) -> str:
    """
    Generate a View struct for a nested struct type, preceded by the views nested inside it.
    Uses generated_types to avoid creating duplicate type definitions.

    View names are keyed on the parent Vec element type *and the full field path*
    (see `view_name_for`), so a DD type reused at several paths gets one view per
    path rather than one shared view reading a single hard-coded path.

    `ancestry` carries the DD types currently being expanded, so a self-referential
    schema stops instead of recursing forever (path-keyed names grow with depth and
    so cannot terminate the recursion the way type-keyed names accidentally did).
    """
    view_name: str = view_name_for(parent_type_name, field_path_prefix)
    if view_name in generated_types:
        return ""

    generated_types.add(view_name)

    field_defs, field_inits, methods, nested_code = collect_view_members(struct_ct, parent_type_name, field_path_prefix, type_map, generated_types, ancestry)
    lines: list[str] = emit_view(
        view_name,
        f"View over `{field_path_prefix}` ({struct_name}) across multiple {parent_type_name}",
        parent_type_name,
        field_defs,
        field_inits,
        methods,
    )

    # Nested code first, then this view
    return "\n".join(nested_code + lines)


def generate_slice_view(
    type_name: str,
    ct: ComplexType,
    type_map: dict[str, ComplexType],
    generated_types: set[str],
) -> str:
    """
    Generate the SliceView for a type used in Vec<T>, preceded by the views nested inside it.
    Uses generated_types to avoid creating duplicate type definitions.
    """
    field_defs, field_inits, methods, nested_code = collect_view_members(ct, type_name, "", type_map, generated_types, (type_name,))

    common_methods: list[list[str]] = [
        [
            "    /// Total number of elements, across every sliced level",
            "    pub fn len(&self) -> usize {",
            "        self.slice_elements.len()",
            "    }",
        ],
        [
            "    pub fn is_empty(&self) -> bool {",
            "        self.slice_elements.is_empty()",
            "    }",
        ],
        [
            "    /// The length of each sliced level, in path order like the gathered arrays",
            "    pub fn shape(&self) -> Vec<usize> {",
            "        self.slice_elements.shape()",
            "    }",
        ],
        [
            "    /// Every element, in path order: the first array of structures on the path varies slowest",
            f"    pub fn iter(&self) -> impl Iterator<Item = &'a {type_name}> + '_ {{",
            "        self.slice_elements.iter()",
            "    }",
        ],
    ]

    lines: list[str] = emit_view(
        f"{type_name}SliceView",
        f"View over multiple {type_name} with field accumulation",
        type_name,
        field_defs,
        field_inits,
        common_methods + methods,
    )

    # What lets an accessor's `Selection` build this slice view, and, where the elements have a `name`,
    # select one by it; see `dd_base_types.rs`
    lines += [
        f"impl ArrayOfStructuresElement for {type_name} {{",
        f"    type SliceView<'a, D> = {type_name}SliceView<'a, D> where Self: 'a;",
        "",
        "    fn slice_view<'a, D: Dimension>(elements: Elements<'a, Self, D>) -> Self::SliceView<'a, D>",
        "    where",
        "        Self: 'a,",
        "    {",
        f"        {type_name}SliceView::new(elements)",
        "    }",
        "}",
        "",
    ]
    if element_has_name(ct):
        lines += [
            f"impl Named for {type_name} {{",
            "    fn name(&self) -> &str {",
            "        &self.name",
            "    }",
            "}",
            "",
        ]

    return "\n".join(nested_code + lines)


def generate_vec_field_impl(parent_ct: ComplexType, field: Field, element_is_named: bool) -> str:
    """
    Generate the impl block for a Vec<T> field: the `.field(selection)` accessor and `.field_len()`.

    The accessor takes any `Selection` (see `dd_base_types.rs`): a range gives the slice view over
    those elements, and, where the elements have a `name`, a name gives the element with that name.
    A single element is reached with ordinary indexing, `.field[i]`, so nothing is generated for
    it, and a single position is not a `Selection`, so there is no second spelling of `.field[i]`.
    """
    inner_type: str = field.inner_type
    field_name: str = sanitize_rust_identifier(field.name)
    parent_name: str = snake_to_pascal_case(parent_ct.name)

    lines = [
        f"impl {parent_name} {{",
        f"    /// Select from `{field.name}`. A range, e.g. `.{field_name}(0..2)` or `.{field_name}(..)`, gives the slice view over",
        "    /// those elements, whose leaves gather one value per element.",
    ]
    if element_is_named:
        lines.append(f"    /// A name, `.{field_name}(name)`, gives the one element with that `name`.")
    lines += [
        f"    /// A single element is `.{field_name}[i]`.",
        "    ///",
        "    /// # Panics",
        "    /// If a range is out of bounds.",
    ]
    if element_is_named:
        lines.append("    /// If no element, or more than one, has the name.")
    lines += [
        f"    pub fn {field_name}<'a, S: Selection<'a, {inner_type}>>(&'a self, selection: S) -> S::Output {{",
        f"        selection.select({rust_string_literal(field.name)}, &self.{field_name})",
        "    }",
        "",
        f"    /// Get the number of {field_name} elements",
        f"    pub fn {field_name}_len(&self) -> usize {{",
        f"        self.{field_name}.len()",
        "    }",
        "}",
        "",
    ]
    return "\n".join(lines)


def generate_all_views_and_accessors(complex_types: list[ComplexType], root_element: Optional[ComplexType]) -> str:
    """Generate all view, accessor, and accumulator code."""
    all_types = complex_types + ([root_element] if root_element else [])
    type_map = {snake_to_pascal_case(ct.name): ct for ct in all_types}

    # Find types used in Vec<T> fields
    vec_types = find_vec_field_types(complex_types, root_element)

    lines = []
    # Global tracking set to avoid generating duplicate types
    generated_types: set[str] = set()

    if vec_types:
        lines.append("// " + "=" * 76)
        lines.append("// View, Accessor, and Accumulator Types")
        lines.append("// " + "=" * 76)
        lines.append("")

        # Generate SliceView and related types for each Vec element type
        for type_name, ct in vec_types.items():
            lines.append(f"// --- {type_name} View Types ---")
            lines.append("")
            lines.append(generate_slice_view(type_name, ct, type_map, generated_types))

        # Generate impl blocks for structs with Vec fields
        lines.append("// " + "=" * 76)
        lines.append("// Struct Impl Blocks for Vec Field Access")
        lines.append("// " + "=" * 76)
        lines.append("")

        for ct in all_types:
            for f in ct.fields:
                if f.is_array and f.inner_type in vec_types:
                    lines.append(generate_vec_field_impl(ct, f, element_has_name(vec_types[f.inner_type])))

    return "\n".join(lines)


def generate_stub_types(referenced_types: set[str]) -> str:
    """Generate stub struct definitions for types that aren't defined locally."""
    lines = []
    if referenced_types:
        lines.append("// " + "=" * 76)
        lines.append("// Stub Types (from dd_support.xsd or other schemas)")
        lines.append("// " + "=" * 76)
        lines.append("")
        for type_name in sorted(referenced_types):
            lines.append(f"/// Stub type - TODO: implement from dd_support.xsd")
            lines.append("#[derive(Debug, Clone, Default)]")
            lines.append(f"pub struct {type_name} {{}}")
            lines.append("")
    return "\n".join(lines)


# What an unset scalar leaf holds. This is the IMAS convention: a float has NaN; an integer has
# no NaN, so IMAS reserves `EMPTY_INT` (-999999999); a string is unset when empty. It is what
# every leaf reads as until something writes it, in Rust and in Python alike, so there is no
# `Option` to unwrap and no `Some` to write - but a value must be tested with `is_nan()`,
# `== EMPTY_INT` or `is_empty()`, never against `None`.
#
# Anything not listed here - the nD arrays, `Vec<T>` arrays of structures, nested structures -
# is unset in its ordinary `Default` state, which is empty.
UNSET_SCALAR_VALUES: dict[str, str] = {
    "FLT_0D": "f64::NAN",
    "INT_0D": "EMPTY_INT",
    "CPX_0D": "CPX_0D::new(f64::NAN, f64::NAN)",
    "STR_0D": "String::new()",
}


def unset_value_for(f: Field) -> str:
    """The Rust expression the generated `Default` initialises a field to."""
    if f.rust_type in UNSET_SCALAR_VALUES:
        return UNSET_SCALAR_VALUES[f.rust_type]
    if f.rust_type.startswith("Vec<"):
        return "Vec::new()"
    if f.is_base_type:
        # `FLT_1D::default()` would resolve to ndarray's inherent `default(shape)` constructor
        # rather than the `Default` trait, so the trait is named explicitly
        return "Default::default()"
    return f"{f.rust_type}::default()"


def generate_rust_struct(ct: ComplexType) -> str:
    """Generate Rust struct code for a ComplexType."""
    lines = []

    # Documentation
    if ct.documentation:
        for line in ct.documentation.split("\n"):
            lines.append(f"/// {line.strip()}")

    # `Default` is derived unless a scalar leaf needs a sentinel: the derived default of a scalar
    # is `0.0` or `0`, and both are valid measurements, so such a struct gets an explicit `impl`
    # below which starts every leaf as its IMAS "unset" value (see `unset_value_for`). A struct
    # made only of arrays, strings and nested structures is unset in its derived state already,
    # and writing that impl out by hand would only draw `clippy::derivable_impls`.
    needs_manual_default: bool = any(f.rust_type in UNSET_SCALAR_VALUES and f.rust_type != "STR_0D" for f in ct.fields)

    # Struct definition
    struct_name = snake_to_pascal_case(ct.name)
    lines.append("#[derive(Debug, Clone)]" if needs_manual_default else "#[derive(Debug, Clone, Default)]")
    lines.append(f"pub struct {struct_name} {{")

    # Fields
    for f in ct.fields:
        if f.documentation:
            doc_lines = f.documentation.split("\n")
            for doc_line in doc_lines:
                lines.append(f"    /// {doc_line.strip()}")
        if f.units and f.units != "1":
            lines.append(f"    /// Units: {f.units}")

        field_name = sanitize_rust_identifier(f.name)
        # A key keeps the name it has where it comes from, e.g. `RAYON_NUM_THREADS` in the settings, so that its path is the same
        if f.name != f.name.lower():
            lines.append("    #[allow(non_snake_case)]")
        lines.append(f"    pub {field_name}: {f.rust_type},")

    lines.append("}")
    lines.append("")

    if not needs_manual_default:
        return "\n".join(lines)

    lines.append(f"impl Default for {struct_name} {{")
    lines.append("    fn default() -> Self {")
    lines.append("        Self {")
    for f in ct.fields:
        field_name = sanitize_rust_identifier(f.name)
        lines.append(f"            {field_name}: {unset_value_for(f)},")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_root_constructors(root_element: Optional[ComplexType], complex_types: list[ComplexType]) -> str:
    """Generate sizing constructors (`with_size`, `with_time`) for the root IDS.

    `with_size(n_time)` pre-populates the `time_slice` array with `n_time` default
    (all-unset) slices. `with_time(&[FLT_0D])` additionally sets each slice's
    `time` field. Only emitted when the root has a `Vec<T>` field named
    `time_slice` (the IMAS convention for a type-3 array of structures).
    """
    if root_element is None:
        return ""

    root_name = snake_to_pascal_case(root_element.name)

    time_slice_field = None
    for f in root_element.fields:
        if f.is_array and f.name == "time_slice":
            time_slice_field = f
            break
    if time_slice_field is None:
        return ""

    inner_type = time_slice_field.inner_type

    # Determine whether the time-slice struct has a scalar `time` field.
    type_map = {snake_to_pascal_case(ct.name): ct for ct in complex_types}
    inner_ct = type_map.get(inner_type)
    has_time_field = inner_ct is not None and any(f.name == "time" and f.is_scalar for f in inner_ct.fields)

    lines = [
        "// " + "=" * 76,
        f"// {root_name} Constructors",
        "// " + "=" * 76,
        "",
        f"impl {root_name} {{",
        f"    /// Create a `{root_name}` pre-populated with `n_time` default (empty) time slices.",
        f"    ///",
        f"    /// Every leaf field in each slice is unset (NaN, `EMPTY_INT`, or empty), ready to be",
        f"    /// filled in,",
        f"    /// e.g. via `time_slice.par_iter_mut()`.",
        f"    pub fn with_size(n_time: usize) -> Self {{",
        f"        Self {{",
        f"            time_slice: (0..n_time).map(|_| {inner_type}::default()).collect(),",
        f"            ..Self::default()",
        f"        }}",
        f"    }}",
    ]

    if has_time_field:
        lines.extend(
            [
                f"",
                f"    /// Create a `{root_name}` with one time slice per entry in `time`,",
                f"    /// setting each slice's `time` field. All other leaf fields are unset.",
                f"    pub fn with_time(time: &FLT_1D) -> Self {{",
                f"        let mut ids = Self::with_size(time.len());",
                f"        ids.allocate_time_slices(time);",
                f"        ids",
                f"    }}",
                f"",
                f"    /// Allocate one default (empty) time slice per entry in `time`, setting each slice's",
                f"    /// `time` field.",
                f"    ///",
                f"    /// Unlike `with_time`, this works in place and leaves the rest of the IDS alone, so an",
                f"    /// IDS which already carries data - `code`, `vacuum_toroidal_field`, ... - keeps it.",
                f"    /// Any time slices already present are replaced.",
                f"    pub fn allocate_time_slices(&mut self, time: &FLT_1D) {{",
                f"        self.time_slice = (0..time.len()).map(|_| {inner_type}::default()).collect();",
                f"        for (slice, &t) in self.time_slice.iter_mut().zip(time.iter()) {{",
                f"            slice.time = t;",
                f"        }}",
                f"    }}",
            ]
        )

    lines.append(f"}}")
    lines.append(f"")

    return "\n".join(lines)


def parse_supporting_schemas(xsd_path: Path) -> dict[str, ComplexType]:
    """Parse the schemas pulled in by `<xs:include>`, returning every complexType they define.

    The data dictionary keeps its shared structures - identifiers, R/Z positions, the generic grid -
    in `dd_support.xsd`, which every IDS schema includes. Types are returned keyed by their Rust
    (PascalCase) name, and only the ones actually referenced are emitted; see
    `resolve_supporting_types`.
    """
    tree = ET.parse(xsd_path)
    root = tree.getroot()

    supporting_types: dict[str, ComplexType] = {}
    for include_element in root.findall(f"{XS_NS}include"):
        schema_location: Optional[str] = include_element.get("schemaLocation")
        if schema_location is None:
            continue
        included_path: Path = (xsd_path.parent / schema_location).resolve()
        if not included_path.is_file():
            raise FileNotFoundError(f"{xsd_path.name} includes {schema_location}, which does not exist")

        included_root = ET.parse(included_path).getroot()
        known_types: set[str] = {ct.get("name") for ct in included_root.findall(f"{XS_NS}complexType") if ct.get("name")}
        for ct_element in included_root.findall(f"{XS_NS}complexType"):
            complex_type: ComplexType = parse_complex_type(ct_element, known_types)
            if complex_type.name:
                supporting_types[snake_to_pascal_case(complex_type.name)] = complex_type

        # Some shared structures are declared as a global element wrapping an inline complexType
        # rather than as a named complexType - `code` is one - so they are collected separately
        for element in included_root.findall(f"{XS_NS}element"):
            element_name: Optional[str] = element.get("name")
            inline_complex = element.find(f"{XS_NS}complexType")
            if element_name is None or inline_complex is None:
                continue
            complex_type = parse_complex_type(inline_complex, known_types)
            complex_type.name = element_name
            complex_type.documentation = extract_documentation(element)
            supporting_types[snake_to_pascal_case(element_name)] = complex_type

        print(f"Parsed {len(supporting_types)} supporting types from {included_path.name}")

    return supporting_types


def resolve_supporting_types(
    complex_types: list[ComplexType],
    root_element: Optional[ComplexType],
    supporting_types: dict[str, ComplexType],
) -> list[str]:
    """Pull in the supporting types the IDS references, transitively, in place.

    A supporting type may reference further supporting types, so this repeats until nothing new is
    found. Only reachable types are pulled in - `dd_support.xsd` defines far more than any one IDS
    uses. Anything still unresolved afterwards is emitted as a stub, as before.
    """
    defined_type_names: set[str] = {snake_to_pascal_case(ct.name) for ct in complex_types}
    if root_element is not None:
        defined_type_names.add(snake_to_pascal_case(root_element.name))

    pulled_type_names: list[str] = []
    n_pass_max: int = 100
    for _i_pass in range(n_pass_max):
        newly_pulled: list[str] = []
        all_types: list[ComplexType] = complex_types + ([root_element] if root_element else [])
        for ct in all_types:
            for f in ct.fields:
                inner_type: str = f.inner_type
                if inner_type in defined_type_names or inner_type in BASE_TYPES or inner_type == "Unknown":
                    continue
                if inner_type in supporting_types:
                    complex_types.append(supporting_types[inner_type])
                    defined_type_names.add(inner_type)
                    newly_pulled.append(inner_type)
        pulled_type_names.extend(newly_pulled)
        if not newly_pulled:
            break
    else:
        raise RuntimeError(f"resolve_supporting_types did not settle within {n_pass_max} passes")

    return pulled_type_names


def add_custom_keys_to_ids(
    complex_types: list[ComplexType],
    root_element: ComplexType | None,
    supporting_types: dict[str, ComplexType],
    ids_name: str,
    path_to_custom_keys_file: Path,
) -> None:
    """Merge GSFit's custom (non-IMAS) keys, declared in the XSD schema `path_to_custom_keys_file`, into the IDS's structs, in place.

    Rust cannot add a field to a struct declared in another file, so custom keys that must sit flat alongside the IMAS ones
    (`profiles_2d.d_psi_d_r`, not `profiles_2d.custom.d_psi_d_r`) have to be merged in before the structs are emitted.

    The schema is laid out as the data dictionary's own schema of the IDS, `dd_<ids>.xsd`, is, and holds only GSFit's additions:
    * a top-level `xs:complexType` named as one of `dd_<ids>.xsd` adds its elements to it, e.g. `equilibrium_profiles_2d`;
    * a top-level `xs:complexType` with any other name declares a new structure, which some key must have as its type;
    * the top-level `xs:element` named as the IDS adds its elements to the IDS's root;
    * inside any of these, an element named as one the data dictionary already has there, holding an `xs:complexType`, adds that
      complexType's elements to the existing element's structure, which is one declared in `dd_support.xsd`, e.g. `code`.
    The schema includes `dd_support.xsd`, so that every type it uses is declared and it is a valid schema, which is also why it cannot
    declare one of `dd_support.xsd`'s structures again to add keys to it. A structure is one struct wherever it is used, so a key added
    to it is on every element of its type. The data dictionary's structures keep its documentation, so a note on the keys added to one is
    an XML comment.

    The keys use the data dictionary's types, written as it writes its own: a leaf holds `<xs:group ref="FLT_1D"/>`, or another base
    type, and a structure names a data dictionary complexType in `type`, e.g. `identifier_static`, or a new structure. Anything else,
    e.g. `xs:double` or a simple type of the schema's own, is rejected. Each key is then read by `parse_element`, exactly as the data
    dictionary's own keys are, documentation and units included, and an unset key reads as the IMAS empty value (NaN, `EMPTY_INT`, or
    empty), just like they do.

    An array of structures has `minOccurs="0"`, as it can be empty, which in IMAS any array can; the data dictionary's own schemas never
    write `minOccurs`, so read as if each held at least one. A leaf or a single structure has no `minOccurs`: like the data dictionary's,
    it is always there, a leaf reading as unset when it has no value. `minOccurs` changes nothing generated: an array is a `Vec` either way.

    Missing file is not an error: an IDS need not have any custom keys.

    Args:
        complex_types: The IDS's structs, which the new structures are added to
        root_element: The IDS's root struct
        supporting_types: Every structure of `dd_support.xsd`, keyed by its Rust name
        ids_name: The IDS's name, e.g. "equilibrium"
        path_to_custom_keys_file: The XSD schema of the custom keys, e.g. `custom_keys/custom_equilibrium_keys.xsd`
    """
    if not path_to_custom_keys_file.is_file():
        return

    file_name: str = path_to_custom_keys_file.name
    schema_root: ET.Element = ET.parse(path_to_custom_keys_file).getroot()

    # Every structure a key can have as its type, by its name in the schemas: those of `dd_support.xsd`, the IDS's own, and the new ones
    # below. The IDS's own are the structs which are not from `dd_support.xsd`, as there is nothing else before the custom keys
    support_types_by_name: dict[str, ComplexType] = {}
    for supporting_type in supporting_types.values():
        support_types_by_name[supporting_type.name] = supporting_type
    ids_types_by_name: dict[str, ComplexType] = {}
    for ct in complex_types:
        if ct.name not in support_types_by_name:
            ids_types_by_name[ct.name] = ct
    types_by_name: dict[str, ComplexType] = dict(support_types_by_name)
    types_by_name.update(ids_types_by_name)

    # The Rust name of every struct, so that a new structure is not named as one which already exists
    struct_names: set[str] = set()
    for ct in types_by_name.values():
        struct_names.add(snake_to_pascal_case(ct.name))
    if root_element is not None:
        struct_names.add(snake_to_pascal_case(root_element.name))

    # Each complexType declaring keys, with the struct they are added to and their path in the schema, for the error messages. Every
    # new structure is found before any key is read, so that a key can have as its type a structure declared after it
    additions: list[tuple[ET.Element, ComplexType, str]] = []
    new_types: list[ComplexType] = []
    complex_type_names: set[str] = set()
    n_includes: int = 0
    for component in schema_root:
        component_name: str | None = component.get("name")
        if component.tag == f"{XS_NS}complexType" and component_name is not None:
            if component_name in complex_type_names:
                raise ValueError(f"{file_name} declares `{component_name}` twice")
            complex_type_names.add(component_name)

        if component.tag == f"{XS_NS}include":
            included_file_path: Path = (path_to_custom_keys_file.parent / component.get("schemaLocation", "")).resolve()
            if included_file_path != DD_SUPPORT_XSD_FILE.resolve():
                raise ValueError(f"{file_name} includes {included_file_path}, but only the data dictionary's `dd_support.xsd` is included, for its types")
            n_includes += 1
        elif component.tag == f"{XS_NS}complexType" and component_name in ids_types_by_name:
            if component.find(f"{XS_NS}annotation") is not None:
                raise ValueError(
                    f"{file_name}: `{component_name}` is the data dictionary's, and so is its documentation. Write a note on the keys added to it as an XML comment"
                )
            additions.append((component, ids_types_by_name[component_name], component_name))
        elif component.tag == f"{XS_NS}complexType" and component_name is not None:
            struct_name: str = snake_to_pascal_case(component_name)
            if component_name in support_types_by_name:
                raise ValueError(
                    f"{file_name} declares `{component_name}`, which `dd_support.xsd` declares already. Add keys to it inside an element of its type: "
                    "an element named as that element, holding an `xs:complexType` of the keys"
                )
            if struct_name in struct_names:
                raise ValueError(
                    f"{file_name} declares the new structure `{component_name}`, but its struct, `{struct_name}`, is already that of a data dictionary structure. "
                    "To add keys to that structure, name it exactly as the data dictionary does"
                )
            documentation: str = extract_documentation(component)
            if documentation == "":
                raise ValueError(f"{file_name}: the new structure `{component_name}` has no `xs:documentation`")
            new_type: ComplexType = ComplexType(name=component_name, documentation=f"{documentation}\nCustom (non-IMAS) structure, declared in {file_name}")
            complex_types.append(new_type)
            types_by_name[component_name] = new_type
            struct_names.add(struct_name)
            new_types.append(new_type)
            additions.append((component, new_type, component_name))
            print(f"  Custom struct: {struct_name}")
        elif component.tag == f"{XS_NS}element" and root_element is not None and component_name == root_element.name:
            root_complex_type: ET.Element | None = component.find(f"{XS_NS}complexType")
            if root_complex_type is None or len(component.attrib) != 1:
                raise NotImplementedError(f"{file_name}: <{component_name}> adds keys to the IDS's root, so holds only an `xs:complexType` of them")
            if component.find(f"{XS_NS}annotation") is not None or root_complex_type.find(f"{XS_NS}annotation") is not None:
                raise ValueError(
                    f"{file_name}: <{component_name}> is the IDS's root, whose documentation is the data dictionary's. Write a note on the keys added to it as an XML comment"
                )
            additions.append((root_complex_type, root_element, component_name))
        else:
            raise NotImplementedError(
                f"{file_name} holds the `{component.tag.removeprefix(XS_NS)}` `{component_name}`, but only the `xs:include` of `dd_support.xsd`, `xs:complexType`s, "
                "and the `xs:element` named as the IDS are handled: the custom keys use the data dictionary's types, rather than types of their own"
            )
    if n_includes != 1:
        dd_support_relative_path: str = os.path.relpath(DD_SUPPORT_XSD_FILE, path_to_custom_keys_file.parent)
        raise ValueError(f'{file_name} must include the data dictionary\'s types, with `<xs:include schemaLocation="{dd_support_relative_path}"/>`, once')

    custom_keys: set[tuple[str, str]] = set()
    for complex_type_element, struct, path in additions:
        add_custom_keys_to_struct(
            complex_type_element=complex_type_element,
            struct=struct,
            path=path,
            types_by_name=types_by_name,
            support_types_by_name=support_types_by_name,
            custom_keys=custom_keys,
            path_to_custom_keys_file=path_to_custom_keys_file,
        )

    # A new structure which no key has as its type is almost certainly a mistyped name
    referenced_struct_names: set[str] = set()
    for ct in complex_types + ([root_element] if root_element else []):
        for f in ct.fields:
            referenced_struct_names.add(f.inner_type)
    for new_type in new_types:
        if snake_to_pascal_case(new_type.name) not in referenced_struct_names:
            raise ValueError(
                f"{file_name} declares `{new_type.name}`, which is neither a data dictionary structure nor the type of any custom key. If it was meant to add "
                "keys to a data dictionary structure, check the spelling; if it is a new structure, add a key of its type"
            )


def add_custom_keys_to_struct(
    complex_type_element: ET.Element,
    struct: ComplexType,
    path: str,
    types_by_name: dict[str, ComplexType],
    support_types_by_name: dict[str, ComplexType],
    custom_keys: set[tuple[str, str]],
    path_to_custom_keys_file: Path,
) -> None:
    """Add the custom keys declared in `complex_type_element` to the struct `struct`, in place; see `add_custom_keys_to_ids`.

    Args:
        complex_type_element: The `xs:complexType` declaring the keys, in an `xs:sequence`
        struct: The struct the keys are added to
        path: Where the keys are in the custom keys' schema, e.g. `equilibrium/code`, for the error messages
        types_by_name: Every structure a key can have as its type, by its name in the schemas: those of `dd_support.xsd`, the IDS's own,
            and the new ones
        support_types_by_name: Every structure of `dd_support.xsd`, by its name in the schemas
        custom_keys: The custom keys added so far, as (the name of the struct, the name of the key), which those added here are added to
        path_to_custom_keys_file: The XSD schema of the custom keys
    """
    file_name: str = path_to_custom_keys_file.name

    sequences: list[ET.Element] = []
    for component in complex_type_element:
        if component.tag == f"{XS_NS}annotation":
            continue
        if component.tag != f"{XS_NS}sequence":
            raise NotImplementedError(f"{file_name}: `{path}` holds an `{component.tag.removeprefix(XS_NS)}`, but only an `xs:sequence` of elements is handled")
        sequences.append(component)
    if len(sequences) != 1:
        raise NotImplementedError(f"{file_name}: `{path}` holds {len(sequences)} `xs:sequence`s of elements, rather than one")

    for element in sequences[0]:
        if element.tag != f"{XS_NS}element":
            raise NotImplementedError(f"{file_name}: `{path}` holds an `{element.tag.removeprefix(XS_NS)}`, but only elements are handled")
        key_name: str | None = element.get("name")
        if key_name is None:
            raise NotImplementedError(f"{file_name}: `{path}` holds an element with no `name`, e.g. an `xs:element ref`, which is not handled")
        key_path: str = f"{path}/{key_name}"
        if (struct.name, key_name) in custom_keys:
            raise ValueError(f"{file_name} declares `{key_path}` twice")

        type_name: str | None = element.get("type")
        inline_complex_type: ET.Element | None = element.find(f"{XS_NS}complexType")

        # An element the data dictionary already has here is not a key: holding an `xs:complexType` of keys, it adds them to its structure
        existing_field: Field | None = None
        for f in struct.fields:
            if f.name == key_name:
                existing_field = f
        if existing_field is not None:
            if inline_complex_type is None or inline_complex_type.find(f"{XS_NS}sequence") is None:
                raise ValueError(
                    f"{file_name} declares the custom key `{key_path}`, but the IMAS data dictionary already defines it. Remove the custom key and use the IMAS one"
                )
            # A structure of the IDS's own schema has its keys added by a top-level `xs:complexType` of its name instead, so that there is one way
            nested_struct: ComplexType | None = None
            for support_type in support_types_by_name.values():
                if snake_to_pascal_case(support_type.name) == existing_field.inner_type:
                    nested_struct = support_type
            if nested_struct is None:
                existing_type_name: str = existing_field.inner_type
                for ct in types_by_name.values():
                    if snake_to_pascal_case(ct.name) == existing_field.inner_type:
                        existing_type_name = ct.name
                raise ValueError(
                    f"{file_name}: `{key_path}` adds keys to the data dictionary's `{key_name}`, whose type, `{existing_type_name}`, the IDS's own "
                    f'schema declares rather than `dd_support.xsd`. Add them with a top-level `<xs:complexType name="{existing_type_name}">` instead'
                )
            if len(element.attrib) != 1 or element.find(f"{XS_NS}annotation") is not None or inline_complex_type.find(f"{XS_NS}annotation") is not None:
                raise ValueError(
                    f"{file_name}: `{key_path}` adds keys to the data dictionary's `{key_name}`, so holds only an `xs:complexType` of them: the rest, "
                    "documentation included, is the data dictionary's. Write a note on the keys added to it as an XML comment"
                )
            add_custom_keys_to_struct(
                complex_type_element=inline_complex_type,
                struct=nested_struct,
                path=key_path,
                types_by_name=types_by_name,
                support_types_by_name=support_types_by_name,
                custom_keys=custom_keys,
                path_to_custom_keys_file=path_to_custom_keys_file,
            )
            continue

        for attribute_name in element.attrib:
            if attribute_name not in ("name", "type", "minOccurs", "maxOccurs"):
                raise NotImplementedError(
                    f"{file_name}: `{key_path}` has the attribute `{attribute_name}`, but only `name`, `type`, `minOccurs` and `maxOccurs` are handled"
                )
        if extract_documentation(element) == "":
            raise ValueError(f"{file_name}: the custom key `{key_path}` has no `xs:documentation`")

        # Settings are not kept in an IDS, as an IDS can also be made by another code
        if element.find(f"{XS_NS}annotation/{XS_NS}appinfo/schema") is not None:
            raise NotImplementedError(
                f"{file_name}: `{key_path}` names the XML schema of settings in `xs:appinfo/schema`, but settings are not kept in an IDS: their "
                "structs are generated by `python/gsfit/settings_schema/_generate_rust_schema.py`, and kept with the Rust class they configure"
            )
        new_field: Field | None
        if inline_complex_type is not None:
            # A leaf, written as the data dictionary writes its own: `<xs:complexType><xs:group ref="FLT_1D"/></xs:complexType>`
            leaf_components: list[ET.Element] = list(inline_complex_type)
            if (
                type_name is not None
                or len(leaf_components) != 1
                or leaf_components[0].tag != f"{XS_NS}group"
                or leaf_components[0].get("ref") not in BASE_TYPES
            ):
                raise ValueError(
                    f'{file_name}: `{key_path}` is a leaf, so holds only `<xs:group ref="..."/>`, naming one of the data dictionary\'s base types: '
                    f"{', '.join(sorted(BASE_TYPES))}. A new structure is a top-level `xs:complexType`, which the key names in `type`"
                )
            if "maxOccurs" in element.attrib:
                raise ValueError(
                    f"{file_name}: `{key_path}` is a leaf, which is not repeated. Give it a base type of one more dimension instead, e.g. `FLT_1D` for `FLT_0D`"
                )
            if "minOccurs" in element.attrib:
                raise ValueError(f"{file_name}: `{key_path}` is a leaf, which is always there, reading as unset when it has no value, so has no `minOccurs`")
            new_field = parse_element(element, set())
        elif type_name is not None:
            # A structure, or an array of structures: one of the data dictionary's, e.g. `identifier_static`, or a new one
            if type_name not in types_by_name:
                raise ValueError(
                    f"{file_name}: `{key_path}` has the type `{type_name}`, which is neither a data dictionary structure, e.g. `identifier_static`, nor a new one "
                    f'declared in {file_name}. A leaf holds `<xs:group ref="FLT_1D"/>`, or another of the data dictionary\'s base types, rather than a type of its own'
                )
            new_field = parse_element(element, set())
        else:
            raise ValueError(f'{file_name}: the custom key `{key_path}` has neither a `type` nor an `xs:complexType` holding `<xs:group ref="..."/>`')
        if new_field is None:
            raise ValueError(f"{file_name}: the custom key `{key_path}` is marked as obsolescent; remove it instead")
        # An array of structures can be empty, which its schema says, unlike the data dictionary's; a single structure is always there
        if type_name is not None and new_field.is_array and element.get("minOccurs") != "0":
            raise ValueError(f'{file_name}: `{key_path}` is an array of structures, which can be empty, so has `minOccurs="0"`')
        if type_name is not None and not new_field.is_array and "minOccurs" in element.attrib:
            raise ValueError(f"{file_name}: `{key_path}` is a single structure, which is always there, so has no `minOccurs`")

        struct.fields.append(new_field)
        custom_keys.add((struct.name, key_name))
        print(f"  Custom key: {snake_to_pascal_case(struct.name)}.{key_name}: {new_field.rust_type}")


# ============================================================================
# Python Path Bindings
#
# Two files are generated from the same walk of the schema, so they cannot drift
# apart:
#   * `src/python/<ids>_paths.rs` - the static description the Rust navigator walks,
#     with a reader function per leaf.
#   * `python/gsfit_rs/imas.pyi` - the type stub, which is what gives editors their
#     completions and lets `mypy` check both attribute names and the type `get`
#     returns.
# ============================================================================


# The Rust type each data dictionary base type projects to, and the Python type it
# reads back as. The Python column has two entries: the first is used when every
# array-of-structures level on the path was indexed with an integer (so one value
# comes back), the second when at least one level was sliced (so values are gathered).
# A gathered leaf is always a numpy array, whose rank grows with the number of sliced
# levels, which is why one type serves every depth; strings included.
#
# A base type missing from here is rejected rather than emitted, because the Rust
# side needs a matching `Gatherable` implementation and a silent omission would
# only show up as a compile error in generated code.
PYTHON_LEAF_TYPES: dict[str, tuple[str, str]] = {
    "FLT_0D": ("float", "npt.NDArray[np.float64]"),
    "FLT_1D": ("npt.NDArray[np.float64]", "npt.NDArray[np.float64]"),
    "FLT_2D": ("npt.NDArray[np.float64]", "npt.NDArray[np.float64]"),
    "FLT_3D": ("npt.NDArray[np.float64]", "npt.NDArray[np.float64]"),
    "FLT_4D": ("npt.NDArray[np.float64]", "npt.NDArray[np.float64]"),
    "INT_0D": ("int", "npt.NDArray[np.int32]"),
    "INT_1D": ("npt.NDArray[np.int32]", "npt.NDArray[np.int32]"),
    "INT_2D": ("npt.NDArray[np.int32]", "npt.NDArray[np.int32]"),
    "STR_0D": ("str", "npt.NDArray[np.str_]"),
    "STR_1D": ("list[str]", "npt.NDArray[np.str_]"),
}

# Names that cannot be written as an attribute in a `.pyi`, which would produce a
# stub that does not parse.
PYTHON_KEYWORDS: set[str] = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}


@dataclass
class PathNode:
    """One node in the data dictionary, as reached by a Python path."""

    name: str  # the data dictionary name, e.g. "global_quantities"
    documentation: str
    units: str
    kind: str  # "structure" | "array_of_structures" | "leaf"
    dd_type: str  # PascalCase DD type for structures and arrays of structures
    data_type: str  # base type for leaves, e.g. "FLT_0D"
    dd_path: str  # full path from the root, e.g. "time_slice/global_quantities/ip"
    children: list["PathNode"] = field(default_factory=list)
    # Leaves only:
    projection: str = ""  # Rust expression reading the leaf, given `equilibrium` and `at`
    lengths_function: str = ""  # name of the generated lengths function
    n_levels: int = 0  # number of array-of-structures levels above this leaf


def rust_string_literal(text: str) -> str:
    """Quote a string for Rust source.

    A newline is escaped, so that the literal is on one line: `rustfmt` gives up on the whole item holding a
    literal which spans several lines when one of them is longer than `max_width`, as documentation written one
    sentence per line often is, and leaves it unformatted.
    """
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def python_docstring(documentation: str, units: str, indent: str) -> list[str]:
    """Render a docstring for the type stub, keeping it valid whatever the DD says.

    Every line after the first is indented as the quotes are, so that the units line up with the
    documentation above them however many lines it has.
    """
    body = documentation.replace('"""', "'''").replace("\\", "\\\\").strip()
    if units:
        body = f"{body}\n\nUnits: {units}" if body else f"Units: {units}"
    if not body:
        return []
    lines: list[str] = body.split("\n")
    n_lines: int = len(lines)
    for i_line in range(1, n_lines):
        if lines[i_line] != "":
            lines[i_line] = f"{indent}{lines[i_line]}"
    lines[0] = f'{indent}"""{lines[0]}'
    lines.append(f'{indent}"""')
    return lines


def build_path_tree(
    ct: ComplexType,
    type_map: dict[str, ComplexType],
    *,
    dd_path: str,
    rust_expression: str,
    vec_expressions: list[str],
    vec_dd_paths: list[str],
    ancestry: tuple[str, ...],
    lengths_registry: dict[tuple[str, ...], str],
) -> list[PathNode]:
    """
    Walk one structure and return its children as `PathNode`s.

    `rust_expression` is the Rust expression that reaches this structure given the
    index list `at`; `vec_expressions` is the expression for each array-of-structures
    level above it, which is what the generated lengths function needs.

    `ancestry` stops a self-referential schema from recursing forever, matching the
    guard in `generate_view_for_struct`.
    """
    nodes: list[PathNode] = []

    for f in ct.fields:
        if f.name in PYTHON_KEYWORDS:
            raise ValueError(f"data dictionary field `{dd_path}/{f.name}` is a Python keyword and cannot be written as an attribute in the type stub")

        child_dd_path = f"{dd_path}/{f.name}" if dd_path else f.name
        rust_name = sanitize_rust_identifier(f.name)

        if f.is_array and f.inner_type in type_map:
            # An array of structures: a new index level.
            vec_expression = f"{rust_expression}.{rust_name}"
            child_vec_expressions = vec_expressions + [vec_expression]
            child_vec_dd_paths = vec_dd_paths + [child_dd_path]
            # `gather` resolves every index against the real length before calling the
            # projection, so plain indexing cannot go out of range here. A single element is
            # always `.field[i]`; the round-bracket form is reserved for the slice view
            element_expression = f"{vec_expression}[at[{len(vec_expressions)}]]"

            if f.inner_type in ancestry:
                continue  # self-referential schema; stop rather than recurse forever

            nodes.append(
                PathNode(
                    name=f.name,
                    documentation=f.documentation,
                    units=f.units,
                    kind="array_of_structures",
                    dd_type=f.inner_type,
                    data_type="",
                    dd_path=child_dd_path,
                    children=build_path_tree(
                        type_map[f.inner_type],
                        type_map,
                        dd_path=child_dd_path,
                        rust_expression=element_expression,
                        vec_expressions=child_vec_expressions,
                        vec_dd_paths=child_vec_dd_paths,
                        ancestry=ancestry + (f.inner_type,),
                        lengths_registry=lengths_registry,
                    ),
                )
            )

        elif f.is_base_type:
            if f.rust_type not in PYTHON_LEAF_TYPES:
                raise ValueError(
                    f"`{child_dd_path}` has base type {f.rust_type}, which has no Python "
                    "mapping. Add it to PYTHON_LEAF_TYPES and give it a `Gatherable` "
                    "implementation in src/python/mod.rs."
                )

            lengths_function = register_lengths_function(vec_expressions, vec_dd_paths, lengths_registry)
            nodes.append(
                PathNode(
                    name=f.name,
                    documentation=f.documentation,
                    units=f.units,
                    kind="leaf",
                    dd_type="",
                    data_type=f.rust_type,
                    dd_path=child_dd_path,
                    projection=f"{rust_expression}.{rust_name}.clone()",
                    lengths_function=lengths_function,
                    n_levels=len(vec_expressions),
                )
            )

        elif f.is_struct and f.rust_type in type_map:
            if f.rust_type in ancestry:
                continue  # self-referential schema
            nodes.append(
                PathNode(
                    name=f.name,
                    documentation=f.documentation,
                    units=f.units,
                    kind="structure",
                    dd_type=f.rust_type,
                    data_type="",
                    dd_path=child_dd_path,
                    children=build_path_tree(
                        type_map[f.rust_type],
                        type_map,
                        dd_path=child_dd_path,
                        rust_expression=f"{rust_expression}.{rust_name}",
                        vec_expressions=vec_expressions,
                        vec_dd_paths=vec_dd_paths,
                        ancestry=ancestry + (f.rust_type,),
                        lengths_registry=lengths_registry,
                    ),
                )
            )

        # Anything else is a reference to a type that could not be resolved; it is
        # emitted as an empty struct by `generate_stub_types`, so there is nothing to
        # navigate into and it is left out of the path tree.

    return nodes


def register_lengths_function(
    vec_expressions: list[str],
    vec_dd_paths: list[str],
    lengths_registry: dict[tuple[str, ...], str],
) -> str:
    """
    Name the function giving the length of each array-of-structures level on a path.

    Chains are shared between every leaf below them, so the same chain is only ever
    emitted once.

    The name comes from the full data dictionary path of the deepest level, not from the
    trailing segments: `description_2d/limiter/unit` and `description_2d/mobile/unit` both
    end in `unit` and would otherwise collide.
    """
    if not vec_expressions:
        return "no_levels"

    key = tuple(vec_expressions)
    if key in lengths_registry:
        return lengths_registry[key]

    name = "lengths_" + vec_dd_paths[-1].replace("/", "_")
    if name in lengths_registry.values():
        raise ValueError(f"two array-of-structures chains both want the name `{name}`")

    lengths_registry[key] = name
    return name


def generate_lengths_functions(lengths_registry: dict[tuple[str, ...], str], root_variable: str, ids_pascal: str) -> list[str]:
    """Emit one function per array-of-structures chain."""
    lines: list[str] = []
    for vec_expressions, name in sorted(lengths_registry.items(), key=lambda item: item[1]):
        # `at` is only read when there is more than one level.
        at_parameter = "at" if len(vec_expressions) > 1 else "_at"
        lines.append(f"/// Length of each array-of-structures level along `{name[len('lengths_') :]}`.")
        lines.append(f"fn {name}({root_variable}: &{ids_pascal}, level: usize, {at_parameter}: &[usize]) -> Option<usize> {{")
        lines.append("    match level {")
        for i_level, expression in enumerate(vec_expressions):
            lines.append(f"        {i_level} => return Some({expression}.len()),")
        lines.append("        _ => return None,")
        lines.append("    }")
        lines.append("}")
        lines.append("")
    return lines


def collect_types_from_path_tree(nodes: list[PathNode], collected: dict[str, list[PathNode]], seen_paths: dict[str, str]) -> None:
    """
    Gather the children of every structure type reached by the path tree.

    A data dictionary type has the same children wherever it appears, so the stub needs
    one class per type rather than one per path. That is checked here rather than
    assumed: if the same type ever turned up with different children the stub would be
    silently wrong.
    """
    for node in nodes:
        if node.kind == "leaf":
            continue
        child_names = [child.name for child in node.children]
        if node.dd_type in collected:
            existing = [child.name for child in collected[node.dd_type]]
            if existing != child_names:
                raise ValueError(
                    f"type {node.dd_type} has different children at `{node.dd_path}` than at "
                    f"`{seen_paths[node.dd_type]}`; the stub cannot use one class per type"
                )
        else:
            collected[node.dd_type] = node.children
            seen_paths[node.dd_type] = node.dd_path
        collect_types_from_path_tree(node.children, collected, seen_paths)


def generate_python_paths_body(
    root_nodes: list[PathNode],
    lengths_registry: dict[tuple[str, ...], str],
    root_ct: ComplexType,
    ids_name: str,
    root_kind: str,
) -> tuple[list[str], list[str], list[str]]:
    """Generate the body of the static description the Rust navigator walks: a function per array-of-structures chain giving its
    lengths, the nodes, and the root, `<NAME>_ROOT`.

    Shared by an IDS's `src/python/<ids>_paths.rs`, see `generate_python_paths_rust`, and a structure's, see `build_structure`, which
    differ only in their header and in where they import from.

    Args:
        root_nodes: The root's children, from `build_path_tree`
        lengths_registry: The array-of-structures chains, from `build_path_tree`
        root_ct: The root struct
        ids_name: The root's name, e.g. "equilibrium", which names the root and the variable each reader takes
        root_kind: What the root is, "IDS" or "structure", for the readers' error and the root's documentation

    Returns:
        The lines; the names they use from the Python bindings' module, `imas_rs::python`; and the base types they use
    """
    ids_pascal = snake_to_pascal_case(ids_name)

    statics: list[str] = []

    def emit_children(nodes: list[PathNode], dd_path: str) -> str:
        """Emit the static array holding `nodes`, and return its name."""
        # Emit children first, so that every name a node refers to already exists.
        entries: list[str] = []
        for node in nodes:
            if node.kind == "leaf":
                at_parameter = "at" if node.n_levels > 0 else "_at"
                kind = (
                    "NodeKind::Leaf(Leaf {\n"
                    f"                data_type: {rust_string_literal(node.data_type)},\n"
                    "                read: |ids: &dyn Any, indices: &[IndexSpec]| {\n"
                    f'                    let {ids_name}: &{ids_pascal} = ids.downcast_ref().ok_or_else(|| "not a {ids_name} {root_kind}".to_string())?;\n'
                    "                    gather(\n"
                    f"                    {ids_name},\n"
                    "                        indices,\n"
                    f"                        {node.n_levels},\n"
                    f"                        {node.lengths_function},\n"
                    f"                    |{ids_name}: &{ids_pascal}, {at_parameter}: &[usize]| -> {node.data_type} {{\n"
                    f"                            {node.projection}\n"
                    "                        },\n"
                    "                    )\n"
                    "                },\n"
                    "            })"
                )
            else:
                child_name = emit_children(node.children, node.dd_path)
                variant = "Structure" if node.kind == "structure" else "ArrayOfStructures"
                kind = f"NodeKind::{variant}({child_name})"

            entries.append(
                "    Node {\n"
                f"        name: {rust_string_literal(node.name)},\n"
                f"        documentation: {rust_string_literal(node.documentation)},\n"
                f"        units: {rust_string_literal(node.units)},\n"
                f"        kind: {kind},\n"
                "    },"
            )

        static_name = "NODES_" + (dd_path.replace("/", "_").upper() if dd_path else "ROOT")
        statics.append(f"static {static_name}: &[Node] = &[\n" + "\n".join(entries) + "\n];\n")
        return static_name

    root_children_name = emit_children(root_nodes, "")

    imports = ["IndexSpec", "Leaf", "Node", "NodeKind", "gather"]
    if any(node.kind == "leaf" and node.n_levels == 0 for node in iterate_path_tree(root_nodes)):
        imports.append("no_levels")
    used_base_types = sorted({node.data_type for node in iterate_path_tree(root_nodes) if node.kind == "leaf"})

    lines: list[str] = []
    lines.append("// " + "=" * 76)
    lines.append("// Array-of-structures lengths")
    lines.append("// " + "=" * 76)
    lines.append("")
    lines.extend(generate_lengths_functions(lengths_registry, ids_name, ids_pascal))
    lines.append("// " + "=" * 76)
    lines.append("// Nodes")
    lines.append("// " + "=" * 76)
    lines.append("")
    lines.extend(statics)
    if root_kind == "IDS":
        lines.append(f"/// The root of the {ids_name} data dictionary.")
    else:
        lines.append(f"/// The root of `{ids_name}`.")
    lines.append(f"pub static {ids_name.upper()}_ROOT: Node = Node {{")
    lines.append(f"    name: {rust_string_literal(ids_name)},")
    lines.append(f"    documentation: {rust_string_literal(root_ct.documentation)},")
    lines.append('    units: "",')
    lines.append(f"    kind: NodeKind::Structure({root_children_name}),")
    lines.append("};")

    return lines, imports, used_base_types


def generate_python_paths_rust(
    root_nodes: list[PathNode],
    lengths_registry: dict[tuple[str, ...], str],
    root_ct: ComplexType,
    ids_name: str,
) -> str:
    """Generate an IDS's static data dictionary description, `src/python/<ids>_paths.rs`, which the Rust navigator walks."""
    ids_pascal = snake_to_pascal_case(ids_name)
    body_lines, imports, used_base_types = generate_python_paths_body(root_nodes, lengths_registry, root_ct, ids_name, "IDS")

    lines: list[str] = []
    lines.append(f"//! Static description of the {ids_name} data dictionary, used to build and resolve paths.")
    lines.append("//!")
    lines.append("//! **WARNING** This file is autogenerated by `rust/imas_rs/imas_updater/build_ids.py`.")
    lines.append("//! Any changes will be overwritten.")
    lines.append("//!")
    lines.append("//! Each leaf carries the function that reads it out of the IDS, so the path a")
    lines.append("//! reader follows is written once and checked by the compiler. The `read` closures")
    lines.append("//! capture nothing, so they coerce to plain `fn` pointers and the whole description")
    lines.append("//! stays a `static`.")
    lines.append("")
    lines.append("#![allow(clippy::all)]")
    lines.append("")
    lines.append(f"use super::{{{', '.join(imports)}}};")
    lines.append(f"use crate::dd_base_types::{{{', '.join(used_base_types)}}};")
    lines.append(f"use crate::ids::{ids_name}::{ids_pascal};")
    lines.append("use std::any::Any;")
    lines.append("")
    lines.extend(body_lines)

    return "\n".join(lines)


def generate_python_pickle_body(complex_types: list[ComplexType], root_element: ComplexType, structs_prefix: str) -> list[str]:
    """Generate the pickling of a structure's structs, an implementation of `PickleState` for each.

    Each converts its struct to and from the `dict` of its nodes which `pickle` stores, keyed by their names, leaving out those which
    are unset; see `src/python/pickle.rs`, which implements it for the leaves and for `Vec`.

    Shared by an IDS's `src/python/<ids>_pickle.rs`, see `generate_python_pickle_rust`, and a structure's module, see `build_structure`,
    which differ in their header and in where the structs are.

    Args:
        complex_types: The structs, besides the root
        root_element: The root struct, e.g. the IDS's
        structs_prefix: What each struct's name is prefixed with to reach it: the module it is in, e.g. `equilibrium::`, which the
            caller imports, or "" when the structs are in the same module

    Returns:
        The lines; they use `PickleState`, `PickleStateError` and `structure_nodes` from `imas_rs::python`, and `PyDict` and pyo3's
        prelude
    """
    lines: list[str] = []

    # The structs a schema refers to but does not define are emitted empty (see `generate_stub_types`), and are pickled too, as
    # the empty `dict`
    _, referenced_types = collect_used_types(complex_types, root_element)
    struct_names: list[str] = []
    struct_fields: list[list[Field]] = []
    for type_name in sorted(referenced_types):
        struct_names.append(type_name)
        struct_fields.append([])
    for ct in complex_types + [root_element]:
        struct_names.append(snake_to_pascal_case(ct.name))
        struct_fields.append(ct.fields)

    n_struct: int = len(struct_names)
    for i_struct in range(n_struct):
        fields: list[Field] = struct_fields[i_struct]
        lines.append("")
        lines.append(f"impl PickleState for {structs_prefix}{struct_names[i_struct]} {{")

        # Unset when every node is
        lines.append("    fn is_unset(&self) -> bool {")
        if not fields:
            lines.append("        true")
        else:
            unset_conditions: list[str] = []
            for f in fields:
                unset_conditions.append(f"self.{sanitize_rust_identifier(f.name)}.is_unset()")
            lines.append(f"        {' && '.join(unset_conditions)}")
        lines.append("    }")
        lines.append("")

        # The `dict` of the nodes which are set
        lines.append("    fn to_pickle_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {")
        lines.append("        let state: Bound<'py, PyDict> = PyDict::new(py);")
        for f in fields:
            field_name: str = sanitize_rust_identifier(f.name)
            lines.append(f"        if !self.{field_name}.is_unset() {{")
            lines.append(f"            state.set_item({rust_string_literal(f.name)}, self.{field_name}.to_pickle_state(py)?)?;")
            lines.append("        }")
        lines.append("        Ok(state.into_any())")
        lines.append("    }")
        lines.append("")

        # Read back onto an unset struct, so that a node left out of the state is unset
        lines.append("    fn from_pickle_state(state: &Bound<'_, PyAny>) -> Result<Self, PickleStateError> {")
        lines.append("        let nodes: Vec<(String, Bound<'_, PyAny>)> = structure_nodes(state)?;")
        if not fields:
            lines.append("        if !nodes.is_empty() {")
            lines.append("            return Err(PickleStateError::unknown_node(&nodes[0].0));")
            lines.append("        }")
            lines.append("        Ok(Self::default())")
        else:
            lines.append("        let mut structure: Self = Self::default();")
            lines.append("        let n_node: usize = nodes.len();")
            lines.append("        for i_node in 0..n_node {")
            lines.append("            let (name, node_state): &(String, Bound<'_, PyAny>) = &nodes[i_node];")
            lines.append("            match name.as_str() {")
            for f in fields:
                field_name = sanitize_rust_identifier(f.name)
                name_literal: str = rust_string_literal(f.name)
                lines.append(f"                {name_literal} => {{")
                lines.append(
                    f"                    structure.{field_name} = PickleState::from_pickle_state(node_state)"
                    f".map_err(|error: PickleStateError| error.in_node({name_literal}))?;"
                )
                lines.append("                }")
            lines.append("                _ => return Err(PickleStateError::unknown_node(name)),")
            lines.append("            }")
            lines.append("        }")
            lines.append("        Ok(structure)")
        lines.append("    }")
        lines.append("}")

    return lines


def generate_python_pickle_rust(complex_types: list[ComplexType], root_element: ComplexType, module_documentation: list[str], ids_name: str) -> str:
    """Generate the pickling of an IDS's structs, `src/python/<ids>_pickle.rs`: its header, and the body which
    `generate_python_pickle_body` generates.

    Args:
        complex_types: The structs, besides the root
        root_element: The root struct, the IDS's
        module_documentation: The module's documentation, one line each, an empty one for a blank line
        ids_name: The IDS's name, e.g. "equilibrium", whose module, `crate::ids::<ids>`, holds the structs
    """
    lines: list[str] = []
    for documentation_line in module_documentation:
        lines.append(f"//! {documentation_line}" if documentation_line else "//!")
    lines.append("")
    lines.append("#![allow(clippy::all)]")
    lines.append("")
    lines.append("use super::{PickleState, PickleStateError, structure_nodes};")
    lines.append(f"use crate::ids::{ids_name};")
    lines.append("use pyo3::prelude::*;")
    lines.append("use pyo3::types::PyDict;")
    lines.extend(generate_python_pickle_body(complex_types, root_element, f"{ids_name}::"))

    return "\n".join(lines)


def iterate_path_tree(nodes: list[PathNode]):
    """Yield every node in the tree, depth first."""
    for node in nodes:
        yield node
        yield from iterate_path_tree(node.children)


def generate_python_stub_preamble() -> str:
    """The part of the type stub shared by every IDS: the module docstring and `Path`."""
    lines: list[str] = []
    lines.append('"""Type stubs for the `gsfit_rs.imas` submodule.')
    lines.append("")
    lines.append("**WARNING** This file is autogenerated by `rust/imas_rs/imas_updater/build_ids.py`.")
    lines.append("Any changes will be overwritten.")
    lines.append("")
    lines.append("At runtime there is a single `Path` class reached through `__getattr__`, walking a")
    lines.append("static description of the data dictionary held in Rust. This stub declares one class")
    lines.append("per node so that editors offer completions and `mypy` checks both the attribute")
    lines.append("names and the type that `get` returns.")
    lines.append("")
    lines.append("Classes ending `Item` are reached when every array-of-structures level was indexed")
    lines.append("with an integer, so one value comes back. Classes ending `Many` are reached once a")
    lines.append("level has been sliced, so values are gathered into a numpy array.")
    lines.append("")
    lines.append("Any number of levels may be sliced. The dimensions are gained from left to right: one")
    lines.append("per sliced level, in the order they appear on the path, then the leaf's own dimensions.")
    lines.append("So `coil[:].element[:].turns_with_sign` is `(n_coil, n_element)` and")
    lines.append("`time_slice[:].profiles_2d[0].psi` is `(n_time, n_z, n_r)`.")
    lines.append('"""')
    lines.append("")
    lines.append("from typing import Generic, TypeVar, overload")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("import numpy.typing as npt")
    lines.append("")
    lines.append('_T = TypeVar("_T")')
    lines.append("# `Path` is covariant because a path only ever hands a value out, through `get`. So a")
    lines.append("# `Path[list[str]]` is also a `Path[Iterable[str]]`, which is what `mypy` asks for when it")
    lines.append("# infers `_T` from the surrounding context, e.g. `list(ids.get(path))`")
    lines.append('_T_co = TypeVar("_T_co", covariant=True)')
    lines.append("")
    lines.append("class Path(Generic[_T_co]):")
    lines.append('    """A data dictionary path. Holds no data; pass it to `get`.')
    lines.append("")
    lines.append("    A path is a value: it can be stored in a list, printed, and reused against")
    lines.append("    several IDSs.")
    lines.append('    """')
    lines.append("")
    lines.append("    @property")
    lines.append("    def units(self) -> str:")
    lines.append('        """The units of this node. Empty when the data dictionary gives none."""')
    lines.append("    @property")
    lines.append("    def documentation(self) -> str:")
    lines.append('        """The data dictionary description of this node."""')
    lines.append("    @property")
    lines.append("    def data_type(self) -> str:")
    lines.append('        """The data dictionary type, e.g. `FLT_0D`. Empty for a structure."""')
    lines.append("    def __repr__(self) -> str: ...")
    lines.append("")
    return "\n".join(lines)


def generate_python_stub_section(
    root_nodes: list[PathNode],
    root_ct: ComplexType,
    type_map: dict[str, ComplexType],
    ids_name: str,
    root_kind: str = "IDS",
) -> str:
    """Generate one IDS's section of the type stub, or one structure's, see `build_structure`, whose `root_kind` is "structure".

    Every IDS shares the one `gsfit_rs.imas` module, so the sections are concatenated
    after `generate_python_stub_preamble`.
    """
    ids_pascal = snake_to_pascal_case(ids_name)

    type_children: dict[str, list[PathNode]] = {}
    collect_types_from_path_tree(root_nodes, type_children, {})

    # Class names are namespaced by IDS: every IDS declares its own `Code`, `Library`,
    # `IdentifierStatic`, ... and they all share the one `imas.pyi`. A type whose name
    # already starts with the IDS name is left alone, so `EquilibriumTimeSlice` does not
    # become `EquilibriumEquilibriumTimeSlice`.
    def namespaced(dd_type: str) -> str:
        if dd_type.startswith(ids_pascal):
            return dd_type
        return f"{ids_pascal}{dd_type}"

    def class_name(dd_type: str, many: bool) -> str:
        suffix = "Many" if many else "Item"
        return f"_{namespaced(dd_type)}{suffix}"

    def array_class_name(dd_type: str, many: bool) -> str:
        suffix = "ArrayFromMany" if many else "ArrayFromItem"
        return f"_{namespaced(dd_type)}{suffix}"

    def emit_members(nodes: list[PathNode], many: bool) -> list[str]:
        lines: list[str] = []
        for node in nodes:
            if node.kind == "leaf":
                item_type, many_type = PYTHON_LEAF_TYPES[node.data_type]
                return_type = f"Path[{many_type if many else item_type}]"
            elif node.kind == "structure":
                return_type = class_name(node.dd_type, many)
            else:
                return_type = array_class_name(node.dd_type, many)
            lines.append("    @property")
            lines.append(f"    def {node.name}(self) -> {return_type}:")
            docstring = python_docstring(node.documentation, node.units, "        ")
            if docstring:
                lines.extend(docstring)
            else:
                lines[-1] = lines[-1] + " ..."
        if not lines:
            lines.append("    pass")
        return lines

    lines: list[str] = []
    lines.append(f"class {ids_pascal}:")
    if root_kind == "IDS":
        lines.append(f'    """{root_ct.documentation}"""')
    else:
        lines.extend(python_docstring(root_ct.documentation, "", "    "))
    lines.append("")
    lines.append("    def get(self, path: Path[_T]) -> _T:")
    lines.append(f'        """Read the data at `path` out of this {root_kind}.')
    lines.append("")
    lines.append("        The shape of the result follows the shape of the index: an integer index")
    lines.append("        gives one value, a slice gathers. The dimensions are gained from left to right:")
    lines.append("        one per sliced level, in the order they appear on the path, then the leaf's own")
    lines.append("        dimensions. A sliced level which selects a different number of elements under")
    lines.append("        different elements above it would be jagged, and raises `IndexError`.")
    lines.append("")
    lines.append("        An unset leaf reads back as the IMAS empty value: NaN for a float,")
    lines.append("        -999999999 (`EMPTY_INT`) for an integer, an empty string, or an empty array.")
    lines.append('        """')
    if root_kind == "IDS":
        lines.append("    def __len__(self) -> int:")
        lines.append('        """The number of time slices held by this IDS."""')
    lines.append("    def __repr__(self) -> str: ...")
    lines.append("")
    lines.append("# " + "=" * 74)
    lines.append(f"# {ids_name} path nodes")
    lines.append("# " + "=" * 74)
    lines.append("")

    root_type = snake_to_pascal_case(root_ct.name)
    for dd_type in sorted(type_children):
        if dd_type == root_type:
            continue
        documentation = type_map[dd_type].documentation if dd_type in type_map else ""
        for many in (False, True):
            lines.append(f"class {class_name(dd_type, many)}:")
            if documentation:
                lines.extend(python_docstring(documentation, "", "    "))
                lines.append("")
            lines.extend(emit_members(type_children[dd_type], many))
            lines.append("")

    lines.append("# " + "-" * 74)
    lines.append(f"# {ids_name} arrays of structures")
    lines.append("# " + "-" * 74)
    lines.append("")

    array_types = sorted({node.dd_type for node in iterate_path_tree(root_nodes) if node.kind == "array_of_structures"})
    for dd_type in array_types:
        lines.append(f"class {array_class_name(dd_type, False)}:")
        lines.append("    @overload")
        lines.append(f"    def __getitem__(self, index: int) -> {class_name(dd_type, False)}: ...")
        lines.append("    @overload")
        lines.append(f"    def __getitem__(self, index: slice | list[int]) -> {class_name(dd_type, True)}: ...")
        lines.append("")
        lines.append(f"class {array_class_name(dd_type, True)}:")
        lines.append("    # A level above has already been sliced, so whatever this one is indexed with,")
        lines.append("    # the values are still gathered.")
        lines.append(f"    def __getitem__(self, index: int | slice | list[int]) -> {class_name(dd_type, True)}: ...")
        lines.append("")

    lines.append("# " + "-" * 74)
    lines.append(f"# {ids_name} root")
    lines.append("# " + "-" * 74)
    lines.append("")
    lines.append(f"class _{ids_pascal}Paths:")
    lines.extend(python_docstring(root_ct.documentation, "", "    "))
    lines.append("")
    lines.extend(emit_members(root_nodes, False))
    lines.append("")
    lines.append(f"{ids_name}_paths: _{ids_pascal}Paths")
    lines.append("")

    return "\n".join(lines)


def generate_rust_body(
    complex_types: list[ComplexType],
    root_element: ComplexType | None,
    root_section_title: str,
) -> tuple[list[str], list[str], list[str]]:
    """Generate the body of a structure's Rust module: its structs, the root's sizing constructors, and the views, accessors and
    accumulators which gather from them.

    Shared by an IDS's `src/ids/<ids>.rs`, see `generate_rust_file`, and a structure's, see `build_structure`, which differ in their
    header and in what else the module holds.

    Args:
        complex_types: The structs, besides the root
        root_element: The root struct, e.g. the IDS's
        root_section_title: The title of the comment heading the root struct, e.g. "Root IDS Structure"

    Returns:
        The lines; the names they use from `dd_base_types.rs`; and those they use from `ndarray`
    """
    lines: list[str] = []

    # Collect used types
    used_base_types, referenced_types = collect_used_types(complex_types, root_element)

    # Generate the view/accessor code up front: the gathering machinery it needs
    # from dd_base_types depends on which accumulators it actually emits, so the
    # import list can only be built once the code exists.
    views_code = generate_all_views_and_accessors(complex_types, root_element)

    # Imports
    imported = set(used_base_types)
    for gathering_type in ("Accumulator", "Elements"):
        if re.search(rf"\b{gathering_type}(::|<)", views_code):
            imported.add(gathering_type)
    # Every accessor takes a `Selection`, every element type builds its slice view for one, and every element
    # type with a `name` can be selected by it
    for selection_type, pattern in (
        ("Selection", r"\bSelection<"),
        ("ArrayOfStructuresElement", r"\bimpl ArrayOfStructuresElement for\b"),
        ("Named", r"\bimpl Named for\b"),
    ):
        if re.search(pattern, views_code):
            imported.add(selection_type)
    # The generated `Default` impls initialise every integer leaf to `EMPTY_INT`
    if "INT_0D" in imported:
        imported.add("EMPTY_INT")
    # The views are generic over the rank they gather
    ndarray_imports: list[str] = []
    if views_code:
        ndarray_imports.append("Dimension")
        if re.search(r"\bIx1\b", views_code):
            ndarray_imports.append("Ix1")

    # Generate stub types for undefined references
    stub_code = generate_stub_types(referenced_types)
    if stub_code:
        lines.append(stub_code)

    # Generate structs for all complex types
    lines.append("// " + "=" * 76)
    lines.append("// Complex Types")
    lines.append("// " + "=" * 76)
    lines.append("")

    for ct in complex_types:
        lines.append(generate_rust_struct(ct))

    # Generate root element struct
    if root_element:
        lines.append("// " + "=" * 76)
        lines.append(f"// {root_section_title}")
        lines.append("// " + "=" * 76)
        lines.append("")
        lines.append(generate_rust_struct(root_element))

        # Sizing constructors for the root IDS (with_size / with_time)
        constructors_code = generate_root_constructors(root_element, complex_types)
        if constructors_code:
            lines.append(constructors_code)

    # Generate view, accessor, and accumulator types
    if views_code:
        lines.append("")
        lines.append(views_code)

    return lines, sorted(imported), ndarray_imports


def generate_rust_file(
    complex_types: list[ComplexType],
    root_element: Optional[ComplexType],
    module_documentation: list[str],
) -> str:
    """Generate an IDS's Rust module, `src/ids/<ids>.rs`: its header, and the body which `generate_rust_body` generates.

    Args:
        complex_types: The structs, besides the root
        root_element: The root struct, the IDS's
        module_documentation: The module's documentation, one line each, an empty one for a blank line
    """
    body_lines, dd_base_types_imports, ndarray_imports = generate_rust_body(complex_types, root_element, "Root IDS Structure")

    lines: list[str] = []

    # Module documentation
    for documentation_line in module_documentation:
        lines.append(f"//! {documentation_line}" if documentation_line else "//!")
    lines.append("")
    lines.append("#![allow(dead_code)]")
    lines.append("#![allow(non_camel_case_types)]")
    lines.append("")

    # Imports
    if dd_base_types_imports:
        lines.append(f"use crate::dd_base_types::{{{', '.join(dd_base_types_imports)}}};")
    if ndarray_imports:
        lines.append(f"use ndarray::{{{', '.join(ndarray_imports)}}};")
    if dd_base_types_imports or ndarray_imports:
        lines.append("")

    lines.extend(body_lines)

    return "\n".join(lines)


def build_ids(
    path_to_ids_schema: Path,
    path_to_rust_ids_file: Path,
    path_to_rust_paths_file: Optional[Path] = None,
    path_to_rust_pickle_file: Optional[Path] = None,
    generate_python_stub_for_this_ids: bool = False,
) -> Optional[str]:
    """
    Build Rust IDS code from IMAS XSD schema.

    Example:
    ```python
    from pathlib import Path
    from build_ids import build_ids

    build_ids(
        path_to_ids_schema=Path("/home/peter.buxton/github/imas_rs/tmp/IMAS-Data-Dictionary/schemas/equilibrium/"),
        path_to_rust_ids_file=Path("/home/peter.buxton/github/imas_rs/src/ids/equilibrium.rs")
    )
    ```

    Args:
        path_to_ids_schema: Path to the schema directory containing dd_<name>.xsd
        path_to_rust_ids_file: Path where the generated Rust file will be written
        path_to_rust_paths_file: (optional) Path for the static data dictionary description
            used by the Python bindings
        path_to_rust_pickle_file: (optional) Path for the pickling of the IDS's structs, used by
            the Python bindings
        generate_python_stub_for_this_ids: whether to return this IDS's section of the
            `.pyi` type stub. Every IDS shares the one `gsfit_rs.imas` module, so the
            caller concatenates the sections rather than each writing its own file.

    Returns:
        This IDS's section of the type stub, or `None` when it was not asked for.
    """
    # Find the main XSD file (dd_<name>.xsd)
    schema_dir = Path(path_to_ids_schema)
    ids_name = schema_dir.name  # e.g., "equilibrium"

    xsd_files = list(schema_dir.glob(f"dd_{ids_name}.xsd"))
    if not xsd_files:
        raise FileNotFoundError(f"No XSD file found matching dd_{ids_name}.xsd in {schema_dir}")

    xsd_path = xsd_files[0]
    print(f"Parsing: {xsd_path}")

    # Parse the XSD
    complex_types, root_element = parse_xsd(xsd_path)
    print(f"Found {len(complex_types)} complex types")

    # Pull in the shared structures the IDS references from `dd_support.xsd`, so that they are
    # emitted properly instead of as empty stubs
    supporting_types = parse_supporting_schemas(xsd_path)
    pulled = resolve_supporting_types(complex_types, root_element, supporting_types)
    print(f"Resolved {len(pulled)} supporting types: {', '.join(sorted(pulled))}")

    # Merge in GSFit's custom, non-IMAS keys before anything is emitted, so that the
    # views and accumulators are generated for them too
    add_custom_keys_to_ids(
        complex_types=complex_types,
        root_element=root_element,
        supporting_types=supporting_types,
        ids_name=ids_name,
        path_to_custom_keys_file=CUSTOM_KEYS_DIR / f"custom_{ids_name}_keys.xsd",
    )
    resolve_supporting_types(complex_types, root_element, supporting_types)

    if root_element:
        print(f"Root element: {root_element.name}")

    # Generate Rust code
    module_documentation: list[str] = [
        f"IMAS {ids_name.capitalize()} IDS",
        "",
        "**WARNING** This file is autogenerated by `rust/imas_rs/imas_updater/build_ids.py`.",
        "Any changes will be overwritten.",
        "",
        f"This module defines the {ids_name} Interface Data Structure (IDS), generated from the",
        "IMAS Data Dictionary XSD schema.",
    ]
    rust_code = generate_rust_file(complex_types, root_element, module_documentation)

    # Write to file
    output_path = Path(path_to_rust_ids_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rust_code)
    print(f"Generated: {output_path}")

    # Format, so that the committed file matches what `cargo fmt` would produce
    subprocess.run(
        [
            "rustfmt",
            "--edition",
            RUST_EDITION,
            "--config-path",
            str(WORKSPACE_DIR),
            str(output_path),
        ],
        check=True,
    )
    print(f"Formatted: {output_path}")

    # The Python bindings: a static description of the same schema, plus the type stub
    # that gives editors their completions. Both come from one walk of the tree, so they
    # cannot drift apart from each other or from the structs above.
    if path_to_rust_paths_file is None and path_to_rust_pickle_file is None and not generate_python_stub_for_this_ids:
        return None

    if root_element is None:
        raise ValueError("cannot generate the Python bindings without a root element")

    # The pickling, from the same structs as the IDS's file
    if path_to_rust_pickle_file is not None:
        pickle_path = Path(path_to_rust_pickle_file)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_module_documentation: list[str] = [
            f"The pickling of the {ids_name} IDS: how each of its structs is converted to and from the Python objects which",
            "`pickle` stores, see `pickle.rs`.",
            "",
            "**WARNING** This file is autogenerated by `rust/imas_rs/imas_updater/build_ids.py`.",
            "Any changes will be overwritten.",
        ]
        pickle_path.write_text(generate_python_pickle_rust(complex_types, root_element, pickle_module_documentation, ids_name))
        print(f"Generated: {pickle_path}")
        subprocess.run(
            [
                "rustfmt",
                "--edition",
                RUST_EDITION,
                "--config-path",
                str(WORKSPACE_DIR),
                str(pickle_path),
            ],
            check=True,
        )
        print(f"Formatted: {pickle_path}")

    all_types = complex_types + [root_element]
    type_map = {snake_to_pascal_case(ct.name): ct for ct in all_types}

    lengths_registry: dict[tuple[str, ...], str] = {}
    root_nodes = build_path_tree(
        root_element,
        type_map,
        dd_path="",
        rust_expression=ids_name,
        vec_expressions=[],
        vec_dd_paths=[],
        ancestry=(snake_to_pascal_case(root_element.name),),
        lengths_registry=lengths_registry,
    )
    n_nodes = sum(1 for _ in iterate_path_tree(root_nodes))
    n_leaves = sum(1 for node in iterate_path_tree(root_nodes) if node.kind == "leaf")
    print(f"Path tree: {n_nodes} nodes, {n_leaves} leaves, {len(lengths_registry)} array-of-structures chains")

    if path_to_rust_paths_file is not None:
        paths_path = Path(path_to_rust_paths_file)
        paths_path.parent.mkdir(parents=True, exist_ok=True)
        paths_path.write_text(generate_python_paths_rust(root_nodes, lengths_registry, root_element, ids_name))
        print(f"Generated: {paths_path}")
        subprocess.run(
            [
                "rustfmt",
                "--edition",
                RUST_EDITION,
                "--config-path",
                str(WORKSPACE_DIR),
                str(paths_path),
            ],
            check=True,
        )
        print(f"Formatted: {paths_path}")

    if generate_python_stub_for_this_ids:
        return generate_python_stub_section(root_nodes, root_element, type_map, ids_name)

    return None


def build_dd_support(element_name: str, path_to_dd_support_schema: Path, path_to_rust_file: Path) -> None:
    """
    Build the Rust struct for a global element of `dd_support.xsd` which is not an IDS.

    `dd_support.xsd` mostly holds structures shared by the IDSs, which `build_ids` pulls into each
    IDS module as needed. A few of its global elements appear in no IDS, e.g. `parameters_input`, the
    code parameters block "passed from the wrapper to the subroutine", which the Data Dictionary
    declares for the Fortran and C++ type definitions. Each of these gets a module of its own in
    `src/dd_support/`.

    Args:
        element_name: Name of the global element, e.g. "parameters_input"
        path_to_dd_support_schema: Path to `dd_support.xsd`
        path_to_rust_file: Path where the generated Rust file will be written, e.g. `src/dd_support/parameters_input.rs`
    """
    print(f"Parsing: {path_to_dd_support_schema} <{element_name}>")
    schema_root = ET.parse(path_to_dd_support_schema).getroot()
    known_types: set[str] = {ct.get("name") for ct in schema_root.findall(f"{XS_NS}complexType") if ct.get("name")}

    element_type: ComplexType | None = None
    for element in schema_root.findall(f"{XS_NS}element"):
        inline_complex = element.find(f"{XS_NS}complexType")
        if element.get("name") == element_name and inline_complex is not None:
            element_type = parse_complex_type(inline_complex, known_types)
            element_type.name = element_name
            element_type.documentation = extract_documentation(element)
    if element_type is None:
        raise ValueError(f"{path_to_dd_support_schema.name} has no global element <{element_name}> holding a complexType")

    # Only fields of the base types are handled: a field holding another structure would need that structure generated too
    for f in element_type.fields:
        if not f.is_base_type:
            raise NotImplementedError(f"<{element_name}>/{f.name} is a `{f.rust_type}`, but only the data dictionary base types are handled")

    base_types: list[str] = sorted({f.rust_type for f in element_type.fields})
    lines: list[str] = [
        f"//! IMAS `{element_name}`",
        "//!",
        "//! **WARNING** This file is autogenerated by `rust/imas_rs/imas_updater/build_ids.py`.",
        "//! Any changes will be overwritten.",
        "//!",
        f"//! This module defines `{element_name}`, generated from `dd_support.xsd` in the IMAS Data Dictionary.",
        "//! It is not an IDS, and does not appear in any IDS.",
        "",
        f"use crate::dd_base_types::{{{', '.join(base_types)}}};",
        "",
        generate_rust_struct(element_type),
    ]

    output_path = Path(path_to_rust_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Generated: {output_path}")

    # Format, so that the committed file matches what `cargo fmt` would produce
    subprocess.run(
        [
            "rustfmt",
            "--edition",
            RUST_EDITION,
            "--config-path",
            str(WORKSPACE_DIR),
            str(output_path),
        ],
        check=True,
    )
    print(f"Formatted: {output_path}")


def build_structure(path_to_schema: Path, path_to_rust_file: Path) -> str:
    """
    Build the Rust module of a structure of GSFit's own, which is not part of any IDS, from its XSD schema, e.g. the Greens tables:
    its structs, their Python bindings, and their pickling.

    The structure is generated exactly as an IDS is, so its structs have the same accessors: an array of structures is selected
    from with a range, e.g. `.flux_loop(..)`, whose leaves then gather one value per element, or, where its elements have a
    `name`, with a name, e.g. `.flux_loop(name)`; see `dd_base_types.rs`. The generated module is written into the crate which holds
    the structure, so it reaches `imas_rs` by its name.

    Its Python bindings are an IDS's too: the static description of its paths, `<ROOT>_ROOT`, which the crate mounts as
    `gsfit_rs.imas.<root>_paths`, and a class, `Py<Root>`, holding a copy of the structure, readable through them, as
    `imas_rs::python::PyEquilibrium` holds an IDS. It is constructed on the Rust side only, e.g. by a getter of the class holding
    the structure. Its structs are pickled as an IDS's are, through `imas_rs::python::PickleState`, which the class pickles with.

    An IDS's structs, Python bindings and pickling are generated into three files, `src/ids/<ids>.rs`, `src/python/<ids>_paths.rs`
    and `src/python/<ids>_pickle.rs`, as `imas_rs` has its Python bindings behind its optional `python` feature. The crate which holds
    a structure always has its Python bindings, so a structure's are generated into one module.

    The schema is written as the custom keys are (see `add_custom_keys_to_ids`), but declares the whole structure:
    * the root is the one `xs:element`, holding an inline `xs:complexType`, which names the root struct, e.g. `greens_tables` is
      `GreensTables`
    * every other structure is a top-level `xs:complexType`, which an element names in `type`, as can be one of `dd_support.xsd`'s
    * a leaf holds `<xs:group ref="FLT_2D"/>`, or another of the data dictionary's base types, which `dd_support.xsd` declares, so
      it is included, once
    * an array of structures has `minOccurs="0" maxOccurs="unbounded"`, as it can be empty, while a leaf or a single structure,
      which is always there, has neither
    * every structure and element is documented, and a leaf's units are in `xs:appinfo/units`

    Args:
        path_to_schema: The XSD schema of the structure, e.g. `imas_updater/greens_storage.xsd`
        path_to_rust_file: Where the module is written, e.g. `rust/gsfit_rs/src/greens_storage.rs`

    Returns:
        The structure's section of the type stub, `python/gsfit_rs/imas.pyi`
    """
    file_name: str = path_to_schema.name
    print(f"Parsing: {path_to_schema}")
    schema_root: ET.Element = ET.parse(path_to_schema).getroot()

    # The schema's components: the include of `dd_support.xsd`, the structures, and the root
    n_includes: int = 0
    root_names: list[str] = []
    structures: list[tuple[str, ET.Element]] = []
    structure_names: set[str] = set()
    for component in schema_root:
        component_name: str | None = component.get("name")
        if component.tag == f"{XS_NS}include":
            included_file_path: Path = (path_to_schema.parent / component.get("schemaLocation", "")).resolve()
            if included_file_path != DD_SUPPORT_XSD_FILE.resolve():
                raise ValueError(f"{file_name} includes {included_file_path}, but only the data dictionary's `dd_support.xsd` is included, for its types")
            n_includes += 1
        elif component.tag == f"{XS_NS}complexType" and component_name is not None:
            if extract_documentation(component) == "":
                raise ValueError(f"{file_name}: the structure `{component_name}` has no `xs:documentation`")
            structures.append((component_name, component))
            structure_names.add(component_name)
        elif component.tag == f"{XS_NS}element" and component_name is not None and component.find(f"{XS_NS}complexType") is not None:
            if extract_documentation(component) == "":
                raise ValueError(f"{file_name}: the root `{component_name}` has no `xs:documentation`")
            structures.append((component_name, component.find(f"{XS_NS}complexType")))
            root_names.append(component_name)
        else:
            raise NotImplementedError(
                f"{file_name} holds the `{component.tag.removeprefix(XS_NS)}` `{component_name}`, but only the `xs:include` of `dd_support.xsd`, named "
                "`xs:complexType`s, and one `xs:element`, the root, holding an `xs:complexType`, are handled"
            )
    if n_includes != 1:
        dd_support_relative_path: str = os.path.relpath(DD_SUPPORT_XSD_FILE, path_to_schema.parent)
        raise ValueError(f'{file_name} must include the data dictionary\'s types, with `<xs:include schemaLocation="{dd_support_relative_path}"/>`, once')
    if len(root_names) != 1:
        raise ValueError(f"{file_name} has {len(root_names)} root elements, {root_names}, but a structure has one")

    # The structures of `dd_support.xsd`, which an element can have as its type too
    supporting_types: dict[str, ComplexType] = parse_supporting_schemas(path_to_schema)
    support_type_names: set[str] = set()
    for supporting_type in supporting_types.values():
        support_type_names.add(supporting_type.name)

    # Each structure's elements, checked as the custom keys are
    for structure_name, complex_type_element in structures:
        sequences: list[ET.Element] = []
        for child in complex_type_element:
            if child.tag != f"{XS_NS}annotation":
                sequences.append(child)
        if len(sequences) != 1 or sequences[0].tag != f"{XS_NS}sequence":
            raise NotImplementedError(f"{file_name}: `{structure_name}` holds something other than one `xs:sequence` of elements, which is not handled")
        for element in sequences[0]:
            element_name: str | None = element.get("name")
            element_path: str = f"{structure_name}/{element_name}"
            if element.tag != f"{XS_NS}element" or element_name is None:
                raise NotImplementedError(f"{file_name}: `{structure_name}` holds an `{element.tag.removeprefix(XS_NS)}` with no `name`, which is not handled")
            for attribute_name in element.attrib:
                if attribute_name not in ("name", "type", "minOccurs", "maxOccurs"):
                    raise NotImplementedError(
                        f"{file_name}: `{element_path}` has the attribute `{attribute_name}`, but only `name`, `type`, `minOccurs` and `maxOccurs` are handled"
                    )
            if extract_documentation(element) == "":
                raise ValueError(f"{file_name}: `{element_path}` has no `xs:documentation`")
            type_name: str | None = element.get("type")
            if type_name is None:
                # A leaf, written as the data dictionary writes its own: `<xs:complexType><xs:group ref="FLT_2D"/></xs:complexType>`
                leaf_complex_type: ET.Element | None = element.find(f"{XS_NS}complexType")
                leaf_components: list[ET.Element] = [] if leaf_complex_type is None else list(leaf_complex_type)
                if len(leaf_components) != 1 or leaf_components[0].tag != f"{XS_NS}group" or leaf_components[0].get("ref") not in BASE_TYPES:
                    raise ValueError(
                        f'{file_name}: `{element_path}` is a leaf, so holds only `<xs:group ref="..."/>`, naming one of the data dictionary\'s base types: '
                        f"{', '.join(sorted(BASE_TYPES))}. A structure is a top-level `xs:complexType`, which the element names in `type`"
                    )
                if "minOccurs" in element.attrib or "maxOccurs" in element.attrib:
                    raise ValueError(
                        f"{file_name}: `{element_path}` is a leaf, which is always there and is not repeated, so has no `minOccurs` or `maxOccurs`"
                    )
            else:
                if type_name not in structure_names and type_name not in support_type_names:
                    raise ValueError(
                        f"{file_name}: `{element_path}` has the type `{type_name}`, which is neither a structure of {file_name} nor of `dd_support.xsd`"
                    )
                is_array: bool = "maxOccurs" in element.attrib
                if is_array and (element.get("minOccurs") != "0" or element.get("maxOccurs") != "unbounded"):
                    raise ValueError(
                        f'{file_name}: `{element_path}` is an array of structures, which can be empty, so has `minOccurs="0" maxOccurs="unbounded"`'
                    )
                if not is_array and "minOccurs" in element.attrib:
                    raise ValueError(f"{file_name}: `{element_path}` is a single structure, which is always there, so has no `minOccurs`")

    complex_types, root_element = parse_xsd(path_to_schema)
    if root_element is None:
        raise ValueError(f"{file_name} has no root element")
    pulled: list[str] = resolve_supporting_types(complex_types, root_element, supporting_types)
    if pulled:
        print(f"Found {len(complex_types) - len(pulled)} structures, and {len(pulled)} of `dd_support.xsd`'s: {', '.join(sorted(pulled))}")
    else:
        print(f"Found {len(complex_types)} structures, and none of `dd_support.xsd`'s")

    # A type which is not resolved would be emitted as an empty stub, so is refused instead
    _used_base_types, unresolved_type_names = collect_used_types(complex_types, root_element)
    if unresolved_type_names:
        raise ValueError(f"{file_name}: {', '.join(sorted(unresolved_type_names))} cannot be resolved")

    root_struct_name: str = snake_to_pascal_case(root_element.name)
    root_name: str = root_element.name
    python_class_name: str = f"Py{root_struct_name}"
    schema_path_from_repository: str = Path(os.path.relpath(path_to_schema, REPO_ROOT)).as_posix()

    # The structs
    struct_lines, struct_dd_base_types_imports, struct_ndarray_imports = generate_rust_body(complex_types, root_element, "Root Structure")

    # The Python bindings: the static description of the paths, and the class holding a copy of the structure
    type_map: dict[str, ComplexType] = {}
    for ct in complex_types + [root_element]:
        type_map[snake_to_pascal_case(ct.name)] = ct
    lengths_registry: dict[tuple[str, ...], str] = {}
    root_nodes: list[PathNode] = build_path_tree(
        root_element,
        type_map,
        dd_path="",
        rust_expression=root_name,
        vec_expressions=[],
        vec_dd_paths=[],
        ancestry=(root_struct_name,),
        lengths_registry=lengths_registry,
    )
    paths_lines, paths_python_imports, paths_dd_base_types_imports = generate_python_paths_body(
        root_nodes, lengths_registry, root_element, root_name, "structure"
    )

    # `__repr__` gives the length of each array of structures at the root
    repr_parts: list[str] = []
    repr_arguments: list[str] = []
    for f in root_element.fields:
        if f.is_array:
            repr_parts.append(f"{f.name}={{}}")
            repr_arguments.append(f"self.inner.{sanitize_rust_identifier(f.name)}.len()")
    repr_format: str = f"{root_struct_name}({', '.join(repr_parts)})"
    repr_expression: str = (
        rust_string_literal(repr_format) if not repr_arguments else f"format!({rust_string_literal(repr_format)}, {', '.join(repr_arguments)})"
    )
    if not repr_arguments:
        repr_expression = f"{repr_expression}.to_string()"

    class_lines: list[str] = [
        "// " + "=" * 76,
        "// The Python class",
        "// " + "=" * 76,
        "",
        f"/// A copy of `{root_struct_name}`, readable from Python through paths from `gsfit_rs.imas.{root_name}_paths`.",
        "///",
        "/// It is constructed on the Rust side only, e.g. by a getter of the class holding the structure; Python cannot build one from",
        "/// scratch. It can be pickled, though, through `__reduce__`, which `_from_pickle_state` reads back.",
        f'#[pyclass(module = "gsfit_rs.imas", name = "{root_struct_name}")]',
        f"pub struct {python_class_name} {{",
        f"    pub inner: {root_struct_name},",
        "}",
        "",
        f"impl {python_class_name} {{",
        f"    pub fn new(inner: {root_struct_name}) -> Self {{",
        "        Self { inner }",
        "    }",
        "}",
        "",
        "#[pymethods]",
        f"impl {python_class_name} {{",
        f"    /// Read the data at `path`, a path from `gsfit_rs.imas.{root_name}_paths`, out of this copy.",
        "    ///",
        "    /// The shape of the result follows the shape of the index: an integer index gives one value, a slice gathers.",
        "    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {",
        f"        read_path(py, &self.inner, {rust_string_literal(root_name)}, path)",
        "    }",
        "",
        "    fn __repr__(&self) -> String {",
        f"        {repr_expression}",
        "    }",
        "",
        "    /// Pickle this copy. Its state is a `dict` of the version of the crate which pickled it, `version`, when, `pickled_datetime`,",
        f"    /// in seconds since 1970, and the `dict` of its nodes, `{root_name}`, see `imas_rs::python::PickleState`; `_from_pickle_state`",
        "    /// reads it back.",
        "    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {",
        "        let state: Bound<'py, PyDict> = PyDict::new(py);",
        '        state.set_item("version", env!("CARGO_PKG_VERSION"))?;',
        "        let pickled_datetime: f64 = SystemTime::now()",
        "            .duration_since(UNIX_EPOCH)",
        '            .map_err(|error: SystemTimeError| PyValueError::new_err(format!("The system clock is before 1970: {error}")))?',
        "            .as_secs_f64();",
        '        state.set_item("pickled_datetime", pickled_datetime)?;',
        f"        state.set_item({rust_string_literal(root_name)}, self.inner.to_pickle_state(py)?)?;",
        '        Ok((py.get_type::<Self>().getattr("_from_pickle_state")?, (state,)))',
        "    }",
        "",
        "    /// Unpickle: read back the state which `__reduce__` made.",
        "    ///",
        "    /// # Errors",
        "    /// `ValueError` if the state is not what `__reduce__` makes, e.g. it was pickled by a version of the crate whose structure",
        "    /// is different. A version whose structure is the same only gives a warning",
        "    #[staticmethod]",
        "    fn _from_pickle_state(state: &Bound<'_, PyDict>) -> PyResult<Self> {",
        "        let version: String = state",
        '            .get_item("version")?',
        f'            .ok_or_else(|| PyValueError::new_err("The pickled `{root_struct_name}` has no `version`"))?',
        "            .extract()?;",
        '        if version != env!("CARGO_PKG_VERSION") {',
        "            eprintln!(",
        f'                "Warning: unpickling `{root_struct_name}` pickled by {{}} v{{version}}, but this is v{{}}",',
        '                env!("CARGO_PKG_NAME"),',
        '                env!("CARGO_PKG_VERSION")',
        "            );",
        "        }",
        "        let inner_state: Bound<'_, PyAny> = state",
        f"            .get_item({rust_string_literal(root_name)})?",
        f'            .ok_or_else(|| PyValueError::new_err("The pickled `{root_struct_name}` has no `{root_name}`"))?;',
        f"        let inner: {root_struct_name} =",
        f'            {root_struct_name}::from_pickle_state(&inner_state).map_err(|error: PickleStateError| error.into_py_err("`{root_struct_name}`"))?;',
        "        Ok(Self::new(inner))",
        "    }",
        "}",
    ]

    # The pickling of the structs, which are in the same module
    pickle_lines: list[str] = generate_python_pickle_body(complex_types, root_element, "")

    # The module: its documentation, the imports of all its parts, then the parts: the structs, the paths, the class, and the pickling
    dd_base_types_imports: list[str] = sorted(set(struct_dd_base_types_imports) | set(paths_dd_base_types_imports))
    python_imports: list[str] = sorted(set(paths_python_imports) | {"PickleState", "PickleStateError", "PyPath", "read_path", "structure_nodes"})
    lines: list[str] = [
        f"//! `{root_struct_name}`, a structure of GSFit's own, which is not part of any IDS: its structs, their Python bindings, and their",
        "//! pickling",
        "//!",
        f"//! **WARNING** This file is autogenerated by `rust/imas_rs/imas_updater/build_ids.py`, from the XSD schema `{schema_path_from_repository}`.",
        "//! Any changes will be overwritten.",
        "//!",
        "//! It is generated as an IDS is, so its structs have the same accessors: an array of structures is selected from with a range,",
        "//! `.<array>(..)`, whose leaves then gather one value per element, or, where its elements have a `name`, with a name,",
        "//! `.<array>(name)`; see `imas_rs::dd_base_types`.",
        "//!",
        f"//! Its Python bindings are an IDS's too: the static description of its paths, `{root_name.upper()}_ROOT`, which",
        f"//! `gsfit_rs.imas.{root_name}_paths` is, and `{python_class_name}`, the class holding a copy of it, readable through them, as",
        "//! `imas_rs::python::PyEquilibrium` is. Each leaf carries the function that reads it out of the structure, so the path a reader",
        "//! follows is written once and checked by the compiler. The structs are pickled as an IDS's are, through",
        "//! `imas_rs::python::PickleState`, which the class pickles with.",
        "//!",
        "//! An IDS's structs, Python bindings and pickling are generated into three files, as `imas_rs` has its Python bindings behind",
        "//! its optional `python` feature. This crate always has them, so they are in one file.",
        "",
        "#![allow(clippy::all)]",
        "#![allow(dead_code)]",
        "#![allow(non_camel_case_types)]",
        "",
        f"use imas_rs::dd_base_types::{{{', '.join(dd_base_types_imports)}}};",
        f"use imas_rs::python::{{{', '.join(python_imports)}}};",
    ]
    if struct_ndarray_imports:
        lines.append(f"use ndarray::{{{', '.join(struct_ndarray_imports)}}};")
    lines += [
        "use pyo3::exceptions::PyValueError;",
        "use pyo3::prelude::*;",
        "use pyo3::types::PyDict;",
        "use std::any::Any;",
        "use std::time::{SystemTime, SystemTimeError, UNIX_EPOCH};",
        "",
    ]
    lines.extend(struct_lines)
    lines.append("")
    lines.extend(paths_lines)
    lines.append("")
    lines.extend(class_lines)
    lines.append("")
    lines.append("// " + "=" * 76)
    lines.append("// Pickling")
    lines.append("// " + "=" * 76)
    lines.extend(pickle_lines)

    output_path: Path = Path(path_to_rust_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Generated: {output_path}")

    # Format, so that the committed file matches what `cargo fmt` would produce
    subprocess.run(
        [
            "rustfmt",
            "--edition",
            RUST_EDITION,
            "--config-path",
            str(WORKSPACE_DIR),
            str(output_path),
        ],
        check=True,
    )
    print(f"Formatted: {output_path}")

    return generate_python_stub_section(root_nodes, root_element, type_map, root_name, root_kind="structure")


def get_imas_dd_version() -> str:
    """
    The version of the local IMAS Data Dictionary clone, as `git describe` reports it,
    e.g. `4.1.1-60-gf5d44e8`. A `-dirty` suffix means the clone had uncommitted changes, so
    the generated files cannot be reproduced from the Data Dictionary repository alone.
    """

    completed_process = subprocess.run(
        ["git", "-C", str(DATA_DICTIONARY_DIR), "describe", "--tags", "--dirty"],
        capture_output=True,
        text=True,
        check=True,
    )

    return completed_process.stdout.strip()


if __name__ == "__main__":
    if not DATA_DICTIONARY_DIR.is_dir():
        raise FileNotFoundError(f"IMAS Data Dictionary not found at {DATA_DICTIONARY_DIR}\nSee rust/imas_rs/imas_updater/README.md for how to clone it.")

    # Read before generating anything, so a clone that `git` cannot describe fails immediately
    imas_dd_version: str = get_imas_dd_version()

    ids_names: list[str] = [
        "equilibrium",
        "pf_active",
        "pf_passive",
        "pulse_schedule",
        "tf",
        "wall",
        "magnetics",
        # For the reconstruction consistent with the neoclassical current (paper/gsfit_consistent_with_neoclassical_current):
        # the kinetic profiles, current sources, conductivity and current diffusion
        "core_profiles",
        "core_sources",
        "core_transport",
        "transport_solver_numerics",
        # the neutral beams and their fast ions
        "nbi",
        "distribution_sources",
        "distributions",
        # the kinetic measurements
        "thomson_scattering",
    ]

    # Which IDSs get Python path bindings (`<ids>_paths` and a `get`) and pickling. Adding one here also
    # needs a matching `#[pyclass]` wrapper, and `mod <ids>_paths;` and `mod <ids>_pickle;` in `src/python/mod.rs`.
    ids_names_with_python_paths: set[str] = {
        "equilibrium",
        "pf_active",
        "pf_passive",
        "pulse_schedule",
        "tf",
        "wall",
        "magnetics",
        "core_profiles",
        "core_sources",
    }

    stub_sections: list[str] = [generate_python_stub_preamble()]

    for ids_name in ids_names:
        with_paths = ids_name in ids_names_with_python_paths
        section = build_ids(
            path_to_ids_schema=DATA_DICTIONARY_DIR / "schemas" / ids_name,
            path_to_rust_ids_file=CRATE_DIR / "src" / "ids" / f"{ids_name}.rs",
            path_to_rust_paths_file=(CRATE_DIR / "src" / "python" / f"{ids_name}_paths.rs") if with_paths else None,
            path_to_rust_pickle_file=(CRATE_DIR / "src" / "python" / f"{ids_name}_pickle.rs") if with_paths else None,
            generate_python_stub_for_this_ids=with_paths,
        )
        if section is not None:
            stub_sections.append(section)

    # The global elements of `dd_support.xsd` which are not IDSs, each generated into `src/dd_support/`.
    # Adding one here also needs a matching `pub mod <name>;` in `src/dd_support/mod.rs`.
    dd_support_element_names: list[str] = ["parameters_input"]

    for dd_support_element_name in dd_support_element_names:
        build_dd_support(
            element_name=dd_support_element_name,
            path_to_dd_support_schema=DD_SUPPORT_XSD_FILE,
            path_to_rust_file=CRATE_DIR / "src" / "dd_support" / f"{dd_support_element_name}.rs",
        )

    # The structures of GSFit's own which are not part of any IDS, each generated into the crate which holds it, with its Python
    # bindings, whose section of the type stub shares `gsfit_rs.imas`'s, as the crate mounts them in it
    stub_sections.append(
        build_structure(
            path_to_schema=GREENS_STORAGE_XSD_FILE,
            path_to_rust_file=GREENS_STORAGE_RUST_FILE,
        )
    )

    PYTHON_STUB_FILE.parent.mkdir(parents=True, exist_ok=True)
    PYTHON_STUB_FILE.write_text("\n".join(stub_sections))
    print(f"Generated: {PYTHON_STUB_FILE}")

    # Written last, so that it only ever describes a run which regenerated every file above
    IMAS_DD_VERSION_FILE.write_text(f"{imas_dd_version}\n")
    print(f"Generated: {IMAS_DD_VERSION_FILE} -> {imas_dd_version}")
