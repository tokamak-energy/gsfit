"""
IMAS IDS Rust Code Generator

Generates Rust struct definitions from IMAS Data Dictionary XSD schema files.
Includes accessor, view, and accumulator types for slicing Vec<T> fields.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import subprocess
import xml.etree.ElementTree as ET
import re


# Location of this script: `rust/imas_rs/imas_updater/`
IMAS_UPDATER_DIR: Path = Path(__file__).resolve().parent
# The `imas_rs` crate root: `rust/imas_rs/`
CRATE_DIR: Path = IMAS_UPDATER_DIR.parent
# The cargo workspace root: `rust/`, which holds `rustfmt.toml`
WORKSPACE_DIR: Path = CRATE_DIR.parent
# Local clone of the IMAS Data Dictionary; see `README.md` for the pinned version
DATA_DICTIONARY_DIR: Path = IMAS_UPDATER_DIR / "IMAS-Data-Dictionary"
# Rust edition of the `imas_rs` crate, needed so that `rustfmt` sorts imports correctly
RUST_EDITION: str = "2024"
# Hand-written, non-IMAS keys spliced into the generated structs; see
# `add_custom_keys_to_equilibrium_ids`
CUSTOM_KEYS_DIR: Path = CRATE_DIR / "src" / "ids"


# XML Schema namespace
XS_NS = "{http://www.w3.org/2001/XMLSchema}"


# Mapping from base types to their Array1 types for accumulators
SCALAR_TO_ARRAY = {
    "FLT_0D": ("f64", "Array1<f64>"),
    "INT_0D": ("i32", "Array1<i32>"),
    "STR_0D": ("String", "Vec<String>"),  # Strings are cloned
    "CPX_0D": ("Complex64", "Array1<Complex64>"),
}

# The generic accumulator each scalar type uses when gathered across an array of
# structures. `Accumulator` gathers into `Array1<U>`; `StringAccumulator` gathers
# into `Vec<String>` because `String` is not `Copy`. Both are hand-written once in
# `dd_base_types.rs` and parameterised by a projection closure, so the DD field
# path is emitted exactly once - inside the closure - instead of being baked into
# a generated per-field struct. `{parent}` is the array-of-structures element type.
#
# The bool is whether the projection has to `.clone()` the value out.
SCALAR_ACCUMULATOR = {
    "FLT_0D": ("Accumulator<'a, {parent}, FLT_0D>", False),
    "INT_0D": ("Accumulator<'a, {parent}, INT_0D>", False),
    "STR_0D": ("StringAccumulator<'a, {parent}>", True),
    "CPX_0D": ("Accumulator<'a, {parent}, CPX_0D>", False),
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


def pascal_case_to_snake(name: str) -> str:
    """Convert PascalCase back to snake_case.

    Custom structs are declared by their Rust (PascalCase) name, while `ComplexType.name` holds
    the data dictionary's snake_case name and is converted on the way out. Storing the converted
    form keeps a custom struct's emitted name identical to what was written by hand.
    """
    return re.sub(r"(?<!^)(?=[A-Z0-9])", "_", name).lower()


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
    """Extract documentation string from xs:annotation/xs:documentation."""
    annotation = element.find(f"{XS_NS}annotation")
    if annotation is not None:
        doc = annotation.find(f"{XS_NS}documentation")
        if doc is not None and doc.text:
            return doc.text.strip()
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

    # Check if it's an array (unbounded)
    max_occurs = element.get("maxOccurs", "1")
    is_array = max_occurs == "unbounded"

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


def collect_used_types(
    complex_types: list[ComplexType], root_element: Optional[ComplexType]
) -> tuple[set[str], set[str]]:
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


def find_vec_field_types(
    complex_types: list[ComplexType], root_element: Optional[ComplexType]
) -> dict[str, ComplexType]:
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


def scalar_accumulator_field(
    f: Field, field_name: str, parent_type_name: str, field_path: str
) -> tuple[str, str]:
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
    accumulator_type, needs_clone = SCALAR_ACCUMULATOR[f.rust_type]
    accumulator_type = accumulator_type.format(parent=parent_type_name)
    accumulator_ctor = accumulator_type.split("<")[0]

    projection = f"|item: &{parent_type_name}| item.{sanitize_field_path(field_path)}"
    if needs_clone:
        projection += ".clone()"

    decl = f"    pub {field_name}: {accumulator_type},"
    init = (
        f"            {field_name}: {accumulator_ctor}::new("
        f'data, {projection}, "{field_path}"),'
    )
    return decl, init


def generate_view_for_struct(
    struct_name: str,
    struct_ct: ComplexType,
    parent_type_name: str,
    field_path_prefix: str,
    type_map: dict[str, ComplexType],
    generated_types: set[str],
    ancestry: tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """
    Generate a View struct for a nested struct type.
    Returns (code, list of field initializers).
    Uses generated_types to avoid creating duplicate type definitions.

    View names are keyed on the parent Vec element type *and the full field path*
    (see `view_name_for`), so a DD type reused at several paths gets one view per
    path rather than one shared view reading a single hard-coded path.

    `ancestry` carries the DD types currently being expanded, so a self-referential
    schema stops instead of recursing forever (path-keyed names grow with depth and
    so cannot terminate the recursion the way type-keyed names accidentally did).
    """
    view_name = view_name_for(parent_type_name, field_path_prefix)
    if view_name in generated_types:
        return "", []

    generated_types.add(view_name)

    lines = []
    field_defs = []
    field_inits = []
    nested_code = []
    has_fields = False

    for f in struct_ct.fields:
        field_name = sanitize_rust_identifier(f.name)
        field_path = f"{field_path_prefix}.{f.name}" if field_path_prefix else f.name

        if f.is_scalar:
            has_fields = True
            decl, init = scalar_accumulator_field(
                f, field_name, parent_type_name, field_path
            )
            field_defs.append(decl)
            field_inits.append(init)

        elif f.is_struct and f.rust_type in type_map:
            if f.rust_type in ancestry:
                # Self-referential DD type: stop expanding rather than recurse forever.
                continue
            has_fields = True
            # Nested struct - one view per field path (not per DD type)
            nested_view_name = view_name_for(parent_type_name, field_path)
            field_defs.append(f"    pub {field_name}: {nested_view_name}<'a>,")
            field_inits.append(
                f"            {field_name}: {nested_view_name}::new(data),"
            )

            # Recursively generate the nested view
            nested_struct_code, _ = generate_view_for_struct(
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

    # If no fields, add PhantomData to use the lifetime
    if not has_fields:
        field_defs.append(
            f"    _phantom: std::marker::PhantomData<&'a {parent_type_name}>,"
        )
        field_inits.append(f"            _phantom: std::marker::PhantomData,")

    # Use _data when data isn't actually used (no scalar or struct fields)
    data_param = "_data" if not has_fields else "data"

    # Generate the view struct
    lines.append(
        f"/// View over `{field_path_prefix}` ({struct_name}) across multiple {parent_type_name}"
    )
    lines.append(f"pub struct {view_name}<'a> {{")
    lines.extend(field_defs)
    lines.append(f"}}")
    lines.append(f"")
    lines.append(f"impl<'a> {view_name}<'a> {{")
    lines.append(f"    pub fn new({data_param}: &'a [{parent_type_name}]) -> Self {{")
    lines.append(f"        Self {{")
    for init in field_inits:
        lines.append(init)
    lines.append(f"        }}")
    lines.append(f"    }}")
    lines.append(f"}}")
    lines.append(f"")

    # Combine nested code first, then this view
    all_code = (
        "\n".join(nested_code) + "\n" + "\n".join(lines)
        if nested_code
        else "\n".join(lines)
    )
    return all_code, field_inits


def generate_slice_view(
    type_name: str,
    ct: ComplexType,
    type_map: dict[str, ComplexType],
    generated_types: set[str],
) -> str:
    """
    Generate SliceView and SliceViewMut for a type used in Vec<T>.
    Uses generated_types to avoid creating duplicate type definitions.
    """
    lines = []

    # First generate accumulators and views for all fields
    view_field_defs = []
    view_field_inits = []
    accumulated_code = []

    for f in ct.fields:
        field_name = sanitize_rust_identifier(f.name)

        if f.is_scalar:
            decl, init = scalar_accumulator_field(f, field_name, type_name, f.name)
            view_field_defs.append(decl)
            view_field_inits.append(init)

        elif f.is_struct and f.rust_type in type_map:
            # Nested struct - one view per field path (not per DD type)
            nested_view_name = view_name_for(type_name, f.name)
            view_field_defs.append(f"    pub {field_name}: {nested_view_name}<'a>,")
            view_field_inits.append(
                f"            {field_name}: {nested_view_name}::new(data),"
            )

            nested_code, _ = generate_view_for_struct(
                f.rust_type,
                type_map[f.rust_type],
                type_name,
                f.name,
                type_map,
                generated_types,  # Pass the shared tracking set
                (type_name, f.rust_type),
            )
            if nested_code:
                accumulated_code.append(nested_code)

    # Add all accumulated code (accumulators and nested views)
    lines.extend(accumulated_code)

    # Generate the SliceView struct
    slice_view_name = f"{type_name}SliceView"
    lines.append(f"/// View over multiple {type_name} with field accumulation")
    lines.append(f"pub struct {slice_view_name}<'a> {{")
    lines.append(f"    data: &'a [{type_name}],")
    for field_def in view_field_defs:
        lines.append(field_def)
    lines.append(f"}}")
    lines.append(f"")
    lines.append(f"impl<'a> {slice_view_name}<'a> {{")
    lines.append(f"    pub fn new(data: &'a [{type_name}]) -> Self {{")
    lines.append(f"        Self {{")
    lines.append(f"            data,")
    for init in view_field_inits:
        lines.append(init)
    lines.append(f"        }}")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    pub fn len(&self) -> usize {{")
    lines.append(f"        self.data.len()")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    pub fn is_empty(&self) -> bool {{")
    lines.append(f"        self.data.is_empty()")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    pub fn iter(&self) -> impl Iterator<Item = &{type_name}> {{")
    lines.append(f"        self.data.iter()")
    lines.append(f"    }}")
    lines.append(f"}}")
    lines.append(f"")

    # Generate SliceViewMut
    slice_view_mut_name = f"{type_name}SliceViewMut"
    lines.append(f"/// Mutable view over multiple {type_name}")
    lines.append(f"pub struct {slice_view_mut_name}<'a> {{")
    lines.append(f"    data: &'a mut [{type_name}],")
    lines.append(f"}}")
    lines.append(f"")
    lines.append(f"impl<'a> {slice_view_mut_name}<'a> {{")
    lines.append(f"    pub fn new(data: &'a mut [{type_name}]) -> Self {{")
    lines.append(f"        Self {{ data }}")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    pub fn len(&self) -> usize {{")
    lines.append(f"        self.data.len()")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    pub fn is_empty(&self) -> bool {{")
    lines.append(f"        self.data.is_empty()")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(
        f"    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut {type_name}> {{"
    )
    lines.append(f"        self.data.iter_mut()")
    lines.append(f"    }}")
    lines.append(f"}}")
    lines.append(f"")

    return "\n".join(lines)


def generate_index_traits(type_name: str) -> str:
    """Generate Index and MutIndex traits for a Vec<T> element type.

    This enables calling .field(0) for single element access and
    .field(0..2) for range access using the same method name.
    """
    slice_view_name = f"{type_name}SliceView"
    slice_view_mut_name = f"{type_name}SliceViewMut"
    index_trait_name = f"{type_name}Index"
    mut_index_trait_name = f"{type_name}MutIndex"

    lines = [
        f"/// Index trait for {type_name} - enables .field(0) and .field(0..2) syntax",
        f"pub trait {index_trait_name}<'a> {{",
        f"    type Output;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output;",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for usize {{",
        f"    type Output = &'a {type_name};",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        &data[self]",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for std::ops::Range<usize> {{",
        f"    type Output = {slice_view_name}<'a>;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        {slice_view_name}::new(&data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for std::ops::RangeFrom<usize> {{",
        f"    type Output = {slice_view_name}<'a>;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        {slice_view_name}::new(&data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for std::ops::RangeTo<usize> {{",
        f"    type Output = {slice_view_name}<'a>;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        {slice_view_name}::new(&data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for std::ops::RangeInclusive<usize> {{",
        f"    type Output = {slice_view_name}<'a>;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        {slice_view_name}::new(&data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for std::ops::RangeToInclusive<usize> {{",
        f"    type Output = {slice_view_name}<'a>;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        {slice_view_name}::new(&data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {index_trait_name}<'a> for std::ops::RangeFull {{",
        f"    type Output = {slice_view_name}<'a>;",
        f"    fn get(self, data: &'a [{type_name}]) -> Self::Output {{",
        f"        {slice_view_name}::new(data)",
        f"    }}",
        f"}}",
        f"",
        f"/// Mutable index trait for {type_name} - enables .field_mut(0) and .field_mut(0..2) syntax",
        f"pub trait {mut_index_trait_name}<'a> {{",
        f"    type Output;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output;",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for usize {{",
        f"    type Output = &'a mut {type_name};",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        &mut data[self]",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for std::ops::Range<usize> {{",
        f"    type Output = {slice_view_mut_name}<'a>;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        {slice_view_mut_name}::new(&mut data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for std::ops::RangeFrom<usize> {{",
        f"    type Output = {slice_view_mut_name}<'a>;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        {slice_view_mut_name}::new(&mut data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for std::ops::RangeTo<usize> {{",
        f"    type Output = {slice_view_mut_name}<'a>;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        {slice_view_mut_name}::new(&mut data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for std::ops::RangeInclusive<usize> {{",
        f"    type Output = {slice_view_mut_name}<'a>;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        {slice_view_mut_name}::new(&mut data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for std::ops::RangeToInclusive<usize> {{",
        f"    type Output = {slice_view_mut_name}<'a>;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        {slice_view_mut_name}::new(&mut data[self])",
        f"    }}",
        f"}}",
        f"",
        f"impl<'a> {mut_index_trait_name}<'a> for std::ops::RangeFull {{",
        f"    type Output = {slice_view_mut_name}<'a>;",
        f"    fn get_mut(self, data: &'a mut [{type_name}]) -> Self::Output {{",
        f"        {slice_view_mut_name}::new(data)",
        f"    }}",
        f"}}",
        f"",
    ]
    return "\n".join(lines)


def generate_vec_field_impl(parent_ct: ComplexType, field: Field) -> str:
    """
    Generate impl block methods for a Vec<T> field in a struct.
    Uses the Index traits to enable .field(0) and .field(0..2) syntax.
    """
    inner_type = field.inner_type
    field_name = sanitize_rust_identifier(field.name)
    index_trait_name = f"{inner_type}Index"
    mut_index_trait_name = f"{inner_type}MutIndex"
    parent_name = snake_to_pascal_case(parent_ct.name)

    lines = [
        f"impl {parent_name} {{",
        f"    /// Access {field_name} - use index for single element or range for slice view",
        f"    /// e.g. `.{field_name}(0)` returns `&{inner_type}`, `.{field_name}(0..2)` returns `{inner_type}SliceView`",
        f"    pub fn {field_name}<'a, I: {index_trait_name}<'a>>(&'a self, index: I) -> I::Output {{",
        f"        index.get(&self.{field_name})",
        f"    }}",
        f"",
        f"    /// Access {field_name} mutably - use index for single element or range for slice view",
        f"    /// e.g. `.{field_name}_mut(0)` returns `&mut {inner_type}`, `.{field_name}_mut(0..2)` returns `{inner_type}SliceViewMut`",
        f"    pub fn {field_name}_mut<'a, I: {mut_index_trait_name}<'a>>(&'a mut self, index: I) -> I::Output {{",
        f"        index.get_mut(&mut self.{field_name})",
        f"    }}",
        f"",
        f"    /// Get the number of {field_name} elements",
        f"    pub fn {field_name}_len(&self) -> usize {{",
        f"        self.{field_name}.len()",
        f"    }}",
        f"}}",
        f"",
    ]
    return "\n".join(lines)


def generate_all_views_and_accessors(
    complex_types: list[ComplexType], root_element: Optional[ComplexType]
) -> str:
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
            lines.append(generate_index_traits(type_name))

        # Generate impl blocks for structs with Vec fields
        lines.append("// " + "=" * 76)
        lines.append("// Struct Impl Blocks for Vec Field Access")
        lines.append("// " + "=" * 76)
        lines.append("")

        for ct in all_types:
            for f in ct.fields:
                if f.is_array and f.inner_type in vec_types:
                    lines.append(generate_vec_field_impl(ct, f))

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


def generate_rust_struct(ct: ComplexType) -> str:
    """Generate Rust struct code for a ComplexType."""
    lines = []

    # Documentation
    if ct.documentation:
        for line in ct.documentation.split("\n"):
            lines.append(f"/// {line.strip()}")

    # Struct definition
    struct_name = snake_to_pascal_case(ct.name)
    lines.append("#[derive(Debug, Clone, Default)]")
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
        # Every base-type leaf is wrapped in Option so that "unset" (None) is unambiguously
        # distinct from a real value - 0.0 is a valid measurement, and a zero-length array is
        # a valid result. This matters most for a freshly constructed IDS, where every leaf
        # must read as absent rather than as an empty array someone might mistake for data.
        # Arrays of structures (Vec<T>) and nested structs keep their natural empty state.
        field_type = f"Option<{f.rust_type}>" if f.is_base_type else f.rust_type
        lines.append(f"    pub {field_name}: {field_type},")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_root_constructors(
    root_element: Optional[ComplexType], complex_types: list[ComplexType]
) -> str:
    """Generate sizing constructors (`with_size`, `with_time`) for the root IDS.

    `with_size(n_time)` pre-populates the `time_slice` array with `n_time` default
    (all-`None`) slices. `with_time(&[FLT_0D])` additionally sets each slice's
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
    has_time_field = inner_ct is not None and any(
        f.name == "time" and f.is_scalar for f in inner_ct.fields
    )

    lines = [
        "// " + "=" * 76,
        f"// {root_name} Constructors",
        "// " + "=" * 76,
        "",
        f"impl {root_name} {{",
        f"    /// Create a `{root_name}` pre-populated with `n_time` default (empty) time slices.",
        f"    ///",
        f"    /// Every leaf field in each slice is unset (`None`), ready to be filled in,",
        f"    /// e.g. via `time_slice.par_iter_mut()`.",
        f"    pub fn with_size(n_time: usize) -> Self {{",
        f"        let mut ids = Self::default();",
        f"        ids.time_slice = (0..n_time).map(|_| {inner_type}::default()).collect();",
        f"        ids",
        f"    }}",
    ]

    if has_time_field:
        lines.extend(
            [
                f"",
                f"    /// Create a `{root_name}` with one time slice per entry in `time`,",
                f"    /// setting each slice's `time` field. All other leaf fields are unset (`None`).",
                f"    pub fn with_time(time: &FLT_1D) -> Self {{",
                f"        let mut ids = Self::with_size(time.len());",
                f"        for (slice, &t) in ids.time_slice.iter_mut().zip(time.iter()) {{",
                f"            slice.time = Some(t);",
                f"        }}",
                f"        ids",
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
        known_types: set[str] = {
            ct.get("name") for ct in included_root.findall(f"{XS_NS}complexType") if ct.get("name")
        }
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


def add_custom_keys_to_equilibrium_ids(
    complex_types: list[ComplexType],
    root_element: Optional[ComplexType],
    path_to_custom_keys_file: Path,
) -> None:
    """Splice hand-written, non-IMAS keys into the parsed IDS structs, in place.

    Rust cannot add a field to a struct declared in another file, so custom keys that must
    sit flat alongside the IMAS ones (`profiles_2d.d_psi_d_r`, not
    `profiles_2d.custom.d_psi_d_r`) have to be merged in before the struct is emitted.

    `path_to_custom_keys_file` is ordinary Rust syntax so that it reads like the generated
    file, but it is never compiled. Each `pub struct <Name>` in it names a *generated*
    struct, and its fields are appended to that struct. Only `pub <name>: <Type>,` lines and
    their preceding `///` comments are read; a `/// Units: <units>` line is taken as the
    field's units, exactly as the XSD parser does.

    Missing file is not an error: an IDS need not have any custom keys.
    """
    if not path_to_custom_keys_file.is_file():
        return

    source: str = path_to_custom_keys_file.read_text()

    # Strip the module-level `//!` header and any `use` lines, so they cannot be mistaken
    # for struct bodies
    source = re.sub(r"^\s*//!.*$", "", source, flags=re.MULTILINE)

    # The root element is included so that custom keys can be added to the IDS itself, e.g. the
    # `code` structure, and not only to the types nested inside it
    all_types: list[ComplexType] = complex_types + ([root_element] if root_element else [])
    types_by_struct_name: dict[str, ComplexType] = {
        snake_to_pascal_case(ct.name): ct for ct in all_types
    }

    struct_pattern = re.compile(r"pub\s+struct\s+(\w+)\s*\{(.*?)\n\}", re.DOTALL)
    field_pattern = re.compile(
        r"((?:\s*///.*\n)*)\s*pub\s+(\w+)\s*:\s*([^,]+?)\s*,", re.MULTILINE
    )

    # A struct name which is not already generated declares a *new* nested type, e.g. the
    # `bounding` sub-structure of `boundary`. It is validated at the end: every new type must be
    # referenced by some custom field, so that a typo'd struct name is still caught.
    new_struct_names: list[str] = []

    for struct_match in struct_pattern.finditer(source):
        struct_name: str = struct_match.group(1)
        struct_body: str = struct_match.group(2)

        if struct_name not in types_by_struct_name:
            new_complex_type = ComplexType(
                name=struct_name,
                documentation=f"Custom (non-IMAS) structure, declared in {path_to_custom_keys_file.name}",
            )
            # `generate_rust_struct` derives the Rust name from `name` via `snake_to_pascal_case`,
            # so store the PascalCase name in a form which round-trips back to itself
            new_complex_type.name = pascal_case_to_snake(struct_name)
            complex_types.append(new_complex_type)
            types_by_struct_name[struct_name] = new_complex_type
            new_struct_names.append(struct_name)
            print(f"  Custom struct: {struct_name}")

        ct: ComplexType = types_by_struct_name[struct_name]

        existing_field_names: list[str] = [f.name for f in ct.fields]

        for field_match in field_pattern.finditer(struct_body):
            doc_block: str = field_match.group(1)
            field_name: str = field_match.group(2)
            rust_type: str = field_match.group(3)

            if field_name in existing_field_names:
                raise ValueError(
                    f"{path_to_custom_keys_file.name} declares custom key "
                    f"`{struct_name}.{field_name}`, but the IMAS data dictionary already "
                    f"defines it. Remove the custom key and use the IMAS one."
                )

            # Split the `///` block into documentation lines and a units line
            documentation_lines: list[str] = []
            units: str = ""
            for doc_line in doc_block.splitlines():
                doc_text: str = doc_line.strip().removeprefix("///").strip()
                if not doc_text:
                    continue
                if doc_text.startswith("Units:"):
                    units = doc_text.removeprefix("Units:").strip()
                else:
                    documentation_lines.append(doc_text)

            ct.fields.append(
                Field(
                    name=field_name,
                    rust_type=rust_type,
                    documentation="\n".join(documentation_lines),
                    units=units,
                    is_array=rust_type.startswith("Vec<"),
                )
            )
            print(f"  Custom key: {struct_name}.{field_name}: {rust_type}")

    # A new struct which nothing refers to is almost certainly a mistyped struct name
    # Recomputed rather than reusing `all_types`, which was snapshotted before the loop: a new
    # struct may be referenced by a field on another new struct, e.g. `code.parameters`
    referenced_types: set[str] = {
        f.inner_type
        for ct in complex_types + ([root_element] if root_element else [])
        for f in ct.fields
    }
    for struct_name in new_struct_names:
        if struct_name not in referenced_types:
            raise ValueError(
                f"{path_to_custom_keys_file.name} declares `{struct_name}`, which is neither a "
                f"generated struct nor used as the type of any custom key. If it was meant to add "
                f"keys to an existing struct, check the spelling; if it is a new nested structure, "
                f"add a field of that type to its parent."
            )


def generate_rust_file(
    complex_types: list[ComplexType], root_element: Optional[ComplexType], ids_name: str
) -> str:
    """Generate complete Rust file content."""
    lines = []

    # Module documentation
    lines.append(f"//! IMAS {ids_name.capitalize()} IDS")
    lines.append("//!")
    lines.append(
        f"//! This module defines the {ids_name} Interface Data Structure (IDS)"
    )
    lines.append("//! Auto-generated from IMAS Data Dictionary XSD schema.")
    lines.append("")
    lines.append("#![allow(dead_code)]")
    lines.append("#![allow(non_camel_case_types)]")
    lines.append("")

    # Collect used types
    used_base_types, referenced_types = collect_used_types(complex_types, root_element)

    # Generate the view/accessor code up front: the gathering machinery it needs
    # from dd_base_types depends on which accumulators it actually emits, so the
    # import list can only be built once the code exists.
    views_code = generate_all_views_and_accessors(complex_types, root_element)

    # Imports
    imported = set(used_base_types)
    for accumulator in ("Accumulator", "StringAccumulator"):
        if re.search(rf"\b{accumulator}(::|<)", views_code):
            imported.add(accumulator)
    if imported:
        lines.append(f"use crate::dd_base_types::{{{', '.join(sorted(imported))}}};")
        lines.append("")

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
        lines.append("// Root IDS Structure")
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

    return "\n".join(lines)


def build_ids(path_to_ids_schema: Path, path_to_rust_ids_file: Path):
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
    """
    # Find the main XSD file (dd_<name>.xsd)
    schema_dir = Path(path_to_ids_schema)
    ids_name = schema_dir.name  # e.g., "equilibrium"

    xsd_files = list(schema_dir.glob(f"dd_{ids_name}.xsd"))
    if not xsd_files:
        raise FileNotFoundError(
            f"No XSD file found matching dd_{ids_name}.xsd in {schema_dir}"
        )

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

    # Merge in the hand-written, non-IMAS keys before anything is emitted, so that the
    # views and accumulators are generated for them too
    add_custom_keys_to_equilibrium_ids(
        complex_types=complex_types,
        root_element=root_element,
        path_to_custom_keys_file=CUSTOM_KEYS_DIR / f"custom_{ids_name}_keys.rs",
    )
    resolve_supporting_types(complex_types, root_element, supporting_types)

    if root_element:
        print(f"Root element: {root_element.name}")

    # Generate Rust code
    rust_code = generate_rust_file(complex_types, root_element, ids_name)

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


if __name__ == "__main__":
    if not DATA_DICTIONARY_DIR.is_dir():
        raise FileNotFoundError(
            f"IMAS Data Dictionary not found at {DATA_DICTIONARY_DIR}\n"
            "See rust/imas_rs/imas_updater/README.md for how to clone it."
        )

    ids_names: list[str] = ["equilibrium"]

    for ids_name in ids_names:
        build_ids(
            path_to_ids_schema=DATA_DICTIONARY_DIR / "schemas" / ids_name,
            path_to_rust_ids_file=CRATE_DIR / "src" / "ids" / f"{ids_name}.rs",
        )
