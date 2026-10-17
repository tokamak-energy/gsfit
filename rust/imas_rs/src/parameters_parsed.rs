//! Reading the leaves of `code/parameters_parsed` from the XML of `code/parameters`.
//!
//! `code/parameters` holds the settings an IDS was built with as XML text, and `code/parameters_parsed` holds the same settings read into
//! structs. `imas_updater/build_ids.py` generates those structs from the settings' XML schema, and the code which reads the XML into them,
//! `src/ids/<ids>_parameters_parsed.rs`. That code calls these functions, one for each kind of leaf. They follow the conventions of
//! GSFit's XML schemas, which are set out in `python/gsfit/settings_schema/xsd/GSFIT_code_settings.xsd`:
//! * `xs:double` is a `FLT_0D`
//! * `xs:nonNegativeInteger` is an `INT_0D`
//! * `xs:boolean` is an `INT_0D`, 1 for true and 0 for false, as the data dictionary has no boolean
//! * `xs:string`, or an enumeration of strings, is a `STR_0D`
//! * a space-separated list of numbers is a `FLT_1D`
//! * a matrix, an element holding only a repeated `row`, each a list of numbers, is a `FLT_2D`, of shape [n_rows, n_columns]
//!
//! Each takes the element's path in the IDS, e.g. `code/parameters_parsed/grid/n_r`, to say where an error is.

use crate::dd_base_types::{FLT_0D, FLT_1D, FLT_2D, INT_0D, STR_0D};
use ndarray::{Array1, Array2};
use roxmltree::Node;

/// The elements inside a struct's element, whose path is `path`.
///
/// # Errors
/// The element holds text other than whitespace, as a leaf does, rather than only elements
pub(crate) fn child_elements<'a, 'input>(node: Node<'a, 'input>, path: &str) -> Result<Vec<Node<'a, 'input>>, String> {
    let mut elements: Vec<Node<'a, 'input>> = Vec::new();
    for child in node.children() {
        if child.is_element() {
            elements.push(child);
        } else if child.is_text() && !child.text().unwrap_or("").trim().is_empty() {
            return Err(format!(
                "`{path}` holds the text \"{}\", but it is a structure, which holds only elements",
                child.text().unwrap_or("").trim()
            ));
        }
    }

    Ok(elements)
}

/// The error for an element which is not in the XML schema the structs were generated from, whose path is `path`
pub(crate) fn unexpected_element(path: &str) -> String {
    format!(
        "`{path}` is not in the XML schema `code/parameters_parsed` was generated from. If the XML schema has changed, re-run `rust/imas_rs/imas_updater/build_ids.py`"
    )
}

/// The text of a leaf's element, whose path is `path`. An element with no text, e.g. `<comment />`, has the text "".
///
/// # Errors
/// The element holds other elements, as a structure does, rather than text
fn leaf_text<'a>(node: Node<'a, '_>, path: &str) -> Result<&'a str, String> {
    for child in node.children() {
        if child.is_element() {
            return Err(format!(
                "`{path}` holds the element <{}>, but it is a leaf, which holds only text",
                child.tag_name().name()
            ));
        }
    }

    Ok(node.text().unwrap_or(""))
}

/// Read an `xs:double`.
///
/// # Errors
/// The text is not a number
pub(crate) fn read_double(node: Node<'_, '_>, path: &str) -> Result<FLT_0D, String> {
    let text: &str = leaf_text(node, path)?.trim();

    text.parse::<f64>().map_err(|_| format!("`{path}` is \"{text}\", which is not a number"))
}

/// Read an `xs:nonNegativeInteger`.
///
/// # Errors
/// The text is not an integer, or is too large for an `INT_0D`
pub(crate) fn read_integer(node: Node<'_, '_>, path: &str) -> Result<INT_0D, String> {
    let text: &str = leaf_text(node, path)?.trim();

    text.parse::<INT_0D>()
        .map_err(|_| format!("`{path}` is \"{text}\", which is not an integer, or is larger than {}", INT_0D::MAX))
}

/// Read an `xs:boolean`, as 1 for true and 0 for false: the data dictionary has no boolean.
///
/// # Errors
/// The text is not "true", "false", "1" or "0"
pub(crate) fn read_boolean(node: Node<'_, '_>, path: &str) -> Result<INT_0D, String> {
    let text: &str = leaf_text(node, path)?.trim();

    match text {
        "true" | "1" => Ok(1),
        "false" | "0" => Ok(0),
        _ => Err(format!("`{path}` is \"{text}\", which is not a boolean: \"true\", \"false\", \"1\" or \"0\"")),
    }
}

/// Read an `xs:string`, or an enumeration of strings. An `xs:string` keeps its whitespace, so the text is not trimmed.
///
/// # Errors
/// The element holds other elements, rather than text
pub(crate) fn read_string(node: Node<'_, '_>, path: &str) -> Result<STR_0D, String> {
    Ok(leaf_text(node, path)?.to_string())
}

/// Read a space-separated list of numbers. An element with no text, e.g. `<regularisations_weight />`, is an empty list.
///
/// # Errors
/// An item of the list is not a number
pub(crate) fn read_double_list(node: Node<'_, '_>, path: &str) -> Result<FLT_1D, String> {
    let text: &str = leaf_text(node, path)?;

    let mut values: Vec<f64> = Vec::new();
    for item in text.split_whitespace() {
        let value: f64 = item.parse::<f64>().map_err(|_| format!("`{path}` holds \"{item}\", which is not a number"))?;
        values.push(value);
    }

    Ok(Array1::from_vec(values))
}

/// Read a matrix: an element holding only a repeated `row`, each a list of numbers, into a `FLT_2D` of shape [n_rows, n_columns].
/// An element with no `row`, e.g. `<regularisations />`, is shape (0, 0).
///
/// # Errors
/// The element holds something other than `row`s, a `row` is empty or holds something which is not a number, or the `row`s are of
/// different lengths
pub(crate) fn read_matrix(node: Node<'_, '_>, path: &str) -> Result<FLT_2D, String> {
    let row_nodes: Vec<Node> = child_elements(node, path)?;
    let n_rows: usize = row_nodes.len();
    if n_rows == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let mut values: Vec<f64> = Vec::new();
    let mut n_columns: usize = 0;
    for i_row in 0..n_rows {
        let row_path: String = format!("{path}/row({i_row})");
        if row_nodes[i_row].tag_name().name() != "row" {
            return Err(format!(
                "`{path}` holds the element <{}>, but it is a matrix, which holds only `row`s",
                row_nodes[i_row].tag_name().name()
            ));
        }
        let row: FLT_1D = read_double_list(row_nodes[i_row], &row_path)?;
        if row.is_empty() {
            return Err(format!("`{row_path}` is empty, but each row of a matrix needs at least one number"));
        }
        if i_row == 0 {
            n_columns = row.len();
        } else if row.len() != n_columns {
            return Err(format!(
                "`{row_path}` has {} number(s), but `{path}/row(0)` has {n_columns}; each row of a matrix needs the same number",
                row.len()
            ));
        }
        values.extend(row.iter());
    }

    Ok(Array2::from_shape_vec((n_rows, n_columns), values).expect("the rows have been checked to be of the same length"))
}
