//! Reading the leaves of settings, for the Rust files generated from XML schemas by `generate_rust_structs`.
//!
//! `generate_rust_structs`, in `python/diagnostic_and_simulation_base/xml_reader_and_writer/generate_rust_structs.py`, generates the structs holding
//! settings, and the code reading their XML into them, e.g. `settings_tf.rs`. That code calls these hand-written functions, one for each kind of leaf,
//! and for its errors. A leaf is read as the Rust type of its XML schema type:
//! * `xs:double`, or a type restricting it, e.g. `FiniteDouble`, is an `f64`
//! * `xs:positiveInteger` or `xs:nonNegativeInteger`, or a type restricting either, is a `usize`
//! * `xs:integer`, or a type restricting it, is an `i64`
//! * `xs:boolean`, or a type restricting it, e.g. `TrueOrFalse`, is a `bool`
//! * `xs:string`, or a type restricting it, e.g. an enumeration, is a `String`
//! * a space-separated list of numbers is an `Array1<f64>`
//! * a matrix, an element holding only `row`s, each a space-separated list of numbers, is an `Array2<f64>`
//! * a nillable element is nil when it has `xsi:nil="true"`, for Rust's `None`
//!
//! Each takes the element's path, e.g. `tf_settings/workflow/tree(0)/pulseNo`, to say where an error is.

use ndarray::{Array1, Array2};
use roxmltree::Node;

/// The namespace of the `xsi:nil` attribute, which marks a nillable element as nil
const XSI_NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema-instance";

/// The elements inside a struct's element, whose path is `path`, in their order in the XML.
///
/// # Errors
/// The element holds text other than whitespace, as a leaf does, rather than only elements
pub(crate) fn child_elements<'a, 'input>(node: Node<'a, 'input>, path: &str) -> Result<Vec<Node<'a, 'input>>, String> {
    let mut elements: Vec<Node<'a, 'input>> = Vec::new();
    for child in node.children() {
        if child.is_element() {
            elements.push(child);
        } else if child.is_text() {
            let text: &str = child.text().unwrap_or("").trim();
            if !text.is_empty() {
                return Err(format!("`{path}` holds the text \"{text}\", but it holds only elements"));
            }
        }
    }

    Ok(elements)
}

/// Whether a nillable element is nil: whether it has `xsi:nil="true"`, or `xsi:nil="1"`, the other way of writing an `xs:boolean`'s true
pub(crate) fn is_nil(node: Node<'_, '_>) -> bool {
    match node.attribute((XSI_NAMESPACE, "nil")) {
        Some(nil) => matches!(nil.trim(), "true" | "1"),
        None => false,
    }
}

/// The text of a leaf's element, whose path is `path`: its pieces of text joined, e.g. those on either side of a comment. An element with no
/// text, e.g. `<boundary_r />`, has the text "".
///
/// # Errors
/// The element holds other elements, as a struct's element does, rather than only text
fn leaf_text(node: Node<'_, '_>, path: &str) -> Result<String, String> {
    let mut text: String = String::new();
    for child in node.children() {
        if child.is_element() {
            return Err(format!(
                "`{path}` holds the element <{}>, but it is a leaf, which holds only text",
                child.tag_name().name()
            ));
        }
        if child.is_text() {
            text.push_str(child.text().unwrap_or(""));
        }
    }

    Ok(text)
}

/// Read an `xs:double`, or a type restricting it, e.g. `FiniteDouble`.
///
/// # Errors
/// The text is not a number
pub(crate) fn read_double(node: Node<'_, '_>, path: &str) -> Result<f64, String> {
    let text: String = leaf_text(node, path)?;
    let value_text: &str = text.trim();

    value_text
        .parse::<f64>()
        .map_err(|_| format!("`{path}` is \"{value_text}\", which is not a number"))
}

/// Read an `xs:positiveInteger` or `xs:nonNegativeInteger`, or a type restricting either, as a `usize`.
///
/// # Errors
/// The text is not an integer of at least 0, or is larger than `usize::MAX`
pub(crate) fn read_unsigned_integer(node: Node<'_, '_>, path: &str) -> Result<usize, String> {
    let text: String = leaf_text(node, path)?;
    let value_text: &str = text.trim();
    // An `xs:nonNegativeInteger` can write zero with a minus sign, e.g. "-0", which `usize` does not parse
    let unsigned_text: &str = match value_text.strip_prefix('-') {
        Some(digits) if !digits.is_empty() && digits.trim_start_matches('0').is_empty() => digits,
        _ => value_text,
    };

    unsigned_text.parse::<usize>().map_err(|_| {
        format!(
            "`{path}` is \"{value_text}\", which is not an integer of at least 0, or is larger than {}",
            usize::MAX
        )
    })
}

/// Read an `xs:integer`, or a type restricting it, as an `i64`.
///
/// # Errors
/// The text is not an integer, or is outside the range of an `i64`
pub(crate) fn read_integer(node: Node<'_, '_>, path: &str) -> Result<i64, String> {
    let text: String = leaf_text(node, path)?;
    let value_text: &str = text.trim();

    value_text.parse::<i64>().map_err(|_| {
        format!(
            "`{path}` is \"{value_text}\", which is not an integer, or is outside the range {} to {}",
            i64::MIN,
            i64::MAX
        )
    })
}

/// Read an `xs:boolean`, or a type restricting it, e.g. `TrueOrFalse`.
///
/// # Errors
/// The text is not "true", "false", "1" or "0"
pub(crate) fn read_boolean(node: Node<'_, '_>, path: &str) -> Result<bool, String> {
    let text: String = leaf_text(node, path)?;
    let value_text: &str = text.trim();

    match value_text {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(format!(
            "`{path}` is \"{value_text}\", which is not a boolean: \"true\", \"false\", \"1\" or \"0\""
        )),
    }
}

/// Read an `xs:string`, or a type restricting it, e.g. an enumeration. An `xs:string` keeps its whitespace, so the text is not trimmed.
///
/// # Errors
/// The element holds other elements, rather than text
pub(crate) fn read_string(node: Node<'_, '_>, path: &str) -> Result<String, String> {
    leaf_text(node, path)
}

/// Read a space-separated list of numbers, an `xs:list` of `xs:double`, or of a type restricting it. An element with no text, e.g.
/// `<boundary_r />`, is an empty list.
///
/// # Errors
/// An item of the list is not a number
pub(crate) fn read_double_list(node: Node<'_, '_>, path: &str) -> Result<Array1<f64>, String> {
    let text: String = leaf_text(node, path)?;

    let mut values: Vec<f64> = Vec::new();
    for item in text.split_whitespace() {
        let value: f64 = item.parse::<f64>().map_err(|_| format!("`{path}` holds \"{item}\", which is not a number"))?;
        values.push(value);
    }

    Ok(Array1::from_vec(values))
}

/// Read a matrix: an element holding only `row`s, each a space-separated list of numbers, as an `Array2<f64>` of shape (n_rows, n_columns).
/// An element with no `row`, e.g. `<regularisations />`, is shape (0, 0).
///
/// # Errors
/// The element holds something other than `row`s, a `row` is empty or holds something which is not a number, or the `row`s are of
/// different lengths
pub(crate) fn read_double_matrix(node: Node<'_, '_>, path: &str) -> Result<Array2<f64>, String> {
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
        let row: Array1<f64> = read_double_list(row_nodes[i_row], &row_path)?;
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

/// The error for an element, whose path is `path`, which is not in the XML schema the settings' structs were generated from, by the script at `generator_path`
pub(crate) fn unexpected_element(path: &str, generator_path: &str) -> String {
    format!("`{path}` is not in the XML schema the settings' structs were generated from. If the XML schema has changed, re-run `{generator_path}`")
}

/// The error for a required element, whose path is `path`, which is missing
pub(crate) fn missing_element(path: &str) -> String {
    format!("`{path}` is required, but it is missing")
}

/// The error for an element, whose path is `path`, which is there more than once, although it can be there only once
pub(crate) fn repeated_element(path: &str) -> String {
    format!("`{path}` is there more than once, but it can be there only once")
}
