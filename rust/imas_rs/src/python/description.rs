//! The static description of a data dictionary.
//!
//! The generated `<ids>_paths.rs` tables are built from these types, and a
//! [`PyPath`](super::PyPath) is a cursor over them.

use super::{IndexSpec, Value};
use std::any::Any;

/// A node in the static data dictionary description.
pub struct Node {
    pub name: &'static str,
    pub documentation: &'static str,
    pub units: &'static str,
    pub kind: NodeKind,
}

pub enum NodeKind {
    /// A nested structure; navigate into its children by name.
    Structure(&'static [Node]),
    /// An array of structures; must be indexed before navigating further.
    ArrayOfStructures(&'static [Node]),
    /// A terminal data node.
    Leaf(Leaf),
}

/// A terminal data node, paired with the function that reads it out of the IDS.
///
/// `read` is a plain `fn` pointer so the whole description stays a `static`. It receives
/// the index selections gathered along the path, one per array-of-structures level
/// crossed, in order.
///
/// The IDS arrives as `&dyn Any` so that one `Node` type serves every IDS; the generated
/// reader downcasts it back. `Path` records which IDS it was built from and `read_path`
/// checks that before calling, so the downcast is a belt-and-braces failure rather than
/// the primary guard.
pub struct Leaf {
    /// The data dictionary type, e.g. `"FLT_0D"`.
    pub data_type: &'static str,
    pub read: fn(&dyn Any, &[IndexSpec]) -> Result<Value, String>,
}

impl Node {
    /// The children reachable from this node, or `None` for a leaf.
    ///
    /// An array of structures only exposes its children once indexed, so that
    /// `time_slice.global_quantities` fails with a clear message rather than silently
    /// meaning `time_slice[:]`.
    pub(super) fn children(&self, indexed: bool) -> Option<&'static [Node]> {
        match &self.kind {
            NodeKind::Structure(children) => Some(children),
            NodeKind::ArrayOfStructures(children) => {
                if indexed {
                    return Some(children);
                }
                None
            }
            NodeKind::Leaf(_) => None,
        }
    }
}
