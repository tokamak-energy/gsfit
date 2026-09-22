//! How an array of structures is indexed, and how a selection is resolved against the real
//! length once the IDS is known.

/// How an array of structures was indexed.
///
/// Resolution is deferred: a path does not know how many time slices exist, so the
/// selection is stored as written and resolved against the real length inside `get`.
#[derive(Clone, Debug)]
pub enum IndexSpec {
    /// `path[3]` - selects one element and drops that dimension.
    One(isize),
    /// `path[:]`, `path[3:7]`, `path[::2]` - keeps the dimension.
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    /// `path[[0, 2, 5]]` - keeps the dimension.
    Many(Vec<isize>),
}

impl IndexSpec {
    /// True when this selection drops the dimension (`path[3]` rather than `path[:]`).
    pub fn is_scalar(&self) -> bool {
        matches!(self, IndexSpec::One(_))
    }

    /// Expand this selection into concrete element indices.
    pub fn resolve(&self, n_elements: usize) -> Result<Vec<usize>, String> {
        match self {
            IndexSpec::One(index) => {
                let resolved: usize = resolve_one_index(*index, n_elements)?;
                Ok(vec![resolved])
            }
            IndexSpec::Many(requested) => {
                let mut indices: Vec<usize> = Vec::with_capacity(requested.len());
                for index in requested {
                    indices.push(resolve_one_index(*index, n_elements)?);
                }
                Ok(indices)
            }
            IndexSpec::Slice { start, stop, step } => resolve_slice(*start, *stop, *step, n_elements),
        }
    }
}

/// Resolve one possibly-negative index against a known length.
fn resolve_one_index(index: isize, n_elements: usize) -> Result<usize, String> {
    let n_elements_signed: isize = n_elements as isize;
    let resolved: isize = if index < 0 { index + n_elements_signed } else { index };

    if resolved < 0 || resolved >= n_elements_signed {
        return Err(format!("index {index} is out of range for an array of structures with {n_elements} element(s)"));
    }

    Ok(resolved as usize)
}

/// Expand a Python slice into concrete indices, following CPython's `slice.indices` rules
/// (negative values count from the end, out-of-range endpoints clamp rather than raise).
fn resolve_slice(start: Option<isize>, stop: Option<isize>, step: Option<isize>, n_elements: usize) -> Result<Vec<usize>, String> {
    let n_elements_signed: isize = n_elements as isize;
    let step: isize = step.unwrap_or(1);

    if step == 0 {
        return Err("slice step cannot be zero".to_string());
    }

    // Clamping bounds differ by direction: a backwards slice may legitimately stop at -1.
    let (default_start, default_stop): (isize, isize) = if step > 0 { (0, n_elements_signed) } else { (n_elements_signed - 1, -1) };

    let start_resolved: isize = match start {
        None => default_start,
        Some(value) => {
            let shifted: isize = if value < 0 { value + n_elements_signed } else { value };
            if step > 0 {
                shifted.clamp(0, n_elements_signed)
            } else {
                shifted.clamp(-1, n_elements_signed - 1)
            }
        }
    };

    let stop_resolved: isize = match stop {
        None => default_stop,
        Some(value) => {
            let shifted: isize = if value < 0 { value + n_elements_signed } else { value };
            if step > 0 {
                shifted.clamp(0, n_elements_signed)
            } else {
                shifted.clamp(-1, n_elements_signed - 1)
            }
        }
    };

    // Compute the count up front so the loop is bounded.
    let n_indices: usize = if step > 0 {
        if stop_resolved > start_resolved {
            ((stop_resolved - start_resolved + step - 1) / step) as usize
        } else {
            0
        }
    } else {
        if stop_resolved < start_resolved {
            ((stop_resolved - start_resolved + step + 1) / step) as usize
        } else {
            0
        }
    };

    let mut indices: Vec<usize> = Vec::with_capacity(n_indices);
    for i_index in 0..n_indices {
        indices.push((start_resolved + (i_index as isize) * step) as usize);
    }

    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slice must be handled exactly as CPython handles it, including negative bounds,
    /// negative steps and out-of-range endpoints clamping rather than raising.
    ///
    /// The expected values were produced by Python itself: `list(range(n))[start:stop:step]`.
    #[test]
    fn slice_semantics_match_python() {
        let cases: Vec<(Vec<usize>, usize, Option<isize>, Option<isize>, Option<isize>)> = vec![
            (vec![0, 1, 2, 3, 4], 5, None, None, None),
            (vec![1, 2], 5, Some(1), Some(3), None),
            (vec![0, 2, 4], 5, None, None, Some(2)),
            (vec![4, 3, 2, 1, 0], 5, None, None, Some(-1)),
            (vec![3, 4], 5, Some(-2), None, None),
            (vec![0, 1, 2, 3], 5, None, Some(-1), None),
            (Vec::<usize>::new(), 5, Some(10), Some(99), None),
            (vec![4, 3, 2], 5, Some(4), Some(1), Some(-1)),
            (vec![4, 2, 0], 5, None, None, Some(-2)),
            (vec![0, 1, 2, 3, 4], 5, Some(-99), Some(99), None),
            (Vec::<usize>::new(), 5, Some(3), Some(1), None),
            (Vec::<usize>::new(), 0, None, None, None),
            (vec![2, 1, 0], 3, None, None, Some(-1)),
            (Vec::<usize>::new(), 5, Some(2), Some(2), None),
        ];

        for (expected, n_elements, start, stop, step) in cases {
            let spec: IndexSpec = IndexSpec::Slice { start, stop, step };
            let resolved: Vec<usize> = spec.resolve(n_elements).expect("slice should resolve");
            assert_eq!(resolved, expected, "[{start:?}:{stop:?}:{step:?}] over {n_elements} element(s)");
        }
    }

    #[test]
    fn a_zero_step_is_rejected() {
        let spec: IndexSpec = IndexSpec::Slice {
            start: None,
            stop: None,
            step: Some(0),
        };
        assert!(spec.resolve(5).is_err());
    }
}
