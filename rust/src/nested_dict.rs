use core::panic;
use ndarray::{Array1, Array2, Array3, Dim, SliceInfo, SliceInfoElem, s};
use std::collections::HashMap;

#[derive(Debug)]
enum SliceType {
    Array3Size(SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 3]>>),
    Array2Size(SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 2]>>),
    Array1Size(SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 1]>>),
    F64Size(SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 0]>>),
}

#[derive(Clone)]
pub enum NestedValue {
    F64(f64),
    String(String),
    Usize(usize),
    Bool(bool),
    Array1(Array1<f64>),
    Array2(Array2<f64>),
    Array3(Array3<f64>),
    VecUsize(Vec<usize>),
    VecBool(Vec<bool>),
    NestedDict(NestedDict), // nested dictionaries
}

pub struct NestedDictAccumulator<'a> {
    pub root: &'a NestedDict,
    pub keys_store: Vec<String>, // TODO: NEED TO CHANGE VARIABLE NAME!!
}

impl<'a> NestedDictAccumulator<'a> {
    pub fn new(root: &'a NestedDict, keys_store: Vec<String>) -> Self {
        Self { root, keys_store }
    }

    pub fn new_scalar(root: &'a NestedDict, key: &str) -> Self {
        Self {
            root,
            keys_store: vec![key.to_string()],
        }
    }

    pub fn get(&self, key: &str) -> NestedDictAccumulator<'a> {
        // Append the key to the existing keys_store
        let mut new_keys = self.keys_store.clone();
        new_keys.push(key.to_string());
        NestedDictAccumulator::new(self.root, new_keys)
    }

    pub fn keys(&self) -> Vec<String> {
        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    // Add all NestedDicts from the current level to the stack
                    for value in dict.data.values() {
                        if let NestedValue::NestedDict(nested_dict) = value {
                            next_stack.push(nested_dict);
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    if let NestedValue::NestedDict(nested_dict) = value {
                        next_stack.push(nested_dict);
                    }
                }
            }
            stack = next_stack;
        }

        let mut collected_keys: Vec<String> = Vec::new();
        for dict in stack {
            collected_keys.extend(dict.data.keys().cloned());
        }

        collected_keys.sort();
        return collected_keys;
    }

    pub fn unwrap_usize(&self) -> usize {
        let mut collected: Vec<usize> = Vec::new();

        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    for value in dict.data.values() {
                        if let NestedValue::NestedDict(nested_dict) = value {
                            next_stack.push(nested_dict);
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            next_stack.push(nested_dict);
                        }
                        NestedValue::Usize(u) => collected.push(*u),
                        _ => {}
                    }
                }
            }
            stack = next_stack;
        }

        if collected.len() == 1 {
            collected[0]
        } else {
            panic!("Expected exactly one usize, found {}: {:?}", collected.len(), collected);
        }
    }

    pub fn unwrap_vec_usize(&self) -> Vec<usize> {
        let mut collected: Vec<usize> = Vec::new();
        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    let mut keys_store: Vec<_> = dict.data.keys().cloned().collect();
                    keys_store.sort();
                    for sorted_key in keys_store {
                        if let Some(value) = dict.data.get(&sorted_key) {
                            match value {
                                NestedValue::NestedDict(nested_dict) => {
                                    next_stack.push(nested_dict);
                                }
                                NestedValue::Usize(u) => collected.push(*u),
                                _ => {}
                            }
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            next_stack.push(nested_dict);
                        }
                        NestedValue::Usize(u) => collected.push(*u),
                        NestedValue::VecUsize(vec_usize) => {
                            return vec_usize.to_owned();
                        }
                        _ => {}
                    }
                }
            }
            stack = next_stack;
        }
        collected
    }

    pub fn unwrap_f64(&self) -> f64 {
        let mut collected: Vec<f64> = Vec::new();

        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    for value in dict.data.values() {
                        if let NestedValue::NestedDict(nested_dict) = value {
                            next_stack.push(nested_dict);
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            next_stack.push(nested_dict);
                        }
                        NestedValue::F64(x) => collected.push(*x),
                        _ => {}
                    }
                }
            }
            stack = next_stack;
        }

        if collected.len() == 1 {
            collected[0]
        } else {
            panic!("Expected exactly one f64, found {}: {:?}", collected.len(), collected);
        }
    }

    pub fn unwrap_bool(&self) -> bool {
        let mut collected: Vec<bool> = Vec::new();

        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    for value in dict.data.values() {
                        if let NestedValue::NestedDict(nested_dict) = value {
                            next_stack.push(nested_dict);
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            next_stack.push(nested_dict);
                        }
                        NestedValue::Bool(x) => collected.push(*x),
                        _ => {}
                    }
                }
            }
            stack = next_stack;
        }

        if collected.len() == 1 {
            collected[0]
        } else {
            panic!("Expected exactly one f64, found {}: {:?}", collected.len(), collected);
        }
    }

    pub fn unwrap_vec_bool(&self) -> Vec<bool> {
        let mut collected: Vec<bool> = Vec::new();
        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    let mut keys_store: Vec<_> = dict.data.keys().cloned().collect();
                    keys_store.sort();
                    for sorted_key in keys_store {
                        if let Some(value) = dict.data.get(&sorted_key) {
                            match value {
                                NestedValue::NestedDict(nested_dict) => {
                                    next_stack.push(nested_dict);
                                }
                                NestedValue::Bool(b) => collected.push(*b),
                                _ => {}
                            }
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            next_stack.push(nested_dict);
                        }
                        NestedValue::Bool(b) => {
                            collected.push(*b);
                        }
                        NestedValue::VecBool(vec_bool) => {
                            return vec_bool.to_owned();
                        }
                        _ => {}
                    }
                }
            }
            stack = next_stack;
        }
        return collected;
    }

    pub fn unwrap_array1(&self) -> Array1<f64> {
        let mut collected: Vec<f64> = Vec::new();

        let mut stack = vec![self.root];
        for key in &self.keys_store {
            let mut next_stack = Vec::new();
            for dict in stack {
                if key == "*" {
                    let mut keys_store: Vec<_> = dict.data.keys().cloned().collect();
                    keys_store.sort();
                    for sorted_key in keys_store {
                        if let Some(value) = dict.data.get(&sorted_key) {
                            match value {
                                NestedValue::NestedDict(nested_dict) => {
                                    next_stack.push(nested_dict);
                                }
                                NestedValue::F64(x) => collected.push(*x),
                                _ => {
                                    panic!("accumulation error!!");
                                }
                            }
                        }
                    }
                } else if let Some(value) = dict.data.get(key) {
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            next_stack.push(nested_dict);
                        }
                        NestedValue::F64(x) => collected.push(*x),
                        NestedValue::Array1(array) => {
                            collected.extend(array.iter().cloned());
                        }
                        _ => {}
                    }
                }
            }
            stack = next_stack;
        }

        Array1::from(collected)
    }

    pub fn unwrap_array2(&self) -> Array2<f64> {
        // Recursive helper function
        fn traverse_recursive(
            node: &NestedDict,
            keys: &[String],
            collected: &mut Vec<Array1<f64>>,
            shape: &mut (usize, usize),
            level: usize,
        ) -> Option<Array2<f64>> {
            if keys.is_empty() {
                return None;
            }

            let current_key: &String = &keys[0];
            let remaining_keys: &[String] = &keys[1..];

            if current_key == "*" {
                // Handle wildcard
                let mut local_count = 0; // Track the count at this level
                let mut keys_tmp: Vec<&String> = node.data.keys().collect();
                keys_tmp.sort();

                // for (key, value) in &node.data {
                for key in keys_tmp {
                    let value: &NestedValue = node.data.get(key).unwrap();
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            // Recurse into nested dictionary
                            if let Some(array2) = traverse_recursive(nested_dict, remaining_keys, collected, shape, level + 1) {
                                return Some(array2);
                            }
                            local_count += 1; // Increment for each nested dictionary
                        }
                        NestedValue::Array1(array) if remaining_keys.is_empty() => {
                            collected.push(array.clone());
                            if level == 0 {
                                shape.0 += 1; // Increment rows for the first wildcard
                            }
                            shape.1 = array.len(); // Columns are determined by array length
                        }
                        NestedValue::F64(value) if remaining_keys.is_empty() => {
                            collected.push(Array1::from(vec![*value]));
                            if level == 0 {
                                shape.0 += 1; // Increment rows for the first wildcard
                            }
                            shape.1 += 1; // Columns are scalar
                        }
                        _ => {}
                    }
                }

                // Update shape based on this wildcard level
                if level == 0 {
                    shape.0 = local_count; // Rows determined by the first wildcard
                }
            } else if let Some(value) = node.data.get(current_key) {
                // Non-wildcard key
                match value {
                    NestedValue::NestedDict(nested_dict) => {
                        return traverse_recursive(nested_dict, remaining_keys, collected, shape, level);
                    }
                    NestedValue::Array2(array2) if remaining_keys.is_empty() => {
                        // Directly return Array2 if no remaining keys
                        shape.0 = array2.nrows();
                        shape.1 = array2.ncols();
                        return Some(array2.clone());
                    }
                    NestedValue::Array1(array) if remaining_keys.is_empty() => {
                        collected.push(array.clone());
                        shape.0 = 1; // Single row
                        shape.1 = array.len(); // Number of columns matches array length
                    }
                    NestedValue::F64(value) if remaining_keys.is_empty() => {
                        collected.push(Array1::from(vec![*value]));
                        shape.0 = 1; // Single row
                        shape.1 += 1; // Single scalar value
                    }
                    _ => {}
                }
            }

            None
        }

        let mut collected: Vec<Array1<f64>> = Vec::new();
        let mut shape: (usize, usize) = (0, 0); // (rows, cols)

        // Start recursion from the root
        if let Some(array2) = traverse_recursive(self.root, &self.keys_store, &mut collected, &mut shape, 0) {
            return array2;
        }

        // Stack collected arrays into an Array2
        if collected.is_empty() {
            panic!("No matching Array1<f64> or Array2<f64> found for keys: {:?}", self.keys_store);
        }

        // Flatten collected arrays into a row-major 2D array
        let n_rows = shape.0;
        let mut n_cols = shape.1;
        let mut flattened_data = Vec::with_capacity(n_rows * n_cols);

        for array in collected {
            flattened_data.extend(array.iter().cloned());
        }

        // Construct the final Array2
        // let keys: Vec<String> = self.keys_store.clone();  // TODO: what was this for????
        let n_wildcards = self.keys_store.iter().filter(|&key| key == "*").count();

        if n_wildcards == 2 {
            n_cols = n_cols / n_rows;
        }

        let results: Array2<f64> = Array2::from_shape_vec((n_rows, n_cols), flattened_data.clone()).expect("Failed to construct Array2 with correct shape.");

        if n_wildcards == 1 {
            return results.t().to_owned();
        } else if n_wildcards == 2 {
            return results.t().to_owned();
        } else {
            return results;
        }
    }

    pub fn unwrap_array3(&self) -> Array3<f64> {
        /// Recursive helper function.
        /// This function finds all the paths to the data and stores where each path should be stored within the Array3
        fn traverse_recursive_to_find_paths_and_slices(
            node: &NestedDict,
            keys: &Vec<String>,
            path: &mut Vec<String>,
            index_right_to_left: &mut Vec<usize>,
            sizes_right_to_left: &mut Vec<usize>,
            paths: &mut Vec<(Vec<String>, SliceType, [usize; 3])>,
        ) {
            let current_key: &String = &keys[0];
            let remaining_keys: &Vec<String> = &keys[1..].to_vec();

            if current_key == "*" {
                // Get all keys at this level
                let mut keys_tmp: Vec<&String> = node.data.keys().collect();
                keys_tmp.sort();

                sizes_right_to_left.push(keys_tmp.len());

                for i_key in 0..keys_tmp.len() {
                    let key: &String = keys_tmp[i_key];

                    let value: &NestedValue = node.data.get(key).unwrap();
                    match value {
                        NestedValue::NestedDict(nested_dict) => {
                            let mut path_new: Vec<String> = path.clone();
                            path_new.push(key.to_string());

                            let mut index_right_to_left_new: Vec<usize> = index_right_to_left.clone();
                            index_right_to_left_new.push(i_key);

                            let mut sizes_right_to_left_new: Vec<usize> = sizes_right_to_left.clone();

                            traverse_recursive_to_find_paths_and_slices(
                                nested_dict,
                                remaining_keys,
                                &mut path_new,
                                &mut index_right_to_left_new,
                                &mut sizes_right_to_left_new,
                                paths,
                            );
                        }
                        NestedValue::Array2(_array2_data) if remaining_keys.is_empty() => {
                            // e.g. ["node0", "node1", "*"]
                            let mut path_new: Vec<String> = path.clone();
                            path_new.push(key.to_string());

                            let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 2]>> = s![.., .., i_key];

                            let array2_shape: &[usize] = _array2_data.shape();
                            let data_shape: [usize; 3] = [array2_shape[0], array2_shape[1], sizes_right_to_left[0]];

                            paths.push((path_new.clone(), SliceType::Array2Size(slice_info), data_shape));
                        }
                        NestedValue::Array1(_array1_data) if remaining_keys.is_empty() => {
                            // e.g. ["*", "node1", "*"]
                            let mut path_new: Vec<String> = path.clone();
                            path_new.push(key.to_string());

                            let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 1]>> = s![.., i_key, index_right_to_left[0]];

                            let array1_shape: &[usize] = _array1_data.shape();
                            let data_shape: [usize; 3] = [array1_shape[0], sizes_right_to_left[1], sizes_right_to_left[0]];

                            paths.push((path_new.clone(), SliceType::Array1Size(slice_info), data_shape));
                        }
                        NestedValue::F64(_f64_data) if remaining_keys.is_empty() => {
                            // e.g. ["*", "*", "*"] (test_09)
                            let mut path_new: Vec<String> = path.clone();
                            path_new.push(key.to_string());

                            let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 0]>> =
                                s![i_key, index_right_to_left[1], index_right_to_left[0]];

                            let data_shape: [usize; 3] = [sizes_right_to_left[2], sizes_right_to_left[1], sizes_right_to_left[0]];

                            paths.push((path_new.clone(), SliceType::F64Size(slice_info), data_shape));
                        }
                        _ => {
                            panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Unknown data type found");
                        }
                    }
                }
            } else {
                path.push(current_key.to_string());
                let value: &NestedValue = node.data.get(current_key).unwrap();
                match value {
                    NestedValue::NestedDict(nested_dict) => {
                        // Recurse into nested dictionary
                        traverse_recursive_to_find_paths_and_slices(nested_dict, remaining_keys, path, index_right_to_left, sizes_right_to_left, paths);
                    }
                    NestedValue::Array3(_array3_data) if remaining_keys.is_empty() => {
                        // e.g. ["node0", "node1", "node2"] (test_07)
                        let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 3]>> = s![.., .., ..];

                        let array3_shape: &[usize] = _array3_data.shape();
                        let data_shape: [usize; 3] = [array3_shape[0], array3_shape[1], array3_shape[2]];

                        paths.push((path.clone(), SliceType::Array3Size(slice_info), data_shape));
                    }
                    NestedValue::Array2(_array2_data) if remaining_keys.is_empty() => {
                        // e.g. ["node0", "*", "node2"] (test_08)
                        let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 2]>> = s![.., .., index_right_to_left[0]];

                        let array2_shape: &[usize] = _array2_data.shape();
                        let data_shape: [usize; 3] = [array2_shape[0], array2_shape[1], sizes_right_to_left[0]];

                        paths.push((path.clone(), SliceType::Array2Size(slice_info), data_shape));
                    }
                    NestedValue::Array1(_array1_data) if remaining_keys.is_empty() => {
                        // e.g. ["*", "*", "node2"]
                        let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 1]>> =
                            s![.., index_right_to_left[1], index_right_to_left[0]];

                        let array1_shape: &[usize] = _array1_data.shape();
                        let data_shape: [usize; 3] = [array1_shape[0], sizes_right_to_left[1], sizes_right_to_left[0]];

                        paths.push((path.clone(), SliceType::Array1Size(slice_info), data_shape));
                    }
                    NestedValue::F64(_f64_data) if remaining_keys.is_empty() => {
                        // e.g. ["*", "*", "*", "node3"]  TODO: check this?
                        let slice_info: SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 0]>> =
                            s![index_right_to_left[2], index_right_to_left[1], index_right_to_left[0]];

                        let data_shape: [usize; 3] = [sizes_right_to_left[2], sizes_right_to_left[1], sizes_right_to_left[0]];

                        paths.push((path.clone(), SliceType::F64Size(slice_info), data_shape));
                    }
                    _ => {
                        panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Unknown data type found");
                    }
                }
            }
        }

        let mut paths_slices_sizes: Vec<(Vec<String>, SliceType, [usize; 3])> = Vec::new();
        let mut path: Vec<String> = Vec::new();
        let mut index_right_to_left: Vec<usize> = Vec::new();
        let mut sizes_right_to_left: Vec<usize> = Vec::new();
        traverse_recursive_to_find_paths_and_slices(
            self.root,
            &self.keys_store,
            &mut path,
            &mut index_right_to_left,
            &mut sizes_right_to_left,
            &mut paths_slices_sizes,
        );

        // Exit if there is no data to collect
        if paths_slices_sizes.is_empty() {
            return Array3::zeros([0, 0, 0]);
        }

        // Retrieve the data size
        let (_1, _2, data_size) = &paths_slices_sizes[0];

        // Loop to collect the data
        let mut final_data: Array3<f64> = Array3::zeros(data_size.to_owned());
        for path_and_slice in paths_slices_sizes {
            let (path, slice_indices, _data_size) = path_and_slice;

            // Loop over all keys, except the last one
            let mut dict: &NestedDict = self.root;
            for key in path[0..path.len() - 1].iter() {
                let value: &NestedValue = dict.data.get(key).unwrap();
                match value {
                    NestedValue::NestedDict(nested_dict) => {
                        dict = nested_dict;
                    }
                    _ => {
                        panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Should not have a non-NestedDict value here");
                    }
                }
            }
            // For the last key we retrieve the data
            match slice_indices {
                SliceType::Array3Size(slice_info) => {
                    let value: &NestedValue = dict.data.get(&path[path.len() - 1]).unwrap();
                    match value {
                        NestedValue::Array3(array3) => {
                            final_data.slice_mut(slice_info).assign(&array3);
                        }
                        _ => {
                            panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Unknown data type found");
                        }
                    }
                }
                SliceType::Array2Size(slice_info) => {
                    let value: &NestedValue = dict.data.get(&path[path.len() - 1]).unwrap();
                    match value {
                        NestedValue::Array2(array2) => {
                            final_data.slice_mut(slice_info).assign(&array2);
                        }
                        _ => {
                            panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Unknown data type found");
                        }
                    }
                }
                SliceType::Array1Size(slice_info) => {
                    let value: &NestedValue = dict.data.get(&path[path.len() - 1]).unwrap();
                    match value {
                        NestedValue::Array1(array1) => {
                            final_data.slice_mut(slice_info).assign(&array1);
                        }
                        _ => {
                            panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Unknown data type found");
                        }
                    }
                }
                SliceType::F64Size(slice_info) => {
                    let value: &NestedValue = dict.data.get(&path[path.len() - 1]).unwrap();
                    match value {
                        NestedValue::F64(f64_val) => {
                            final_data.slice_mut(slice_info).fill(*f64_val);
                        }
                        _ => {
                            panic!("unwrap_array3.traverse_recursive_to_find_paths_and_slices: Unknown data type found");
                        }
                    }
                }
            }
        }

        return final_data;
    }
}

#[derive(Clone)]
pub struct NestedDict {
    pub data: HashMap<String, NestedValue>,
}

impl NestedDict {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    pub fn insert<T: Into<NestedValue>>(&mut self, key: &str, value: T) {
        self.data.insert(key.to_string(), value.into());
    }

    pub fn get(&self, key: &str) -> NestedDictAccumulator {
        NestedDictAccumulator::new(self, vec![key.to_string()])
    }

    pub fn get_or_insert(&mut self, key: &str) -> &mut NestedDict {
        if let NestedValue::NestedDict(ref mut dict) = *self.data.entry(key.to_string()).or_insert_with(|| NestedValue::NestedDict(NestedDict::new())) {
            dict
        } else {
            panic!("Stored value is not a NestedDict");
        }
    }

    pub fn keys(&self) -> Vec<String> {
        // pub fn keys(&self, path: Option<Vec<String>>) -> Vec<String> {

        let mut keys_vec: Vec<String> = self.data.keys().cloned().collect();
        keys_vec.sort();
        return keys_vec;
    }

    pub fn print_keys(&self) {
        let mut results: Vec<(Vec<String>, String)> = Vec::<(Vec<String>, String)>::new();

        fn traverse(current_dict: &NestedDict, prefix: Vec<String>, results: &mut Vec<(Vec<String>, String)>) {
            let mut sorted_keys: Vec<&String> = current_dict.data.keys().collect();
            sorted_keys.sort();

            for key in sorted_keys {
                let mut current_path: Vec<String> = prefix.clone();
                current_path.push(key.clone());

                let value: &NestedValue = current_dict.data.get(key).expect("Key should exist in the dictionary");

                match value {
                    NestedValue::NestedDict(nested_dict) => {
                        traverse(nested_dict, current_path, results);
                    }
                    NestedValue::Array1(array1) => {
                        let dim: usize = array1.dim();
                        let description: String = format!("Array1<f64>;  shape=({:?})", dim);
                        results.push((current_path, description));
                    }
                    NestedValue::Array2(array2) => {
                        let dim: (usize, usize) = array2.dim();
                        let description: String = format!("Array2<f64>;  shape={:?}", dim);
                        results.push((current_path, description));
                    }
                    NestedValue::Array3(array3) => {
                        let dim: (usize, usize, usize) = array3.dim();
                        let description: String = format!("Array3<f64>;  shape={:?}", dim);
                        results.push((current_path, description));
                    }
                    NestedValue::F64(f64_val) => {
                        let description: String = format!("f64;  value={:?}", f64_val);
                        results.push((current_path, description));
                    }
                    NestedValue::String(value) => {
                        let description: String = format!("String;  value={:?}", value);
                        results.push((current_path, description));
                    }
                    NestedValue::Usize(usize_val) => {
                        let description: String = format!("usize;  value={:?}", usize_val);
                        results.push((current_path, description));
                    }
                    NestedValue::Bool(bool_val) => {
                        let description: String = format!("bool;  value={:?}", bool_val);
                        results.push((current_path, description));
                    }
                    NestedValue::VecUsize(vec_usize) => {
                        let dim: usize = vec_usize.len();
                        let description: String = format!("Vec<usize>;  shape={:?}", dim);
                        results.push((current_path, description));
                    }
                    NestedValue::VecBool(vec_bool) => {
                        let dim: usize = vec_bool.len();
                        let description: String = format!("Vec<bool>;  shape={:?}", dim);
                        results.push((current_path, description));
                    }
                }
            }
        }

        traverse(self, vec![], &mut results);

        let max_key_length: usize = results
            .iter()
            .map(|(key, _)| {
                let formatted_key: String = format!("{:?}", key);
                formatted_key.len()
            })
            .max()
            .unwrap_or(0);

        for (key, value) in results {
            let formatted_key: String = format!("{:?}", key);
            println!("{:<width$} -> {}", formatted_key, value, width = max_key_length);
        }
    }
}

impl From<String> for NestedValue {
    fn from(value: String) -> Self {
        NestedValue::String(value)
    }
}

impl From<f64> for NestedValue {
    fn from(value: f64) -> Self {
        NestedValue::F64(value)
    }
}

impl From<usize> for NestedValue {
    fn from(value: usize) -> Self {
        NestedValue::Usize(value)
    }
}

impl From<bool> for NestedValue {
    fn from(value: bool) -> Self {
        NestedValue::Bool(value)
    }
}

impl From<Array1<f64>> for NestedValue {
    fn from(value: Array1<f64>) -> Self {
        NestedValue::Array1(value)
    }
}

impl From<Array2<f64>> for NestedValue {
    fn from(value: Array2<f64>) -> Self {
        NestedValue::Array2(value)
    }
}

impl From<Array3<f64>> for NestedValue {
    fn from(value: Array3<f64>) -> Self {
        NestedValue::Array3(value)
    }
}

impl From<Vec<usize>> for NestedValue {
    fn from(value: Vec<usize>) -> Self {
        NestedValue::VecUsize(value)
    }
}

impl From<Vec<bool>> for NestedValue {
    fn from(value: Vec<bool>) -> Self {
        NestedValue::VecBool(value)
    }
}

impl From<NestedDict> for NestedValue {
    fn from(value: NestedDict) -> Self {
        NestedValue::NestedDict(value)
    }
}

#[test]
fn test_nested_dict() {
    // Lazy loading for the tests
    use ndarray::s;

    // Construct restuls
    // Probe 1
    let mut results: NestedDict = NestedDict::new();
    results.get_or_insert("P101").get_or_insert("geometry").insert("r", 1.1f64);
    results.get_or_insert("P101").get_or_insert("geometry").insert("z", 0.1f64);
    results
        .get_or_insert("P101")
        .get_or_insert("b")
        .insert("measured", Array1::from_vec(vec![1.0, 1.1, 1.2, 1.3, 1.4]));
    let p101_green_psi: Array2<f64> = Array2::from_shape_vec([5, 2], vec![1.11, 1.11, 1.12, 1.13, 1.14, 1.21, 1.21, 1.22, 1.23, 1.24]).expect("shape mismatch");
    results.get_or_insert("P101").get_or_insert("greens").insert("psi", p101_green_psi.clone());
    results.get_or_insert("P101").get_or_insert("greens").get_or_insert("pf").insert("BVL", 0.0);
    results.get_or_insert("P101").get_or_insert("greens").get_or_insert("pf").insert("BVLB", 0.1);
    results.get_or_insert("P101").get_or_insert("greens").get_or_insert("pf").insert("BVLT", 0.2);
    results.get_or_insert("P101").get_or_insert("greens").get_or_insert("pf").insert("SOL", 0.3);
    let mut three_d_data: Array3<f64> = Array3::zeros([11, 12, 13]);
    three_d_data[[8, 2, 3]] = 99.0;
    results.get_or_insert("P101").get_or_insert("three_d").insert("data", three_d_data.clone());

    // Probe 2
    results.get_or_insert("P102").get_or_insert("geometry").insert("r", 1.2f64);
    results.get_or_insert("P102").get_or_insert("geometry").insert("z", 0.2f64);
    results
        .get_or_insert("P102")
        .get_or_insert("b")
        .insert("measured", Array1::from_vec(vec![2.0, 2.1, 2.2, 2.3, 2.4]));
    let p102_green_psi: Array2<f64> = Array2::from_shape_vec([5, 2], vec![2.11, 2.11, 2.12, 2.13, 2.14, 2.21, 2.21, 2.22, 2.23, 2.24]).expect("shape mismatch");
    results.get_or_insert("P102").get_or_insert("greens").insert("psi", p102_green_psi.clone());
    results.get_or_insert("P102").get_or_insert("greens").get_or_insert("pf").insert("BVL", 1.0);
    results.get_or_insert("P102").get_or_insert("greens").get_or_insert("pf").insert("BVLB", 1.1);
    results.get_or_insert("P102").get_or_insert("greens").get_or_insert("pf").insert("BVLT", 1.2);
    results.get_or_insert("P102").get_or_insert("greens").get_or_insert("pf").insert("SOL", 1.3);

    // Probe 3
    results.get_or_insert("P103").get_or_insert("geometry").insert("r", 1.3f64);
    results.get_or_insert("P103").get_or_insert("geometry").insert("z", 0.3f64);
    results
        .get_or_insert("P103")
        .get_or_insert("b")
        .insert("measured", Array1::from_vec(vec![3.0, 3.1, 3.2, 3.3, 3.4]));
    let p103_green_psi: Array2<f64> = Array2::from_shape_vec([5, 2], vec![3.11, 3.11, 3.12, 3.13, 3.14, 3.21, 3.21, 3.22, 3.23, 3.24]).expect("shape mismatch");
    results.get_or_insert("P103").get_or_insert("greens").insert("psi", p103_green_psi.clone());
    results.get_or_insert("P103").get_or_insert("greens").get_or_insert("pf").insert("BVLB", 2.0);
    results.get_or_insert("P103").get_or_insert("greens").get_or_insert("pf").insert("BVLT", 2.1);
    results.get_or_insert("P103").get_or_insert("greens").get_or_insert("pf").insert("BVL", 2.2);
    results.get_or_insert("P103").get_or_insert("greens").get_or_insert("pf").insert("SOL", 2.3);

    // test_01: Retrieve f64, no wildcard, from f64 data
    let data_retrieved: f64 = results.get("P102").get("geometry").get("r").unwrap_f64();
    let data_expected: f64 = 1.2;
    assert_eq!(data_retrieved, data_expected, "test_01");

    // test_02: Retrieve Array1<f64>, no wildcard, from Array1<f64> data
    let data_retrieved: Array1<f64> = results.get("P103").get("b").get("measured").unwrap_array1();
    let data_expected: Array1<f64> = Array1::from(vec![3.0, 3.1, 3.2, 3.3, 3.4]);
    assert_eq!(data_retrieved, data_expected, "test_02");

    // test_03: Retrieve Array1<f64>, one wildcard, from f64 data
    let data_retrieved: Array1<f64> = results.get("*").get("geometry").get("r").unwrap_array1();
    let data_expected: Array1<f64> = Array1::from_vec(vec![1.1, 1.2, 1.3]);
    assert_eq!(data_retrieved, data_expected, "test_03");

    // test_04: Retrieve Array2<f64>, no wildcard, from Array2<f64> data
    let data_retrieved: Array2<f64> = results.get("P102").get("greens").get("psi").unwrap_array2();
    assert_eq!(data_retrieved, p102_green_psi, "test_04");

    // test_05: Retrieve Array2<f64>, one wildcard, from Array1<f64> data
    let data_retrieved: Array2<f64> = results.get("*").get("b").get("measured").unwrap_array2();
    let data_expected: Array2<f64> =
        Array2::from_shape_vec([5, 3], vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 1.3, 2.3, 3.3, 1.4, 2.4, 3.4]).expect("test5, error unravelling");
    assert_eq!(data_retrieved.shape(), &[5, 3], "test_05 shape"); // shape = [n_time, n_accumulation]
    assert_eq!(data_retrieved, data_expected, "test_05");

    // test_06: Retrieve Array2<f64>, two wildcards, from f64 data
    let data_retrieved: Array2<f64> = results.get("*").get("greens").get("pf").get("*").unwrap_array2();
    let data_expected: Array2<f64> =
        Array2::from_shape_vec([4, 3], vec![0.0, 1.0, 2.2, 0.1, 1.1, 2.0, 0.2, 1.2, 2.1, 0.3, 1.3, 2.3]).expect("test6, error unravelling");
    assert_eq!(data_retrieved, data_expected, "test_06");

    // Note: Array3<f64> requires 7 tests to be complete
    // test_07: Retrieve Array3<f64>, no wildcard, from Array3<f64> data
    let data_retrieved: Array3<f64> = results.get("P101").get("three_d").get("data").unwrap_array3();
    assert_eq!(data_retrieved, three_d_data, "test_07");

    // test_08: Retrieve Array3<f64>, one wildcard, from Array2<f64> data
    let data_retrieved: Array3<f64> = results.get("*").get("greens").get("psi").unwrap_array3();
    assert_eq!(data_retrieved.shape(), &[5usize, 2usize, 3usize], "test_08, shape"); // shape = [n_z, n_r, n_sensors]
    println!("data_retrieved.shape()={:?}", data_retrieved.shape());
    assert_eq!(data_retrieved.slice(s![.., .., 0]), p101_green_psi, "test_08, slice 0");
    assert_eq!(data_retrieved.slice(s![.., .., 1]), p102_green_psi, "test_08, slice 1");
    assert_eq!(data_retrieved.slice(s![.., .., 2]), p103_green_psi, "test_08, slice 2");

    // test_09: Retrieve Array3<f64>, two wildcards, from Array1<f64> data
    let mut test_data: NestedDict = NestedDict::new();
    let data: Array1<f64> = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    test_data
        .get_or_insert("level_00_key_00")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_00", data.clone());
    test_data
        .get_or_insert("level_00_key_00")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_01", data.clone() + 1.0);
    test_data
        .get_or_insert("level_00_key_00")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_02", data.clone() + 2.0); // s![.., 2, 0]
    test_data
        .get_or_insert("level_00_key_00")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_03", data.clone() + 3.0);

    test_data
        .get_or_insert("level_00_key_01")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_00", data.clone());
    test_data
        .get_or_insert("level_00_key_01")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_01", data.clone());
    test_data
        .get_or_insert("level_00_key_01")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_02", data.clone());
    test_data
        .get_or_insert("level_00_key_01")
        .get_or_insert("level_01_key_00")
        .insert("level_02_key_03", data.clone());

    let data_retrieved: Array3<f64> = test_data.get("*").get("level_01_key_00").get("*").unwrap_array3(); // shape = [6, 4, 2]
    assert_eq!(data_retrieved.shape(), &[6usize, 4usize, 2usize], "test_09, shape");
    assert_eq!(data_retrieved.slice(s![.., 2, 0]), data.clone() + 2.0, "test_09, values");
}
