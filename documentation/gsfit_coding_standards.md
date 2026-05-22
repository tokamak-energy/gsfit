# GSFit standards

This file provides guidance when working with code in this repository.

## Variable naming
### Hierarchical variable naming
Variables should be named using a "hierarchical" structure.

As an example, we often use the $(R, Z)$ coordinate system for many variables, e.g. the plasma boundary, limit points, and magnetic axis location.
The following is **wrong** (note, I have alphabetically sorted these):
* `r_boundary`
* `r_lim`
* `r_mag`
* `z_boundary`
* `z_lim`
* `z_mag`

Whereas the following are **correct**:
* `boundary_r`
* `boundary_z`
* `lim_r`
* `lim_z`
* `mag_r`
* `mag_z`

By sticking to a hierarchical structure we can clearly distinguish what belongs to what, also we know that `boundary_r` and `boundary_z` have the same data type, `Array1<f64>`, and length.
Whereas `mag_r` and `mag_z` are simply `f64`.
We might also want to have other data such as `boundary_n` for the number of points along the boundary.

### Naming within a for loop using indexing
Loops should be indexed with `i_` followed by the quantity they are iterating over.
Example:
```rust
use ndarray::Array1;

let n_x: usize = 100;
let x: Array1<f64> = Array1::linspace(0.0, 1.0, n_x);

for i_x in 0..n_x {
    println!("x = {}", x[i_x]);
}
```
You should always use `n_` followed by the quantity, **not** `for i_x in 0..x.len() {`

For doubly nested for loops start with `i_`, then `j_`, then `k_`.
```rust
for i_x in 0..n_x {
    for j_x in 0..n_x {
        for k_x in 0..n_x {
            unimplemented!("some logic");
        }
    }
}
```

If the parameter is vs time, the `i_time` should be used, not something like `i_current`.

### Naming within a for loop over an iterator
You can also loop over lists of strings, e.g.

```rust
let pf_names: Vec<String> = vec!["BVL".to_string(), "DIV".to_string(), "SOL".to_string()];

for pf_name in pf_names {
    println!("pf_name = {pf_name}");
}
```

Try to keep meaning with the plural "name**s**"

### Breaking out of a for loop
I prefer breaking a for loop using its name, not a simple break

Example:
```rust
let some_condition: bool = true;

'iteration_loop: for i_iter in 0..self.n_iter_max {
    if some_condition {
        break 'iteration_loop;
    }
}
```

### Type hints
You should always give the variable type, not rely on the autodetected value.
This is wrong:
```rust
use ndarray::Array1;
let x: Array1<f64> = Array1::linspace(0.0, 1.0, 100);
let n_x = x.len();
```
This is correct:
```rust
use ndarray::Array1;
let x: Array1<f64> = Array1::linspace(0.0, 1.0, 100);
let n_x: usize = x.len();
```

With `ndarray` I prefer to use `Array1`, `Array2`, `Array3`, ..., instead of using `Array` paired with the dimension.

### Types
* For floats we should use `f64`
* For other data use appropriate data types, e.g. `usize`, ...

### `d_r`/`d_z` vs `delta_r`/`delta_z`
* `d_r`/`d_z` are for grid spacing
* `delta_r`/`delta_z` are for distances between points

## Looping style preference
I prefer for loops rather than `.map` or `.iter` for clarity.

## Units
Units should be specified for type hints into / out of functions, e.g.
```python
import numpy as np
import numpy.typing as npt

def greens_py(
    r: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    r_prime: npt.NDArray[np.float64],
    z_prime: npt.NDArray[np.float64],
    d_r: npt.NDArray[np.float64] | None = None,
    d_z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    :param r: (by convention) Sensor radial positions [metre]
    :param z: (by convention) Sensor vertical positions [metre]
    :param r_prime: (by convention) Current source radial positions [metre]
    :param z_prime: (by convention) Current source vertical positions [metre]
    :param d_r: (optional) Radial widths [metre]
    :param d_z: (optional) Vertical heights [metre]

    Note: the inputs are symmetrical
    """
    ...
```

### Base units
Within code, quantities should be written using the so called "MKS+eV" units.
We should write units using their "full names" in plain text, the most common units are:
* `ampere`
* `count`
* `dimensionless`
* `electron_volt`
* `hertz`
* `joule`
* `kelvin`
* `kilogram`
* `metre`
* `ohm`
* `radian`
* `second`
* `tesla`
* `volt`
* `weber`

We have chosen to write units using the "full names" to increase readability and to prevent any potential confusion.

**Example 1**: The symbol for kelvin is `K` (uppercase), if `k` (lowercase) were used this would be interpreted as the `boltzmann_constant` (extremely easy to make mistakes)!

**Example 2**: Generally, the ångström should not be used because it's not MKS+eV. But given a good reason we may record data using the ångström, if we do use it, it should be written in "plain text" as:
* `angstrom`
which is unambiguous.

### Compound units
Quantities containing more than one unit need to include the mathematical operation, for example the following would be correct:
* `ampere * metre`
* `ampere / metre`
* `1 / metre ** 3`
The following would be incorrect:
* `metre ampere` would not work, missing the multiplication operator (and not alphabetically sorted)
* `ampere * 1 / metre` would work, but not helpful extra mathematical operations
* `metre ** -3` would work, but not using standard division

### `metre` vs `meter`
We choose to use `metre`, because it's the SI standard spelling.
Only the United States and Philippines use `meter`.

### For plotting
In plotting the compact form should be used with a square bracket, e.g. `[m]` for `metre`, `[s]` for `second`, `[T]` for `tesla`, etc.

## Initializing empty arrays
For safety I like to initialize numerical arrays full with `f64::NAN` this way if an index is missed an error will most likely be thrown, e.g.

```rust
use ndarray::Array1;

let n_time: usize = 100;

let mut matrix: Array1<f64> = Array1::from_elem(n_time, f64::NAN);

for i_time in 0..n_time - 1 { // will miss the last element of the matrix
    matrix[i_time] = i_time as f64;
}
```

This is not a hard rule, if we are only going to assign to the matrix diagonal then we can initialize the matrix to zeros.

## Alphabetically sorting variables
Where possible variables should be alphabetically sorted, this is not a hard rule, but it can help with readability.

Try to use a sensible order for function arguments, for example derivatives should be ordered in a way that matches the mathematical notation, e.g. `d2_psi_d_r2`, `d2_psi_d_r_d_z`, `d2_psi_d_z2` instead of `d2_psi_d_r2`, `d2_psi_d_z2`, `d2_psi_d_r_d_z`.
An example is the Hessian function, which correctly takes:
```rust
let (hessian_determinant, hessian_trace): (f64, f64) = hessian(d2_psi_d_r2, d2_psi_d_r_d_z, d2_psi_d_z2);
```

# Python
When using strings I prefer a double backslash `\\` instead of raw, e.g.
```python
# This is wrong
mds_path = r"\GSFIT::TOP.BEST.GLOBAL:IP"
# This is correct
mds_path = "\\GSFIT::TOP.BEST.GLOBAL:IP"
```
