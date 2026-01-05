use crate::errors::Error;
use ndarray::Array1;

pub struct Dim1Linear<'a> {
    x: &'a Array1<f64>,
    f: &'a Array1<f64>,
}

impl<'a> Dim1Linear<'a> {
    /// Create a new 1D linear interpolator
    ///
    /// # Arguments
    /// - `x`: The x values at which the function is known
    /// - `f`: The function values at the known x values
    ///
    /// # Returns
    /// A new `Dim1Linear` interpolator
    ///
    /// # Notes
    /// To improve memory efficiency, I decided to use references to `x` and `f` rather than
    /// owning them, which would typically involve a `.clone()`.
    /// This does mean that the lifetime of the `Dim1Linear` struct is tied to the lifetime
    /// of the input arrays, which adds some complexity.
    ///
    pub fn new(x: &'a Array1<f64>, f: &'a Array1<f64>) -> Result<Self, Error> {
        if x.len() != f.len() {
            return Err(Error::FunctionAndXLengthMismatch {
                f_len: f.len(),
                x_len: x.len(),
            });
        }

        // Check for the same `x` values
        for i_x in 0..x.len() - 1 {
            if (x[i_x + 1] - x[i_x]).abs() < std::f64::EPSILON {
                return Err(Error::DuplicateXValues {
                    x_value: x[i_x],
                    index: i_x,
                });
            }
        }

        // Check that `x` is increasing
        for i_x in 0..x.len() - 1 {
            if x[i_x + 1] <= x[i_x] {
                return Err(Error::XNotIncreasing {
                    x_value1: x[i_x],
                    x_value2: x[i_x + 1],
                    index: i_x,
                });
            }
        }

        return Ok(Dim1Linear { x, f });
    }

    pub fn interpolate_array1(&self, x_new: &Array1<f64>) -> Result<Array1<f64>, Error> {
        let n_x_new: usize = x_new.len();

        // Check bounds and exit if out of bounds
        for i_x_new in 0..n_x_new {
            if x_new[i_x_new] < self.x[0] || x_new[i_x_new] > self.x[self.x.len() - 1] {
                return Err(Error::XOutOfBounds {
                    x_desired: x_new[i_x_new],
                    x_min: self.x[0],
                    x_max: self.x[self.x.len() - 1],
                });
            }
        }

        let n_x: usize = self.x.len();
        let mut f_new: Array1<f64> = Array1::<f64>::zeros(n_x_new);

        // Loop over the new x values
        for i_x_new in 0..n_x_new {
            // Loop over `x` to find the interval over which to interpolate
            for i_x in 0..n_x - 1 {
                if x_new[i_x_new] >= self.x[i_x] && x_new[i_x_new] <= self.x[i_x + 1] { // TODO: I think this can be simplified?
                    let segment_fraction: f64 = (x_new[i_x_new] - self.x[i_x]) / (self.x[i_x + 1] - self.x[i_x]);
                    f_new[i_x_new] = self.f[i_x] * (1.0 - segment_fraction) + self.f[i_x + 1] * segment_fraction;
                    break;
                }
            }
        }

        return Ok(f_new);
    }
}
