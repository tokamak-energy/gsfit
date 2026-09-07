//! Plasma grid-to-grid Green's convolution, evaluated with FFTs along the vertical axis.
//!
//! The poloidal flux (and its derivatives) produced on the `(R, Z)` grid by the plasma current is
//! ```text
//!     field_k[(i_z, i_r)] = sum_{i_cur_z, i_cur_r} g_k[(|i_z - i_cur_z|, i_r, i_cur_r)] * j_2d[(i_cur_z, i_cur_r)] * d_area
//! ```
//! because the grid is uniform in `z`, so a Green's table only depends on the vertical **offset**
//! between the grid point and the current source (see `Plasma::new`). For every pair of radial
//! indices `(i_r, i_cur_r)` this is a one-dimensional convolution along `z` of the current column
//! `j_2d[(.., i_cur_r)]` with the kernel `g_k[(.., i_r, i_cur_r)]`, extended to negative offsets as
//! an even function (`psi`, `d_psi_d_r`, `d2_psi_d_r2`, `d2_psi_d_z2`, `d3_psi_d_r_d_z2`) or an odd
//! function (`d_psi_d_z`, `d2_psi_d_r_d_z`, `d3_psi_d_r2_d_z`, `d3_psi_d_z3`; the sign is that of
//! `i_z - i_cur_z`, i.e. sources below the grid point count positive).
//!
//! Convolutions are products in Fourier space, so the kernels are transformed **once** per
//! reconstruction (they depend only on the grid), and each Picard iteration only transforms the
//! current columns, multiplies, and transforms back:
//! ```text
//!     field_k[(.., i_r)] = irfft( sum_{i_cur_r} G_k[(i_r, i_cur_r, ..)] * rfft(j_2d[(.., i_cur_r)] * d_area) )
//! ```
//! Zero-padding to `fft_length >= 2 * n_z - 1` makes the circular convolution equal to the linear one.
//! Even kernels have purely real spectra and odd kernels purely imaginary ones, so one `f64` per
//! frequency bin is stored for each kernel and radial pair.
//!
//! Per iteration this costs `O(n_r^2 * n_z * log(n_z))` instead of the `O(n_r^2 * n_z^2)` of the
//! dense matrix product it replaces; the result is the same sum evaluated in a different
//! (floating-point rounding level) order.
use crate::Plasma;
use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Even kernels, in the column-block order produced by `convolve`
pub const EVEN_KERNEL_NAMES: [&str; 5] = ["psi", "d_psi_d_r", "d2_psi_d_r2", "d2_psi_d_z2", "d3_psi_d_r_d_z2"];
/// Odd kernels, in the column-block order produced by `convolve`
pub const ODD_KERNEL_NAMES: [&str; 4] = ["d_psi_d_z", "d2_psi_d_r_d_z", "d3_psi_d_r2_d_z", "d3_psi_d_z3"];

const N_EVEN_KERNELS: usize = 5;
const N_ODD_KERNELS: usize = 4;

/// Fourier transforms of the plasma grid-to-grid Green's tables (see the module documentation)
pub struct PlasmaGreensSpectra {
    n_r: usize,
    n_z: usize,
    fft_length: usize,
    n_bins: usize,
    /// Real part of the even-kernel spectra, divided by `fft_length` (the inverse transform is unnormalised);
    /// index = `((i_kernel * n_r + i_r) * n_r + i_cur_r) * n_bins + i_bin`
    even_spectra: Vec<f64>,
    /// Imaginary part of the odd-kernel spectra, divided by `fft_length`; same indexing as `even_spectra`
    odd_spectra: Vec<f64>,
    fft_forward: Arc<dyn RealToComplex<f64>>,
    fft_inverse: Arc<dyn ComplexToReal<f64>>,
}

/// Smallest even integer `>= n_min` whose prime factors are all 2, 3 or 5, so that the FFT is fast
fn smooth_even_fft_length(n_min: usize) -> usize {
    let n_candidate_max: usize = 2 * n_min.max(2) + 2;
    for n_candidate in n_min.max(2)..=n_candidate_max {
        if n_candidate % 2 != 0 {
            continue;
        }
        let mut remainder: usize = n_candidate;
        for prime in [2, 3, 5] {
            let n_divisions_max: usize = 64;
            for _i_division in 0..n_divisions_max {
                if remainder % prime == 0 {
                    remainder /= prime;
                } else {
                    break;
                }
            }
        }
        if remainder == 1 {
            return n_candidate;
        }
    }
    // Unreachable in practice (a power of two always lies within the search range); fall back to the next power of two
    return n_min.next_power_of_two();
}

impl PlasmaGreensSpectra {
    /// Build the spectra from the `greens/grid_grid` tables stored in `plasma`
    pub fn new(plasma: &Plasma) -> Self {
        let n_r: usize = plasma.results.get("grid").get("n_r").unwrap_usize();
        let n_z: usize = plasma.results.get("grid").get("n_z").unwrap_usize();

        // Stored shape = (n_z * n_r, n_r), which unflattens to (i_offset_z, i_r, i_cur_r)
        let grid_grid = |key: &str| -> Array3<f64> {
            plasma
                .results
                .get("greens")
                .get("grid_grid")
                .get(key)
                .unwrap_array2()
                .to_shape((n_z, n_r, n_r))
                .expect("PlasmaGreensSpectra: failed to reshape grid_grid table into (n_z, n_r, n_r)")
                .to_owned()
        };
        let mut even_tables: Vec<Array3<f64>> = Vec::with_capacity(N_EVEN_KERNELS);
        for kernel_name in EVEN_KERNEL_NAMES {
            even_tables.push(grid_grid(kernel_name));
        }
        let mut odd_tables: Vec<Array3<f64>> = Vec::with_capacity(N_ODD_KERNELS);
        for kernel_name in ODD_KERNEL_NAMES {
            odd_tables.push(grid_grid(kernel_name));
        }
        let even_views: Vec<ArrayView3<f64>> = even_tables.iter().map(|table| table.view()).collect();
        let odd_views: Vec<ArrayView3<f64>> = odd_tables.iter().map(|table| table.view()).collect();

        return Self::from_tables(n_r, n_z, &even_views, &odd_views);
    }

    /// Build the spectra from Green's tables of shape `(n_z, n_r, n_r)` = `(i_offset_z, i_r, i_cur_r)`
    ///
    /// # Arguments
    /// * `n_r` - number of radial grid points, [dimensionless]
    /// * `n_z` - number of vertical grid points, [dimensionless]
    /// * `even_tables` - the five kernels which are even in `z - z_current_source`, in `EVEN_KERNEL_NAMES` order
    /// * `odd_tables` - the four kernels which are odd in `z - z_current_source`, in `ODD_KERNEL_NAMES` order
    pub fn from_tables(n_r: usize, n_z: usize, even_tables: &[ArrayView3<f64>], odd_tables: &[ArrayView3<f64>]) -> Self {
        assert!(
            even_tables.len() == N_EVEN_KERNELS,
            "PlasmaGreensSpectra: expected {N_EVEN_KERNELS} even kernels"
        );
        assert!(odd_tables.len() == N_ODD_KERNELS, "PlasmaGreensSpectra: expected {N_ODD_KERNELS} odd kernels");
        for table in even_tables.iter().chain(odd_tables.iter()) {
            assert!(table.dim() == (n_z, n_r, n_r), "PlasmaGreensSpectra: Green's table has the wrong shape");
        }

        // Zero-padding so the circular convolution equals the linear convolution over all offsets
        let fft_length: usize = smooth_even_fft_length(2 * n_z - 1);
        let n_bins: usize = fft_length / 2 + 1;

        let mut planner: RealFftPlanner<f64> = RealFftPlanner::<f64>::new();
        let fft_forward: Arc<dyn RealToComplex<f64>> = planner.plan_fft_forward(fft_length);
        let fft_inverse: Arc<dyn ComplexToReal<f64>> = planner.plan_fft_inverse(fft_length);

        // Transform every (kernel, i_r, i_cur_r) kernel column; the inverse transform's normalisation is folded in here
        let normalisation: f64 = 1.0 / (fft_length as f64);
        let transform_kernels = |tables: &[ArrayView3<f64>], is_odd: bool| -> Vec<f64> {
            let n_kernels: usize = tables.len();
            let spectra_per_row: Vec<Vec<f64>> = (0..n_kernels * n_r)
                .into_par_iter()
                .map(|i_kernel_and_r: usize| {
                    let i_kernel: usize = i_kernel_and_r / n_r;
                    let i_r: usize = i_kernel_and_r % n_r;
                    let table: &ArrayView3<f64> = &tables[i_kernel];
                    let mut spectra_this_row: Vec<f64> = vec![0.0; n_r * n_bins];
                    let mut sequence: Vec<f64> = fft_forward.make_input_vec();
                    let mut spectrum: Vec<Complex<f64>> = fft_forward.make_output_vec();
                    for i_cur_r in 0..n_r {
                        // Kernel as a function of the signed offset `m = i_z - i_cur_z`, stored circularly:
                        // index `m` for `m >= 0`, index `fft_length + m` for `m < 0`
                        for value in sequence.iter_mut() {
                            *value = 0.0;
                        }
                        if is_odd {
                            // Odd kernels: sources below the grid point (`m > 0`) count positive, sources at or
                            // above (`m <= 0`) count negative, matching the sign convention of the direct sum.
                            // At zero offset an odd kernel vanishes by symmetry, so its spectrum is purely imaginary.
                            let g_zero_offset: f64 = table[(0, i_r, i_cur_r)];
                            assert!(
                                g_zero_offset == 0.0,
                                "PlasmaGreensSpectra: odd Green's kernel is not zero at zero vertical offset ({g_zero_offset})"
                            );
                            sequence[0] = -g_zero_offset;
                            for i_offset_z in 1..n_z {
                                let g_value: f64 = table[(i_offset_z, i_r, i_cur_r)];
                                sequence[i_offset_z] = g_value;
                                sequence[fft_length - i_offset_z] = -g_value;
                            }
                        } else {
                            sequence[0] = table[(0, i_r, i_cur_r)];
                            for i_offset_z in 1..n_z {
                                let g_value: f64 = table[(i_offset_z, i_r, i_cur_r)];
                                sequence[i_offset_z] = g_value;
                                sequence[fft_length - i_offset_z] = g_value;
                            }
                        }
                        fft_forward
                            .process(&mut sequence, &mut spectrum)
                            .expect("PlasmaGreensSpectra: forward FFT of a Green's kernel failed");
                        for i_bin in 0..n_bins {
                            spectra_this_row[i_cur_r * n_bins + i_bin] = if is_odd {
                                spectrum[i_bin].im * normalisation
                            } else {
                                spectrum[i_bin].re * normalisation
                            };
                        }
                    }
                    spectra_this_row
                })
                .collect();
            let mut spectra: Vec<f64> = Vec::with_capacity(n_kernels * n_r * n_r * n_bins);
            for spectra_this_row in spectra_per_row {
                spectra.extend_from_slice(&spectra_this_row);
            }
            spectra
        };
        let even_spectra: Vec<f64> = transform_kernels(even_tables, false);
        let odd_spectra: Vec<f64> = transform_kernels(odd_tables, true);

        return PlasmaGreensSpectra {
            n_r,
            n_z,
            fft_length,
            n_bins,
            even_spectra,
            odd_spectra,
            fft_forward,
            fft_inverse,
        };
    }

    /// Plasma contribution to the nine fields, for the current density `j_2d`, shape = (n_z, n_r), [ampere / metre ** 2]
    ///
    /// # Returns
    /// * `plasma_even` - the five even kernels concatenated column-wise in `EVEN_KERNEL_NAMES` order; shape = (n_z, 5 * n_r)
    /// * `plasma_odd` - the four odd kernels concatenated column-wise in `ODD_KERNEL_NAMES` order; shape = (n_z, 4 * n_r)
    pub fn convolve(&self, j_2d: &Array2<f64>, d_area: f64) -> (Array2<f64>, Array2<f64>) {
        let n_r: usize = self.n_r;
        let n_z: usize = self.n_z;
        let n_bins: usize = self.n_bins;
        let fft_length: usize = self.fft_length;
        assert!(j_2d.dim() == (n_z, n_r), "PlasmaGreensSpectra::convolve: `j_2d` has the wrong shape");

        // Spectra of the current columns; columns without current are skipped in the products below
        let mut source_spectra: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_r * n_bins];
        let mut active_source_columns: Vec<usize> = Vec::with_capacity(n_r);
        let mut sequence: Vec<f64> = self.fft_forward.make_input_vec();
        let mut spectrum: Vec<Complex<f64>> = self.fft_forward.make_output_vec();
        for i_cur_r in 0..n_r {
            let mut column_has_current: bool = false;
            for value in sequence.iter_mut() {
                *value = 0.0;
            }
            for i_cur_z in 0..n_z {
                let current: f64 = d_area * j_2d[(i_cur_z, i_cur_r)];
                sequence[i_cur_z] = current;
                if current != 0.0 {
                    column_has_current = true;
                }
            }
            if !column_has_current {
                continue;
            }
            self.fft_forward
                .process(&mut sequence, &mut spectrum)
                .expect("PlasmaGreensSpectra::convolve: forward FFT of a current column failed");
            source_spectra[i_cur_r * n_bins..(i_cur_r + 1) * n_bins].copy_from_slice(&spectrum);
            active_source_columns.push(i_cur_r);
        }

        // One task per (kernel, i_r): accumulate the spectral products over the active current columns,
        // then transform back; the first `n_z` samples of the padded result are the field column
        let field_columns = |spectra: &Vec<f64>, n_kernels: usize, is_odd: bool| -> Vec<Vec<f64>> {
            (0..n_kernels * n_r)
                .into_par_iter()
                .map(|i_kernel_and_r: usize| {
                    let mut accumulator: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_bins];
                    for &i_cur_r in &active_source_columns {
                        let i_start: usize = (i_kernel_and_r * n_r + i_cur_r) * n_bins;
                        let kernel_spectrum: &[f64] = &spectra[i_start..i_start + n_bins];
                        let source_spectrum: &[Complex<f64>] = &source_spectra[i_cur_r * n_bins..(i_cur_r + 1) * n_bins];
                        if is_odd {
                            // kernel spectrum = i * g_im; (i * g_im) * (s_re + i * s_im) = -g_im * s_im + i * g_im * s_re
                            for i_bin in 0..n_bins {
                                accumulator[i_bin].re -= kernel_spectrum[i_bin] * source_spectrum[i_bin].im;
                                accumulator[i_bin].im += kernel_spectrum[i_bin] * source_spectrum[i_bin].re;
                            }
                        } else {
                            for i_bin in 0..n_bins {
                                accumulator[i_bin].re += kernel_spectrum[i_bin] * source_spectrum[i_bin].re;
                                accumulator[i_bin].im += kernel_spectrum[i_bin] * source_spectrum[i_bin].im;
                            }
                        }
                    }
                    // A real signal has real DC and Nyquist bins; rounding leaves tiny imaginary parts which
                    // the inverse real transform rejects, so they are cleared explicitly
                    accumulator[0].im = 0.0;
                    if fft_length % 2 == 0 {
                        accumulator[n_bins - 1].im = 0.0;
                    }
                    let mut padded_field: Vec<f64> = self.fft_inverse.make_output_vec();
                    self.fft_inverse
                        .process(&mut accumulator, &mut padded_field)
                        .expect("PlasmaGreensSpectra::convolve: inverse FFT failed");
                    padded_field.truncate(n_z);
                    padded_field
                })
                .collect()
        };
        let even_columns: Vec<Vec<f64>> = field_columns(&self.even_spectra, N_EVEN_KERNELS, false);
        let odd_columns: Vec<Vec<f64>> = field_columns(&self.odd_spectra, N_ODD_KERNELS, true);

        // Column block `i_kernel` holds kernel `i_kernel`, matching the layout consumed by `calculate_psi_and_derivatives`
        let mut plasma_even: Array2<f64> = Array2::from_elem((n_z, N_EVEN_KERNELS * n_r), f64::NAN);
        for i_kernel_and_r in 0..N_EVEN_KERNELS * n_r {
            for i_z in 0..n_z {
                plasma_even[(i_z, i_kernel_and_r)] = even_columns[i_kernel_and_r][i_z];
            }
        }
        let mut plasma_odd: Array2<f64> = Array2::from_elem((n_z, N_ODD_KERNELS * n_r), f64::NAN);
        for i_kernel_and_r in 0..N_ODD_KERNELS * n_r {
            for i_z in 0..n_z {
                plasma_odd[(i_z, i_kernel_and_r)] = odd_columns[i_kernel_and_r][i_z];
            }
        }

        return (plasma_even, plasma_odd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Direct evaluation of the double sum at one grid point, for comparison with the FFT convolution
    fn direct_sum_at(table: &Array3<f64>, j_2d: &Array2<f64>, d_area: f64, is_odd: bool, i_z: usize, i_r: usize) -> f64 {
        let (n_z, n_r, _): (usize, usize, usize) = table.dim();
        let mut field: f64 = 0.0;
        for i_cur_z in 0..n_z {
            for i_cur_r in 0..n_r {
                let i_offset_z: usize = i_z.abs_diff(i_cur_z);
                let mut sign: f64 = 1.0;
                if is_odd && i_cur_z >= i_z {
                    sign = -1.0;
                }
                field += sign * table[(i_offset_z, i_r, i_cur_r)] * j_2d[(i_cur_z, i_cur_r)] * d_area;
            }
        }
        field
    }

    /// Deterministic pseudo-random values in [-1, 1)
    fn pseudo_random(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn test_smooth_even_fft_length() {
        assert_eq!(smooth_even_fft_length(321), 324);
        assert_eq!(smooth_even_fft_length(257), 270);
        assert_eq!(smooth_even_fft_length(641), 648);
        assert_eq!(smooth_even_fft_length(1), 2);
    }

    #[test]
    fn test_convolve_matches_direct_sum() {
        convolve_matches_direct_sum(5, 7, 12345);
    }

    /// The MAST-U example grid (65 x 129, FFT length 270) and the ST40 default grid (81 x 161, FFT length 324)
    #[test]
    fn test_convolve_matches_direct_sum_at_production_sizes() {
        convolve_matches_direct_sum(65, 129, 777);
        convolve_matches_direct_sum(81, 161, 999);
    }

    fn convolve_matches_direct_sum(n_r: usize, n_z: usize, seed_start: u64) {
        let d_area: f64 = 0.25;
        let mut seed: u64 = seed_start;

        let mut even_tables: Vec<Array3<f64>> = Vec::new();
        for _i_kernel in 0..N_EVEN_KERNELS {
            let mut table: Array3<f64> = Array3::zeros((n_z, n_r, n_r));
            for value in table.iter_mut() {
                *value = pseudo_random(&mut seed);
            }
            even_tables.push(table);
        }
        let mut odd_tables: Vec<Array3<f64>> = Vec::new();
        for _i_kernel in 0..N_ODD_KERNELS {
            let mut table: Array3<f64> = Array3::zeros((n_z, n_r, n_r));
            for value in table.iter_mut() {
                *value = pseudo_random(&mut seed);
            }
            // Odd kernels vanish at zero vertical offset
            for i_r in 0..n_r {
                for i_cur_r in 0..n_r {
                    table[(0, i_r, i_cur_r)] = 0.0;
                }
            }
            odd_tables.push(table);
        }
        let mut j_2d: Array2<f64> = Array2::zeros((n_z, n_r));
        for value in j_2d.iter_mut() {
            *value = pseudo_random(&mut seed);
        }
        // An empty current column exercises the active-column selection
        for i_z in 0..n_z {
            j_2d[(i_z, 2)] = 0.0;
        }

        let even_views: Vec<ArrayView3<f64>> = even_tables.iter().map(|table| table.view()).collect();
        let odd_views: Vec<ArrayView3<f64>> = odd_tables.iter().map(|table| table.view()).collect();
        let spectra: PlasmaGreensSpectra = PlasmaGreensSpectra::from_tables(n_r, n_z, &even_views, &odd_views);
        let (plasma_even, plasma_odd): (Array2<f64>, Array2<f64>) = spectra.convolve(&j_2d, d_area);

        // Check every grid point on small grids, and a sample of grid points on production-sized grids
        // (the direct double sum is O(n_z^2 n_r^2))
        let n_points_max: usize = 200;
        let mut points: Vec<(usize, usize)> = Vec::new();
        if n_z * n_r <= n_points_max {
            for i_z in 0..n_z {
                for i_r in 0..n_r {
                    points.push((i_z, i_r));
                }
            }
        } else {
            for _i_point in 0..n_points_max {
                let i_z: usize = ((pseudo_random(&mut seed) + 1.0) / 2.0 * (n_z as f64)) as usize % n_z;
                let i_r: usize = ((pseudo_random(&mut seed) + 1.0) / 2.0 * (n_r as f64)) as usize % n_r;
                points.push((i_z, i_r));
            }
            // Always include the corners and the centre
            points.push((0, 0));
            points.push((n_z - 1, n_r - 1));
            points.push((n_z / 2, n_r / 2));
        }
        let scale_even: f64 = plasma_even.iter().fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let scale_odd: f64 = plasma_odd.iter().fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        for i_kernel in 0..N_EVEN_KERNELS {
            for &(i_z, i_r) in &points {
                let expected: f64 = direct_sum_at(&even_tables[i_kernel], &j_2d, d_area, false, i_z, i_r);
                let difference: f64 = (plasma_even[(i_z, i_kernel * n_r + i_r)] - expected).abs();
                assert!(
                    difference < 1.0e-12 * scale_even.max(1.0),
                    "even kernel {i_kernel}: difference {difference} at ({i_z}, {i_r})"
                );
            }
        }
        for i_kernel in 0..N_ODD_KERNELS {
            for &(i_z, i_r) in &points {
                let expected: f64 = direct_sum_at(&odd_tables[i_kernel], &j_2d, d_area, true, i_z, i_r);
                let difference: f64 = (plasma_odd[(i_z, i_kernel * n_r + i_r)] - expected).abs();
                assert!(
                    difference < 1.0e-12 * scale_odd.max(1.0),
                    "odd kernel {i_kernel}: difference {difference} at ({i_z}, {i_r})"
                );
            }
        }
    }
}
