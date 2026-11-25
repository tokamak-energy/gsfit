//! Lattice series: finite exponential series in `Z` whose coefficients are power series in the scaled radial variable,
//!
//! ```text
//! (1 - psi_N)(R, Z) = Re sum_j sum_m d_{j,m} radial(R)**m exp(i k_j Z),    k_j = j . spacings,
//! ```
//!
//! over every lattice point `j` of a box `|j_n| <= L_n`. The spacing is that of the paper's equally spaced wavenumbers,
//! `k_spacing k_max` (a one-dimensional lattice), `Z` is measured from `z_geo`, and `radial` is the scaled radial variable of
//! `radial_series`. Since products of exponentials are exponentials at the summed wavenumbers, products of such series are again
//! lattice series: a convolution over the lattice axes and a Cauchy product over the powers, done by FFT (see `series_product`).
//!
//! `(1 - psi_N)` is real: `d_{j*} = conj(d_j)`, `j*` being `-j` with its imaginary components restored (for a real lattice, `j* = -j`).
//! An up-down symmetric `(1 - psi_N)` is also even in `Z`, `d_{-j} = d_j`, so that it is a cosine series with real coefficients on a
//! real lattice; an up-down asymmetric one (single null, different upper and lower shapes) has complex coefficients, whose imaginary
//! parts are its sine terms.
//!
//! A spacing can also be imaginary, `i k_spacing_cosh k_max` (the paper's `cosh` terms, `k_j**2 < 0`, for the X-point corners of a
//! spherical tokamak): its terms are `cosh(|k_j| Z)`, which grow towards the top and bottom, with radial functions that oscillate.
//! Products then reach complex wavenumbers, with complex `k_j**2`, complex radial functions and complex coefficients. Every function
//! of the series is evaluated as the real part of the sum, which is all there is.
//!
//! The lattice points of a box are stored in row-major order (the last lattice axis fastest), so that flat index `i` and
//! `n_point - 1 - i` are `j` and `-j`.

use super::numerics::binomial;
use super::radial_series::{RadialDerivatives, RadialVariable, RadialVariableKind};
use faer::c64;
use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use ndarray::{Array1, Array2, ArrayView1, s};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Largest power of 3, and of 5, in the FFT lengths [dimensionless]
const N_FFT_FACTOR_POWER_MAX: u32 = 8;

/// The smallest length `>= length_min` whose only prime factors are 2, 3 and 5, on which rustfft is fast
fn fft_length(length_min: usize) -> usize {
    let length_min: usize = length_min.max(1);
    let mut length: usize = length_min.next_power_of_two();
    'factor_3_loop: for i_power_3 in 0..=N_FFT_FACTOR_POWER_MAX {
        let factor_3: usize = 3_usize.pow(i_power_3);
        if factor_3 > length {
            break 'factor_3_loop;
        }
        'factor_5_loop: for i_power_5 in 0..=N_FFT_FACTOR_POWER_MAX {
            let odd_factor: usize = factor_3 * 5_usize.pow(i_power_5);
            if odd_factor > length {
                break 'factor_5_loop;
            }
            length = length.min(odd_factor * length_min.div_ceil(odd_factor).next_power_of_two());
        }
    }
    length
}

/// Points per block in the evaluation of a series: small enough that a few hundred points are shared among the threads
const N_POINT_BLOCK: usize = 128;

/// Rows of the result per task in the transposes of the multidimensional FFT: four complex values fill a cache line
const N_ROTATION_BLOCK: usize = 4;

/// The parity of a single term in `Z`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    /// `cos(k_j Z)` (`cosh(|k_j| Z)` for an imaginary `k_j`), even in `Z`
    Cos,
    /// `sin(k_j Z)` (`sinh(|k_j| Z)` for an imaginary `k_j`), odd in `Z`
    Sin,
}

/// `(1 - psi_N)` and its first and second derivatives in `R` and `Z`, at a set of points
pub struct FluxValues {
    /// [dimensionless]
    pub one_minus_psi_n: Array1<f64>,
    /// [1 / metre]
    pub d_d_r: Array1<f64>,
    /// [1 / metre]
    pub d_d_z: Array1<f64>,
    /// [1 / metre ** 2]
    pub d2_d_r2: Array1<f64>,
    /// [1 / metre ** 2]
    pub d2_d_r_d_z: Array1<f64>,
    /// [1 / metre ** 2]
    pub d2_d_z2: Array1<f64>,
}

impl FluxValues {
    /// Zeros at `n_point` points
    pub fn zeros(n_point: usize) -> FluxValues {
        FluxValues {
            one_minus_psi_n: Array1::zeros(n_point),
            d_d_r: Array1::zeros(n_point),
            d_d_z: Array1::zeros(n_point),
            d2_d_r2: Array1::zeros(n_point),
            d2_d_r_d_z: Array1::zeros(n_point),
            d2_d_z2: Array1::zeros(n_point),
        }
    }

    /// Every value divided by `divisor`
    pub fn divided_by(&self, divisor: f64) -> FluxValues {
        FluxValues {
            one_minus_psi_n: &self.one_minus_psi_n / divisor,
            d_d_r: &self.d_d_r / divisor,
            d_d_z: &self.d_d_z / divisor,
            d2_d_r2: &self.d2_d_r2 / divisor,
            d2_d_r_d_z: &self.d2_d_r_d_z / divisor,
            d2_d_z2: &self.d2_d_z2 / divisor,
        }
    }
}

/// The wavenumber lattice and the radial variable shared by every series of one solution.
#[derive(Clone)]
pub struct Lattice {
    /// The spacings of the wavenumbers along the lattice axes: real, `k_spacing k_max`, or imaginary, `i k_spacing_cosh k_max`, for the
    /// `cosh` terms [1 / metre]
    pub wavenumber_spacings: Vec<c64>,
    /// Whether a spacing is imaginary
    pub is_complex: bool,
    /// Whether the series are even in `Z` (cosine series)
    pub is_up_down_symmetric: bool,
    /// Whether every coefficient is real: a real lattice of up-down symmetric series, which keeps real arithmetic throughout
    pub is_real: bool,
    /// Number of lattice axes [dimensionless]
    pub n_base: usize,
    /// The radial variable of the power series
    pub radial: Arc<RadialVariable>,
    /// Number of terms in every radial power series [dimensionless]
    pub n_radial: usize,
    /// Largest `|Z|` of the plasma, where `cosh` terms are largest [metre]
    pub z_top: f64,
    /// The plasma's half-height at the radial quadrature nodes, if known: the minimum-norm projections of the wavenumbers with an
    /// imaginary part `b` then weight `R**2` by `cosh(b half_height(R**2))**2`, their largest amplification over the plasma's column
    /// there (`reference_operator::particular_solution`) [metre]
    pub plasma_heights: Option<Array1<f64>>,
}

impl Lattice {
    /// # Arguments
    /// * `wavenumber_spacings` - the spacings of the wavenumbers: real, or imaginary for the `cosh` terms [1 / metre]
    /// * `radial` - the radial variable of the power series
    /// * `z_top` - largest `|Z|` of the plasma [metre]
    /// * `is_up_down_symmetric` - whether the series are even in `Z`; if not, they have complex coefficients
    pub fn new(wavenumber_spacings: Vec<c64>, radial: Arc<RadialVariable>, z_top: f64, is_up_down_symmetric: bool) -> Lattice {
        let n_base: usize = wavenumber_spacings.len();
        let mut is_complex: bool = false;
        for i_base in 0..n_base {
            is_complex |= wavenumber_spacings[i_base].im != 0.0;
        }
        let n_radial: usize = radial.n_radial;
        Lattice {
            wavenumber_spacings,
            is_complex,
            is_up_down_symmetric,
            is_real: !is_complex && is_up_down_symmetric,
            n_base,
            radial,
            n_radial,
            z_top,
            plasma_heights: None,
        }
    }

    /// `k_j = j . spacings` at every lattice point of a box, in row-major order; complex with imaginary spacings [1 / metre]
    pub fn box_wavenumbers(&self, half_widths: &[usize]) -> Vec<c64> {
        let offsets: Vec<Vec<i64>> = box_offsets(half_widths);
        let n_point: usize = offsets.len();
        let mut wavenumbers: Vec<c64> = vec![c64::new(0.0, 0.0); n_point];
        for i_point in 0..n_point {
            for i_base in 0..self.n_base {
                wavenumbers[i_point] += self.wavenumber_spacings[i_base] * (offsets[i_point][i_base] as f64);
            }
        }
        wavenumbers
    }

    /// `k_j = j . spacings` of one lattice point [1 / metre]
    pub fn wavenumber(&self, key: &[i64]) -> c64 {
        let mut wavenumber: c64 = c64::new(0.0, 0.0);
        for i_base in 0..self.n_base {
            wavenumber += self.wavenumber_spacings[i_base] * (key[i_base] as f64);
        }
        wavenumber
    }
}

/// `(2 L_1 + 1) ... (2 L_n + 1)`, the number of lattice points of a box
pub fn box_size(half_widths: &[usize]) -> usize {
    let n_axis: usize = half_widths.len();
    let mut size: usize = 1;
    for i_axis in 0..n_axis {
        size *= 2 * half_widths[i_axis] + 1;
    }
    size
}

/// The offsets `j` of every lattice point of a box, in row-major order
pub fn box_offsets(half_widths: &[usize]) -> Vec<Vec<i64>> {
    let n_axis: usize = half_widths.len();
    let n_point: usize = box_size(half_widths);
    let mut offsets: Vec<Vec<i64>> = Vec::with_capacity(n_point);
    for i_point in 0..n_point {
        let mut remainder: usize = i_point;
        let mut offset: Vec<i64> = vec![0; n_axis];
        for i_axis in (0..n_axis).rev() {
            let width: usize = 2 * half_widths[i_axis] + 1;
            offset[i_axis] = (remainder % width) as i64 - half_widths[i_axis] as i64;
            remainder /= width;
        }
        offsets.push(offset);
    }
    offsets
}

/// The flat index of the lattice point `offset` in a box, or `None` when it lies outside
pub fn box_index(half_widths: &[usize], offset: &[i64]) -> Option<usize> {
    let n_axis: usize = half_widths.len();
    let mut index: usize = 0;
    for i_axis in 0..n_axis {
        let half_width: i64 = half_widths[i_axis] as i64;
        if offset[i_axis] < -half_width || offset[i_axis] > half_width {
            return None;
        }
        index = index * (2 * half_widths[i_axis] + 1) + (offset[i_axis] + half_width) as usize;
    }
    Some(index)
}

/// `|j_1| + ... + |j_n|`
pub fn l1_norm(offset: &[i64]) -> usize {
    let n_axis: usize = offset.len();
    let mut norm: usize = 0;
    for i_axis in 0..n_axis {
        norm += offset[i_axis].unsigned_abs() as usize;
    }
    norm
}

/// `Re sum_j sum_m coefficients[j, m] radial**m exp(i k_j Z)` over the box `|j_n| <= L_n`, `L = half_widths`: the paper's `d_{j,m}`.
#[derive(Clone, Debug)]
pub struct LatticeSeries {
    /// Half-widths `L_n` of the box, one per lattice axis [dimensionless]
    pub half_widths: Vec<usize>,
    /// Shape `(n_lattice_point, n_radial)`, the lattice points in row-major order [dimensionless]
    pub coefficients: Array2<c64>,
    /// Largest `|j_1| + ... + |j_n|` that can be non-zero [dimensionless]
    pub support: usize,
}

impl LatticeSeries {
    /// Whether a lattice point's row has a non-zero coefficient
    fn is_row_nonzero(&self, i_row: usize) -> bool {
        let n_radial: usize = self.coefficients.ncols();
        for i_radial in 0..n_radial {
            let coefficient: c64 = self.coefficients[[i_row, i_radial]];
            if coefficient.re != 0.0 || coefficient.im != 0.0 {
                return true;
            }
        }
        false
    }
}

/// The zero series in a box
pub fn series_zero(lattice: &Lattice, half_widths: &[usize]) -> LatticeSeries {
    LatticeSeries {
        half_widths: half_widths.to_vec(),
        coefficients: Array2::from_elem((box_size(half_widths), lattice.n_radial), c64::new(0.0, 0.0)),
        support: 0,
    }
}

/// The single term `cos(k_j Z) X_j` (`Parity::Cos`), or its odd partner (`Parity::Sin`): `sin(k_j Z) X_j` for a real wavenumber,
/// `sinh(|k_j| Z) X_j` for an imaginary one (`sin` itself would be imaginary there).
///
/// # Arguments
/// * `lattice` - the lattice
/// * `key` - the lattice point `j`, not 0 for `Sin` [dimensionless]
/// * `radial_coefficients` - power-series coefficients of `X_j` in the scaled radial variable [dimensionless]
/// * `parity` - `Cos` (even in `Z`) or `Sin` (odd in `Z`)
///
/// # Errors
/// A sine term at the zero wavenumber, or on an up-down symmetric (real) lattice
pub fn series_from_term(lattice: &Lattice, key: &[i64], radial_coefficients: ArrayView1<c64>, parity: Parity) -> Result<LatticeSeries, String> {
    let n_base: usize = lattice.n_base;
    let mut half_widths: Vec<usize> = vec![0; n_base];
    let mut support: usize = 0;
    let mut minus_key: Vec<i64> = vec![0; n_base];
    for i_base in 0..n_base {
        half_widths[i_base] = key[i_base].unsigned_abs() as usize;
        support += half_widths[i_base];
        minus_key[i_base] = -key[i_base];
    }
    let mut series: LatticeSeries = series_zero(lattice, &half_widths);
    let plus: usize = box_index(&half_widths, key).expect("series_from_term: the key lies in its own box");
    let minus: usize = box_index(&half_widths, &minus_key).expect("series_from_term: the key lies in its own box");
    match parity {
        Parity::Cos => {
            if plus == minus {
                series.coefficients.row_mut(plus).assign(&radial_coefficients);
            } else {
                series.coefficients.row_mut(plus).assign(&radial_coefficients.mapv(|value: c64| value * 0.5));
                series.coefficients.row_mut(minus).assign(&radial_coefficients.mapv(|value: c64| value * 0.5));
            }
        }
        Parity::Sin => {
            if plus == minus {
                return Err("series_from_term: a sine term needs a non-zero wavenumber".to_string());
            }
            if lattice.is_real {
                return Err("series_from_term: sine terms need an up-down asymmetric (complex) lattice".to_string());
            }
            let wavenumber: c64 = lattice.wavenumber(key);
            if wavenumber.re == 0.0 {
                // sinh(|k| Z) = (exp(-i k Z) - exp(i k Z)) / 2 for k = i |k|, |k| > 0
                let sign: f64 = if wavenumber.im > 0.0 { 1.0 } else { -1.0 };
                series
                    .coefficients
                    .row_mut(plus)
                    .assign(&radial_coefficients.mapv(|value: c64| value * (-0.5 * sign)));
                series
                    .coefficients
                    .row_mut(minus)
                    .assign(&radial_coefficients.mapv(|value: c64| value * (0.5 * sign)));
            } else {
                // sin(k Z) = (exp(i k Z) - exp(-i k Z)) / (2 i)
                let minus_half_i: c64 = c64::new(0.0, -0.5);
                let half_i: c64 = c64::new(0.0, 0.5);
                series
                    .coefficients
                    .row_mut(plus)
                    .assign(&radial_coefficients.mapv(|value: c64| minus_half_i * value));
                series
                    .coefficients
                    .row_mut(minus)
                    .assign(&radial_coefficients.mapv(|value: c64| half_i * value));
            }
        }
    }
    series.support = support;
    Ok(series)
}

/// Pad with zeros, or crop, to a new box, axis by axis.
pub fn series_resize(series: &LatticeSeries, half_widths: &[usize]) -> LatticeSeries {
    if series.half_widths == half_widths {
        return series.clone();
    }
    let n_radial: usize = series.coefficients.ncols();
    let n_axis: usize = half_widths.len();
    let mut support: usize = 0;
    for i_axis in 0..n_axis {
        support += half_widths[i_axis];
    }
    let mut coefficients: Array2<c64> = Array2::from_elem((box_size(half_widths), n_radial), c64::new(0.0, 0.0));
    let offsets: Vec<Vec<i64>> = box_offsets(&series.half_widths);
    let n_row: usize = offsets.len();
    for i_row in 0..n_row {
        if let Some(i_target) = box_index(half_widths, &offsets[i_row]) {
            coefficients.row_mut(i_target).assign(&series.coefficients.row(i_row));
        }
    }
    LatticeSeries {
        half_widths: half_widths.to_vec(),
        coefficients,
        support: series.support.min(support),
    }
}

/// `series_a + scale_b * series_b`, in the box that holds both
pub fn series_add(series_a: &LatticeSeries, series_b: &LatticeSeries, scale_b: f64) -> LatticeSeries {
    let n_axis: usize = series_a.half_widths.len();
    let mut half_widths: Vec<usize> = vec![0; n_axis];
    for i_axis in 0..n_axis {
        half_widths[i_axis] = series_a.half_widths[i_axis].max(series_b.half_widths[i_axis]);
    }
    let resized_a: LatticeSeries = series_resize(series_a, &half_widths);
    let resized_b: LatticeSeries = series_resize(series_b, &half_widths);
    let mut coefficients: Array2<c64> = resized_a.coefficients;
    coefficients.zip_mut_with(&resized_b.coefficients, |value_a: &mut c64, value_b: &c64| *value_a += *value_b * scale_b);
    LatticeSeries {
        half_widths,
        coefficients,
        support: series_a.support.max(series_b.support),
    }
}

/// `total += scale * series`, in the box that holds both: in place when the box of `total` holds that of `series` (the same sum as
/// `series_add`, without its copies)
pub fn series_add_assign(total: &mut LatticeSeries, series: &LatticeSeries, scale: f64) {
    let n_axis: usize = total.half_widths.len();
    let mut half_widths: Vec<usize> = vec![0; n_axis];
    for i_axis in 0..n_axis {
        half_widths[i_axis] = total.half_widths[i_axis].max(series.half_widths[i_axis]);
    }
    if half_widths != total.half_widths {
        *total = series_resize(total, &half_widths);
    }
    if series.half_widths == half_widths {
        total
            .coefficients
            .zip_mut_with(&series.coefficients, |value_total: &mut c64, value: &c64| *value_total += *value * scale);
    } else {
        let n_radial: usize = series.coefficients.ncols();
        let offsets: Vec<Vec<i64>> = box_offsets(&series.half_widths);
        let n_row: usize = offsets.len();
        for i_row in 0..n_row {
            let i_total: usize = box_index(&half_widths, &offsets[i_row]).expect("series_add_assign: the box holds both");
            for i_radial in 0..n_radial {
                total.coefficients[[i_total, i_radial]] += series.coefficients[[i_row, i_radial]] * scale;
            }
        }
    }
    total.support = total.support.max(series.support);
}

/// `scale * series`
pub fn series_scale(series: &LatticeSeries, scale: f64) -> LatticeSeries {
    LatticeSeries {
        half_widths: series.half_widths.clone(),
        coefficients: series.coefficients.mapv(|value: c64| value * scale),
        support: series.support,
    }
}

/// One pass of a multidimensional FFT: the transforms along the last axis of a row-major array, that axis then moved to the front.
/// The result is row-major in the shape `(n_last, n_0, ..., n_{last - 1})`, so that as many passes as axes transform along every
/// axis and restore their order.
///
/// # Arguments
/// * `data` - the array, row-major, its last axis `fft.len()` long
/// * `fft` - the transform of the last axis' length
fn transform_last_axis_and_rotate(mut data: Vec<c64>, fft: &dyn Fft<f64>) -> Vec<c64> {
    let n_last: usize = fft.len();
    let n_rest: usize = data.len() / n_last;
    let n_scratch: usize = fft.get_inplace_scratch_len();
    data.par_chunks_mut(n_last).for_each_init(
        || vec![c64::new(0.0, 0.0); n_scratch],
        |scratch: &mut Vec<c64>, row: &mut [c64]| fft.process_with_scratch(row, scratch),
    );
    // The transpose, N_ROTATION_BLOCK rows of the result at a time: the values they take from a row of `data` are adjacent
    let mut rotated: Vec<c64> = vec![c64::new(0.0, 0.0); data.len()];
    rotated
        .par_chunks_mut(N_ROTATION_BLOCK * n_rest)
        .enumerate()
        .for_each(|(i_block, rows): (usize, &mut [c64])| {
            let i_last_start: usize = i_block * N_ROTATION_BLOCK;
            let n_row: usize = rows.len() / n_rest;
            for i_rest in 0..n_rest {
                for i_row in 0..n_row {
                    rows[i_row * n_rest + i_rest] = data[i_rest * n_last + i_last_start + i_row];
                }
            }
        });
    rotated
}

/// The multidimensional FFT of a row-major array, along every axis.
///
/// # Arguments
/// * `data` - the array, row-major in the shape of the transforms' lengths
/// * `ffts` - one transform per axis, all forward or all inverse (rustfft's inverse is not divided by the length)
fn transform_all_axes(data: Vec<c64>, ffts: &[Arc<dyn Fft<f64>>]) -> Vec<c64> {
    let n_axis: usize = ffts.len();
    let mut transformed: Vec<c64> = data;
    for i_pass in 0..n_axis {
        transformed = transform_last_axis_and_rotate(transformed, ffts[n_axis - 1 - i_pass].as_ref());
    }
    transformed
}

/// Product of two series: a full convolution over the lattice axes and a Cauchy product over the powers, truncated at `n_radial`.
///
/// By FFT over the powers and the real lattice axes, and on a complex lattice exactly along its imaginary
/// axis (the last), slice by slice: an FFT over it would put rounding noise of the largest slice into the high `cosh` terms, which are
/// `exp(|k_j| z_top)` larger at the top than their coefficients. The transforms of the factors' slices are multiplied and summed
/// per slice of the product, which is transformed back once. The exact product vanishes outside the L1 ball
/// `|j|_1 <= support_a + support_b`; there the FFT leaves only rounding noise, which is zeroed (unmasked, it would fill every mode).
///
/// # Returns
/// * the product, in the box of half-widths `L_a + L_b` [dimensionless]
pub fn series_product(lattice: &Lattice, series_a: &LatticeSeries, series_b: &LatticeSeries) -> LatticeSeries {
    let n_radial: usize = lattice.n_radial;
    let n_base: usize = lattice.n_base;
    let is_complex: bool = lattice.is_complex;
    let mut half_widths: Vec<usize> = vec![0; n_base];
    for i_base in 0..n_base {
        half_widths[i_base] = series_a.half_widths[i_base] + series_b.half_widths[i_base];
    }
    let support: usize = series_a.support + series_b.support;
    // The transforms: over the lattice axes but an imaginary one, then over the powers, each long enough for the full convolution
    let n_axis_transformed: usize = if is_complex { n_base - 1 } else { n_base };
    let mut fft_lengths: Vec<usize> = Vec::with_capacity(n_axis_transformed + 1);
    for i_base in 0..n_axis_transformed {
        fft_lengths.push(fft_length(2 * half_widths[i_base] + 1));
    }
    let n_power_fft: usize = fft_length(2 * n_radial - 1);
    fft_lengths.push(n_power_fft);
    let n_fft_axis: usize = fft_lengths.len();
    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let mut ffts_forward: Vec<Arc<dyn Fft<f64>>> = Vec::with_capacity(n_fft_axis);
    let mut ffts_inverse: Vec<Arc<dyn Fft<f64>>> = Vec::with_capacity(n_fft_axis);
    let mut n_fft_point: usize = 1;
    for i_fft_axis in 0..n_fft_axis {
        ffts_forward.push(planner.plan_fft_forward(fft_lengths[i_fft_axis]));
        ffts_inverse.push(planner.plan_fft_inverse(fft_lengths[i_fft_axis]));
        n_fft_point *= fft_lengths[i_fft_axis];
    }
    // The inverse transforms are not divided by the lengths: the product is, once, as it is taken out
    let inverse_scale: f64 = 1.0 / (n_fft_point as f64);

    // The transform of every non-zero slice of a factor along the imaginary axis (a single slice on a lattice without one)
    let transform_slices = |series: &LatticeSeries| -> Vec<Option<Vec<c64>>> {
        let n_slice: usize = if is_complex { 2 * series.half_widths[n_base - 1] + 1 } else { 1 };
        let mut slices: Vec<Option<Vec<c64>>> = vec![None; n_slice];
        let mut index: Vec<usize> = vec![0; n_base];
        let n_row: usize = series.coefficients.nrows();
        for i_row in 0..n_row {
            if !series.is_row_nonzero(i_row) {
                continue;
            }
            // The row's index along every axis of its box, the last axis fastest
            let mut remainder: usize = i_row;
            for i_base in (0..n_base).rev() {
                let width: usize = 2 * series.half_widths[i_base] + 1;
                index[i_base] = remainder % width;
                remainder /= width;
            }
            let i_slice: usize = if is_complex { index[n_base - 1] } else { 0 };
            let mut position: usize = 0;
            for i_base in 0..n_axis_transformed {
                position = position * fft_lengths[i_base] + index[i_base];
            }
            let slice: &mut Vec<c64> = slices[i_slice].get_or_insert_with(|| vec![c64::new(0.0, 0.0); n_fft_point]);
            for i_radial in 0..n_radial {
                slice[position * n_power_fft + i_radial] = series.coefficients[[i_row, i_radial]];
            }
        }
        slices.par_iter_mut().for_each(|slice: &mut Option<Vec<c64>>| {
            if let Some(data) = slice.take() {
                *slice = Some(transform_all_axes(data, &ffts_forward));
            }
        });
        slices
    };
    let slices_a: Vec<Option<Vec<c64>>> = transform_slices(series_a);
    let slices_b: Vec<Option<Vec<c64>>> = transform_slices(series_b);
    let n_slice_a: usize = slices_a.len();
    let n_slice_b: usize = slices_b.len();

    // Per slice of the product, the sum of the products of the factors' slices, transformed back
    let slices_product: Vec<Option<Vec<c64>>> = (0..n_slice_a + n_slice_b - 1)
        .into_par_iter()
        .map(|i_slice_product: usize| {
            let mut total: Option<Vec<c64>> = None;
            for i_slice_a in 0..n_slice_a {
                if i_slice_a > i_slice_product || i_slice_product - i_slice_a >= n_slice_b {
                    continue;
                }
                if let (Some(transform_a), Some(transform_b)) = (&slices_a[i_slice_a], &slices_b[i_slice_product - i_slice_a]) {
                    let accumulated: &mut Vec<c64> = total.get_or_insert_with(|| vec![c64::new(0.0, 0.0); n_fft_point]);
                    for i_point in 0..n_fft_point {
                        accumulated[i_point] += transform_a[i_point] * transform_b[i_point];
                    }
                }
            }
            total.map(|accumulated: Vec<c64>| transform_all_axes(accumulated, &ffts_inverse))
        })
        .collect();

    // The product's box, zero outside the L1 ball. A real lattice's products are real: the imaginary parts are rounding
    let offsets: Vec<Vec<i64>> = box_offsets(&half_widths);
    let n_row: usize = offsets.len();
    let is_real: bool = lattice.is_real;
    let mut coefficients: Array2<c64> = Array2::from_elem((n_row, n_radial), c64::new(0.0, 0.0));
    for i_row in 0..n_row {
        if l1_norm(&offsets[i_row]) > support {
            continue;
        }
        let i_slice: usize = if is_complex {
            (offsets[i_row][n_base - 1] + half_widths[n_base - 1] as i64) as usize
        } else {
            0
        };
        let Some(slice) = &slices_product[i_slice] else {
            continue;
        };
        let mut position: usize = 0;
        for i_base in 0..n_axis_transformed {
            position = position * fft_lengths[i_base] + (offsets[i_row][i_base] + half_widths[i_base] as i64) as usize;
        }
        for i_radial in 0..n_radial {
            let value: c64 = slice[position * n_power_fft + i_radial] * inverse_scale;
            coefficients[[i_row, i_radial]] = if is_real { c64::new(value.re, 0.0) } else { value };
        }
    }
    LatticeSeries {
        half_widths,
        coefficients,
        support,
    }
}

/// Multiply by the linear function `constant + slope (R**2 - R_c**2)`, truncated at `n_radial`: a shift of the powers in `R**2`; in
/// `ln(R**2 / R_c**2)`, a Cauchy product with the series of `exp(w radial)` (see `RadialVariable::linear_function_coefficients`).
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series` - the series [any unit]
/// * `constant` - [any unit]
/// * `slope` - [same unit / metre ** 2]
pub fn series_multiply_linear(lattice: &Lattice, series: &LatticeSeries, constant: f64, slope: f64) -> LatticeSeries {
    let radial: &RadialVariable = &lattice.radial;
    let linear: super::radial_series::LinearFunctionInRadial = radial.linear_function_coefficients(constant, slope);
    let n_row: usize = series.coefficients.nrows();
    let n_radial: usize = series.coefficients.ncols();
    let mut coefficients: Array2<c64> = Array2::from_elem((n_row, n_radial), c64::new(f64::NAN, f64::NAN));
    coefficients
        .as_slice_mut()
        .expect("series_multiply_linear: a new array is contiguous")
        .par_chunks_mut(n_radial)
        .enumerate()
        .for_each(|(i_row, row): (usize, &mut [c64])| {
            let source: ArrayView1<c64> = series.coefficients.row(i_row);
            for i_radial in 0..n_radial {
                let other: c64 = match radial.kind {
                    RadialVariableKind::LogRSquared => {
                        let mut with_exp: c64 = c64::new(0.0, 0.0);
                        for j_radial in 0..=i_radial {
                            with_exp += source[j_radial] * radial.exp_series[i_radial - j_radial];
                        }
                        with_exp
                    }
                    RadialVariableKind::RSquared => {
                        if i_radial >= 1 {
                            source[i_radial - 1]
                        } else {
                            c64::new(0.0, 0.0)
                        }
                    }
                };
                row[i_radial] = source[i_radial] * linear.coefficient_one + other * linear.coefficient_other;
            }
        });
    LatticeSeries {
        half_widths: series.half_widths.clone(),
        coefficients,
        support: series.support,
    }
}

/// `sum_m |d_{j,m}|` at every lattice point `j`, a bound on `|X_j|` on the plasma, where `|radial| <= 1` [dimensionless]
pub fn series_magnitudes(series: &LatticeSeries) -> Array1<f64> {
    let n_row: usize = series.coefficients.nrows();
    let n_radial: usize = series.coefficients.ncols();
    let mut magnitudes: Array1<f64> = Array1::zeros(n_row);
    for i_row in 0..n_row {
        for i_radial in 0..n_radial {
            magnitudes[i_row] += series.coefficients[[i_row, i_radial]].norm();
        }
    }
    magnitudes
}

/// `sum_j |d_{j,m}|` for every power `m` of the radial variable, a bound on that power's part of the series on the plasma, where
/// `|radial| <= 1` (as `series_magnitudes`, at the midplane for the `cosh` terms) [dimensionless]
pub fn series_power_sizes(series: &LatticeSeries) -> Array1<f64> {
    let n_row: usize = series.coefficients.nrows();
    let n_radial: usize = series.coefficients.ncols();
    let mut sizes: Array1<f64> = Array1::zeros(n_radial);
    for i_row in 0..n_row {
        for i_radial in 0..n_radial {
            sizes[i_radial] += series.coefficients[[i_row, i_radial]].norm();
        }
    }
    sizes
}

/// Zero the lattice points whose magnitude is below `tolerance`, then crop to the smallest box that holds the rest.
///
/// # Arguments
/// * `series` - the series [dimensionless]
/// * `tolerance` - absolute threshold on `sum_m |d_{j,m}|` [dimensionless]
pub fn series_trim(series: &LatticeSeries, tolerance: f64) -> LatticeSeries {
    let magnitudes: Array1<f64> = series_magnitudes(series);
    let n_row: usize = magnitudes.len();
    let n_axis: usize = series.half_widths.len();
    let n_radial: usize = series.coefficients.ncols();
    let offsets: Vec<Vec<i64>> = box_offsets(&series.half_widths);
    let mut coefficients: Array2<c64> = series.coefficients.clone();
    let mut is_any_kept: bool = false;
    let mut half_widths_needed: Vec<usize> = vec![0; n_axis];
    for i_row in 0..n_row {
        if magnitudes[i_row] >= tolerance {
            is_any_kept = true;
            for i_axis in 0..n_axis {
                half_widths_needed[i_axis] = half_widths_needed[i_axis].max(offsets[i_row][i_axis].unsigned_abs() as usize);
            }
        } else {
            coefficients.row_mut(i_row).fill(c64::new(0.0, 0.0));
        }
    }
    if !is_any_kept {
        return LatticeSeries {
            half_widths: vec![0; n_axis],
            coefficients: Array2::from_elem((1, n_radial), c64::new(0.0, 0.0)),
            support: 0,
        };
    }
    let zeroed: LatticeSeries = LatticeSeries {
        half_widths: series.half_widths.clone(),
        coefficients,
        support: series.support,
    };
    series_resize(&zeroed, &half_widths_needed)
}

/// `sum_j sum_m |d_{j,m}|`, a bound on the series on the plasma [dimensionless]
pub fn series_size(series: &LatticeSeries) -> f64 {
    let mut size: f64 = 0.0;
    for coefficient in series.coefficients.iter() {
        size += coefficient.norm();
    }
    size
}

/// Number of distinct non-zero wavenumbers, counting `j` and `-j` once [dimensionless]
pub fn series_count(series: &LatticeSeries) -> usize {
    let magnitudes: Array1<f64> = series_magnitudes(series);
    let mut n_nonzero: usize = 0;
    for magnitude in magnitudes.iter() {
        if *magnitude > 0.0 {
            n_nonzero += 1;
        }
    }
    n_nonzero.div_ceil(2)
}

/// One folded lattice row of a series being evaluated: `E_j cos(k_j Z)` or `i O_j sin(k_j Z)`
struct FoldedRow {
    /// [1 / metre]
    wavenumber: c64,
    is_odd: bool,
    /// The first of its columns in the matrix of radial coefficients: value, d1, d2, each as (re, im)
    column: usize,
}

/// The folded rows of a series, and their radial coefficients as the columns of `columns` from `column_start` on.
///
/// Each pair of lattice points `j` and `-j` is folded into `E_j cos(k_j Z) + i O_j sin(k_j Z)`, with the even part
/// `E_j = d_j + d_{-j}` and the odd part `O_j = d_j - d_{-j}` (zero for an up-down symmetric series).
fn fold_series(lattice: &Lattice, series: &LatticeSeries, values_only: bool, columns: &mut Vec<Array1<f64>>) -> Vec<FoldedRow> {
    let n_radial: usize = lattice.n_radial;
    let n_row: usize = series.coefficients.nrows();
    let centre: usize = (n_row - 1) / 2;
    let wavenumbers: Vec<c64> = lattice.box_wavenumbers(&series.half_widths);
    // The even and odd parts of every folded row, i = 0 (j = 0) to centre
    let mut parts_even: Vec<Array1<c64>> = Vec::with_capacity(centre + 1);
    let mut parts_odd: Vec<Array1<c64>> = Vec::with_capacity(centre + 1);
    let mut has_odd: bool = false;
    for i_fold in 0..=centre {
        let row_plus: ArrayView1<c64> = series.coefficients.row(centre + i_fold);
        let row_minus: ArrayView1<c64> = series.coefficients.row(centre - i_fold);
        let mut even: Array1<c64> = Array1::from_elem(n_radial, c64::new(0.0, 0.0));
        let mut odd: Array1<c64> = Array1::from_elem(n_radial, c64::new(0.0, 0.0));
        for i_radial in 0..n_radial {
            if i_fold == 0 {
                even[i_radial] = row_plus[i_radial];
            } else {
                even[i_radial] = row_plus[i_radial] + row_minus[i_radial];
                odd[i_radial] = row_plus[i_radial] - row_minus[i_radial];
                has_odd |= odd[i_radial].re != 0.0 || odd[i_radial].im != 0.0;
            }
        }
        parts_even.push(even);
        parts_odd.push(odd);
    }
    let mut folded: Vec<FoldedRow> = Vec::new();
    for i_fold in 0..=centre {
        let n_part: usize = if has_odd { 2 } else { 1 };
        let mut is_even_nonzero: bool = false;
        let mut is_odd_nonzero: bool = false;
        for i_radial in 0..n_radial {
            is_even_nonzero |= parts_even[i_fold][i_radial].re != 0.0 || parts_even[i_fold][i_radial].im != 0.0;
            is_odd_nonzero |= parts_odd[i_fold][i_radial].re != 0.0 || parts_odd[i_fold][i_radial].im != 0.0;
        }
        // A folded row is active when either part is non-zero, and then both of its parts are evaluated
        if !(is_even_nonzero || is_odd_nonzero) {
            continue;
        }
        for i_part in 0..n_part {
            let part: &Array1<c64> = if i_part == 0 { &parts_even[i_fold] } else { &parts_odd[i_fold] };
            let mut part_d1: Array1<c64> = Array1::from_elem(n_radial, c64::new(0.0, 0.0));
            let mut part_d2: Array1<c64> = Array1::from_elem(n_radial, c64::new(0.0, 0.0));
            for i_radial in 0..n_radial - 1 {
                part_d1[i_radial] = part[i_radial + 1] * ((i_radial + 1) as f64);
            }
            for i_radial in 0..n_radial - 1 {
                part_d2[i_radial] = part_d1[i_radial + 1] * ((i_radial + 1) as f64);
            }
            let column: usize = columns.len();
            columns.push(part.mapv(|value: c64| value.re));
            columns.push(part.mapv(|value: c64| value.im));
            if !values_only {
                columns.push(part_d1.mapv(|value: c64| value.re));
                columns.push(part_d1.mapv(|value: c64| value.im));
                columns.push(part_d2.mapv(|value: c64| value.re));
                columns.push(part_d2.mapv(|value: c64| value.im));
            }
            folded.push(FoldedRow {
                wavenumber: wavenumbers[centre + i_fold],
                is_odd: i_part == 1,
                column,
            });
        }
    }
    folded
}

/// The value and derivatives of several series at the same points, in blocks of points: the radial powers once per block, every
/// series' radial parts by one matrix product, then the vertical factors.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series_list` - the series [dimensionless]
/// * `r` - major radius [metre]
/// * `z` - height above `z_geo`, as many [metre]
/// * `values_only` - compute the values only (the derivatives are left zero)
///
/// # Returns
/// * one `FluxValues` per series
fn evaluate_many(lattice: &Lattice, series_list: &[&LatticeSeries], r: &[f64], z: &[f64], values_only: bool) -> Vec<FluxValues> {
    let n_point: usize = r.len();
    let n_series: usize = series_list.len();
    let n_radial: usize = lattice.n_radial;
    let radial: &RadialVariable = &lattice.radial;
    let mut columns: Vec<Array1<f64>> = Vec::new();
    let mut folded_list: Vec<Vec<FoldedRow>> = Vec::with_capacity(n_series);
    for i_series in 0..n_series {
        folded_list.push(fold_series(lattice, series_list[i_series], values_only, &mut columns));
    }
    let n_column: usize = columns.len();
    let mut results: Vec<FluxValues> = Vec::with_capacity(n_series);
    for _i_series in 0..n_series {
        results.push(FluxValues::zeros(n_point));
    }
    if n_column == 0 || n_point == 0 {
        return results;
    }
    // The radial coefficients, (n_radial, n_column), column-major in faer
    let coefficients_faer: faer::Mat<f64> = faer::Mat::from_fn(n_radial, n_column, |i_radial, i_column| columns[i_column][i_radial]);
    let is_complex: bool = lattice.is_complex;
    let n_block: usize = n_point.div_ceil(N_POINT_BLOCK);
    // Per block, per series: (value, d/dR, d/dZ, d2/dR2, d2/dRdZ, d2/dZ2) at the block's points
    let blocks: Vec<Vec<[Vec<f64>; 6]>> = (0..n_block)
        .into_par_iter()
        .map(|i_block: usize| {
            let i_start: usize = i_block * N_POINT_BLOCK;
            let i_end: usize = (i_start + N_POINT_BLOCK).min(n_point);
            let n_block_point: usize = i_end - i_start;
            let mut powers: faer::Mat<f64> = faer::Mat::zeros(n_block_point, n_radial);
            let mut derivatives: Vec<RadialDerivatives> = Vec::with_capacity(n_block_point);
            for i_block_point in 0..n_block_point {
                let r_point: f64 = r[i_start + i_block_point];
                let radial_point: f64 = radial.of_r(r_point);
                let mut power: f64 = 1.0;
                for i_radial in 0..n_radial {
                    powers[(i_block_point, i_radial)] = power;
                    power *= radial_point;
                }
                derivatives.push(radial.derivatives_of_radial(r_point));
            }
            // A single block shares the matrix product among the threads (which does not change its rounding)
            let parallelism: Par = if n_block < rayon::current_num_threads() { Par::rayon(0) } else { Par::Seq };
            let mut radial_parts: faer::Mat<f64> = faer::Mat::zeros(n_block_point, n_column);
            matmul(
                radial_parts.as_mut(),
                Accum::Replace,
                powers.as_ref(),
                coefficients_faer.as_ref(),
                1.0,
                parallelism,
            );
            let mut block_results: Vec<[Vec<f64>; 6]> = Vec::with_capacity(n_series);
            for i_series in 0..n_series {
                // cos(k_j Z) and sin(k_j Z) of every folded row at the block's points, complex with imaginary spacings: the
                // transcendental functions, in parallel over the rows
                let vertical_factors: Vec<Vec<(c64, c64)>> = folded_list[i_series]
                    .par_iter()
                    .map(|folded: &FoldedRow| {
                        let mut factors: Vec<(c64, c64)> = Vec::with_capacity(n_block_point);
                        for i_block_point in 0..n_block_point {
                            let z_point: f64 = z[i_start + i_block_point];
                            if is_complex {
                                let phase: c64 = folded.wavenumber * z_point;
                                factors.push((phase.cos(), phase.sin()));
                            } else {
                                let phase: f64 = folded.wavenumber.re * z_point;
                                factors.push((c64::new(phase.cos(), 0.0), c64::new(phase.sin(), 0.0)));
                            }
                        }
                        factors
                    })
                    .collect();
                let mut value: Vec<f64> = vec![0.0; n_block_point];
                let mut d_d_r: Vec<f64> = vec![0.0; n_block_point];
                let mut d_d_z: Vec<f64> = vec![0.0; n_block_point];
                let mut d2_d_r2: Vec<f64> = vec![0.0; n_block_point];
                let mut d2_d_r_d_z: Vec<f64> = vec![0.0; n_block_point];
                let mut d2_d_z2: Vec<f64> = vec![0.0; n_block_point];
                let n_folded: usize = folded_list[i_series].len();
                for i_folded in 0..n_folded {
                    let folded: &FoldedRow = &folded_list[i_series][i_folded];
                    let wavenumber: c64 = folded.wavenumber;
                    for i_block_point in 0..n_block_point {
                        let (cos_phase, sin_phase): (c64, c64) = vertical_factors[i_folded][i_block_point];
                        // Even part: E cos(k Z); odd part: i O sin(k Z)
                        let i_unit: c64 = c64::new(0.0, 1.0);
                        let vertical: c64 = if folded.is_odd { i_unit * sin_phase } else { cos_phase };
                        let radial_value: c64 = c64::new(radial_parts[(i_block_point, folded.column)], radial_parts[(i_block_point, folded.column + 1)]);
                        value[i_block_point] += (vertical * radial_value).re;
                        if values_only {
                            continue;
                        }
                        let vertical_d1: c64 = if folded.is_odd {
                            i_unit * wavenumber * cos_phase
                        } else {
                            -wavenumber * sin_phase
                        };
                        let vertical_d2: c64 = -(wavenumber * wavenumber) * vertical;
                        let d1_r: c64 = c64::new(
                            radial_parts[(i_block_point, folded.column + 2)],
                            radial_parts[(i_block_point, folded.column + 3)],
                        );
                        let d2_r: c64 = c64::new(
                            radial_parts[(i_block_point, folded.column + 4)],
                            radial_parts[(i_block_point, folded.column + 5)],
                        );
                        let d_radial_d_r: f64 = derivatives[i_block_point].d_radial_d_r;
                        let d2_radial_d_r2: f64 = derivatives[i_block_point].d2_radial_d_r2;
                        let radial_d2: c64 = d2_r * (d_radial_d_r * d_radial_d_r) + d1_r * d2_radial_d_r2;
                        let radial_d1: c64 = d1_r * d_radial_d_r;
                        d_d_r[i_block_point] += (vertical * radial_d1).re;
                        d_d_z[i_block_point] += (vertical_d1 * radial_value).re;
                        d2_d_r2[i_block_point] += (vertical * radial_d2).re;
                        d2_d_r_d_z[i_block_point] += (vertical_d1 * radial_d1).re;
                        d2_d_z2[i_block_point] += (vertical_d2 * radial_value).re;
                    }
                }
                block_results.push([value, d_d_r, d_d_z, d2_d_r2, d2_d_r_d_z, d2_d_z2]);
            }
            block_results
        })
        .collect();
    for i_block in 0..n_block {
        let i_start: usize = i_block * N_POINT_BLOCK;
        for i_series in 0..n_series {
            let block: &[Vec<f64>; 6] = &blocks[i_block][i_series];
            let n_block_point: usize = block[0].len();
            for i_block_point in 0..n_block_point {
                results[i_series].one_minus_psi_n[i_start + i_block_point] = block[0][i_block_point];
                results[i_series].d_d_r[i_start + i_block_point] = block[1][i_block_point];
                results[i_series].d_d_z[i_start + i_block_point] = block[2][i_block_point];
                results[i_series].d2_d_r2[i_start + i_block_point] = block[3][i_block_point];
                results[i_series].d2_d_r_d_z[i_start + i_block_point] = block[4][i_block_point];
                results[i_series].d2_d_z2[i_start + i_block_point] = block[5][i_block_point];
            }
        }
    }
    results
}

/// Value and first and second derivatives of a series at points.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series` - the series [dimensionless]
/// * `r` - major radius [metre]
/// * `z` - height above `z_geo`, as many [metre]
pub fn series_evaluate(lattice: &Lattice, series: &LatticeSeries, r: &[f64], z: &[f64]) -> FluxValues {
    evaluate_many(lattice, &[series], r, z, false).pop().expect("series_evaluate: one series")
}

/// The value and derivatives of several series at the same points (cheaper than one `series_evaluate` per series)
pub fn series_evaluate_many(lattice: &Lattice, series_list: &[&LatticeSeries], r: &[f64], z: &[f64]) -> Vec<FluxValues> {
    evaluate_many(lattice, series_list, r, z, false)
}

/// The value only of a series at points [dimensionless]
pub fn series_values(lattice: &Lattice, series: &LatticeSeries, r: &[f64], z: &[f64]) -> Array1<f64> {
    evaluate_many(lattice, &[series], r, z, true)
        .pop()
        .expect("series_values: one series")
        .one_minus_psi_n
}

/// The values only of several series at the same points [dimensionless]
pub fn series_values_many(lattice: &Lattice, series_list: &[&LatticeSeries], r: &[f64], z: &[f64]) -> Vec<Array1<f64>> {
    let flux_list: Vec<FluxValues> = evaluate_many(lattice, series_list, r, z, true);
    let mut values_list: Vec<Array1<f64>> = Vec::with_capacity(flux_list.len());
    for flux in flux_list {
        values_list.push(flux.one_minus_psi_n);
    }
    values_list
}

/// Taylor coefficients of a series about `(radial_0, z_0)`, in the scaled radial variable and `Z`:
/// `sum_{a, b} T[a, b] (radial - radial_0)**a (Z - z_0)**b`, from `sum_m d_{j,m} (radial_0 + shift)**m = sum_a S_j[a] shift**a` and
/// `exp(i k_j (z_0 + shift)) = exp(i k_j z_0) sum_b (i k_j shift)**b / b!`.
///
/// # Arguments
/// * `lattice` - the lattice
/// * `series` - the series [dimensionless]
/// * `radial_0` - expansion point, in the scaled radial variable [dimensionless]
/// * `z_0` - expansion point [metre]
/// * `n_degree` - number of coefficients in each variable [dimensionless]
///
/// # Returns
/// * `T`, shape `(n_degree, n_degree)` [1 / metre ** b]
pub fn series_taylor(lattice: &Lattice, series: &LatticeSeries, radial_0: f64, z_0: f64, n_degree: usize) -> Array2<f64> {
    let n_radial: usize = series.coefficients.ncols();
    let n_row: usize = series.coefficients.nrows();
    let wavenumbers: Vec<c64> = lattice.box_wavenumbers(&series.half_widths);
    let mut active: Vec<usize> = Vec::new();
    for i_row in 0..n_row {
        if series.is_row_nonzero(i_row) {
            active.push(i_row);
        }
    }
    let mut taylor: Array2<f64> = Array2::zeros((n_degree, n_degree));
    let n_active: usize = active.len();
    if n_active == 0 {
        return taylor;
    }
    // B[m, a] = C(m, a) radial_0**(m - a)
    let mut shift: Array2<f64> = Array2::zeros((n_radial, n_degree));
    for i_degree in 0..n_degree {
        for i_radial in i_degree..n_radial {
            shift[[i_radial, i_degree]] = binomial(i_radial, i_degree) * radial_0.powf((i_radial - i_degree) as f64);
        }
    }
    let i_unit: c64 = c64::new(0.0, 1.0);
    for i_active in 0..n_active {
        let i_row: usize = active[i_active];
        let mut radial_taylor: Vec<c64> = vec![c64::new(0.0, 0.0); n_degree];
        for i_degree in 0..n_degree {
            for i_radial in i_degree..n_radial {
                radial_taylor[i_degree] += series.coefficients[[i_row, i_radial]] * shift[[i_radial, i_degree]];
            }
        }
        let mut vertical_taylor: Vec<c64> = vec![c64::new(0.0, 0.0); n_degree];
        vertical_taylor[0] = (i_unit * wavenumbers[i_row] * z_0).exp();
        for i_degree in 1..n_degree {
            vertical_taylor[i_degree] = vertical_taylor[i_degree - 1] * i_unit * wavenumbers[i_row] / (i_degree as f64);
        }
        for i_degree_a in 0..n_degree {
            for i_degree_b in 0..n_degree {
                taylor[[i_degree_a, i_degree_b]] += (radial_taylor[i_degree_a] * vertical_taylor[i_degree_b]).re;
            }
        }
    }
    taylor
}

/// Power-series coefficients in the scaled radial variable of the series on the midplane, `Z = 0` (real) [dimensionless]
pub fn midplane_coefficients(series: &LatticeSeries) -> Array1<f64> {
    let n_radial: usize = series.coefficients.ncols();
    let mut midplane: Array1<f64> = Array1::zeros(n_radial);
    for i_radial in 0..n_radial {
        let mut total: c64 = c64::new(0.0, 0.0);
        for coefficient in series.coefficients.slice(s![.., i_radial]).iter() {
            total += *coefficient;
        }
        midplane[i_radial] = total.re;
    }
    midplane
}

/// The value of a power series at a point, by Horner's rule
pub fn polynomial_value(coefficients: &Array1<f64>, point: f64) -> f64 {
    let n_coefficient: usize = coefficients.len();
    let mut value: f64 = 0.0;
    for i_coefficient in (0..n_coefficient).rev() {
        value = value * point + coefficients[i_coefficient];
    }
    value
}

/// The series at `(R_c, 0)` [dimensionless]
pub fn value_at_centre(lattice: &Lattice, series: &LatticeSeries) -> f64 {
    polynomial_value(&midplane_coefficients(series), lattice.radial.of_r_squared(lattice.radial.r_c_squared))
}
