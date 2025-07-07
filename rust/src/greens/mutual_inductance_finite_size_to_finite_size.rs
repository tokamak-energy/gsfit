use crate::greens::greens;
use approx::abs_diff_eq;
use ndarray::{Array1, Array2, s};

pub fn mutual_inductance_finite_size_to_finite_size(
    r: &Array1<f64>,
    z: &Array1<f64>,
    d_r: &Array1<f64>,
    d_z: &Array1<f64>,
    angle1: &Array1<f64>,
    angle2: &Array1<f64>,
    r_prime: &Array1<f64>,
    z_prime: &Array1<f64>,
    d_r_prime: &Array1<f64>,
    d_z_prime: &Array1<f64>,
    angle1_prime: &Array1<f64>,
    angle2_prime: &Array1<f64>,
) -> Array2<f64> {
    let n_filaments: usize = r.len();
    let n_filaments_prime: usize = r_prime.len();

    let mut g: Array2<f64> = Array2::from_elem((n_filaments, n_filaments_prime), f64::NAN);

    // Number of sub-filaments to divide each filament into
    // TODO: can this can be adjusted automatically to achieve a desired accuracy ??
    let n_sub_filaments: usize = 10;

    for i_filament in 0..n_filaments {
        // TODO: Make this a parallel loop
        for i_filament_prime in 0..n_filaments_prime {
            // Discretise the first filament
            let (r_sub_filament, z_sub_filament, d_r_sub_filament, d_z_sub_filament, area): (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, f64) =
                discretise_parallelogram(
                    r[i_filament],
                    z[i_filament],
                    d_r[i_filament],
                    d_z[i_filament],
                    angle1[i_filament],
                    angle2[i_filament],
                    n_sub_filaments,
                );

            // Discretise the second filament
            let (r_sub_filament_prime, z_sub_filament_prime, d_r_sub_filament_prime, d_z_sub_filament_prime, area_prime): (
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                f64,
            ) = discretise_parallelogram(
                r_prime[i_filament_prime],
                z_prime[i_filament_prime],
                d_r_prime[i_filament_prime],
                d_z_prime[i_filament_prime],
                angle1_prime[i_filament_prime],
                angle2_prime[i_filament_prime],
                n_sub_filaments,
            );

            // Calculate the greens function for the sub-filaments to sub-filaments
            let g_sub_filaments: Array2<f64> = greens(
                r_sub_filament.clone(),
                z_sub_filament.clone(),
                r_sub_filament_prime.clone(),
                z_sub_filament_prime.clone(),
                r_sub_filament.clone() * 0.0 + d_r[i_filament] / (n_sub_filaments as f64), // THIS LOOKS WRONG!!! do I need it??
                z_sub_filament.clone() * 0.0 + d_z[i_filament] / (n_sub_filaments as f64), // THIS LOOKS WRONG!!! do I need it??
            ); // shape = [n_sub_filaments * n_sub_filaments, n_sub_filaments * n_sub_filaments]

            g[[i_filament, i_filament_prime]] = g_sub_filaments.sum() / ((n_sub_filaments as f64).powi(4));
        }
    }

    return g;
}

fn discretise_parallelogram(
    r: f64,
    z: f64,
    d_r: f64,
    d_z: f64,
    angle1: f64,
    angle2: f64,
    n_sub_filaments: usize,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, f64) {
    // First coordinate in the parallelogram
    let r1: f64;
    let z1: f64;
    if abs_diff_eq!(angle1, 0.0) {
        r1 = r - d_r / 2.0;
    } else {
        r1 = r - d_r / 2.0 - d_z / (2.0 * angle1.tan());
    }
    if abs_diff_eq!(angle2, 0.0) {
        z1 = z - d_z / 2.0;
    } else {
        z1 = z - d_z / 2.0 - d_r * angle2.tan() / 2.0;
    }

    // Second coordinate in the parallelogram
    let r2: f64;
    let z2: f64;
    if abs_diff_eq!(angle1, 0.0) {
        r2 = r + d_r / 2.0;
    } else {
        r2 = r + d_r / 2.0 - d_z / (2.0 * angle1.tan());
    }
    if abs_diff_eq!(angle2, 0.0) {
        z2 = z - d_z / 2.0;
    } else {
        z2 = z - d_z / 2.0 + d_r * angle2.tan() / 2.0;
    }

    // Third coordinate in the parallelogram
    let r3: f64;
    let z3: f64;
    if abs_diff_eq!(angle1, 0.0) {
        r3 = r + d_r / 2.0;
    } else {
        r3 = r + d_r / 2.0 + d_z / (2.0 * angle1.tan());
    }
    if abs_diff_eq!(angle2, 0.0) {
        z3 = z - d_z / 2.0;
    } else {
        z3 = z - d_z / 2.0 + d_r * angle2.tan() / 2.0;
    }

    // Fourth coordinate in the parallelogram
    let r4: f64;
    let z4: f64;
    if abs_diff_eq!(angle1, 0.0) {
        r4 = r - d_r / 2.0;
    } else {
        r4 = r - d_r / 2.0 + d_z / (2.0 * angle1.tan());
    }
    if abs_diff_eq!(angle2, 0.0) {
        z4 = z + d_z / 2.0;
    } else {
        z4 = z + d_z / 2.0 - d_r * angle2.tan() / 2.0;
    }

    // Equation for the parallelelogram:
    // (R, Z) = (R1, Z1) + (R2-R1, Z2-Z1) * u + (R4-R1, Z4-Z1) * v
    let u: Array1<f64> = Array1::linspace(0.0, 1.0, n_sub_filaments + 2).slice(s![1..n_sub_filaments + 1]).to_owned();
    let v: Array1<f64> = Array1::linspace(0.0, 1.0, n_sub_filaments + 2).slice(s![1..n_sub_filaments + 1]).to_owned();

    // Create a meshgrid
    let mut mesh_r: Array2<f64> = Array2::from_elem((n_sub_filaments, n_sub_filaments), f64::NAN);
    let mut mesh_z: Array2<f64> = Array2::from_elem((n_sub_filaments, n_sub_filaments), f64::NAN);
    for i_u in 0..n_sub_filaments {
        for i_v in 0..n_sub_filaments {
            let r_sub_filament: f64 = r1 + (r2 - r1) * u[i_u] + (r4 - r1) * v[i_v];
            let z_sub_filament: f64 = z1 + (z2 - z1) * u[i_u] + (z4 - z1) * v[i_v];
            mesh_r[[i_u, i_v]] = r_sub_filament;
            mesh_z[[i_u, i_v]] = z_sub_filament;
        }
    }

    let r_sub_filament_flat: Array1<f64> = mesh_r.flatten().to_owned();
    let z_sub_filament_flat: Array1<f64> = mesh_z.flatten().to_owned();

    let d_r_sub_filament_flat: Array1<f64> = Array1::from_elem(n_sub_filaments * n_sub_filaments, d_r);
    let d_z_sub_filament_flat: Array1<f64> = Array1::from_elem(n_sub_filaments * n_sub_filaments, d_z);

    // let area_sub_filament_flat: Array1<f64> = &d_r_sub_filament_flat * &d_z_sub_filament_flat;

    let area: f64 = (r1 * z2 - r2 * z1 + r2 * z3 - r3 * z2 + r3 * z4 - r4 * z3 + r4 * z1 - r1 * z4) / 2.0;

    return (r_sub_filament_flat, z_sub_filament_flat, d_r_sub_filament_flat, d_z_sub_filament_flat, area);
}
