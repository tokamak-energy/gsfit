use super::filament_geometry::FilamentGeometry;
use crate::greens::greens_psi;
use ndarray::{Array1, Array2, s};

/// Mutual inductance between filaments of finite size
///
/// # Arguments
/// * `r` - R coordinate of the first set of filaments, shape = [n_filaments], metre
/// * `z` - Z coordinate of the first set of filaments, shape = [n_filaments], metre
/// * `d_r` - Radial width of the first set of filaments (note, area = d_r * d_z), shape = [n_filaments], metre
/// * `d_z` - Vertical height of the first set of filaments, shape = [n_filaments], metre
/// * `angle1` - Angle of the first side of the first set of filaments, shape = [n_filaments], radian
/// * `angle2` - Angle of the second side of the first set of filaments, shape = [n_filaments], radian
/// * `r_prime` - R coordinate of the second set of filaments, shape = [n_filaments_prime], metre
/// * `z_prime` - Z coordinate of the second set of filaments, shape = [n_filaments_prime], metre
/// * `d_r_prime` - Radial width of the second set of filaments (note, area = d_r_prime * d_z_prime), shape = [n_filaments_prime], metre
/// * `d_z_prime` - Vertical height of the second set of filaments, shape = [n_filaments_prime], metre
/// * `angle1_prime` - Angle of the first side of the second set of filaments, shape = [n_filaments_prime], radian
/// * `angle2_prime` - Angle of the second side of the second set of filaments, shape = [n_filaments_prime], radian
///
/// # Returns
/// * `g_psi` - Mutual inductance between the two sets of filaments, shape = [n_filaments, n_filaments_prime], henry
///
pub fn mutual_inductance_finite_size_to_finite_size(
    r: &Array1<f64>,
    z: &Array1<f64>,
    d_r: &Array1<f64>,
    d_z: &Array1<f64>,
    angle_1: &Array1<f64>,
    angle_2: &Array1<f64>,
    r_prime: &Array1<f64>,
    z_prime: &Array1<f64>,
    d_r_prime: &Array1<f64>,
    d_z_prime: &Array1<f64>,
    angle_1_prime: &Array1<f64>,
    angle_2_prime: &Array1<f64>,
) -> Array2<f64> {
    let n_filaments: usize = r.len();
    let n_filaments_prime: usize = r_prime.len();

    let mut g_psi: Array2<f64> = Array2::from_elem((n_filaments, n_filaments_prime), f64::NAN);

    // Number of sub-filaments to divide each filament into
    // TODO: can this can be adjusted automatically to achieve a desired accuracy ??
    let n_sub_filaments: usize = 10;

    for i_filament in 0..n_filaments {
        // TODO: Make this a parallel loop
        for i_filament_prime in 0..n_filaments_prime {
            // Discretise the first filament
            let (r_sub_filament, z_sub_filament, _d_r_sub_filament, _d_z_sub_filament, _area): (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, f64) =
                discretise_parallelogram(
                    r[i_filament],
                    z[i_filament],
                    d_r[i_filament],
                    d_z[i_filament],
                    angle_1[i_filament],
                    angle_2[i_filament],
                    n_sub_filaments,
                );

            // Discretise the second filament
            let (r_sub_filament_prime, z_sub_filament_prime, _d_r_sub_filament_prime, _d_z_sub_filament_prime, _area_prime): (
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
                angle_1_prime[i_filament_prime],
                angle_2_prime[i_filament_prime],
                n_sub_filaments,
            );

            // Calculate the greens function for the sub-filaments to sub-filaments
            let g_sub_filaments: Array2<f64> = greens_psi(
                r_sub_filament.clone(),
                z_sub_filament.clone(),
                r_sub_filament_prime.clone(),
                z_sub_filament_prime.clone(),
                r_sub_filament.clone() * 0.0 + d_r[i_filament] / (n_sub_filaments as f64), // TODO: Check this!!
                z_sub_filament.clone() * 0.0 + d_z[i_filament] / (n_sub_filaments as f64), // TODO: Check this!!
            ); // shape = [n_sub_filaments * n_sub_filaments, n_sub_filaments * n_sub_filaments]

            g_psi[(i_filament, i_filament_prime)] = g_sub_filaments.sum() / ((n_sub_filaments as f64).powi(4));
        }
    }

    return g_psi;
}

fn discretise_parallelogram(
    r: f64,
    z: f64,
    d_r: f64,
    d_z: f64,
    angle_1: f64,
    angle_2: f64,
    n_sub_filaments: usize,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, f64) {
    let filament_geometry: FilamentGeometry = FilamentGeometry::new(angle_1, angle_2, d_r, d_z, r, z);

    // Equation for the parallelelogram:
    // (R, Z) = (R1, Z1) + (R2-R1, Z2-Z1) * u + (R4-R1, Z4-Z1) * v
    let u: Array1<f64> = Array1::linspace(0.0, 1.0, n_sub_filaments + 2).slice(s![1..n_sub_filaments + 1]).to_owned();
    let v: Array1<f64> = Array1::linspace(0.0, 1.0, n_sub_filaments + 2).slice(s![1..n_sub_filaments + 1]).to_owned();

    // Create a meshgrid
    let mut mesh_r: Array2<f64> = Array2::from_elem((n_sub_filaments, n_sub_filaments), f64::NAN);
    let mut mesh_z: Array2<f64> = Array2::from_elem((n_sub_filaments, n_sub_filaments), f64::NAN);
    for i_u in 0..n_sub_filaments {
        for i_v in 0..n_sub_filaments {
            let r_sub_filament: f64 =
                filament_geometry.r_1 + (filament_geometry.r_2 - filament_geometry.r_1) * u[i_u] + (filament_geometry.r_4 - filament_geometry.r_1) * v[i_v];
            let z_sub_filament: f64 =
                filament_geometry.z_1 + (filament_geometry.z_2 - filament_geometry.z_1) * u[i_u] + (filament_geometry.z_4 - filament_geometry.z_1) * v[i_v];
            mesh_r[(i_u, i_v)] = r_sub_filament;
            mesh_z[(i_u, i_v)] = z_sub_filament;
        }
    }

    let r_sub_filament_flat: Array1<f64> = mesh_r.flatten().to_owned();
    let z_sub_filament_flat: Array1<f64> = mesh_z.flatten().to_owned();

    let d_r_sub_filament_flat: Array1<f64> = Array1::from_elem(n_sub_filaments * n_sub_filaments, d_r);
    let d_z_sub_filament_flat: Array1<f64> = Array1::from_elem(n_sub_filaments * n_sub_filaments, d_z);

    let area: f64 = filament_geometry.calculate_area();

    return (r_sub_filament_flat, z_sub_filament_flat, d_r_sub_filament_flat, d_z_sub_filament_flat, area);
}
