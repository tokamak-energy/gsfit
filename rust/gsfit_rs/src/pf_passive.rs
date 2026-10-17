use crate::greens::FilamentGeometry;
use crate::greens::mutual_inductance_finite_size_to_finite_size;
use faer::Side;
use imas_rs::dd_support::parameters_input::ParametersInput;
use imas_rs::ids::pf_passive::{
    Code, IdentifierStatic, ObliqueStatic, Outline2dGeometryStatic, PfCoilsElements, PfPassive as PfPassiveIds, PfPassiveLoops, PfPassiveLoopsDof,
};
use imas_rs::python::{PyParametersInput, PyPath, PyPfPassive, read_path};
use ndarray::{Array1, Array2, s};
use numpy::PyArrayMethods;
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

/// The machine's passive conductors, e.g. the vacuum vessel, the passive plates and the PF coil cases, stored as an IMAS
/// `pf_passive` IDS.
///
/// For each passive conductor, `loop(i1)`, these are filled:
/// * `name`
/// * `resistivity`
/// * `element(i2)/geometry/oblique/r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta` - one element per filament,
///   with `element(i2)/geometry/geometry_type` = 3, oblique (a parallelogram)
/// * `element(i2)/area` - the area of the parallelogram
/// * `element(i2)/turns_with_sign` = 1
/// * `current_distribution_type` - a custom key, saying how the current is distributed over the elements
/// * `dof` - a custom key, the degrees of freedom of the current, which the reconstruction fits
/// * `regularisations` and `regularisations_weight` - custom keys, the regularisation of the degrees of freedom
///
/// # Geometry
///
/// The data dictionary describes a parallelogram from one of its corners, the reference point `oblique/r` and `z`. One side,
/// `length_alpha` long, leaves it at angle `alpha` counter-clockwise from the R axis; the other, `length_beta` long, leaves it at
/// angle `beta` counter-clockwise from the Z axis. The database readers convert into this.
///
/// GSFit's Greens functions describe a filament instead as EFIT does: by its centre, its radial width `d_r`, its vertical height
/// `d_z`, and the angles `angle_1` and `angle_2` of its sides from the R axis, where an angle of zero means the side is not sheared.
/// `passive_filaments` converts back into this.
///
/// # Degrees of freedom
///
/// The passive currents are not measured, so the reconstruction fits them, as the amplitudes of a few current distributions for
/// each loop. `current_distribution_type` says which:
/// * `constant_current_density` (`index` 1): one degree of freedom, with the same current density in every element
/// * `eig` (`index` 2): the loop's `n_dof` slowest decaying eigenmodes, from its mutual inductances and resistances
///
/// The degrees of freedom of every loop, in the order of `loop` and then of `dof`, are the passive columns of the reconstruction's
/// fit: `Plasma.greens_with_pf_passive` fills `greens/grid/pf_passive` in this order, and `time_slice(itime)/passive_dof_values` follows it.
///
/// `Plasma.copy_from_pf_passive_ids` copies each loop's degrees of freedom and regularisations onto every time-slice of the
/// equilibrium IDS, as `constraints/pf_passive_current(i1)`, which is where the solver reads the regularisations from.
///
/// Read the data back through `pf_passive_ids` and a path from `gsfit_rs.imas.pf_passive_paths`; there are no bespoke accessors.
///
/// # Settings
///
/// The passive conductors are constructed with the settings they are built from, e.g. which run of which database they are read
/// from, as an IMAS `parameters_input`. They are kept in `parameters_input`, and their XML is also written to
/// `pf_passive/code/parameters`, where IMAS records the parameters of the code which made an IDS.
#[pyclass(module = "gsfit_rs")]
pub struct PfPassive {
    pub pf_passive_ids: PfPassiveIds,
    /// The settings the passive conductors are built from: their XML, the XML schema they follow, and the defaults
    pub parameters_input: ParametersInput,
}

/// Python accessible methods
#[pymethods]
impl PfPassive {
    /// Construct empty passive conductors, ready for `add_loop` to be called.
    ///
    /// # Arguments
    /// * `parameters_input` - the settings the passive conductors are built from, as a
    ///   `gsfit_rs.dd_support.ParametersInput`; their XML is written to `pf_passive/code/parameters`
    #[new]
    pub fn new(parameters_input: &PyParametersInput) -> Self {
        let pf_passive_ids: PfPassiveIds = PfPassiveIds {
            code: Code {
                parameters: parameters_input.inner.parameters_value.clone(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            pf_passive_ids,
            parameters_input: parameters_input.inner.clone(),
        }
    }

    /// The settings the passive conductors were built from, as a `gsfit_rs.dd_support.ParametersInput`.
    ///
    /// This is a copy: changing it does not change the passive conductors.
    #[getter]
    fn parameters_input(&self) -> PyParametersInput {
        PyParametersInput::new(self.parameters_input.clone())
    }

    /// Append a passive conductor to `pf_passive/loop`.
    ///
    /// # Arguments
    /// * `name` - short identifier for the passive conductor, unique within the passive conductors, e.g. `"IVC"`
    /// * `r` - major radius of each filament's reference point, the corner which `alpha` and `beta` are measured from (1d array), [metre]
    /// * `z` - height of each filament's reference point (1d array), [metre]
    /// * `length_alpha` - length of each filament's side inclined at `alpha` (1d array), [metre]
    /// * `length_beta` - length of each filament's side inclined at `beta` (1d array), [metre]
    /// * `alpha` - inclination of each filament's first side, counter-clockwise from the R axis (1d array), [radian]
    /// * `beta` - inclination of each filament's second side, counter-clockwise from the Z axis (1d array), [radian]
    /// * `resistivity` - resistivity of the conductor, the same for every filament, [ohm * metre]
    /// * `current_distribution_type` - `"constant_current_density"` or `"eig"`; see the `PfPassive` documentation
    /// * `n_dof` - number of degrees of freedom: 1 for `"constant_current_density"`, the number of eigenmodes for `"eig"`
    /// * `regularisations` - the regularisations of the degrees of freedom, shape = [n_regularisations, n_dof]; no rows for none
    /// * `regularisations_weight` - the weight of each regularisation, shape = [n_regularisations], [dimensionless]
    ///
    /// Each filament becomes one `element(i2)`: a parallelogram of one turn.
    #[allow(clippy::too_many_arguments)]
    pub fn add_loop(
        &mut self,
        name: &str,
        r: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
        length_alpha: PyReadonlyArray1<f64>,
        length_beta: PyReadonlyArray1<f64>,
        alpha: PyReadonlyArray1<f64>,
        beta: PyReadonlyArray1<f64>,
        resistivity: f64,
        current_distribution_type: &str,
        n_dof: usize,
        regularisations: PyReadonlyArray2<f64>,
        regularisations_weight: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        // Change Python types into Rust types
        let element_r: Array1<f64> = r.to_owned_array();
        let element_z: Array1<f64> = z.to_owned_array();
        let element_length_alpha: Array1<f64> = length_alpha.to_owned_array();
        let element_length_beta: Array1<f64> = length_beta.to_owned_array();
        let element_alpha: Array1<f64> = alpha.to_owned_array();
        let element_beta: Array1<f64> = beta.to_owned_array();
        let regularisations_ndarray: Array2<f64> = regularisations.to_owned_array();
        let regularisations_weight_ndarray: Array1<f64> = regularisations_weight.to_owned_array();

        // Run the Rust method
        self.add_loop_rs(
            name,
            element_r,
            element_z,
            element_length_alpha,
            element_length_beta,
            element_alpha,
            element_beta,
            resistivity,
            current_distribution_type,
            n_dof,
            regularisations_ndarray,
            regularisations_weight_ndarray,
        )
        .map_err(PyValueError::new_err)
    }

    /// Read the data at `path`, a path from `gsfit_rs.imas.pf_passive_paths`, straight out of the
    /// pf_passive IDS.
    ///
    /// This is how the data is read: there are no bespoke accessors, so every quantity is reached
    /// by its data dictionary path. A path holds no data, so the IDS is only borrowed for the
    /// read, never copied.
    fn get<'py>(&self, py: Python<'py>, path: &PyPath) -> PyResult<Bound<'py, PyAny>> {
        read_path(py, &self.pf_passive_ids, "pf_passive", path)
    }

    /// A copy of the whole pf_passive IDS, for reading with `gsfit_rs.imas.pf_passive_paths`.
    ///
    /// Read the data with `get` instead. This copies the IDS on every access. It is for when a
    /// detached snapshot is wanted: changes made on the Rust side afterwards are not seen by it.
    #[getter]
    fn pf_passive_ids(&self) -> PyPfPassive {
        PyPfPassive::new(self.pf_passive_ids.clone())
    }

    /// Print to screen, to be used within Python
    fn __repr__(&self) -> String {
        let version: &str = env!("CARGO_PKG_VERSION");

        let mut string_output = String::from("╔═════════════════════════════════════════════════════════════════════════════╗\n");
        string_output += &format!("║  {:<74} ║\n", "<gsfit_rs.PfPassive>");
        string_output += &format!("║  {:<74} ║\n", version);

        let loops: &Vec<PfPassiveLoops> = &self.pf_passive_ids.r#loop;
        let n_loop: usize = loops.len();
        if n_loop == 0 {
            string_output += &format!("║  {:<74} ║\n", " `pf_passive/loop` is empty");
        }
        for i_loop in 0..n_loop {
            let n_element: usize = loops[i_loop].element.len();
            let n_dof: usize = loops[i_loop].dof.len();
            string_output += &format!(
                "║  {:<74} ║\n",
                format!(
                    " loop({i_loop}) = {}; elements = {n_element}; {} with {n_dof} dof",
                    loops[i_loop].name, loops[i_loop].current_distribution_type.name
                )
            );
        }

        string_output.push_str("╚═════════════════════════════════════════════════════════════════════════════╝");

        string_output
    }
}

/// Rust only methods
impl PfPassive {
    /// Append a passive conductor to `pf_passive/loop`. This does the work of the Python method `add_loop`, so that Rust code can
    /// add passive conductors too; see it for the arguments.
    ///
    /// # Errors
    /// Why the passive conductor cannot be added: an empty or repeated name, arrays of different lengths, no filaments, a filament
    /// without a positive radius and size, an unknown `current_distribution_type`, or regularisations of the wrong shape.
    #[allow(clippy::too_many_arguments)]
    pub fn add_loop_rs(
        &mut self,
        name: &str,
        element_r: Array1<f64>,
        element_z: Array1<f64>,
        element_length_alpha: Array1<f64>,
        element_length_beta: Array1<f64>,
        element_alpha: Array1<f64>,
        element_beta: Array1<f64>,
        resistivity: f64,
        current_distribution_type: &str,
        n_dof: usize,
        regularisations: Array2<f64>,
        regularisations_weight: Array1<f64>,
    ) -> Result<(), String> {
        if name.is_empty() {
            return Err("passive conductor: `name` is empty; a passive conductor needs a short identifier".to_string());
        }

        // The data dictionary asks for the name to be "unique for a given device", and the sensors find their Greens with each
        // passive conductor by it
        let n_loop: usize = self.pf_passive_ids.r#loop.len();
        for i_loop in 0..n_loop {
            if self.pf_passive_ids.r#loop[i_loop].name == name {
                return Err(format!(
                    "passive conductor `{name}`: a passive conductor with this name already exists, at `pf_passive/loop({i_loop})`"
                ));
            }
        }

        let n_element: usize = element_r.len();
        if element_z.len() != n_element
            || element_length_alpha.len() != n_element
            || element_length_beta.len() != n_element
            || element_alpha.len() != n_element
            || element_beta.len() != n_element
        {
            return Err(format!(
                "passive conductor `{name}`: `r`, `z`, `length_alpha`, `length_beta`, `alpha` and `beta` have {}, {}, {}, {}, {} and {} point(s); each filament needs one of each",
                n_element,
                element_z.len(),
                element_length_alpha.len(),
                element_length_beta.len(),
                element_alpha.len(),
                element_beta.len()
            ));
        }

        if n_element == 0 {
            return Err(format!("passive conductor `{name}`: there are no filaments"));
        }

        // No regularisations is a matrix with no rows, which may arrive with no columns either
        let n_regularisation: usize = regularisations.nrows();
        let regularisations: Array2<f64> = if n_regularisation == 0 { Array2::zeros((0, n_dof)) } else { regularisations };
        if regularisations.ncols() != n_dof || regularisations_weight.len() != n_regularisation {
            return Err(format!(
                "passive conductor `{name}`: `regularisations` has shape {:?} and `regularisations_weight` has {} value(s); they need shape \
                 (n_regularisations, n_dof) = (n_regularisations, {n_dof}) and n_regularisations values",
                regularisations.shape(),
                regularisations_weight.len()
            ));
        }

        // One element per filament: a parallelogram of one turn
        let mut elements: Vec<PfCoilsElements> = Vec::with_capacity(n_element);
        for i_element in 0..n_element {
            // The Greens functions need every filament at a positive radius, and of a positive size
            let is_usable_filament: bool = element_r[i_element] > 0.0
                && element_z[i_element].is_finite()
                && element_length_alpha[i_element] > 0.0
                && element_length_beta[i_element] > 0.0
                && element_alpha[i_element].is_finite()
                && element_beta[i_element].is_finite();
            if !is_usable_filament {
                return Err(format!(
                    "passive conductor `{name}`: filament {i_element} has r = {}, z = {} metre, length_alpha = {}, length_beta = {} metre, alpha = {} \
                     and beta = {} radian; r and the lengths must be positive, and z and the angles finite",
                    element_r[i_element],
                    element_z[i_element],
                    element_length_alpha[i_element],
                    element_length_beta[i_element],
                    element_alpha[i_element],
                    element_beta[i_element]
                ));
            }

            elements.push(PfCoilsElements {
                turns_with_sign: 1.0,
                geometry: Outline2dGeometryStatic {
                    geometry_type: 3, // oblique
                    oblique: ObliqueStatic {
                        r: element_r[i_element],
                        z: element_z[i_element],
                        length_alpha: element_length_alpha[i_element],
                        length_beta: element_length_beta[i_element],
                        alpha: element_alpha[i_element],
                        beta: element_beta[i_element],
                    },
                    ..Default::default()
                },
                ..Default::default()
            });
        }

        // The filaments as GSFit's Greens functions describe them
        let filaments: PassiveFilaments = passive_filaments(&elements);

        // The area of each parallelogram
        let mut area: Array1<f64> = Array1::from_elem(n_element, f64::NAN);
        for i_element in 0..n_element {
            let filament_geometry: FilamentGeometry = FilamentGeometry::new(
                filaments.angle_1[i_element],
                filaments.angle_2[i_element],
                filaments.d_r[i_element],
                filaments.d_z[i_element],
                filaments.r[i_element],
                filaments.z[i_element],
            );
            area[i_element] = filament_geometry.calculate_area();
            elements[i_element].area = area[i_element];
        }

        // The degrees of freedom
        let current_distribution_type_identifier: IdentifierStatic;
        let mut dofs: Vec<PfPassiveLoopsDof> = Vec::with_capacity(n_dof);
        if current_distribution_type == "constant_current_density" {
            if n_dof != 1 {
                return Err(format!(
                    "passive conductor `{name}`: `current_distribution_type` is constant_current_density, which has 1 degree of freedom, but `n_dof` is {n_dof}"
                ));
            }

            current_distribution_type_identifier = IdentifierStatic {
                name: "constant_current_density".to_string(),
                index: 1,
                description: "One degree of freedom, with the same current density in every element".to_string(),
            };

            // The current in each element is its area, so that the degree of freedom is the current density
            dofs.push(PfPassiveLoopsDof {
                name: "constant_current_density".to_string(),
                current_distribution: Array1::ones(n_element) * &area,
                ..Default::default()
            });
        } else if current_distribution_type == "eig" {
            if n_dof > n_element {
                return Err(format!(
                    "passive conductor `{name}`: `n_dof` is {n_dof}, but there are only {n_element} filaments, and so only {n_element} eigenmodes"
                ));
            }

            current_distribution_type_identifier = IdentifierStatic {
                name: "eig".to_string(),
                index: 2,
                description: "One degree of freedom for each of the slowest decaying eigenmodes".to_string(),
            };

            // Mutual inductance matrix
            let mutual_inductance_matrix: Array2<f64> = mutual_inductance_finite_size_to_finite_size(
                &filaments.r,
                &filaments.z,
                &filaments.d_r,
                &filaments.d_z,
                &filaments.angle_1,
                &filaments.angle_2,
                &filaments.r,
                &filaments.z,
                &filaments.d_r,
                &filaments.d_z,
                &filaments.angle_1,
                &filaments.angle_2,
            );

            // Resistance matrix
            let length: Array1<f64> = 2.0 * PI * &filaments.r;
            let resistance_matrix: Array2<f64> = Array2::eye(n_element) * resistivity * length / &area;

            // Generalized symmetric eigenproblem: A x = λ B x
            // where A = mutual_inductance_matrix, B = resistance_matrix
            // Solved via: L L^T = B (Cholesky), then standard eigenproblem on L^{-1} A L^{-T}

            // Convert ndarray matrices to faer matrices
            let n: usize = n_element;
            let a_faer: faer::Mat<f64> = faer::Mat::from_fn(n, n, |i, j| mutual_inductance_matrix[[i, j]]);
            let b_faer: faer::Mat<f64> = faer::Mat::from_fn(n, n, |i, j| resistance_matrix[[i, j]]);

            // Cholesky factorization of B: B = L L^T
            let b_llt: faer::linalg::solvers::Llt<f64> = b_faer
                .llt(Side::Lower)
                .map_err(|error| format!("passive conductor `{name}`: Cholesky factorisation of the resistance matrix failed: {error:?}"))?;
            let l_factor: faer::MatRef<'_, f64> = b_llt.L();

            // Form C = L^{-1} A L^{-T} for the standard symmetric eigenproblem
            // Step 1: solve L Y = A  (forward substitution, Y = L^{-1} A)
            let mut y: faer::Mat<f64> = a_faer.clone();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l_factor, y.as_mut(), faer::Par::Seq);
            // Step 2: C = Y L^{-T}, equivalent to solving L^T C^T = Y^T
            let mut c: faer::Mat<f64> = y.transpose().to_owned();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l_factor, c.as_mut(), faer::Par::Seq);
            let c: faer::Mat<f64> = c.transpose().to_owned();

            // Symmetric eigendecomposition of C: eigenvalues in nondecreasing order
            let eigen: faer::linalg::solvers::SelfAdjointEigen<f64> = c
                .self_adjoint_eigen(Side::Lower)
                .map_err(|error| format!("passive conductor `{name}`: symmetric eigendecomposition failed: {error:?}"))?;
            let eigenvalues_faer: faer::Col<f64> = eigen.S().column_vector().to_owned();
            let eigenvectors_c: faer::Mat<f64> = eigen.U().to_owned();

            // Back-transform eigenvectors: x = L^{-T} u
            // Solve L^T X = U  (back substitution)
            let mut eigenvectors_faer: faer::Mat<f64> = eigenvectors_c;
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l_factor.transpose(), eigenvectors_faer.as_mut(), faer::Par::Seq);

            // Eigenvalues are already sorted nondecreasing (smallest to largest)
            let mut eigenvalues: Array1<f64> = Array1::from_elem(n, f64::NAN);
            for i_eig in 0..n {
                eigenvalues[i_eig] = eigenvalues_faer[i_eig];
            }

            // Normalize eigenvectors using "2-norm" and store in ndarray (row-major: row i = eigenvector i)
            // This will panic if the eigenvector is itself zero, which should never happen, so let's keep the panic in place as it's the correct behaviour
            let mut eigenvectors: Array2<f64> = Array2::from_elem((n, n), f64::NAN);
            for i_eig in 0..n {
                let eigenvector_col: faer::ColRef<'_, f64> = eigenvectors_faer.col(i_eig);
                let norm: f64 = eigenvector_col.norm_l2();
                for j_elem in 0..n {
                    eigenvectors[[i_eig, j_elem]] = eigenvector_col[j_elem] / norm;
                }
            }

            // The slowest decaying eigenmodes, which have the largest eigenvalues, first
            for i_eig in 0..n_dof {
                dofs.push(PfPassiveLoopsDof {
                    name: format!("eig_{:02}", i_eig + 1),
                    current_distribution: eigenvectors.slice(s![n_element - i_eig - 1, ..]).to_owned() * 100.0,
                    time_constant: eigenvalues[n_element - i_eig - 1],
                });
            }
        } else {
            return Err(format!(
                "passive conductor `{name}`: unknown `current_distribution_type` {current_distribution_type:?}; it must be \"constant_current_density\" or \"eig\""
            ));
        }

        self.pf_passive_ids.r#loop.push(PfPassiveLoops {
            name: name.to_string(),
            resistivity,
            element: elements,
            current_distribution_type: current_distribution_type_identifier,
            dof: dofs,
            regularisations,
            regularisations_weight,
            ..Default::default()
        });

        Ok(())
    }
}

/// The filaments of one passive conductor, described as GSFit's Greens functions describe them, in the order of `element`.
///
/// See the `PfPassive` documentation for how this differs from the data dictionary's description.
pub struct PassiveFilaments {
    /// Major radius of each filament's centre, [metre]
    pub r: Array1<f64>,
    /// Height of each filament's centre, [metre]
    pub z: Array1<f64>,
    /// Radial width of each filament: the radial extent of its bottom and top sides, [metre]
    pub d_r: Array1<f64>,
    /// Vertical height of each filament: the vertical extent of its left and right sides, [metre]
    pub d_z: Array1<f64>,
    /// Angle between the R axis and each filament's left and right sides; zero when they are not sheared, so vertical, [radian]
    pub angle_1: Array1<f64>,
    /// Angle between the R axis and each filament's bottom and top sides; zero when they are not sheared, so horizontal, [radian]
    pub angle_2: Array1<f64>,
}

/// Describe a passive conductor's filaments as GSFit's Greens functions describe them, from their data dictionary `oblique`
/// description.
///
/// The reference point is the bottom left corner. The side inclined at `alpha` is the bottom side, and the side inclined at `beta`
/// is the left side, which is at angle `beta + pi / 2` from the R axis. An angle of zero means the side is not sheared.
///
/// # Arguments
/// * `elements` - the passive conductor's elements, `loop(i1)/element`
///
/// # Returns
/// * The filaments, in the order of `elements`
pub fn passive_filaments(elements: &[PfCoilsElements]) -> PassiveFilaments {
    let n_element: usize = elements.len();

    let mut filaments: PassiveFilaments = PassiveFilaments {
        r: Array1::from_elem(n_element, f64::NAN),
        z: Array1::from_elem(n_element, f64::NAN),
        d_r: Array1::from_elem(n_element, f64::NAN),
        d_z: Array1::from_elem(n_element, f64::NAN),
        angle_1: Array1::from_elem(n_element, f64::NAN),
        angle_2: Array1::from_elem(n_element, f64::NAN),
    };

    for i_element in 0..n_element {
        let oblique: &ObliqueStatic = &elements[i_element].geometry.oblique;

        // The bottom side, from the reference point, and the left side, from the reference point
        let side_alpha_r: f64 = oblique.length_alpha * oblique.alpha.cos();
        let side_alpha_z: f64 = oblique.length_alpha * oblique.alpha.sin();
        let side_beta_r: f64 = -oblique.length_beta * oblique.beta.sin();
        let side_beta_z: f64 = oblique.length_beta * oblique.beta.cos();

        // The centre is half of each side on from the reference point
        filaments.r[i_element] = oblique.r + 0.5 * (side_alpha_r + side_beta_r);
        filaments.z[i_element] = oblique.z + 0.5 * (side_alpha_z + side_beta_z);
        filaments.d_r[i_element] = side_alpha_r;
        filaments.d_z[i_element] = side_beta_z;
        filaments.angle_1[i_element] = if oblique.beta == 0.0 { 0.0 } else { oblique.beta + PI / 2.0 };
        filaments.angle_2[i_element] = if oblique.alpha == 0.0 { 0.0 } else { oblique.alpha };
    }

    filaments
}
