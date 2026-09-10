//! Calculations written to `time_slice(itime)/profiles_1d`.

#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod area__volume__derivatives;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod b_field_average__b_field_max__b_field_min;
pub(super) mod beta_pol;
pub(super) mod dpressure_dpsi;
pub(super) mod dpsi_drho_tor;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod elongation__squareness__triangularity;
pub(super) mod f;
pub(super) mod f_df_dpsi;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod geometric_axis__r__z;
pub(super) mod gm1_to_gm9;
pub(super) mod j_parallel;
pub(super) mod j_phi;
pub(super) mod magnetic_shear;
pub(super) mod phi;
pub(super) mod pressure;
pub(super) mod psi;
pub(super) mod q;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod r_inboard__r_outboard;
pub(super) mod rho_pol;
pub(super) mod rho_tor;
pub(super) mod rho_tor_norm;
pub(super) mod rho_volume_norm;
