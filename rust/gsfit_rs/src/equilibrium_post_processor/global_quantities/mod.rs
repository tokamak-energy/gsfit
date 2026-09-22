//! Calculations written to `time_slice(itime)/global_quantities`.

#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod area__length_pol__surface__volume;
pub(super) mod beta_pol;
pub(super) mod beta_tor;
pub(super) mod bt_vac_at_r_geo;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod current_centre__r__velocity_z__z;
pub(super) mod delta_r_sep;
pub(super) mod energy_mhd;
pub(super) mod f_x;
pub(super) mod li;
pub(super) mod magnetic_axis_b_field_phi;
pub(super) mod q_95;
pub(super) mod q_axis;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod q_min__psi__psi_norm__rho_tor_norm__value;
pub(super) mod v_loop;
