//! Calculations written to `time_slice(itime)/profiles_2d`.

pub(super) mod b_field_phi;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod b_field_r__b_field_z;
pub(super) mod d_b_field_z_d_z;
pub(super) mod grid_volume_element;
pub(super) mod j_parallel;
pub(super) mod phi;
pub(super) mod pressure;
pub(super) mod theta;
#[expect(non_snake_case, reason = "double underscores separate IDS node names")]
pub(super) mod type__description__index__name;
