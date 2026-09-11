//! `time_slice(itime)/profiles_2d(0)/type/description`, `.../type/index` and `.../type/name`

use super::super::constant_values::ConstantValues;
use super::super::intermediate_values::IntermediateValues;
use imas_rs::EquilibriumTimeSlice;

/// Identify the 2D profiles as total fields.
///
/// These values are the `total` entry in the data dictionary's
/// `equilibrium_profiles_2d_identifier`:
///
/// ```text
/// name        = "total"
/// index       = 0
/// description = "Total fields"
/// ```
///
/// The identifier applies to the complete `profiles_2d(0)` structure: its magnetic field,
/// current density and poloidal flux include all active-coil, passive-element and plasma
/// contributions. This is metadata, so it is filled even when the equilibrium solve did not
/// converge.
///
/// # Arguments
/// * `time_slice` - the time-slice whose `profiles_2d(0)/type` identifier is written
pub fn calculate(time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues, _intermediate_values: &mut IntermediateValues) {
    // `profiles_2d[0]` because GSFit solves on one rectangular (R, Z) grid.
    time_slice.profiles_2d[0].r#type.description = "Total fields".to_string();
    time_slice.profiles_2d[0].r#type.index = 0;
    time_slice.profiles_2d[0].r#type.name = "total".to_string();
}

#[cfg(test)]
mod tests {
    use super::super::super::constant_values::constant_values_for_test;
    use super::super::super::intermediate_values::intermediate_values_for_test;
    use super::*;
    use imas_rs::EquilibriumProfiles2d;

    #[test]
    fn identifier_records_total_fields() {
        let mut time_slice: EquilibriumTimeSlice = EquilibriumTimeSlice::default();
        time_slice.profiles_2d = vec![EquilibriumProfiles2d::default()];

        calculate(&mut time_slice, &constant_values_for_test(), &mut intermediate_values_for_test());

        assert_eq!(time_slice.profiles_2d[0].r#type.description, "Total fields");
        assert_eq!(time_slice.profiles_2d[0].r#type.index, 0);
        assert_eq!(time_slice.profiles_2d[0].r#type.name, "total");
    }
}
