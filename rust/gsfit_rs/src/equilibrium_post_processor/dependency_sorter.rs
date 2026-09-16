//! Orders the calculator table so that every calculator runs after the calculators it depends on.
//!
//! The sort is deliberately simple: repeated passes over the table, each placing, in table order,
//! every calculator whose dependencies have all been placed. Calculators which do not constrain
//! each other therefore keep the order they were written in, so the execution order only changes
//! when a `dependencies` column changes.
//!
//! The table has a few dozen rows, so nothing here is optimised.

use super::CalculatorIdentifier as CI;
use super::equilibrium_post_processor::CalculatorEntry;

/// Sort the calculator table into an order in which every calculator runs after its dependencies.
///
/// # Arguments
/// * `calculators_unsorted` - the calculator table, in the order it was written
///
/// # Panics
/// The table is a compile-time constant, so an inconsistent one is a programming error rather than
/// a runtime condition. Panics, naming the calculators involved, when:
/// * two rows share an identifier
/// * a row depends on an identifier which has no row in the table
/// * the dependencies contain a cycle (a row depending on itself included)
pub(super) fn sort(calculators_unsorted: Vec<CalculatorEntry>) -> Vec<CalculatorEntry> {
    let n_calculators: usize = calculators_unsorted.len();

    // Resolve each dependency to the index of the row which fills it, so that the passes below only
    // ever test a `bool`. This is also where an inconsistent table is caught
    let mut dependency_indices: Vec<Vec<usize>> = Vec::with_capacity(n_calculators);
    for i_calculator in 0..n_calculators {
        let calculator: &CalculatorEntry = &calculators_unsorted[i_calculator];

        for j_calculator in (i_calculator + 1)..n_calculators {
            if calculators_unsorted[j_calculator].identifier == calculator.identifier {
                panic!("dependency_sorter: `{:?}` appears twice in the calculator table", calculator.identifier);
            }
        }

        let mut calculator_dependency_indices: Vec<usize> = Vec::with_capacity(calculator.dependencies.len());
        for dependency in calculator.dependencies {
            let mut dependency_index: Option<usize> = None;
            'search_loop: for j_calculator in 0..n_calculators {
                if calculators_unsorted[j_calculator].identifier == *dependency {
                    dependency_index = Some(j_calculator);
                    break 'search_loop;
                }
            }
            match dependency_index {
                Some(j_calculator) => calculator_dependency_indices.push(j_calculator),
                None => panic!(
                    "dependency_sorter: `{:?}` depends on `{:?}`, which has no row in the calculator table",
                    calculator.identifier, dependency
                ),
            }
        }
        dependency_indices.push(calculator_dependency_indices);
    }

    // Each pass places, in table order, every calculator whose dependencies have all been placed. A
    // consistent table places at least one calculator per pass, so `n_calculators` passes always
    // suffice; a pass which places nothing means the calculators still waiting form a cycle, or
    // wait on one
    let mut calculators_sorted: Vec<CalculatorEntry> = Vec::with_capacity(n_calculators);
    let mut is_placed: Vec<bool> = vec![false; n_calculators];
    let n_pass_max: usize = n_calculators;
    'pass_loop: for _i_pass in 0..n_pass_max {
        let n_placed_before_pass: usize = calculators_sorted.len();

        for i_calculator in 0..n_calculators {
            if is_placed[i_calculator] {
                continue;
            }
            let mut all_dependencies_placed: bool = true;
            for &j_calculator in &dependency_indices[i_calculator] {
                if !is_placed[j_calculator] {
                    all_dependencies_placed = false;
                }
            }
            if all_dependencies_placed {
                calculators_sorted.push(calculators_unsorted[i_calculator]);
                is_placed[i_calculator] = true;
            }
        }

        if calculators_sorted.len() == n_calculators {
            break 'pass_loop;
        }
        if calculators_sorted.len() == n_placed_before_pass {
            let mut identifiers_unplaced: Vec<CI> = Vec::new();
            for i_calculator in 0..n_calculators {
                if !is_placed[i_calculator] {
                    identifiers_unplaced.push(calculators_unsorted[i_calculator].identifier);
                }
            }
            panic!("dependency_sorter: the dependencies contain a cycle; the calculators which could not be placed are {identifiers_unplaced:?}");
        }
    }

    calculators_sorted
}

#[cfg(test)]
mod tests {
    use super::super::constant_values::ConstantValues;
    use super::super::intermediate_values::IntermediateValues;
    use super::*;
    use imas_rs::EquilibriumTimeSlice;

    /// Stands in for a calculator: the sorter never calls it
    fn no_calculation(_time_slice: &mut EquilibriumTimeSlice, _constant_values: &ConstantValues<'_>, _intermediate_values: &mut IntermediateValues) {}

    fn entry(identifier: CI, dependencies: &'static [CI]) -> CalculatorEntry {
        return CalculatorEntry {
            identifier,
            calculator_function: no_calculation,
            dependencies,
        };
    }

    fn identifiers(calculators: &[CalculatorEntry]) -> Vec<CI> {
        let mut identifiers: Vec<CI> = Vec::with_capacity(calculators.len());
        for calculator in calculators {
            identifiers.push(calculator.identifier);
        }
        return identifiers;
    }

    #[test]
    fn a_table_already_in_dependency_order_is_unchanged() {
        let calculators_unsorted: Vec<CalculatorEntry> = vec![
            entry(CI::boundary__outline__r__z, &[]),
            entry(CI::intermediate_values__flux_surfaces, &[CI::boundary__outline__r__z]),
            entry(CI::profiles_1d__f, &[]),
            entry(CI::profiles_1d__q, &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__f]),
        ];
        let identifiers_expected: Vec<CI> = identifiers(&calculators_unsorted);

        let calculators_sorted: Vec<CalculatorEntry> = sort(calculators_unsorted);

        assert_eq!(identifiers(&calculators_sorted), identifiers_expected);
    }

    #[test]
    fn a_calculator_written_before_its_dependencies_is_moved_after_them() {
        // `q` is written first but needs `flux_surfaces`, which in turn needs `outline`; the rows
        // with nothing to wait for keep their written order
        let calculators_unsorted: Vec<CalculatorEntry> = vec![
            entry(CI::profiles_1d__q, &[CI::intermediate_values__flux_surfaces, CI::profiles_1d__f]),
            entry(CI::intermediate_values__flux_surfaces, &[CI::boundary__outline__r__z]),
            entry(CI::profiles_1d__f, &[]),
            entry(CI::boundary__outline__r__z, &[]),
        ];

        let calculators_sorted: Vec<CalculatorEntry> = sort(calculators_unsorted);

        let identifiers_expected: Vec<CI> = vec![
            CI::profiles_1d__f,
            CI::boundary__outline__r__z,
            CI::intermediate_values__flux_surfaces,
            CI::profiles_1d__q,
        ];
        assert_eq!(identifiers(&calculators_sorted), identifiers_expected);
    }

    #[test]
    fn an_empty_table_sorts_to_an_empty_table() {
        let calculators_sorted: Vec<CalculatorEntry> = sort(Vec::new());

        assert!(calculators_sorted.is_empty());
    }

    #[test]
    #[should_panic(expected = "appears twice in the calculator table")]
    fn a_duplicated_identifier_panics() {
        let calculators_unsorted: Vec<CalculatorEntry> = vec![entry(CI::profiles_1d__f, &[]), entry(CI::profiles_1d__f, &[])];

        let _calculators_sorted: Vec<CalculatorEntry> = sort(calculators_unsorted);
    }

    #[test]
    #[should_panic(expected = "has no row in the calculator table")]
    fn a_dependency_with_no_row_panics() {
        let calculators_unsorted: Vec<CalculatorEntry> = vec![entry(CI::profiles_1d__q, &[CI::profiles_1d__f])];

        let _calculators_sorted: Vec<CalculatorEntry> = sort(calculators_unsorted);
    }

    #[test]
    #[should_panic(expected = "the dependencies contain a cycle")]
    fn a_cycle_panics() {
        let calculators_unsorted: Vec<CalculatorEntry> = vec![
            entry(CI::profiles_1d__phi, &[CI::profiles_1d__rho_tor]),
            entry(CI::profiles_1d__rho_tor, &[CI::profiles_1d__phi]),
        ];

        let _calculators_sorted: Vec<CalculatorEntry> = sort(calculators_unsorted);
    }

    #[test]
    #[should_panic(expected = "the dependencies contain a cycle")]
    fn a_self_dependency_panics() {
        let calculators_unsorted: Vec<CalculatorEntry> = vec![entry(CI::profiles_1d__f, &[CI::profiles_1d__f])];

        let _calculators_sorted: Vec<CalculatorEntry> = sort(calculators_unsorted);
    }
}
