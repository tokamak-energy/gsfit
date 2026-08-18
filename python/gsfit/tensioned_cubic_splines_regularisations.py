import typing

import gsfit_rs
import numpy as np
import numpy.typing as npt


def make_tensioned_cubic_b_spline_regularisations(
    interior_knots: npt.NDArray[np.float64],
    interval_tensions: npt.NDArray[np.float64],
    regularisation_builder: dict[str, typing.Any],
) -> npt.NDArray[np.float64]:
    """
    Build the regularisation matrix for a tensioned cubic B-spline source function.

    This gathers the boiler-plate that used to be copied into the examples (see the previous
    `examples/scan_gsfit__st40__real_data_pressure_constrained.py` and
    `examples/example_07__mastu__freegsnke_data__tensioned_cubic_b_spline_test.ipynb`) into a single
    reusable function which is driven purely by settings.

    :param interior_knots: Interior knot locations in normalised poloidal flux [dimensionless].
        The boundary knots at psi_n = 0 and psi_n = 1 are added automatically and must not be included here.
    :param interval_tensions: Tension over each interval between knots [dimensionless].
    :param regularisation_builder: Dictionary describing how to build the regularisation matrix. Keys:
        * `regularisation_scale` [dimensionless]: Multiplies the "bulk" regularisation rows.
        * `regularisation_mode` [str]: How to regularise the bulk of the profile. Presently only
          `"second_derivative_zero"` is supported, which drives the second derivative of the source
          function to zero at every knot.
        * `left_boundary_condition` / `right_boundary_condition` [str]: One of `"free"`, `"dirichlet"`,
          or `"neumann"`. `"free"` adds no boundary row, `"dirichlet"` drives the source function value to
          zero at the boundary, and `"neumann"` drives the first derivative of the source function to zero
          at the boundary.
        * `left_boundary_weight` / `right_boundary_weight` [dimensionless]: Weight applied to the boundary
          rows (only used when the corresponding boundary condition is not `"free"`).

    :return: The regularisation matrix, shape = (n_regularisation, n_dof).

    The analytic formulae for the value, first derivative, and second derivative of a tensioned cubic
    B-spline at a knot are derived in the following LaTeX document:
    https://www.overleaf.com/read/vbydpqkjtmds#910342
    which builds on results from P. E. Koch and T. Lyche, "Interpolation with Exponential B-Splines in
    Tension", in *Geometric Modelling*, pp. 173-190.
    """

    # Ensure numpy arrays (the values may arrive as Python lists from the JSON settings)
    interior_knots = np.asarray(interior_knots, dtype=np.float64)
    interval_tensions = np.asarray(interval_tensions, dtype=np.float64)

    # Number of degrees of freedom for a cubic spline = number of interior knots plus 4
    n_dof: int = len(interior_knots) + 4

    # A dummy source function is used purely to access the `sigma1_array` and `sigma2_array` values,
    # which are needed to evaluate the analytic derivative formulae below.
    regularisations_dummy: npt.NDArray[np.float64] = np.zeros((1, n_dof), dtype=np.float64)
    source_function_dummy = gsfit_rs.TensionedCubicBSpline(regularisations_dummy, interior_knots, interval_tensions)
    sigma1_array: npt.NDArray[np.float64] = source_function_dummy.get_array1(["sigma1_array"])
    sigma2_array: npt.NDArray[np.float64] = source_function_dummy.get_array1(["sigma2_array"])

    # All knots, including the boundary knots at psi_n = 0 and psi_n = 1
    knots: npt.NDArray[np.float64] = np.concatenate(([0.0], interior_knots, [1.0]))
    n_knots: int = len(knots)

    regularisation_scale: float = regularisation_builder["regularisation_scale"]
    regularisation_mode: str = regularisation_builder["regularisation_mode"]

    # "Bulk" regularisation rows, one per knot (including the boundary knots)
    bulk_regularisations: npt.NDArray[np.float64] = np.zeros((n_knots, n_dof), dtype=np.float64)
    if regularisation_mode == "second_derivative_zero":
        # Impose d^2(source_function)/d(psi_n)^2 = 0 at each knot location
        for i_knot in range(n_knots):
            j: int = i_knot + 3
            bulk_regularisations[i_knot, j - 3] = 1.0 / (sigma2_array[j - 2] * sigma1_array[j - 1])
            bulk_regularisations[i_knot, j - 2] = -(1.0 / sigma2_array[j - 2] + 1.0 / sigma2_array[j - 1]) / sigma1_array[j - 1]
            bulk_regularisations[i_knot, j - 1] = 1.0 / (sigma2_array[j - 1] * sigma1_array[j - 1])
    else:
        raise ValueError(f"Unknown regularisation_mode: '{regularisation_mode}'. Supported values are: 'second_derivative_zero'.")

    bulk_regularisations *= regularisation_scale

    # Boundary rows are stacked before and after the bulk rows
    left_boundary_regularisation: npt.NDArray[np.float64] | None = _make_boundary_regularisation(
        side="left",
        boundary_condition=regularisation_builder["left_boundary_condition"],
        boundary_weight=regularisation_builder["left_boundary_weight"],
        n_dof=n_dof,
    )
    right_boundary_regularisation: npt.NDArray[np.float64] | None = _make_boundary_regularisation(
        side="right",
        boundary_condition=regularisation_builder["right_boundary_condition"],
        boundary_weight=regularisation_builder["right_boundary_weight"],
        n_dof=n_dof,
    )

    regularisations_list: list[npt.NDArray[np.float64]] = []
    if left_boundary_regularisation is not None:
        regularisations_list.append(left_boundary_regularisation)
    regularisations_list.append(bulk_regularisations)
    if right_boundary_regularisation is not None:
        regularisations_list.append(right_boundary_regularisation)

    regularisations: npt.NDArray[np.float64] = np.concatenate(regularisations_list, axis=0)

    return regularisations


def _make_boundary_regularisation(
    side: str,
    boundary_condition: str,
    boundary_weight: float,
    n_dof: int,
) -> npt.NDArray[np.float64] | None:
    """
    Build a single boundary regularisation row.

    :param side: `"left"` (psi_n = 0) or `"right"` (psi_n = 1).
    :param boundary_condition: One of `"free"`, `"dirichlet"`, or `"neumann"`.
    :param boundary_weight: Weight applied to the boundary row [dimensionless].
    :param n_dof: Number of degrees of freedom (columns of the regularisation matrix).

    :return: A row of shape (1, n_dof), or `None` when the boundary condition is `"free"`.

    At a clamped boundary only a single B-spline is non-zero, so the source function value equals the
    first (psi_n = 0) or last (psi_n = 1) degree of freedom. The first derivative at the boundary is
    proportional to the difference of the two degrees of freedom nearest the boundary (see the LaTeX
    document referenced in `make_tensioned_cubic_b_spline_regularisations`).
    """

    if boundary_condition == "free":
        return None

    boundary_regularisation: npt.NDArray[np.float64] = np.zeros((1, n_dof), dtype=np.float64)

    if side == "left":
        if boundary_condition == "dirichlet":
            # Impose source_function = 0 at psi_n = 0
            boundary_regularisation[0, 0] = boundary_weight
        elif boundary_condition == "neumann":
            # Impose d(source_function)/d(psi_n) = 0 at psi_n = 0
            boundary_regularisation[0, 0] = boundary_weight
            boundary_regularisation[0, 1] = -boundary_weight
        else:
            raise ValueError(f"Unknown boundary_condition: '{boundary_condition}'. Supported values are: 'free', 'dirichlet', 'neumann'.")
    elif side == "right":
        if boundary_condition == "dirichlet":
            # Impose source_function = 0 at psi_n = 1
            boundary_regularisation[0, -1] = boundary_weight
        elif boundary_condition == "neumann":
            # Impose d(source_function)/d(psi_n) = 0 at psi_n = 1
            boundary_regularisation[0, -1] = boundary_weight
            boundary_regularisation[0, -2] = -boundary_weight
        else:
            raise ValueError(f"Unknown boundary_condition: '{boundary_condition}'. Supported values are: 'free', 'dirichlet', 'neumann'.")
    else:
        raise ValueError(f"Unknown side: '{side}'. Supported values are: 'left', 'right'.")

    return boundary_regularisation
