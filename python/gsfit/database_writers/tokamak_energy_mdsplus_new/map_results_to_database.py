import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs.imas import equilibrium_paths as ep
from gsfit_rs.imas import pf_passive_paths as pp
from scipy.constants import mu_0

from ..workflow_names import workflow_from_settings

if TYPE_CHECKING:
    from ...gsfit import Gsfit
    from . import DatabaseWriterTokamakEnergyMDSplusNew


# Pairs of (MDSplus node path, IMAS path) for everything which comes out of the `equilibrium` IDS
# one value per time-slice. An IMAS path is a value: it holds no data, so the table below can be
# built once, at import, and read against any IDS.
#
# `time_slice[:]` gathers over every time-slice, giving the `[n_time, ...]` arrays MDSplus expects.
_TIME_SERIES_PATH_PAIRS: list[tuple[tuple[str, ...], typing.Any]] = [
    # Plasma boundary
    (("BOUNDARY", "GEO_AXIS", "R"), ep.time_slice[:].boundary.geometric_axis.r),
    (("BOUNDARY", "GEO_AXIS", "Z"), ep.time_slice[:].boundary.geometric_axis.z),
    (("BOUNDARY", "MINOR_RADIUS"), ep.time_slice[:].boundary.minor_radius),
    (("BOUNDARY", "ELONGATION"), ep.time_slice[:].boundary.elongation),
    (("BOUNDARY", "TRIANG"), ep.time_slice[:].boundary.triangularity),
    (("BOUNDARY", "TRIANG_L"), ep.time_slice[:].boundary.triangularity_lower),
    (("BOUNDARY", "TRIANG_U"), ep.time_slice[:].boundary.triangularity_upper),
    (("BOUNDARY", "SQUARE_L_I"), ep.time_slice[:].boundary.squareness_lower_inner),
    (("BOUNDARY", "SQUARE_L_O"), ep.time_slice[:].boundary.squareness_lower_outer),
    (("BOUNDARY", "SQUARE_U_I"), ep.time_slice[:].boundary.squareness_upper_inner),
    (("BOUNDARY", "SQUARE_U_O"), ep.time_slice[:].boundary.squareness_upper_outer),
    (("BOUNDARY", "PSI"), ep.time_slice[:].boundary.psi),
    (("BOUNDARY", "OUTLINE", "R"), ep.time_slice[:].boundary.outline.r),
    (("BOUNDARY", "OUTLINE", "Z"), ep.time_slice[:].boundary.outline.z),
    (("BOUNDARY", "BOUNDING", "R"), ep.time_slice[:].boundary.bounding.r),
    (("BOUNDARY", "BOUNDING", "Z"), ep.time_slice[:].boundary.bounding.z),
    # Convergence
    (("CONVERGENCE", "GS_ERROR"), ep.time_slice[:].convergence.grad_shafranov_deviation_value),
    (("CONVERGENCE", "ITERATIONS_N"), ep.time_slice[:].convergence.iterations_n),
    # Global
    (("GLOBAL", "CURRENT_CENT", "R"), ep.time_slice[:].global_quantities.current_centre.r),
    (("GLOBAL", "CURRENT_CENT", "VELOCITY_Z"), ep.time_slice[:].global_quantities.current_centre.velocity_z),
    (("GLOBAL", "CURRENT_CENT", "Z"), ep.time_slice[:].global_quantities.current_centre.z),
    (("GLOBAL", "MAG_AXIS", "R"), ep.time_slice[:].global_quantities.magnetic_axis.r),
    (("GLOBAL", "MAG_AXIS", "B_FIELD_PHI"), ep.time_slice[:].global_quantities.magnetic_axis.b_field_phi),
    (("GLOBAL", "MAG_AXIS", "Z"), ep.time_slice[:].global_quantities.magnetic_axis.z),
    (("GLOBAL", "Q_MIN", "VALUE"), ep.time_slice[:].global_quantities.q_min.value),
    (("GLOBAL", "Q_MIN", "RHO_TOR_NORM"), ep.time_slice[:].global_quantities.q_min.rho_tor_norm),
    (("GLOBAL", "Q_MIN", "PSI_NORM"), ep.time_slice[:].global_quantities.q_min.psi_norm),
    (("GLOBAL", "Q_MIN", "PSI"), ep.time_slice[:].global_quantities.q_min.psi),
    (("GLOBAL", "AREA"), ep.time_slice[:].global_quantities.area),
    (("GLOBAL", "BETA_N"), ep.time_slice[:].global_quantities.beta_tor_norm),
    (("GLOBAL", "BETA_P_1"), ep.time_slice[:].global_quantities.beta_pol_1),
    (("GLOBAL", "BETA_P_2"), ep.time_slice[:].global_quantities.beta_pol_2),
    (("GLOBAL", "BETA_P_3"), ep.time_slice[:].global_quantities.beta_pol_3),
    (("GLOBAL", "BP_OMP"), ep.time_slice[:].global_quantities.bp_omp),
    (("GLOBAL", "BT_VAC_RGEO"), ep.time_slice[:].global_quantities.bt_vac_at_r_geo),
    (("GLOBAL", "DELTA_R_SEP"), ep.time_slice[:].global_quantities.delta_r_sep),
    (("GLOBAL", "DELTA_Z"), ep.time_slice[:].convergence.delta_z),
    (("GLOBAL", "ENERGY_MHD"), ep.time_slice[:].global_quantities.energy_mhd),
    (("GLOBAL", "FX"), ep.time_slice[:].global_quantities.f_x),
    (("GLOBAL", "IP"), ep.time_slice[:].global_quantities.ip),
    (("GLOBAL", "LI_1"), ep.time_slice[:].global_quantities.li_1),
    (("GLOBAL", "LI_2"), ep.time_slice[:].global_quantities.li_2),
    (("GLOBAL", "LI_3"), ep.time_slice[:].global_quantities.li_3),
    (("GLOBAL", "LENGTH_POL"), ep.time_slice[:].global_quantities.length_pol),
    (("GLOBAL", "PHI_DIA"), ep.time_slice[:].constraints.diamagnetic_flux.reconstructed),
    (("GLOBAL", "PSI_MAG_AXIS"), ep.time_slice[:].global_quantities.psi_magnetic_axis),
    (("GLOBAL", "Q_AXIS"), ep.time_slice[:].global_quantities.q_axis),
    (("GLOBAL", "Q_95"), ep.time_slice[:].global_quantities.q_95),
    (("GLOBAL", "SURFACE"), ep.time_slice[:].global_quantities.surface),
    (("GLOBAL", "V_LOOP"), ep.time_slice[:].global_quantities.v_loop),
    (("GLOBAL", "VOLUME"), ep.time_slice[:].global_quantities.volume),
    (("GLOBAL", "XPT_DIVERTED"), ep.time_slice[:].boundary.type),
    # Profiles_1d, on the psi_norm grid
    (("PROFILES_1D", "PSI_NORM", "GEO_AXIS", "R"), ep.time_slice[:].profiles_1d.geometric_axis.r),
    (("PROFILES_1D", "PSI_NORM", "GEO_AXIS", "Z"), ep.time_slice[:].profiles_1d.geometric_axis.z),
    (("PROFILES_1D", "PSI_NORM", "AREA"), ep.time_slice[:].profiles_1d.area),
    (("PROFILES_1D", "PSI_NORM", "DAREA_DPSI"), ep.time_slice[:].profiles_1d.darea_dpsi),
    (("PROFILES_1D", "PSI_NORM", "DAREA_DRHO_T"), ep.time_slice[:].profiles_1d.darea_drho_tor),
    (("PROFILES_1D", "PSI_NORM", "ELONGATION"), ep.time_slice[:].profiles_1d.elongation),
    (("PROFILES_1D", "PSI_NORM", "F"), ep.time_slice[:].profiles_1d.f),
    (("PROFILES_1D", "PSI_NORM", "FF_PRIME"), ep.time_slice[:].profiles_1d.f_df_dpsi),
    (("PROFILES_1D", "PSI_NORM", "FLUX_TOR"), ep.time_slice[:].profiles_1d.phi),
    (("PROFILES_1D", "PSI_NORM", "MAG_SHEAR"), ep.time_slice[:].profiles_1d.magnetic_shear),
    (("PROFILES_1D", "PSI_NORM", "P_PRIME"), ep.time_slice[:].profiles_1d.dpressure_dpsi),
    (("PROFILES_1D", "PSI_NORM", "PRESSURE"), ep.time_slice[:].profiles_1d.pressure),
    (("PROFILES_1D", "PSI_NORM", "PSI"), ep.time_slice[:].profiles_1d.psi),
    (("PROFILES_1D", "PSI_NORM", "PHI"), ep.time_slice[:].profiles_1d.phi),
    (("PROFILES_1D", "PSI_NORM", "J_PHI"), ep.time_slice[:].profiles_1d.j_phi),
    (("PROFILES_1D", "PSI_NORM", "J_PARALLEL"), ep.time_slice[:].profiles_1d.j_parallel),
    (("PROFILES_1D", "PSI_NORM", "Q"), ep.time_slice[:].profiles_1d.q),
    (("PROFILES_1D", "PSI_NORM", "RHO_POL"), ep.time_slice[:].profiles_1d.rho_pol),
    (("PROFILES_1D", "PSI_NORM", "R_INBOARD"), ep.time_slice[:].profiles_1d.r_inboard),
    (("PROFILES_1D", "PSI_NORM", "R_OUTBOARD"), ep.time_slice[:].profiles_1d.r_outboard),
    (("PROFILES_1D", "PSI_NORM", "R_INBOARD_0"), ep.time_slice[:].profiles_1d.r_inboard_z_0),
    (("PROFILES_1D", "PSI_NORM", "R_OUTBOARD_0"), ep.time_slice[:].profiles_1d.r_outboard_z_0),
    (("PROFILES_1D", "PSI_NORM", "RHO_TOR"), ep.time_slice[:].profiles_1d.rho_tor),
    (("PROFILES_1D", "PSI_NORM", "RHO_TOR_NORM"), ep.time_slice[:].profiles_1d.rho_tor_norm),
    (("PROFILES_1D", "PSI_NORM", "SQUARE_L_I"), ep.time_slice[:].profiles_1d.squareness_lower_inner),
    (("PROFILES_1D", "PSI_NORM", "SQUARE_L_O"), ep.time_slice[:].profiles_1d.squareness_lower_outer),
    (("PROFILES_1D", "PSI_NORM", "SQUARE_U_I"), ep.time_slice[:].profiles_1d.squareness_upper_inner),
    (("PROFILES_1D", "PSI_NORM", "SQUARE_U_O"), ep.time_slice[:].profiles_1d.squareness_upper_outer),
    (("PROFILES_1D", "PSI_NORM", "TRIANG"), ep.time_slice[:].profiles_1d.triangularity),
    (("PROFILES_1D", "PSI_NORM", "TRIANG_L"), ep.time_slice[:].profiles_1d.triangularity_lower),
    (("PROFILES_1D", "PSI_NORM", "TRIANG_U"), ep.time_slice[:].profiles_1d.triangularity_upper),
    (("PROFILES_1D", "PSI_NORM", "VOL"), ep.time_slice[:].profiles_1d.volume),
    (("PROFILES_1D", "PSI_NORM", "VOL_PRIME"), ep.time_slice[:].profiles_1d.dvolume_dpsi),
    (("PROFILES_1D", "PSI_NORM", "RHO_VOL_NORM"), ep.time_slice[:].profiles_1d.rho_volume_norm),
    (("PROFILES_1D", "PSI_NORM", "DVOL_DPSI"), ep.time_slice[:].profiles_1d.dvolume_dpsi),
    (("PROFILES_1D", "PSI_NORM", "DVOL_DRHO_T"), ep.time_slice[:].profiles_1d.dvolume_drho_tor),
    (("PROFILES_1D", "PSI_NORM", "SURFACE"), ep.time_slice[:].profiles_1d.surface),
    (("PROFILES_1D", "PSI_NORM", "GM1"), ep.time_slice[:].profiles_1d.gm1),
    (("PROFILES_1D", "PSI_NORM", "GM2"), ep.time_slice[:].profiles_1d.gm2),
    (("PROFILES_1D", "PSI_NORM", "GM3"), ep.time_slice[:].profiles_1d.gm3),
    (("PROFILES_1D", "PSI_NORM", "GM4"), ep.time_slice[:].profiles_1d.gm4),
    (("PROFILES_1D", "PSI_NORM", "GM5"), ep.time_slice[:].profiles_1d.gm5),
    (("PROFILES_1D", "PSI_NORM", "GM6"), ep.time_slice[:].profiles_1d.gm6),
    (("PROFILES_1D", "PSI_NORM", "GM7"), ep.time_slice[:].profiles_1d.gm7),
    (("PROFILES_1D", "PSI_NORM", "GM8"), ep.time_slice[:].profiles_1d.gm8),
    (("PROFILES_1D", "PSI_NORM", "GM9"), ep.time_slice[:].profiles_1d.gm9),
    (("PROFILES_1D", "PSI_NORM", "B_FIELD_AV"), ep.time_slice[:].profiles_1d.b_field_average),
    (("PROFILES_1D", "PSI_NORM", "B_FIELD_MIN"), ep.time_slice[:].profiles_1d.b_field_min),
    (("PROFILES_1D", "PSI_NORM", "B_FIELD_MAX"), ep.time_slice[:].profiles_1d.b_field_max),
    # Mid-plane profiles
    (("PROFILES_1D", "R_MIDPLANE", "P_PRIME"), ep.time_slice[:].profiles_1d_r_midplane.dpressure_dpsi),
    (("PROFILES_1D", "R_MIDPLANE", "F"), ep.time_slice[:].profiles_1d_r_midplane.f),
    (("PROFILES_1D", "R_MIDPLANE", "FF_PRIME"), ep.time_slice[:].profiles_1d_r_midplane.f_df_dpsi),
    (("PROFILES_1D", "R_MIDPLANE", "J_PHI"), ep.time_slice[:].profiles_1d_r_midplane.j_phi),
    (("PROFILES_1D", "R_MIDPLANE", "PRESSURE"), ep.time_slice[:].profiles_1d_r_midplane.pressure),
    (("PROFILES_1D", "R_MIDPLANE", "Q"), ep.time_slice[:].profiles_1d_r_midplane.q),
    # SOL
    (("PROFILES_1D", "R_MIDPLANE_H", "PSI_NORM"), ep.time_slice[:].profiles_1d_r_midplane_h.psi_norm),
    (("PROFILES_1D", "R_MIDPLANE_H", "PSI"), ep.time_slice[:].profiles_1d_r_midplane_h.psi),
    # Profiles_2d. `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    (("PROFILES_2D", "R_Z", "B_FIELD_PHI"), ep.time_slice[:].profiles_2d[0].b_field_phi),
    (("PROFILES_2D", "R_Z", "B_FIELD_R"), ep.time_slice[:].profiles_2d[0].b_field_r),
    (("PROFILES_2D", "R_Z", "B_FIELD_Z"), ep.time_slice[:].profiles_2d[0].b_field_z),
    (("PROFILES_2D", "R_Z", "MASK"), ep.time_slice[:].profiles_2d[0].mask),
    (("PROFILES_2D", "R_Z", "PRESSURE"), ep.time_slice[:].profiles_2d[0].pressure),
    (("PROFILES_2D", "R_Z", "PSI"), ep.time_slice[:].profiles_2d[0].psi),
    (("PROFILES_2D", "R_Z", "PHI"), ep.time_slice[:].profiles_2d[0].phi),
    (("PROFILES_2D", "R_Z", "THETA"), ep.time_slice[:].profiles_2d[0].theta),
    (("PROFILES_2D", "R_Z", "J_PHI"), ep.time_slice[:].profiles_2d[0].j_phi),
    (("PROFILES_2D", "R_Z", "J_PARALLEL"), ep.time_slice[:].profiles_2d[0].j_parallel),
    # Constraints
    (("CONSTRAINTS", "CHI_SQ_MAG"), ep.time_slice[:].constraints.chi_squared_reduced),
    # Scrape off layer (SOL)
    (("SOL", "HFS", "CONTOUR", "R"), ep.time_slice[:].sol.hfs.contour.r),
    (("SOL", "HFS", "CONTOUR", "Z"), ep.time_slice[:].sol.hfs.contour.z),
    (("SOL", "HFS", "STRIKE_POINT", "R"), ep.time_slice[:].sol.hfs.strike_point.r),
    (("SOL", "HFS", "STRIKE_POINT", "Z"), ep.time_slice[:].sol.hfs.strike_point.z),
    (("SOL", "LFS", "CONTOUR", "R"), ep.time_slice[:].sol.lfs.contour.r),
    (("SOL", "LFS", "CONTOUR", "Z"), ep.time_slice[:].sol.lfs.contour.z),
    (("SOL", "LFS", "STRIKE_POINT", "R"), ep.time_slice[:].sol.lfs.strike_point.r),
    (("SOL", "LFS", "STRIKE_POINT", "Z"), ep.time_slice[:].sol.lfs.strike_point.z),
]

# The same, for quantities which are the same on every time-slice. The IDS stores them per
# time-slice all the same, so the gather returns `[n_time, n_points]` and the first row is taken.
_TIME_INDEPENDENT_PATH_PAIRS: list[tuple[tuple[str, ...], typing.Any]] = [
    (("PROFILES_1D", "PSI_NORM", "PSI_NORM"), ep.time_slice[:].profiles_1d.psi_norm),
    (("PROFILES_1D", "R_MIDPLANE", "R"), ep.time_slice[:].profiles_1d_r_midplane.r),
    (("PROFILES_1D", "R_MIDPLANE_H", "R"), ep.time_slice[:].profiles_1d_r_midplane_h.r),
    (("PROFILES_2D", "R_Z", "R"), ep.time_slice[:].profiles_2d[0].grid.dim1),
    (("PROFILES_2D", "R_Z", "Z"), ep.time_slice[:].profiles_2d[0].grid.dim2),
]


def _assign(results: typing.Any, mdsplus_path: tuple[str, ...], value: typing.Any) -> None:
    """Assign `value` into the nested `results` object at `mdsplus_path`."""
    node = results
    for key in mdsplus_path[:-1]:
        node = node[key]
    node[mdsplus_path[-1]] = value


def _n_points_per_time(padded: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """Number of real points in each row of a NaN-padded `[n_time, n_points]` array.

    The IDS stores one contour per time-slice, each its own length; gathering over time pads the
    short rows with NaN to make a rectangle. MDSplus needs that rectangle plus a separate count of
    how much of each row is real, which is what this recovers. Contour coordinates are always
    finite, so a NaN can only be padding.
    """
    n_points: npt.NDArray[np.int32] = np.isfinite(padded).sum(axis=1).astype(np.int32)

    return n_points


def map_results_to_database(
    self: "DatabaseWriterTokamakEnergyMDSplusNew",
    gsfit_controller: "Gsfit",
) -> None:
    """Map the results to MDSplus structure.
    `gsfit_controller.results` is a `NestedDict` object, which is a 1:1 mapping to the MDSplus structure.
    This function will mutate the `gsfit_controller` object.
    """

    # Take class object out of the `gsfit_controller` object
    pulseNo = gsfit_controller.pulseNo
    plasma = gsfit_controller.plasma
    bp_probes = gsfit_controller.bp_probes
    flux_loops = gsfit_controller.flux_loops
    dialoop = gsfit_controller.dialoop
    rogowski_coils = gsfit_controller.rogowski_coils
    pf_passive = gsfit_controller.pf_passive
    pressure_sensors = gsfit_controller.pressure_sensors
    results = gsfit_controller.results

    # Everything below comes out of the `equilibrium` IDS through the (MDSplus path, IMAS path)
    # tables at the top of this file. `plasma.get` reads in place, so reading quantity by quantity
    # costs nothing extra

    for mdsplus_path, imas_path in _TIME_SERIES_PATH_PAIRS:
        _assign(results, mdsplus_path, plasma.get(imas_path))

    for mdsplus_path, imas_path in _TIME_INDEPENDENT_PATH_PAIRS:
        _assign(results, mdsplus_path, np.asarray(plasma.get(imas_path))[0])

    # The rest need something done to them on the way out, so they are not in the tables

    # Flux defining the LCFS. SPIDER has `psi_norm = 0.9999`
    results["BOUNDARY"]["PSI_NORM"] = np.ones_like(results["BOUNDARY"]["PSI"])

    # Contour lengths. The IDS stores each contour at its own length and the gather pads them into
    # a rectangle; MDSplus wants the rectangle plus the count of real points in each row
    results["BOUNDARY"]["OUTLINE"]["N"] = _n_points_per_time(results["BOUNDARY"]["OUTLINE"]["R"])
    results["SOL"]["HFS"]["CONTOUR"]["N"] = _n_points_per_time(results["SOL"]["HFS"]["CONTOUR"]["R"])
    results["SOL"]["LFS"]["CONTOUR"]["N"] = _n_points_per_time(results["SOL"]["LFS"]["CONTOUR"]["R"])

    # Contour tree: the stationary points of psi (O-points and X-points). How many there are changes
    # from one time-slice to the next, so `time_slice[:].contour_tree.node[:]` would be jagged, which
    # the IDS refuses to gather. Each time-slice is read on its own, and its nodes are padded with NaN
    # to the most found on any time-slice. `N` counts the real nodes, with an entry for every
    # time-slice: 0 for one without nodes, e.g. one which failed. `CRIT_TYPE` is an integer in the
    # IDS, but is stored as a float so that it can be padded with NaN like the others
    n_time: int = len(results["TIME"])
    contour_tree_node_n: npt.NDArray[np.int32] = np.zeros(n_time, dtype=np.int32)
    for i_time in range(n_time):
        contour_tree_node_n[i_time] = len(plasma.get(ep.time_slice[i_time].contour_tree.node[:].r))
    n_node_max: int = int(np.max(contour_tree_node_n, initial=0))
    contour_tree_node_critical_type: npt.NDArray[np.float64] = np.full((n_time, n_node_max), np.nan)
    contour_tree_node_psi: npt.NDArray[np.float64] = np.full((n_time, n_node_max), np.nan)
    contour_tree_node_r: npt.NDArray[np.float64] = np.full((n_time, n_node_max), np.nan)
    contour_tree_node_z: npt.NDArray[np.float64] = np.full((n_time, n_node_max), np.nan)
    for i_time in range(n_time):
        n_node: int = int(contour_tree_node_n[i_time])
        contour_tree_node_critical_type[i_time, :n_node] = plasma.get(ep.time_slice[i_time].contour_tree.node[:].critical_type)
        contour_tree_node_psi[i_time, :n_node] = plasma.get(ep.time_slice[i_time].contour_tree.node[:].psi)
        contour_tree_node_r[i_time, :n_node] = plasma.get(ep.time_slice[i_time].contour_tree.node[:].r)
        contour_tree_node_z[i_time, :n_node] = plasma.get(ep.time_slice[i_time].contour_tree.node[:].z)
    results["CONTOUR_TREE"]["NODE"]["CRIT_TYPE"] = contour_tree_node_critical_type  # shape = [n_time, n_node_max]
    results["CONTOUR_TREE"]["NODE"]["N"] = contour_tree_node_n  # shape = [n_time]
    results["CONTOUR_TREE"]["NODE"]["PSI"] = contour_tree_node_psi  # shape = [n_time, n_node_max]
    results["CONTOUR_TREE"]["NODE"]["R"] = contour_tree_node_r  # shape = [n_time, n_node_max]
    results["CONTOUR_TREE"]["NODE"]["Z"] = contour_tree_node_z  # shape = [n_time, n_node_max]

    # Secondary X-point: the four legs of its separatrix, each from the secondary X-point to where it
    # meets the limiter. The equilibrium post-processor stores them as the secondary X-point's
    # `contour_tree/node(i_node)/levelset`, the only node with a `levelset`, in GSFit's own order (see
    # `contour_tree/node_levelset__r__z.rs`): contours 0 and 1 leave away from the plasma core, and 2
    # and 3 towards it, the low field side first in each pair. Whether a leg goes up or down therefore
    # depends on which side of the magnetic axis the secondary X-point is, which can change during a
    # pulse, so each time-slice is mapped on its own. A time-slice without a secondary X-point, e.g. a
    # limited one, has no legs: its rows are all NaN and its `N` is 0. Each leg is padded with NaN to
    # its own longest, rather than all four to the longest of any, which keeps the padding down
    secondary_xpt_leg_names: list[str] = ["HFS_LOWER", "HFS_UPPER", "LFS_LOWER", "LFS_UPPER"]
    # The leg which each `levelset` contour is, when the secondary X-point is above the magnetic axis,
    # and when it is below
    levelset_leg_names_above_axis: list[str] = ["LFS_UPPER", "HFS_UPPER", "LFS_LOWER", "HFS_LOWER"]
    levelset_leg_names_below_axis: list[str] = ["LFS_LOWER", "HFS_LOWER", "LFS_UPPER", "HFS_UPPER"]
    n_levelset: int = len(levelset_leg_names_above_axis)
    mag_z: npt.NDArray[np.float64] = np.asarray(plasma.get(ep.time_slice[:].global_quantities.magnetic_axis.z))  # shape = [n_time] [metre]
    # Each leg on each time-slice, without padding; empty on a time-slice without legs
    secondary_xpt_leg_r: dict[str, list[npt.NDArray[np.float64]]] = {}  # [metre]
    secondary_xpt_leg_z: dict[str, list[npt.NDArray[np.float64]]] = {}  # [metre]
    for leg_name in secondary_xpt_leg_names:
        secondary_xpt_leg_r[leg_name] = [np.zeros(0)] * n_time
        secondary_xpt_leg_z[leg_name] = [np.zeros(0)] * n_time
    for i_time in range(n_time):
        n_node = int(contour_tree_node_n[i_time])
        for i_node in range(n_node):
            # One row per contour, the shorter ones padded with NaN; no rows for a node without a `levelset`
            levelset_r: npt.NDArray[np.float64] = np.asarray(plasma.get(ep.time_slice[i_time].contour_tree.node[i_node].levelset[:].r))  # [metre]
            if levelset_r.shape[0] == 0:
                continue
            if levelset_r.shape[0] != n_levelset:
                raise ValueError(
                    f"`time_slice({i_time})/contour_tree/node({i_node})/levelset` has {levelset_r.shape[0]} contours, but the secondary "
                    f"X-point has {n_levelset} legs: see `contour_tree/node_levelset__r__z.rs`"
                )
            levelset_z: npt.NDArray[np.float64] = np.asarray(plasma.get(ep.time_slice[i_time].contour_tree.node[i_node].levelset[:].z))  # [metre]
            secondary_xpt_z: float = plasma.get(ep.time_slice[i_time].contour_tree.node[i_node].z)  # [metre]
            if secondary_xpt_z > mag_z[i_time]:
                levelset_leg_names: list[str] = levelset_leg_names_above_axis
            else:
                levelset_leg_names = levelset_leg_names_below_axis
            for i_levelset in range(n_levelset):
                # Contour coordinates are always finite, so a NaN can only be padding
                is_point: npt.NDArray[np.bool_] = np.isfinite(levelset_r[i_levelset, :])
                secondary_xpt_leg_r[levelset_leg_names[i_levelset]][i_time] = levelset_r[i_levelset, is_point]
                secondary_xpt_leg_z[levelset_leg_names[i_levelset]][i_time] = levelset_z[i_levelset, is_point]
    for leg_name in secondary_xpt_leg_names:
        leg_n: npt.NDArray[np.int32] = np.zeros(n_time, dtype=np.int32)
        for i_time in range(n_time):
            leg_n[i_time] = len(secondary_xpt_leg_r[leg_name][i_time])
        n_point_max: int = int(np.max(leg_n, initial=0))
        leg_r: npt.NDArray[np.float64] = np.full((n_time, n_point_max), np.nan)
        leg_z: npt.NDArray[np.float64] = np.full((n_time, n_point_max), np.nan)
        for i_time in range(n_time):
            n_point: int = int(leg_n[i_time])
            leg_r[i_time, :n_point] = secondary_xpt_leg_r[leg_name][i_time]
            leg_z[i_time, :n_point] = secondary_xpt_leg_z[leg_name][i_time]
        results["SEC_XPT"][leg_name]["CONTOUR"]["N"] = leg_n  # shape = [n_time]
        results["SEC_XPT"][leg_name]["CONTOUR"]["R"] = leg_r  # shape = [n_time, n_point_max]
        results["SEC_XPT"][leg_name]["CONTOUR"]["Z"] = leg_z  # shape = [n_time, n_point_max]

    # The data dictionary defines `beta_tor` as a fraction, but this MDSplus node has always held a
    # percentage, so the factor of 100 is put back here rather than changing what consumers read
    results["GLOBAL"]["BETA_T"] = 100.0 * np.asarray(plasma.get(ep.time_slice[:].global_quantities.beta_tor))

    # The rod current is not a data dictionary node. It is recovered from the vacuum toroidal
    # field, `f_vac = r0 * b0 = mu_0 * i_rod / (2 * pi)`
    r0 = plasma.get(ep.vacuum_toroidal_field.r0)
    b0 = np.asarray(plasma.get(ep.vacuum_toroidal_field.b0))
    results["GLOBAL"]["I_ROD"] = 2.0 * np.pi * r0 * b0 / mu_0


    # The magnetic sensors, from `constraints/b_field_pol_probe`, `constraints/flux_loop` and `constraints/rogowski_coil`, which
    # `Plasma.copy_from_magnetics_ids` filled on every time-slice with every sensor, in the order of the magnetics IDS: `measured`, the
    # measurement interpolated onto the time-slice's time, and `reconstructed`, the reconstruction's value of it.
    # Whether each sensor is fitted, and its weight, are still read from the sensor objects
    b_field_pol_probe_names: npt.NDArray[np.str_] = plasma.get(ep.time_slice[0].constraints.b_field_pol_probe[:].name)
    # shape = [n_time, n_b_field_pol_probe]
    b_field_pol_probe_measured: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].constraints.b_field_pol_probe[:].measured)
    b_field_pol_probe_reconstructed: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].constraints.b_field_pol_probe[:].reconstructed)
    n_b_field_pol_probe: int = len(b_field_pol_probe_names)
    for i_b_field_pol_probe in range(n_b_field_pol_probe):
        sensor_name: str = str(b_field_pol_probe_names[i_b_field_pol_probe])
        # results["CONSTRAINTS"]["BP_PROBE"][pf_name]["EXACT"]
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["INCLUDE"] = np.int32(bp_probes.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["MEASURED"] = b_field_pol_probe_measured[:, i_b_field_pol_probe]
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["RECONSTRUCT"] = b_field_pol_probe_reconstructed[:, i_b_field_pol_probe]
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["WEIGHT"] = bp_probes.get_f64([sensor_name, "fit_settings", "weight"])

    flux_loop_names: npt.NDArray[np.str_] = plasma.get(ep.time_slice[0].constraints.flux_loop[:].name)
    # shape = [n_time, n_flux_loop]
    flux_loop_measured: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].constraints.flux_loop[:].measured)
    flux_loop_reconstructed: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].constraints.flux_loop[:].reconstructed)
    n_flux_loop: int = len(flux_loop_names)
    for i_flux_loop in range(n_flux_loop):
        sensor_name = str(flux_loop_names[i_flux_loop])
        # results["CONSTRAINTS"]["FLUX_LOOP"][pf_name]["EXACT"]
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["INCLUDE"] = np.int32(flux_loops.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["MEASURED"] = flux_loop_measured[:, i_flux_loop]
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["RECONSTRUCT"] = flux_loop_reconstructed[:, i_flux_loop]
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["WEIGHT"] = flux_loops.get_f64([sensor_name, "fit_settings", "weight"])

    rogowski_coil_names: npt.NDArray[np.str_] = plasma.get(ep.time_slice[0].constraints.rogowski_coil[:].name)
    # shape = [n_time, n_rogowski_coil]
    rogowski_coil_measured: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].constraints.rogowski_coil[:].measured)
    rogowski_coil_reconstructed: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].constraints.rogowski_coil[:].reconstructed)
    n_rogowski_coil: int = len(rogowski_coil_names)
    for i_rogowski_coil in range(n_rogowski_coil):
        sensor_name = str(rogowski_coil_names[i_rogowski_coil])
        # results["CONSTRAINTS"]["ROGOWSKI"][pf_name]["EXACT"]
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["INCLUDE"] = np.int32(rogowski_coils.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["MEASURED"] = rogowski_coil_measured[:, i_rogowski_coil]
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["RECONSTRUCT"] = rogowski_coil_reconstructed[:, i_rogowski_coil]
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["WEIGHT"] = rogowski_coils.get_f64([sensor_name, "fit_settings", "weight"])

    # Diamagnetic flux (single diamagnetic flux loop "DIALOOP"), from `constraints/diamagnetic_flux`, whose `reconstructed` the equilibrium
    # post-processor calculates. Its name is empty when the magnetics IDS has no diamagnetic flux, e.g. on a pulse without a DIALOOP run
    diamagnetic_flux_name: str = str(plasma.get(ep.time_slice[0].constraints.diamagnetic_flux.name))
    if diamagnetic_flux_name != "":
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["INCLUDE"] = np.int32(dialoop.get_bool([diamagnetic_flux_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["MEASURED"] = plasma.get(ep.time_slice[:].constraints.diamagnetic_flux.measured)
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["RECONSTRUCT"] = plasma.get(ep.time_slice[:].constraints.diamagnetic_flux.reconstructed)
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["WEIGHT"] = dialoop.get_f64([diamagnetic_flux_name, "fit_settings", "weight"])

    # The PF coil currents, from `constraints/pf_current`, which `Plasma.copy_from_pf_active_ids` filled on every time-slice
    pf_current_names = [str(pf_name) for pf_name in plasma.get(ep.time_slice[0].constraints.pf_current[:].name)]
    pf_current_measured = plasma.get(ep.time_slice[:].constraints.pf_current[:].measured)  # shape = (n_time, n_pf)
    for i_pf, pf_name in enumerate(pf_current_names):
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["EXACT"]
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["INCLUDE"]
        results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["MEASURED"] = pf_current_measured[:, i_pf]
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["RECONSTRUCT"] = plasma.get(ep.time_slice[:].constraints.pf_current[i_pf].reconstructed)
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["WEIGHT"] = plasma.get(ep.time_slice[:].constraints.pf_current[i_pf].weight)
    # TODO: need to handle circuits vs coils better
    results["CONSTRAINTS"]["PF_CURRENT"]["BVL"]["MEASURED"] = pf_current_measured[:, pf_current_names.index("BVLT")]
    results["CONSTRAINTS"]["PF_CURRENT"]["DIV"]["MEASURED"] = pf_current_measured[:, pf_current_names.index("DIVT")]
    results["CONSTRAINTS"]["PF_CURRENT"]["PSH"]["MEASURED"] = pf_current_measured[:, pf_current_names.index("PSHT")]

    # Passive conductors: for each loop, its degrees of freedom, which are the amplitudes the reconstruction fitted and their current
    # distributions, and the geometry of its filaments.
    # The fitted amplitudes are each time-slice's `passive_dof_values`, NaN on a time-slice which failed. They follow the Greens tables,
    # `greens/grid/pf_passive`, which are in the order of `pf_passive/loop`, and then of `loop(i1)/dof`
    passive_dof_values: npt.NDArray[np.float64] = plasma.get(ep.time_slice[:].passive_dof_values)  # shape = [n_time, n_dof_total]
    n_loop: int = len(pf_passive.get(pp.loop[:].name))
    i_dof_total: int = 0
    for i_loop in range(n_loop):
        passive_name: str = pf_passive.get(pp.loop[i_loop].name)

        n_dof: int = len(pf_passive.get(pp.loop[i_loop].dof[:].name))
        for i_dof in range(n_dof):
            # The MDSplus nodes are named `CONSTANT_J`, and `EIG_01`, `EIG_02`, ... for the eigenmodes
            dof_name: str = pf_passive.get(pp.loop[i_loop].dof[i_dof].name)
            if dof_name == "constant_current_density":
                dof_node_name: str = "CONSTANT_J"
            else:
                dof_node_name = dof_name.upper()
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["DOF"][dof_node_name]["RECONSTRUCT"] = np.ascontiguousarray(passive_dof_values[:, i_dof_total])
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["DOF"][dof_node_name]["I_DIST"] = pf_passive.get(pp.loop[i_loop].dof[i_dof].current_distribution)
            i_dof_total += 1

        # The filaments, which the pf_passive IDS describes as parallelograms from one corner (`oblique`), converted back into the centre, the
        # radial width and vertical height, and the angles of the sides from the R axis, zero for a side which is not sheared, which these nodes
        # have always held. This is the conversion `passive_filaments` does in `gsfit_rs`
        oblique_r: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.r)  # [metre]
        oblique_z: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.z)  # [metre]
        oblique_length_alpha: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.length_alpha)  # [metre]
        oblique_length_beta: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.length_beta)  # [metre]
        oblique_alpha: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.alpha)  # [radian]
        oblique_beta: npt.NDArray[np.float64] = pf_passive.get(pp.loop[i_loop].element[:].geometry.oblique.beta)  # [radian]
        # The bottom side, inclined at `alpha` from the R axis, and the left side, inclined at `beta` from the Z axis, from the reference point,
        # the bottom left corner
        side_alpha_r: npt.NDArray[np.float64] = oblique_length_alpha * np.cos(oblique_alpha)  # [metre]
        side_alpha_z: npt.NDArray[np.float64] = oblique_length_alpha * np.sin(oblique_alpha)  # [metre]
        side_beta_r: npt.NDArray[np.float64] = -oblique_length_beta * np.sin(oblique_beta)  # [metre]
        side_beta_z: npt.NDArray[np.float64] = oblique_length_beta * np.cos(oblique_beta)  # [metre]
        results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["ANGLE_1"] = np.where(oblique_beta == 0.0, 0.0, oblique_beta + np.pi / 2.0)
        results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["ANGLE_2"] = np.where(oblique_alpha == 0.0, 0.0, oblique_alpha)
        results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["D_R"] = side_alpha_r
        results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["D_Z"] = side_beta_z
        results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["R"] = oblique_r + 0.5 * (side_alpha_r + side_beta_r)
        results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["Z"] = oblique_z + 0.5 * (side_alpha_z + side_beta_z)

    if len(pressure_sensors.keys()) > 0:
        sensor_names = list(pressure_sensors.keys())

        # Per-sensor nodes (keyed by sensor name)
        for sensor_name in sensor_names:
            results["CONSTRAINTS"]["PRESSURE"][sensor_name]["MEASURED"] = pressure_sensors.get_array1(
                [sensor_name, "pressure", "measured", "value"]
            )  # shape = [n_time]
            results["CONSTRAINTS"]["PRESSURE"][sensor_name]["RECONSTRUCT"] = pressure_sensors.get_array1(
                [sensor_name, "pressure", "calculated", "value"]
            )  # shape = [n_time]
            results["CONSTRAINTS"]["PRESSURE"][sensor_name]["WEIGHT"] = pressure_sensors.get_f64([sensor_name, "fit_settings", "weight"])  # scalar
            results["CONSTRAINTS"]["PRESSURE"][sensor_name]["POSITION"]["R"] = pressure_sensors.get_f64([sensor_name, "geometry", "r"])  # scalar
            results["CONSTRAINTS"]["PRESSURE"][sensor_name]["POSITION"]["Z"] = pressure_sensors.get_f64([sensor_name, "geometry", "z"])  # scalar
            results["CONSTRAINTS"]["PRESSURE"][sensor_name]["POSITION"]["PSI"] = pressure_sensors.get_array1(
                [sensor_name, "pressure", "calculated", "psi"]
            )  # shape = [n_time]

        # ALL aggregate node
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["MEASURED"] = pressure_sensors.get_array2(
            ["*", "pressure", "measured", "value"]
        )  # shape = [n_time, n_points]
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["RECONSTRUCT"] = pressure_sensors.get_array2(
            ["*", "pressure", "calculated", "value"]
        )  # shape = [n_time, n_points]
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["WEIGHT"] = pressure_sensors.get_array1(["*", "fit_settings", "weight"])  # shape = [n_points]
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["POSITION"]["R"] = pressure_sensors.get_array1(["*", "geometry", "r"])  # shape = [n_points]
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["POSITION"]["Z"] = pressure_sensors.get_array1(["*", "geometry", "z"])  # shape = [n_points]
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["POSITION"]["PSI"] = pressure_sensors.get_array2(
            ["*", "pressure", "calculated", "psi"]
        )  # shape = [n_time, n_points]
        results["CONSTRAINTS"]["PRESSURE"]["ALL"]["NAMES"] = np.array(sensor_names)

    # Store "WORKFLOW": which run of each MDSplus tree the database reader read, from the `workflow.xml` which each IDS's file links to
    for tree_name, workflow_tree in workflow_from_settings(gsfit_controller.settings_xml).items():
        # A nil `pulseNo` reads the pulse being reconstructed
        if workflow_tree.pulseNo is None:
            results["INPUT"]["WORKFLOW"][tree_name]["PULSE"] = pulseNo
        else:
            results["INPUT"]["WORKFLOW"][tree_name]["PULSE"] = workflow_tree.pulseNo
        results["INPUT"]["WORKFLOW"][tree_name]["RUN"] = workflow_tree.run_name
        results["INPUT"]["WORKFLOW"][tree_name]["USAGE"] = workflow_tree.usage
