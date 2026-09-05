import typing
from typing import TYPE_CHECKING

import numpy as np
from gsfit_rs.imas import equilibrium_paths as ep

# from st40_database import GetData

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
    # Global
    (("GLOBAL", "CURRENT_CENT", "R"), ep.time_slice[:].global_quantities.current_centre.r),
    (("GLOBAL", "CURRENT_CENT", "Z"), ep.time_slice[:].global_quantities.current_centre.z),
    (("GLOBAL", "MAG_AXIS", "R"), ep.time_slice[:].global_quantities.magnetic_axis.r),
    (("GLOBAL", "MAG_AXIS", "Z"), ep.time_slice[:].global_quantities.magnetic_axis.z),
    (("GLOBAL", "AREA"), ep.time_slice[:].global_quantities.area),
    (("GLOBAL", "BETA_N"), ep.time_slice[:].global_quantities.beta_tor_norm),
    (("GLOBAL", "BETA_P_1"), ep.time_slice[:].global_quantities.beta_pol_1),
    (("GLOBAL", "BETA_P_2"), ep.time_slice[:].global_quantities.beta_pol_2),
    (("GLOBAL", "BETA_P_3"), ep.time_slice[:].global_quantities.beta_pol_3),
    (("GLOBAL", "BT_VAC_RGEO"), ep.time_slice[:].global_quantities.bt_vac_at_r_geo),
    (("GLOBAL", "DELTA_Z"), ep.time_slice[:].convergence.delta_z),
    (("GLOBAL", "ENERGY_MHD"), ep.time_slice[:].global_quantities.energy_mhd),
    (("GLOBAL", "IP"), ep.time_slice[:].global_quantities.ip),
    (("GLOBAL", "I_ROD"), ep.time_slice[:].global_quantities.i_rod),
    (("GLOBAL", "LI_1"), ep.time_slice[:].global_quantities.li_1),
    (("GLOBAL", "LI_2"), ep.time_slice[:].global_quantities.li_2),
    (("GLOBAL", "LI_3"), ep.time_slice[:].global_quantities.li_3),
    (("GLOBAL", "PHI_DIA"), ep.time_slice[:].constraints.diamagnetic_flux.reconstructed),
    (("GLOBAL", "PSI_MAG_AXIS"), ep.time_slice[:].global_quantities.psi_magnetic_axis),
    (("GLOBAL", "Q_AXIS"), ep.time_slice[:].global_quantities.q_axis),
    (("GLOBAL", "Q_95"), ep.time_slice[:].global_quantities.q_95),
    (("GLOBAL", "V_LOOP"), ep.time_slice[:].global_quantities.v_loop),
    (("GLOBAL", "VOLUME"), ep.time_slice[:].global_quantities.volume),
    # Profiles_1d, on the psi_norm grid
    (("PROFILES_1D", "PSI_NORM", "AREA"), ep.time_slice[:].profiles_1d.area),
    (("PROFILES_1D", "PSI_NORM", "AREA_PRIME"), ep.time_slice[:].profiles_1d.darea_dpsi),
    (("PROFILES_1D", "PSI_NORM", "F"), ep.time_slice[:].profiles_1d.f),
    (("PROFILES_1D", "PSI_NORM", "FF_PRIME"), ep.time_slice[:].profiles_1d.f_df_dpsi),
    (("PROFILES_1D", "PSI_NORM", "FLUX_TOR"), ep.time_slice[:].profiles_1d.phi),
    (("PROFILES_1D", "PSI_NORM", "P_PRIME"), ep.time_slice[:].profiles_1d.dpressure_dpsi),
    (("PROFILES_1D", "PSI_NORM", "PRESSURE"), ep.time_slice[:].profiles_1d.pressure),
    (("PROFILES_1D", "PSI_NORM", "Q"), ep.time_slice[:].profiles_1d.q),
    (("PROFILES_1D", "PSI_NORM", "RHO_POL"), ep.time_slice[:].profiles_1d.rho_pol),
    (("PROFILES_1D", "PSI_NORM", "RHO_TOR"), ep.time_slice[:].profiles_1d.rho_tor),
    (("PROFILES_1D", "PSI_NORM", "VOL"), ep.time_slice[:].profiles_1d.volume),
    (("PROFILES_1D", "PSI_NORM", "VOL_PRIME"), ep.time_slice[:].profiles_1d.dvolume_dpsi),
    # Mid-plane profiles
    (("PROFILES_1D", "R_MIDPLANE", "PRESSURE"), ep.time_slice[:].profiles_r_midplane.pressure),
    # Profiles_2d. `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    (("PROFILES_2D", "R_Z", "B_FIELD_PHI"), ep.time_slice[:].profiles_2d[0].b_field_phi),
    (("PROFILES_2D", "R_Z", "B_FIELD_R"), ep.time_slice[:].profiles_2d[0].b_field_r),
    (("PROFILES_2D", "R_Z", "B_FIELD_Z"), ep.time_slice[:].profiles_2d[0].b_field_z),
    (("PROFILES_2D", "R_Z", "MASK"), ep.time_slice[:].profiles_2d[0].mask),
    (("PROFILES_2D", "R_Z", "PRESSURE"), ep.time_slice[:].profiles_2d[0].pressure),
    (("PROFILES_2D", "R_Z", "PSI"), ep.time_slice[:].profiles_2d[0].psi),
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
    (("PROFILES_1D", "R_MIDPLANE", "R"), ep.time_slice[:].profiles_r_midplane.r),
    (("PROFILES_2D", "R_Z", "R"), ep.time_slice[:].profiles_2d[0].grid.dim1),
    (("PROFILES_2D", "R_Z", "Z"), ep.time_slice[:].profiles_2d[0].grid.dim2),
]


def _assign(results: typing.Any, mdsplus_path: tuple[str, ...], value: typing.Any) -> None:
    """Assign `value` into the nested `results` object at `mdsplus_path`."""
    node = results
    for key in mdsplus_path[:-1]:
        node = node[key]
    node[mdsplus_path[-1]] = value


def _gather_int(
    equilibrium_ids: typing.Any,
    node_for_slice: typing.Callable[[int], typing.Any],
    n_time: int,
    unset: int,
) -> "np.ndarray":
    """Gather an integer node over time, substituting `unset` where the IDS holds no value.

    Integers have no NaN, so gathering with `time_slice[:]` refuses a node which is unset on any
    time-slice rather than inventing a sentinel. A time-slice which did not converge has neither an
    iteration count nor a boundary type, so those are read one slice at a time and the sentinel is
    chosen here, where it is visible.
    """
    values: "np.ndarray" = np.full(n_time, unset, dtype=np.int32)
    for i_time in range(n_time):
        value = equilibrium_ids.get(node_for_slice(i_time))
        if value is not None:
            values[i_time] = value
    return values


def _n_points_per_time(padded: "np.ndarray") -> "np.ndarray":
    """Number of real points in each row of a NaN-padded `[n_time, n_points]` array.

    The IDS stores one contour per time-slice, each its own length; gathering over time pads the
    short rows with NaN to make a rectangle. MDSplus needs that rectangle plus a separate count of
    how much of each row is real, which is what this recovers. Contour coordinates are always
    finite, so a NaN can only be padding.
    """
    return np.isfinite(padded).sum(axis=1).astype(np.int32)


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
    settings = gsfit_controller.settings
    plasma = gsfit_controller.plasma
    bp_probes = gsfit_controller.bp_probes
    flux_loops = gsfit_controller.flux_loops
    dialoop = gsfit_controller.dialoop
    rogowski_coils = gsfit_controller.rogowski_coils
    passives = gsfit_controller.passives
    coils = gsfit_controller.coils
    pressure_sensors = gsfit_controller.pressure_sensors
    results = gsfit_controller.results

    # Everything below comes out of the `equilibrium` IDS through the (MDSplus path, IMAS path)
    # tables at the top of this file.
    #
    # Read once and reused: the `equilibrium_ids` getter copies the whole IDS out of Rust, so
    # calling it per quantity would copy every 2D profile on every time-slice, once each
    equilibrium_ids = plasma.equilibrium_ids

    for mdsplus_path, imas_path in _TIME_SERIES_PATH_PAIRS:
        _assign(results, mdsplus_path, equilibrium_ids.get(imas_path))

    for mdsplus_path, imas_path in _TIME_INDEPENDENT_PATH_PAIRS:
        _assign(results, mdsplus_path, np.asarray(equilibrium_ids.get(imas_path))[0])

    # The rest need something done to them on the way out, so they are not in the tables

    # Flux defining the LCFS. SPIDER has `psi_norm = 0.9999`
    results["BOUNDARY"]["PSI_NORM"] = np.ones_like(results["BOUNDARY"]["PSI"])

    # Contour lengths. The IDS stores each contour at its own length and the gather pads them into
    # a rectangle; MDSplus wants the rectangle plus the count of real points in each row
    results["BOUNDARY"]["OUTLINE"]["N"] = _n_points_per_time(results["BOUNDARY"]["OUTLINE"]["R"])
    results["SOL"]["HFS"]["CONTOUR"]["N"] = _n_points_per_time(results["SOL"]["HFS"]["CONTOUR"]["R"])
    results["SOL"]["LFS"]["CONTOUR"]["N"] = _n_points_per_time(results["SOL"]["LFS"]["CONTOUR"]["R"])

    # Integer nodes. A time-slice which did not converge has no iteration count and no boundary
    # type. `-1` is what the old `DataTree` path wrote for the iteration count, because its
    # `usize::MAX` wraps to `-1` on the cast to `int32`; `0` matches its `xpt_diverted = false`
    n_time: int = len(equilibrium_ids)
    results["CONVERGENCE"]["ITERATIONS_N"] = _gather_int(
        equilibrium_ids, lambda i_time: ep.time_slice[i_time].convergence.iterations_n, n_time, unset=-1
    )
    results["GLOBAL"]["XPT_DIVERTED"] = _gather_int(equilibrium_ids, lambda i_time: ep.time_slice[i_time].boundary.type, n_time, unset=0)

    # The data dictionary defines `beta_tor` as a fraction, but this MDSplus node has always held a
    # percentage, so the factor of 100 is put back here rather than changing what consumers read
    results["GLOBAL"]["BETA_T"] = 100.0 * np.asarray(equilibrium_ids.get(ep.time_slice[:].global_quantities.beta_tor))


    for sensor_name in bp_probes.keys():
        # results["CONSTRAINTS"]["BP_PROBE"][pf_name]["EXACT"]
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["INCLUDE"] = np.int32(bp_probes.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["MEASURED"] = bp_probes.get_array1([sensor_name, "b", "measured", "value"])
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["RECONSTRUCT"] = bp_probes.get_array1([sensor_name, "b", "calculated", "value"])
        results["CONSTRAINTS"]["BP_PROBE"][sensor_name]["WEIGHT"] = bp_probes.get_f64([sensor_name, "fit_settings", "weight"])

    for sensor_name in flux_loops.keys():
        # results["CONSTRAINTS"]["FLUX_LOOP"][pf_name]["EXACT"]
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["INCLUDE"] = np.int32(flux_loops.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["MEASURED"] = flux_loops.get_array1([sensor_name, "psi", "measured", "value"])
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["RECONSTRUCT"] = flux_loops.get_array1([sensor_name, "psi", "calculated", "value"])
        results["CONSTRAINTS"]["FLUX_LOOP"][sensor_name]["WEIGHT"] = flux_loops.get_f64([sensor_name, "fit_settings", "weight"])

    for sensor_name in rogowski_coils.keys():
        # results["CONSTRAINTS"]["ROGOWSKI"][pf_name]["EXACT"]
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["INCLUDE"] = np.int32(rogowski_coils.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["MEASURED"] = rogowski_coils.get_array1([sensor_name, "i", "measured", "value"])
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["RECONSTRUCT"] = rogowski_coils.get_array1([sensor_name, "i", "calculated", "value"])
        results["CONSTRAINTS"]["ROGOWSKI"][sensor_name]["WEIGHT"] = rogowski_coils.get_f64([sensor_name, "fit_settings", "weight"])

    # Diamagnetic flux (single diamagnetic flux loop "DIALOOP")
    for sensor_name in dialoop.keys():
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["INCLUDE"] = np.int32(dialoop.get_bool([sensor_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["MEASURED"] = dialoop.get_array1([sensor_name, "b", "measured", "value"])
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["RECONSTRUCT"] = dialoop.get_array1([sensor_name, "b", "calculated", "value"])
        results["CONSTRAINTS"]["DIAMAG_FLUX"]["WEIGHT"] = dialoop.get_f64([sensor_name, "fit_settings", "weight"])

    for pf_name in coils.keys(["pf"]):
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["EXACT"]
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["INCLUDE"] = np.int32(coils.get_bool(["pf", pf_name, "fit_settings", "include"]))
        results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["MEASURED"] = coils.get_array1(["pf", pf_name, "i", "measured", "value"])
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["RECONSTRUCT"] = coils.get_array1(["pf", pf_name, "i", "calculated", "value"])
        # results["CONSTRAINTS"]["PF_CURRENT"][pf_name]["WEIGHT"] = coils.get_f64(["pf", pf_name, "fit_settings", "weight"])
    # TODO: need to handle circuits vs coils better
    results["CONSTRAINTS"]["PF_CURRENT"]["BVL"]["MEASURED"] = coils.get_array1(["pf", "BVLT", "i", "measured", "value"])
    results["CONSTRAINTS"]["PF_CURRENT"]["DIV"]["MEASURED"] = coils.get_array1(["pf", "DIVT", "i", "measured", "value"])
    results["CONSTRAINTS"]["PF_CURRENT"]["PSH"]["MEASURED"] = coils.get_array1(["pf", "PSHT", "i", "measured", "value"])

    # # X-points
    # # TODO!!!!!!!!!!!
    # results["XPOINTS"]["UPPER"]["R"] = plasma.get_array1(["xpoints", "upper", "r"])
    # results["XPOINTS"]["UPPER"]["Z"] = plasma.get_array1(["xpoints", "upper", "z"])
    # results["XPOINTS"]["LOWER"]["R"] = plasma.get_array1(["xpoints", "lower", "r"])
    # results["XPOINTS"]["LOWER"]["Z"] = plasma.get_array1(["xpoints", "lower", "z"])

    # Passives
    for passive_name in passives.keys():
        if passive_name == "IVC":
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_01"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_01", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_01"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_01", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_02"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_02", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_02"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_02", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_03"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_03", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_03"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_03", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_04"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_04", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_04"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_04", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_05"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_05", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_05"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_05", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_06"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_06", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_06"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_06", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_07"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_07", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_07"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_07", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_08"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_08", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_08"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_08", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_09"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_09", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_09"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_09", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_10"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_10", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_10"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_10", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_11"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_11", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_11"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_11", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_12"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_12", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_12"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_12", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_13"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_13", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_13"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_13", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_14"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_14", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_14"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_14", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_15"]["RECONSTRUCT"] = passives.get_array1(["IVC", "dof", "eig_15", "calculated"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["DOF"]["EIG_15"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_15", "current_distribution"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["GEOMETRY"]["ANGLE_1"] = passives.get_array1(["IVC", "geometry", "angle_1"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["GEOMETRY"]["ANGLE_2"] = passives.get_array1(["IVC", "geometry", "angle_2"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["GEOMETRY"]["D_R"] = passives.get_array1(["IVC", "geometry", "d_r"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["GEOMETRY"]["D_Z"] = passives.get_array1(["IVC", "geometry", "d_z"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["GEOMETRY"]["R"] = passives.get_array1(["IVC", "geometry", "r"])
            results["CONSTRAINTS"]["PF_PASSIVE"]["IVC"]["GEOMETRY"]["Z"] = passives.get_array1(["IVC", "geometry", "z"])
        else:
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["DOF"]["CONSTANT_J"]["RECONSTRUCT"] = passives.get_array1(
                [passive_name, "dof", "constant_current_density", "calculated"]
            )
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["DOF"]["CONSTANT_J"]["I_DIST"] = passives.get_array1(
                [passive_name, "dof", "constant_current_density", "current_distribution"]
            )
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["ANGLE_1"] = passives.get_array1([passive_name, "geometry", "angle_1"])
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["ANGLE_2"] = passives.get_array1([passive_name, "geometry", "angle_2"])
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["D_R"] = passives.get_array1([passive_name, "geometry", "d_r"])
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["D_Z"] = passives.get_array1([passive_name, "geometry", "d_z"])
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["R"] = passives.get_array1([passive_name, "geometry", "r"])
            results["CONSTRAINTS"]["PF_PASSIVE"][passive_name]["GEOMETRY"]["Z"] = passives.get_array1([passive_name, "geometry", "z"])


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

    # Store "WORKFLOW"
    database_reader_method = settings["GSFIT_code_settings.json"]["database_reader"]["method"]

    code_names = settings["GSFIT_code_settings.json"]["database_reader"][database_reader_method]["workflow"].keys()

    for code_name in code_names:
        pulseNo_json = settings["GSFIT_code_settings.json"]["database_reader"][database_reader_method]["workflow"][code_name]["pulseNo"]
        if pulseNo_json is not None:
            results["INPUT"]["WORKFLOW"][code_name]["PULSE"] = pulseNo_json
        else:
            results["INPUT"]["WORKFLOW"][code_name]["PULSE"] = pulseNo

        run_name = settings["GSFIT_code_settings.json"]["database_reader"][database_reader_method]["workflow"][code_name]["run_name"]
        results["INPUT"]["WORKFLOW"][code_name]["RUN"] = run_name

        usage = settings["GSFIT_code_settings.json"]["database_reader"][database_reader_method]["workflow"][code_name]["usage"]
        results["INPUT"]["WORKFLOW"][code_name]["USAGE"] = usage
