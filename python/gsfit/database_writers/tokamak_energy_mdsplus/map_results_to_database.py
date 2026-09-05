import typing
from typing import TYPE_CHECKING

import numpy as np
from gsfit_rs.imas import equilibrium_paths as ep

# from st40_database import GetData

if TYPE_CHECKING:
    from ...gsfit import Gsfit
    from . import DatabaseWriterTokamakEnergyMDSplus


# Pairs of (MDSplus node path, IMAS path) for everything which comes out of the `equilibrium` IDS
# one value per time-slice. See the same tables in `tokamak_energy_mdsplus_new`; this writer maps
# the same quantities onto the older MDSplus node layout.
_TIME_SERIES_PATH_PAIRS: list[tuple[tuple[str, ...], typing.Any]] = [
    # Two-d. `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    (("TWO_D", "BR"), ep.time_slice[:].profiles_2d[0].b_field_r),
    (("TWO_D", "BT"), ep.time_slice[:].profiles_2d[0].b_field_phi),
    (("TWO_D", "BZ"), ep.time_slice[:].profiles_2d[0].b_field_z),
    (("TWO_D", "MASK"), ep.time_slice[:].profiles_2d[0].mask),
    (("TWO_D", "P"), ep.time_slice[:].profiles_2d[0].pressure),
    (("TWO_D", "PSI"), ep.time_slice[:].profiles_2d[0].psi),
    # Global
    (("GLOBAL", "BETA_N"), ep.time_slice[:].global_quantities.beta_tor_norm),
    (("GLOBAL", "BETA_P_1"), ep.time_slice[:].global_quantities.beta_pol_1),
    (("GLOBAL", "BETA_P_2"), ep.time_slice[:].global_quantities.beta_pol_2),
    (("GLOBAL", "BETA_P_3"), ep.time_slice[:].global_quantities.beta_pol_3),
    (("GLOBAL", "BT_VAC_RGEO"), ep.time_slice[:].global_quantities.bt_vac_at_r_geo),
    (("GLOBAL", "CHI_MAG"), ep.time_slice[:].constraints.chi_squared_reduced),
    (("GLOBAL", "LI_1"), ep.time_slice[:].global_quantities.li_1),
    (("GLOBAL", "LI_2"), ep.time_slice[:].global_quantities.li_2),
    (("GLOBAL", "LI_3"), ep.time_slice[:].global_quantities.li_3),
    (("GLOBAL", "DELTA_Z"), ep.time_slice[:].convergence.delta_z),
    (("GLOBAL", "ELON"), ep.time_slice[:].boundary.elongation),
    (("GLOBAL", "PHI_DIA"), ep.time_slice[:].constraints.diamagnetic_flux.reconstructed),
    (("GLOBAL", "GS_ERROR"), ep.time_slice[:].convergence.grad_shafranov_deviation_value),
    (("GLOBAL", "I_ROD"), ep.time_slice[:].global_quantities.i_rod),
    (("GLOBAL", "IP"), ep.time_slice[:].global_quantities.ip),
    (("GLOBAL", "N_ITER"), ep.time_slice[:].convergence.iterations_n),
    (("GLOBAL", "P"), ep.time_slice[:].global_quantities.pressure_2d_sum),
    (("GLOBAL", "PSI_A"), ep.time_slice[:].global_quantities.psi_magnetic_axis),
    (("GLOBAL", "PSI_B"), ep.time_slice[:].boundary.psi),
    (("GLOBAL", "Q0"), ep.time_slice[:].global_quantities.q_axis),
    (("GLOBAL", "Q95"), ep.time_slice[:].global_quantities.q_95),
    (("GLOBAL", "R_CUR"), ep.time_slice[:].global_quantities.current_centre.r),
    (("GLOBAL", "Z_CUR"), ep.time_slice[:].global_quantities.current_centre.z),
    (("GLOBAL", "R_GEO"), ep.time_slice[:].boundary.geometric_axis.r),
    (("GLOBAL", "Z_GEO"), ep.time_slice[:].boundary.geometric_axis.z),
    (("GLOBAL", "R_MAG"), ep.time_slice[:].global_quantities.magnetic_axis.r),
    (("GLOBAL", "Z_MAG"), ep.time_slice[:].global_quantities.magnetic_axis.z),
    (("GLOBAL", "R_MINOR"), ep.time_slice[:].boundary.minor_radius),
    (("GLOBAL", "V_LOOP"), ep.time_slice[:].global_quantities.v_loop),
    (("GLOBAL", "VPLASMA"), ep.time_slice[:].global_quantities.volume),
    (("GLOBAL", "W_MHD"), ep.time_slice[:].global_quantities.energy_mhd),
    (("GLOBAL", "XPT_DIVERTED"), ep.time_slice[:].boundary.type),
    # Plasma boundary
    (("P_BOUNDARY", "RBND"), ep.time_slice[:].boundary.outline.r),
    (("P_BOUNDARY", "ZBND"), ep.time_slice[:].boundary.outline.z),
    (("P_BOUNDARY", "BOUNDING_R"), ep.time_slice[:].boundary.bounding.r),
    (("P_BOUNDARY", "BOUNDING_Z"), ep.time_slice[:].boundary.bounding.z),
    # Profiles, on the psi_norm grid
    (("PROFILES", "RHO", "AREA"), ep.time_slice[:].profiles_1d.area),
    (("PROFILES", "RHO", "AREA_PRIME"), ep.time_slice[:].profiles_1d.darea_dpsi),
    (("PROFILES", "RHO", "F"), ep.time_slice[:].profiles_1d.f),
    (("PROFILES", "RHO", "FF_PRIME"), ep.time_slice[:].profiles_1d.f_df_dpsi),
    (("PROFILES", "RHO", "FLUX_TOR"), ep.time_slice[:].profiles_1d.phi),
    (("PROFILES", "RHO", "P"), ep.time_slice[:].profiles_1d.pressure),
    (("PROFILES", "RHO", "P_PRIME"), ep.time_slice[:].profiles_1d.dpressure_dpsi),
    (("PROFILES", "RHO", "Q"), ep.time_slice[:].profiles_1d.q),
    (("PROFILES", "RHO", "RHO_POL"), ep.time_slice[:].profiles_1d.rho_pol),
    (("PROFILES", "RHO", "RHO_TOR"), ep.time_slice[:].profiles_1d.rho_tor),
    (("PROFILES", "RHO", "VOL"), ep.time_slice[:].profiles_1d.volume),
    (("PROFILES", "RHO", "VOL_PRIME"), ep.time_slice[:].profiles_1d.dvolume_dpsi),
    (("PROFILES", "R_MIDPLANE", "P"), ep.time_slice[:].profiles_r_midplane.pressure),
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

# The same, for quantities which are the same on every time-slice
_TIME_INDEPENDENT_PATH_PAIRS: list[tuple[tuple[str, ...], typing.Any]] = [
    (("TWO_D", "RGRID"), ep.time_slice[:].profiles_2d[0].grid.dim1),
    (("TWO_D", "ZGRID"), ep.time_slice[:].profiles_2d[0].grid.dim2),
    (("PROFILES", "RHO", "PSI_N"), ep.time_slice[:].profiles_1d.psi_norm),
    (("PROFILES", "R_MIDPLANE", "R"), ep.time_slice[:].profiles_r_midplane.r),
]


def _assign(results: typing.Any, mdsplus_path: tuple[str, ...], value: typing.Any) -> None:
    """Assign `value` into the nested `results` object at `mdsplus_path`."""
    node = results
    for key in mdsplus_path[:-1]:
        node = node[key]
    node[mdsplus_path[-1]] = value


def _n_points_per_time(padded: "np.ndarray") -> "np.ndarray":
    """Number of real points in each row of a NaN-padded `[n_time, n_points]` array.

    The IDS stores one contour per time-slice, each its own length; gathering over time pads the
    short rows with NaN to make a rectangle. MDSplus needs that rectangle plus a separate count of
    how much of each row is real. Contour coordinates are always finite, so a NaN can only be
    padding.
    """
    return np.isfinite(padded).sum(axis=1).astype(np.int32)


def map_results_to_database(
    self: "DatabaseWriterTokamakEnergyMDSplus",
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
    rogowski_coils = gsfit_controller.rogowski_coils
    passives = gsfit_controller.passives
    results = gsfit_controller.results
    pressure_sensors = gsfit_controller.pressure_sensors

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

    results["SOL"]["HFS"]["CONTOUR"]["N"] = _n_points_per_time(results["SOL"]["HFS"]["CONTOUR"]["R"])
    results["SOL"]["LFS"]["CONTOUR"]["N"] = _n_points_per_time(results["SOL"]["LFS"]["CONTOUR"]["R"])

    # The data dictionary defines `beta_tor` as a fraction, but this MDSplus node has always held a
    # percentage, so the factor of 100 is put back here rather than changing what consumers read
    results["GLOBAL"]["BETA_T"] = 100.0 * np.asarray(equilibrium_ids.get(ep.time_slice[:].global_quantities.beta_tor))


    # Bp probes (note, this is all the sensors, both the ones we fit and the ones we don't)
    bp_names = bp_probes.keys()  # list of strings; len(bp_names) = n_sensors
    bp_names = [bp_name.replace("P", "B_BPPROBE_") for bp_name in bp_names]
    results["CONSTRAINTS"]["BPPROBE"]["NAME"] = np.array(bp_names)  # MDSplus requires numpy objects, not lists of strings; shape = [n_sensors]
    results["CONSTRAINTS"]["BPPROBE"]["CVALUE"] = bp_probes.get_array2(["*", "b", "calculated", "value"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["BPPROBE"]["INCLUDE"] = np.array(bp_probes.get_vec_bool(["*", "fit_settings", "include"])).astype(np.int32)
    results["CONSTRAINTS"]["BPPROBE"]["MVALUE"] = bp_probes.get_array2(["*", "b", "measured", "value"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["BPPROBE"]["WEIGHT"] = bp_probes.get_array1(["*", "fit_settings", "weight"])  # shape = [n_sensors]

    # Flux loops (note, this is all the sensors, both the ones we fit and the ones we don't)
    fl_names = flux_loops.keys()  # list of strings; len(fl_names) = n_sensors
    fl_names = [fl_name.replace("L", "PSI_FLOOP_") for fl_name in fl_names]
    results["CONSTRAINTS"]["FLOOP"]["NAME"] = np.array(fl_names)  # MDSplus requires numpy objects, not lists of strings; shape = [n_sensors]
    results["CONSTRAINTS"]["FLOOP"]["CVALUE"] = flux_loops.get_array2(["*", "psi", "calculated", "value"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["FLOOP"]["INCLUDE"] = np.array(flux_loops.get_vec_bool(["*", "fit_settings", "include"])).astype(np.int32)
    results["CONSTRAINTS"]["FLOOP"]["MVALUE"] = flux_loops.get_array2(["*", "psi", "measured", "value"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["FLOOP"]["WEIGHT"] = flux_loops.get_array1(["*", "fit_settings", "weight"])  # shape = [n_sensors]

    # Rogowski coils (note, this is all the sensors, both the ones we fit and the ones we don't)
    rog_names = rogowski_coils.keys()  # list of strings; len(rog_names) = n_sensors
    rog_names = [f"I_ROG_{rog_name}" for rog_name in rog_names]
    results["CONSTRAINTS"]["ROG"]["NAME"] = np.array(rog_names)  # MDSplus requires numpy objects, not lists of strings; shape = [n_sensors]
    results["CONSTRAINTS"]["ROG"]["CVALUE"] = rogowski_coils.get_array2(["*", "i", "calculated", "value"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["ROG"]["INCLUDE"] = np.array(rogowski_coils.get_vec_bool(["*", "fit_settings", "include"])).astype(np.int32)
    results["CONSTRAINTS"]["ROG"]["MVALUE"] = rogowski_coils.get_array2(["*", "i", "measured", "value"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["ROG"]["WEIGHT"] = rogowski_coils.get_array1(["*", "fit_settings", "weight"])  # shape = [n_sensors]

    # Plasma boundary
    # The IDS stores each contour at its own length and the gather pads them into a rectangle;
    # MDSplus wants the rectangle plus the count of real points in each row
    results["P_BOUNDARY"]["NBND"] = _n_points_per_time(results["P_BOUNDARY"]["RBND"])

    # X-points
    # TODO: the upper and lower x-points came from `gs_solution.rs`, which has been removed. The
    # solver does not put them on the IDS, so there is nowhere to read them from; these nodes are
    # left unwritten until the solver stores them
    # results["XPOINTS"]["UPPER"]["R"] = ...
    # results["XPOINTS"]["UPPER"]["Z"] = ...
    # results["XPOINTS"]["LOWER"]["R"] = ...
    # results["XPOINTS"]["LOWER"]["Z"] = ...


    # Passives
    for passive_name in passives.keys():
        if passive_name == "IVC":
            results["PASSIVES"]["IVC"]["DOF"]["EIG_01"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_01", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_01"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_01", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_02"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_02", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_02"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_02", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_03"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_03", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_03"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_03", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_04"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_04", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_04"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_04", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_05"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_05", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_05"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_05", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_06"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_06", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_06"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_06", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_07"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_07", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_07"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_07", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_08"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_08", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_08"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_08", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_09"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_09", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_09"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_09", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_10"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_10", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_10"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_10", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_11"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_11", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_11"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_11", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_12"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_12", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_12"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_12", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_13"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_13", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_13"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_13", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_14"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_14", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_14"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_14", "current_distribution"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_15"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_15", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_15"]["I_DIST"] = passives.get_array1(["IVC", "dof", "eig_15", "current_distribution"])
            results["PASSIVES"]["IVC"]["GEOMETRY"]["ANGLE_1"] = passives.get_array1(["IVC", "geometry", "angle_1"])
            results["PASSIVES"]["IVC"]["GEOMETRY"]["ANGLE_2"] = passives.get_array1(["IVC", "geometry", "angle_2"])
            results["PASSIVES"]["IVC"]["GEOMETRY"]["D_R"] = passives.get_array1(["IVC", "geometry", "d_r"])
            results["PASSIVES"]["IVC"]["GEOMETRY"]["D_Z"] = passives.get_array1(["IVC", "geometry", "d_z"])
            results["PASSIVES"]["IVC"]["GEOMETRY"]["R"] = passives.get_array1(["IVC", "geometry", "r"])
            results["PASSIVES"]["IVC"]["GEOMETRY"]["Z"] = passives.get_array1(["IVC", "geometry", "z"])
        else:
            results["PASSIVES"][passive_name]["DOF"]["CONSTANT_J"]["CVALUE"] = passives.get_array1(
                [passive_name, "dof", "constant_current_density", "calculated"]
            )
            results["PASSIVES"][passive_name]["DOF"]["CONSTANT_J"]["I_DIST"] = passives.get_array1(
                [passive_name, "dof", "constant_current_density", "current_distribution"]
            )
            results["PASSIVES"][passive_name]["GEOMETRY"]["ANGLE_1"] = passives.get_array1([passive_name, "geometry", "angle_1"])
            results["PASSIVES"][passive_name]["GEOMETRY"]["ANGLE_2"] = passives.get_array1([passive_name, "geometry", "angle_2"])
            results["PASSIVES"][passive_name]["GEOMETRY"]["D_R"] = passives.get_array1([passive_name, "geometry", "d_r"])
            results["PASSIVES"][passive_name]["GEOMETRY"]["D_Z"] = passives.get_array1([passive_name, "geometry", "d_z"])
            results["PASSIVES"][passive_name]["GEOMETRY"]["R"] = passives.get_array1([passive_name, "geometry", "r"])
            results["PASSIVES"][passive_name]["GEOMETRY"]["Z"] = passives.get_array1([passive_name, "geometry", "z"])

    # Scrape off layer (SOL)

    if len(pressure_sensors.keys()) > 0:
        results["CONSTRAINTS"]["PRESSURE"]["RECONSTRUCTED"] = pressure_sensors.get_array2(
            ["*", "pressure", "calculated", "value"]
        )  # shape = [n_time, n_points]
        results["CONSTRAINTS"]["PRESSURE"]["MEASURED"] = pressure_sensors.get_array2(["*", "pressure", "measured", "value"])  # shape = [n_time, n_points]
        results["CONSTRAINTS"]["PRESSURE"]["WEIGHT"] = pressure_sensors.get_array1(["*", "fit_settings", "weight"])  # shape = [n_points]
        results["CONSTRAINTS"]["PRESSURE"]["POSITION"]["R"] = pressure_sensors.get_array1(["*", "geometry", "r"])  # shape = [n_points]
        results["CONSTRAINTS"]["PRESSURE"]["POSITION"]["Z"] = pressure_sensors.get_array1(["*", "geometry", "z"])  # shape = [n_points]
        results["CONSTRAINTS"]["PRESSURE"]["POSITION"]["PSI"] = pressure_sensors.get_array2(
            ["*", "pressure", "calculated", "psi"]
        )  # shape = [n_time, n_points]

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
