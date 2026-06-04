from typing import TYPE_CHECKING

import numpy as np

# from st40_database import GetData

if TYPE_CHECKING:
    from ...gsfit import Gsfit
    from . import DatabaseWriterTokamakEnergyMDSplusNew


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
    rogowski_coils = gsfit_controller.rogowski_coils
    passives = gsfit_controller.passives
    coils = gsfit_controller.coils
    results = gsfit_controller.results

    # Plasma boundary
    results["BOUNDARY"]["GEO_AXIS"]["R"] = plasma.get_array1(["global", "r_geo"])
    results["BOUNDARY"]["GEO_AXIS"]["Z"] = plasma.get_array1(["global", "z_geo"])
    results["BOUNDARY"]["MINOR_RADIUS"] = plasma.get_array1(["global", "r_minor"])
    results["BOUNDARY"]["ELONGATION"] = plasma.get_array1(["global", "elongation"])
    results["BOUNDARY"]["PSI"] = plasma.get_array1(["global", "psi_b"])
    results["BOUNDARY"]["PSI_NORM"] = np.ones_like(results["BOUNDARY"]["PSI"])  # flux defining LCFS, SPIDER has `psi_norm = 0.9999`
    results["BOUNDARY"]["OUTLINE"]["N"] = np.array(plasma.get_vec_usize(["p_boundary", "nbnd"]))
    results["BOUNDARY"]["OUTLINE"]["R"] = plasma.get_array2(["p_boundary", "rbnd"])
    results["BOUNDARY"]["OUTLINE"]["Z"] = plasma.get_array2(["p_boundary", "zbnd"])
    results["BOUNDARY"]["BOUNDING"]["R"] = plasma.get_array1(["p_boundary", "bounding_r"])
    results["BOUNDARY"]["BOUNDING"]["Z"] = plasma.get_array1(["p_boundary", "bounding_z"])

    # Convergence
    results["CONVERGENCE"]["GS_ERROR"] = plasma.get_array1(["global", "gs_error"])
    results["CONVERGENCE"]["ITERATIONS_N"] = np.array(plasma.get_vec_usize(["global", "n_iter"])).astype(np.int32)

    # Global
    results["GLOBAL"]["CURRENT_CENT"]["R"] = plasma.get_array1(["global", "r_cur"])
    # results["GLOBAL"]["CURRENT_CENT"]["VELOCITY_Z"] = plasma.get_array1(["global", "current_cent", "velocity_z"])
    results["GLOBAL"]["CURRENT_CENT"]["Z"] = plasma.get_array1(["global", "z_cur"])
    # results["GLOBAL"]["MAG_AXIS"]["B_FIELD_PHI"] = plasma.get_array1(["global", "bt_at_r_mag"])
    results["GLOBAL"]["MAG_AXIS"]["R"] = plasma.get_array1(["global", "r_mag"])
    results["GLOBAL"]["MAG_AXIS"]["Z"] = plasma.get_array1(["global", "z_mag"])
    results["GLOBAL"]["AREA"] = plasma.get_array1(["global", "area"])
    results["GLOBAL"]["BETA_N"] = plasma.get_array1(["global", "beta_n"])
    results["GLOBAL"]["BETA_P_1"] = plasma.get_array1(["global", "beta_p_1"])
    results["GLOBAL"]["BETA_P_2"] = plasma.get_array1(["global", "beta_p_2"])
    results["GLOBAL"]["BETA_P_3"] = plasma.get_array1(["global", "beta_p_3"])
    results["GLOBAL"]["BETA_T"] = plasma.get_array1(["global", "beta_t"])
    # results["GLOBAL"]["BP_OMP"] = plasma.get_array1(["global", "bp_omp"])
    results["GLOBAL"]["BT_VAC_RGEO"] = plasma.get_array1(["global", "bt_vac_at_r_geo"])
    # results["GLOBAL"]["CONN_LENGTH"] = plasma.get_array1(["global", "conn_length"])
    # results["GLOBAL"]["DELTA_R_SEP"] = plasma.get_array1(["global", "delta_r_sep"])
    results["GLOBAL"]["DELTA_Z"] = plasma.get_array1(["global", "delta_z"])
    results["GLOBAL"]["ENERGY_MHD"] = plasma.get_array1(["global", "w_mhd"]) # TODO: something wrong with energy calculation
    # results["GLOBAL"]["FX"] = plasma.get_array1(["global", "fx"])
    results["GLOBAL"]["IP"] = plasma.get_array1(["global", "ip"])
    results["GLOBAL"]["I_ROD"] = plasma.get_array1(["global", "i_rod"])
    results["GLOBAL"]["LI_1"] = plasma.get_array1(["global", "li_1"])
    results["GLOBAL"]["LI_2"] = plasma.get_array1(["global", "li_2"])
    results["GLOBAL"]["LI_3"] = plasma.get_array1(["global", "li_3"])
    results["GLOBAL"]["PHI_DIA"] = plasma.get_array1(["global", "phi_dia"])
    results["GLOBAL"]["PSI_MAG_AXIS"] = plasma.get_array1(["global", "psi_a"])
    results["GLOBAL"]["Q_AXIS"] = plasma.get_array1(["global", "q0"])
    results["GLOBAL"]["Q_95"] = plasma.get_array1(["global", "q95"])
    results["GLOBAL"]["V_LOOP"] = plasma.get_array1(["global", "v_loop"])
    results["GLOBAL"]["VOLUME"] = plasma.get_array1(["global", "plasma_volume"]) # TODO: something wrong with plasma volume
    results["GLOBAL"]["XPT_DIVERTED"] = np.array(plasma.get_vec_bool(["global", "xpt_diverted"])).astype(np.int32)
    
    # Profiles_1d, psi_norm
    results["PROFILES_1D"]["PSI_NORM"]["AREA"] = plasma.get_array2(["profiles", "area"])
    results["PROFILES_1D"]["PSI_NORM"]["AREA_PRIME"] = plasma.get_array2(["profiles", "area_prime"])
    # results["PROFILES_1D"]["PSI_NORM"]["ELONGATION"] = plasma.get_array2(["profiles", "elongation"])
    results["PROFILES_1D"]["PSI_NORM"]["F"] = plasma.get_array2(["profiles", "f"])
    results["PROFILES_1D"]["PSI_NORM"]["FF_PRIME"] = plasma.get_array2(["profiles", "ff_prime"])
    results["PROFILES_1D"]["PSI_NORM"]["FLUX_TOR"] = plasma.get_array2(["profiles", "flux_tor"])
    results["PROFILES_1D"]["PSI_NORM"]["P_PRIME"] = plasma.get_array2(["profiles", "p_prime"])
    results["PROFILES_1D"]["PSI_NORM"]["PRESSURE"] = plasma.get_array2(["profiles", "p"])
    # results["PROFILES_1D"]["PSI_NORM"]["PSI"] = plasma.get_array2(["profiles", "psi"])
    results["PROFILES_1D"]["PSI_NORM"]["Q"] = plasma.get_array2(["profiles", "q"])
    results["PROFILES_1D"]["PSI_NORM"]["RHO_POL"] = plasma.get_array2(["profiles", "rho_pol"])
    results["PROFILES_1D"]["PSI_NORM"]["RHO_TOR"] = plasma.get_array2(["profiles", "rho_tor"])
    results["PROFILES_1D"]["PSI_NORM"]["PSI_NORM"] = plasma.get_array1(["profiles", "psi_n"])
    results["PROFILES_1D"]["PSI_NORM"]["VOL"] = plasma.get_array2(["profiles", "vol"])
    results["PROFILES_1D"]["PSI_NORM"]["VOL_PRIME"] = plasma.get_array2(["profiles", "vol_prime"])

    # Mid-plane profiles
    results["PROFILES_1D"]["MID_PLANE"]["PRESSURE"] = plasma.get_array2(["profiles", "mid_plane", "p"])
    results["PROFILES_1D"]["MID_PLANE"]["R"] = plasma.get_array1(["profiles", "mid_plane", "r"])

    # Profiles_2d
    results["PROFILES_2D"]["B_FIELD_PHI"] = plasma.get_array3(["two_d", "bt"])
    results["PROFILES_2D"]["B_FIELD_R"] = plasma.get_array3(["two_d", "br"])
    results["PROFILES_2D"]["B_FIELD_Z"] = plasma.get_array3(["two_d", "bz"])
    results["PROFILES_2D"]["MASK"] = plasma.get_array3(["two_d", "mask"])
    results["PROFILES_2D"]["PRESSURE"] = plasma.get_array3(["two_d", "p"])
    results["PROFILES_2D"]["PSI"] = plasma.get_array3(["two_d", "psi"])
    results["PROFILES_2D"]["R"] = plasma.get_array1(["grid", "r"])
    results["PROFILES_2D"]["Z"] = plasma.get_array1(["grid", "z"])

    # Constraints
    results["CONSTRAINTS"]["CHI_SQ_MAG"] = plasma.get_array1(["global", "chi_mag"])

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
    results["SOL"]["HFS"]["CONTOUR"]["R"] = plasma.get_array2(["sol", "hfs", "contour", "r"])  # shape = [n_time, n_points]
    results["SOL"]["HFS"]["CONTOUR"]["Z"] = plasma.get_array2(["sol", "hfs", "contour", "z"])  # shape = [n_time, n_points]
    results["SOL"]["HFS"]["CONTOUR"]["N"] = np.array(plasma.get_vec_usize(["sol", "hfs", "contour", "n"])).astype(np.int32)  # shape = [n_time]
    results["SOL"]["HFS"]["STRIKE_POINT"]["R"] = plasma.get_array1(["sol", "hfs", "strike_point", "r"])  # shape = [n_time]
    results["SOL"]["HFS"]["STRIKE_POINT"]["Z"] = plasma.get_array1(["sol", "hfs", "strike_point", "z"])  # shape = [n_time]
    results["SOL"]["LFS"]["CONTOUR"]["R"] = plasma.get_array2(["sol", "lfs", "contour", "r"])  # shape = [n_time, n_points]
    results["SOL"]["LFS"]["CONTOUR"]["Z"] = plasma.get_array2(["sol", "lfs", "contour", "z"])  # shape = [n_time, n_points]
    results["SOL"]["LFS"]["CONTOUR"]["N"] = np.array(plasma.get_vec_usize(["sol", "lfs", "contour", "n"])).astype(np.int32)  # shape = [n_time]
    results["SOL"]["LFS"]["STRIKE_POINT"]["R"] = plasma.get_array1(["sol", "lfs", "strike_point", "r"])  # shape = [n_time]
    results["SOL"]["LFS"]["STRIKE_POINT"]["Z"] = plasma.get_array1(["sol", "lfs", "strike_point", "z"])  # shape = [n_time]

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
