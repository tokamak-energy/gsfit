import typing
from typing import TYPE_CHECKING

import numpy as np

# from st40_database import GetData

if TYPE_CHECKING:
    from ...gsfit import Gsfit
    from . import DatabaseWriterTokamakEnergyMDSplus


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
    settings = settings = gsfit_controller.settings
    plasma = gsfit_controller.plasma
    bp_probes = gsfit_controller.bp_probes
    flux_loops = gsfit_controller.flux_loops
    rogowski_coils = gsfit_controller.rogowski_coils
    passives = gsfit_controller.passives
    results = gsfit_controller.results

    # Two-d
    results["TWO_D"]["BR"] = plasma.get_array3(["two_d", "br"])
    # results["TWO_D"]["BT"] = plasma.get_array3(["two_d", "bt"])
    results["TWO_D"]["BZ"] = plasma.get_array3(["two_d", "bz"])
    results["TWO_D"]["MASK"] = plasma.get_array3(["two_d", "mask"])
    results["TWO_D"]["P"] = plasma.get_array3(["two_d", "p"])
    results["TWO_D"]["PSI"] = plasma.get_array3(["two_d", "psi"])
    results["TWO_D"]["RGRID"] = plasma.get_array1(["grid", "r"])
    results["TWO_D"]["ZGRID"] = plasma.get_array1(["grid", "z"])

    # Global
    results["GLOBAL"]["BETA_N"] = plasma.get_array1(["global", "beta_n"])
    results["GLOBAL"]["BETA_P_1"] = plasma.get_array1(["global", "beta_p_1"])
    results["GLOBAL"]["BETA_P_2"] = plasma.get_array1(["global", "beta_p_2"])
    results["GLOBAL"]["BETA_P_3"] = plasma.get_array1(["global", "beta_p_3"])
    results["GLOBAL"]["BETA_T"] = plasma.get_array1(["global", "beta_t"])
    results["GLOBAL"]["BT_VAC_RGEO"] = plasma.get_array1(["global", "bt_vac_at_r_geo"])
    results["GLOBAL"]["CHI_MAG"] = plasma.get_array1(["global", "chi_mag"])
    results["GLOBAL"]["LI_1"] = plasma.get_array1(["global", "li_1"])
    results["GLOBAL"]["LI_2"] = plasma.get_array1(["global", "li_2"])
    results["GLOBAL"]["LI_3"] = plasma.get_array1(["global", "li_3"])
    results["GLOBAL"]["DELTA_Z"] = plasma.get_array1(["global", "delta_z"])
    results["GLOBAL"]["ELON"] = plasma.get_array1(["global", "elongation"])
    results["GLOBAL"]["PHI_DIA"] = plasma.get_array1(["global", "phi_dia"])
    results["GLOBAL"]["GS_ERROR"] = plasma.get_array1(["global", "gs_error"])
    results["GLOBAL"]["I_ROD"] = plasma.get_array1(["global", "i_rod"])
    results["GLOBAL"]["IP"] = plasma.get_array1(["global", "ip"])
    results["GLOBAL"]["N_ITER"] = np.array(plasma.get_vec_usize(["global", "n_iter"])).astype(np.int32)
    results["GLOBAL"]["P"] = plasma.get_array1(["global", "p"])
    results["GLOBAL"]["PSI_A"] = plasma.get_array1(["global", "psi_a"])
    results["GLOBAL"]["PSI_B"] = plasma.get_array1(["global", "psi_b"])
    results["GLOBAL"]["Q0"] = plasma.get_array1(["global", "q0"])
    results["GLOBAL"]["Q95"] = plasma.get_array1(["global", "q95"])
    results["GLOBAL"]["R_CUR"] = plasma.get_array1(["global", "r_cur"])
    results["GLOBAL"]["Z_CUR"] = plasma.get_array1(["global", "z_cur"])
    results["GLOBAL"]["R_GEO"] = plasma.get_array1(["global", "r_geo"])
    results["GLOBAL"]["Z_GEO"] = plasma.get_array1(["global", "z_geo"])
    results["GLOBAL"]["R_MAG"] = plasma.get_array1(["global", "r_mag"])
    results["GLOBAL"]["Z_MAG"] = plasma.get_array1(["global", "z_mag"])
    results["GLOBAL"]["R_MINOR"] = plasma.get_array1(["global", "r_minor"])
    results["GLOBAL"]["V_LOOP"] = plasma.get_array1(["global", "v_loop"])
    results["GLOBAL"]["VPLASMA"] = plasma.get_array1(["global", "plasma_volume"])
    results["GLOBAL"]["W_MHD"] = plasma.get_array1(["global", "w_mhd"])
    results["GLOBAL"]["XPT_DIVERTED"] = np.array(plasma.get_vec_bool(["global", "xpt_diverted"])).astype(np.int32)

    # Bp probes (note, this is all the sensors, both the ones we fit and the ones we don't)
    bp_names = bp_probes.keys()
    bp_names = [bp_name.replace("P", "B_BPPROBE_") for bp_name in bp_names]
    results["CONSTRAINTS"]["BPPROBE"]["NAME"] = bp_names  # shape = [n_sensors]
    results["CONSTRAINTS"]["BPPROBE"]["CVALUE"] = bp_probes.get_array2(["*", "b", "calculated"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["BPPROBE"]["INCLUDE"] = np.array(bp_probes.get_vec_bool(["*", "fit_settings", "include"]))
    results["CONSTRAINTS"]["BPPROBE"]["MVALUE"] = bp_probes.get_array2(["*", "b", "measured"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["BPPROBE"]["WEIGHT"] = bp_probes.get_array1(["*", "fit_settings", "weight"])  # shape = [n_sensors]

    # Flux loops (note, this is all the sensors, both the ones we fit and the ones we don't)
    fl_names = flux_loops.keys()
    fl_names = [fl_name.replace("L", "PSI_FLOOP_") for fl_name in fl_names]
    results["CONSTRAINTS"]["FLOOP"]["NAME"] = fl_names  # shape = [n_sensors]
    results["CONSTRAINTS"]["FLOOP"]["CVALUE"] = flux_loops.get_array2(["*", "psi", "calculated"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["FLOOP"]["INCLUDE"] = np.array(flux_loops.get_vec_bool(["*", "fit_settings", "include"]))
    results["CONSTRAINTS"]["FLOOP"]["MVALUE"] = flux_loops.get_array2(["*", "psi", "measured"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["FLOOP"]["WEIGHT"] = flux_loops.get_array1(["*", "fit_settings", "weight"])  # shape = [n_sensors]

    # Rogowski coils (note, this is all the sensors, both the ones we fit and the ones we don't)
    rog_names = rogowski_coils.keys()
    rog_names = [f"I_ROG_{rog_name}" for rog_name in rog_names]
    results["CONSTRAINTS"]["ROG"]["NAME"] = rog_names  # shape = [n_sensors]
    results["CONSTRAINTS"]["ROG"]["CVALUE"] = rogowski_coils.get_array2(["*", "i", "calculated"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["ROG"]["INCLUDE"] = np.array(rogowski_coils.get_vec_bool(["*", "fit_settings", "include"]))
    results["CONSTRAINTS"]["ROG"]["MVALUE"] = rogowski_coils.get_array2(["*", "i", "measured"])  # shape = [n_time, n_sensors]
    results["CONSTRAINTS"]["ROG"]["WEIGHT"] = rogowski_coils.get_array1(["*", "fit_settings", "weight"])  # shape = [n_sensors]

    # Plasma boundary
    results["P_BOUNDARY"]["NBND"] = np.array(plasma.get_vec_usize(["p_boundary", "nbnd"]))
    results["P_BOUNDARY"]["RBND"] = plasma.get_array2(["p_boundary", "rbnd"])
    results["P_BOUNDARY"]["ZBND"] = plasma.get_array2(["p_boundary", "zbnd"])
    results["P_BOUNDARY"]["BOUNDING_R"] = plasma.get_array1(["p_boundary", "bounding_r"])
    results["P_BOUNDARY"]["BOUNDING_Z"] = plasma.get_array1(["p_boundary", "bounding_z"])

    # X-points
    results["XPOINTS"]["UPPER"]["R"] = plasma.get_array1(["xpoints", "upper", "r"])
    results["XPOINTS"]["UPPER"]["Z"] = plasma.get_array1(["xpoints", "upper", "z"])
    results["XPOINTS"]["LOWER"]["R"] = plasma.get_array1(["xpoints", "lower", "r"])
    results["XPOINTS"]["LOWER"]["Z"] = plasma.get_array1(["xpoints", "lower", "z"])

    # Profiles
    results["PROFILES"]["RHO"]["AREA"] = plasma.get_array2(["profiles", "area"])
    results["PROFILES"]["RHO"]["AREA_PRIME"] = plasma.get_array2(["profiles", "area_prime"])
    results["PROFILES"]["RHO"]["F"] = plasma.get_array2(["profiles", "f"])
    results["PROFILES"]["RHO"]["FF_PRIME"] = plasma.get_array2(["profiles", "ff_prime"])
    results["PROFILES"]["RHO"]["FLUX_TOR"] = plasma.get_array2(["profiles", "flux_tor"])
    results["PROFILES"]["RHO"]["P"] = plasma.get_array2(["profiles", "p"])
    results["PROFILES"]["RHO"]["P_PRIME"] = plasma.get_array2(["profiles", "p_prime"])
    results["PROFILES"]["RHO"]["Q"] = plasma.get_array2(["profiles", "q"])
    results["PROFILES"]["RHO"]["RHO_POL"] = plasma.get_array2(["profiles", "rho_pol"])
    results["PROFILES"]["RHO"]["RHO_TOR"] = plasma.get_array2(["profiles", "rho_tor"])
    results["PROFILES"]["RHO"]["PSI_N"] = plasma.get_array1(["profiles", "psi_n"])
    results["PROFILES"]["RHO"]["VOL"] = plasma.get_array2(["profiles", "vol"])
    results["PROFILES"]["RHO"]["VOL_PRIME"] = plasma.get_array2(["profiles", "vol_prime"])

    # Mid-plane profiles
    results["PROFILES"]["MID_PLANE"]["P"] = plasma.get_array2(["profiles", "mid_plane", "p"])
    results["PROFILES"]["MID_PLANE"]["R"] = plasma.get_array1(["profiles", "mid_plane", "r"])

    # Passives
    for passive_name in passives.keys():
        if passive_name == "IVC":
            results["PASSIVES"]["IVC"]["DOF"]["EIG_01"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_01", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_02"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_02", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_03"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_03", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_04"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_04", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_05"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_05", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_06"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_06", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_07"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_07", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_08"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_08", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_09"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_09", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_10"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_10", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_11"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_11", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_12"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_12", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_13"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_13", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_14"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_14", "calculated"])
            results["PASSIVES"]["IVC"]["DOF"]["EIG_15"]["CVALUE"] = passives.get_array1(["IVC", "dof", "eig_15", "calculated"])
        else:
            dof_name = "constant_current_density"
            results["PASSIVES"][passive_name]["DOF"][dof_name.upper()] = passives.get_array1([passive_name, "dof", dof_name, "calculated"])

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
