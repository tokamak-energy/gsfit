import copy
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import shapely
from scipy.constants import mu_0

from .greens_with_boundary_points import greens_with_boundary_points
from .poisson_matrix import compute_lup_bands

if TYPE_CHECKING:
    from ...gsfit import Gsfit
    from . import DatabaseWriterRTGSFitMDSplus


def map_results_to_database(self: "DatabaseWriterRTGSFitMDSplus", gsfit_controller: "Gsfit") -> None:
    """
    Map the results to MDSplus structure.

    `gsfit_controller.results` is a `NestedDict` object, which has a 1:1 mapping to the MDSplus structure.

    This function mutates the `gsfit_controller` object.
    """

    print("rtgsfit_mdsplus running")

    # TODO: move this to a *.json file. This format hasn't yet settled down...
    # TODO: alphabetical order
    rtgsfit_psus = [
        {"power_supply_name": "SOL", "coils": ["SOL"]},
        {"power_supply_name": "MCT", "coils": ["MCT"]},
        {"power_supply_name": "MCB", "coils": ["MCB"]},
        {"power_supply_name": "DIV", "coils": ["DIVT", "DIVB"]},
        {"power_supply_name": "BVL", "coils": ["BVLT", "BVLB"]},
        {"power_supply_name": "BVUT", "coils": ["BVUT"]},
        {"power_supply_name": "BVUB", "coils": ["BVUB"]},
        {"power_supply_name": "PSH", "coils": ["PSHT", "PSHB"]},
    ]

    # Get objects out of `gsfit_controller`
    plasma = gsfit_controller.plasma
    passives = gsfit_controller.passives
    flux_loops = gsfit_controller.flux_loops
    bp_probes = gsfit_controller.bp_probes
    rogowski_coils = gsfit_controller.rogowski_coils
    results = gsfit_controller.results

    # Geometry
    r = plasma.get_array1(["grid", "r"])
    z = plasma.get_array1(["grid", "z"])
    n_z = plasma.get_usize(["grid", "n_z"])
    n_r = plasma.get_usize(["grid", "n_r"])
    d_r = np.mean(r[1:] - r[0:-1])
    d_z = np.mean(z[1:] - z[0:-1])

    # Number of power supplies
    n_psu = len(rtgsfit_psus)

    # Store geomery
    results["PRESHOT"]["DR"] = d_r
    results["PRESHOT"]["DZ"] = d_z
    results["PRESHOT"]["R_VEC"] = r
    results["PRESHOT"]["Z_VEC"] = z
    results["PRESHOT"]["N_R"] = np.int32(n_r)
    results["PRESHOT"]["N_Z"] = np.int32(n_z)

    r_grid, z_grid = np.meshgrid(r, z)
    r_flat = r_grid.flatten()
    z_flat = z_grid.flatten()
    results["PRESHOT"]["R_GRID"] = r_flat
    results["PRESHOT"]["Z_GRID"] = z_flat
    inv_r_mu0 = 1.0 / (r_flat * mu_0)
    results["PRESHOT"]["INV_R_MU0"] = inv_r_mu0
    r_mu0_dz2 = 2 * np.pi * r_flat * mu_0 * d_z**2
    results["PRESHOT"]["R_MU0_DZ2"] = r_mu0_dz2
    results["PRESHOT"]["N_COIL"] = np.int32(n_psu)
    n_grid = n_r * n_z
    results["PRESHOT"]["N_GRID"] = np.int32(n_grid)
    n_ltrb = 2 * n_r + 2 * n_z - 4  # Number of points on the boundary, removing the double counting at the 4 corners
    results["PRESHOT"]["N_LTRB"] = np.int32(n_ltrb)

    # Collect the greens for "grid-coils"
    g_grid_coil = np.zeros((n_z, n_r, n_psu))
    psu_names = []
    for i_psu, power_supply in enumerate(rtgsfit_psus):
        psu_names.append(power_supply["power_supply_name"])
        coil_names = power_supply["coils"]
        for coil_name in coil_names:
            g_grid_coil[:, :, i_psu] += plasma.get_array2(["greens", "pf", coil_name, "psi"])

    # Store in MDSplus
    results["PRESHOT"]["GREENS"]["GRID_COIL"] = g_grid_coil.flatten()
    results["PRESHOT"]["COIL_NAMES"] = np.array(psu_names)

    # Get the "included" sensors
    flux_loops_to_include = flux_loops.get_vec_bool(["*", "fit_settings", "include"])
    n_flux_loops_to_include = np.sum(flux_loops_to_include)
    bp_probes_to_include = bp_probes.get_vec_bool(["*", "fit_settings", "include"])
    n_bp_probes_to_include = np.sum(bp_probes_to_include)
    rogowski_coils_to_include = rogowski_coils.get_vec_bool(["*", "fit_settings", "include"])
    n_rogowski_coils_to_include = np.sum(rogowski_coils_to_include)

    # Count the number of passive degrees of freedom, and regularisations
    n_passive_dofs = 0
    n_regularisations = 0
    passive_names = passives.keys()  # Note: this includes the IVC
    print(passive_names)
    # passive_names = ["IVC", "OVC", "BVLTCASE", "BVLBCASE", "DIVPSRT", "DIVPSRB", "HFSPSRT", "HFSPSRB"] # TODO: TEMPORARY while debugging
    for passive_name in passive_names:
        # The data structure looks like this:
        # passives(["BVLBCASE", "dof", "constant_current_density"])
        # passives(["BVLTCASE", "dof", "constant_current_density"])
        # ...
        # passives(["IVC", "dof", "eig_01"])
        # passives(["IVC", "dof", "eig_02"])
        # ...
        n_passive_dofs += len(passives.keys([passive_name, "dof"]))

        # The data structure looks like this:
        # passives(["BVLBCASE", "regularisations"]) # shape = [n_regularisations, n_dof]
        # passives(["BVLTCASE", "regularisations"])
        # ...
        # passives(["IVC", "regularisations"])
        # ...
        [n_regularisations_local, _] = passives.get_array2([passive_name, "regularisations"]).shape
        n_regularisations += n_regularisations_local

    # Total number of constraints
    n_constraints = n_flux_loops_to_include + n_bp_probes_to_include + n_rogowski_coils_to_include + n_regularisations

    # Store the number of constraints and regularisations
    results["PRESHOT"]["N_F_LOOPS"] = np.int32(n_flux_loops_to_include)
    results["PRESHOT"]["N_BP_PROBES"] = np.int32(n_bp_probes_to_include)
    results["PRESHOT"]["N_ROG_COILS"] = np.int32(n_rogowski_coils_to_include)
    results["PRESHOT"]["N_MEAS"] = np.int32(n_constraints)
    results["PRESHOT"]["N_REG"] = np.int32(n_regularisations)

    # Store number of plasma degrees of freedom
    _, n_p_prime = plasma.get_array2(["source_functions", "p_prime", "coefficients"]).shape  # TODO: this could be done a bit better
    _, n_ff_prime = plasma.get_array2(["source_functions", "ff_prime", "coefficients"]).shape
    n_delta_z = 1
    n_plasma_dof = n_p_prime + n_ff_prime + n_delta_z
    results["PRESHOT"]["N_PLS"] = np.int32(n_plasma_dof)
    # Total number of degrees of freedom
    n_coef = n_passive_dofs + n_plasma_dof
    results["PRESHOT"]["N_COEF"] = np.int32(n_coef)

    # Greens for "sensors-coils"
    g_measured_coil = np.zeros((n_constraints, n_psu))

    # Weights for the constraints
    constraints_weight = np.zeros(n_constraints)

    # Lis of constraint names
    constraint_names = []

    # Greens between the measurements and the degrees of freedom
    # Note: the plasma's dof's are calculated during real-time and are set to zero.
    # So g_dof_meas[:, 0 : n_plasma_dof] will be zero.
    g_dof_meas = np.zeros((n_constraints, n_coef))

    # Collect the greens for "grid-measurements"
    g_grid_meas = np.zeros((n_constraints, n_r * n_z))

    # flux_loops are the first set of constraints
    i_constraint = 0
    for i_flux_loop, floop_name in enumerate(flux_loops.keys()):
        if flux_loops_to_include[i_flux_loop]:
            # Add flux loop name in PCS (Plasma Control System) format
            floop_name_pcs = floop_name.replace("L", "PSI_FLOOP_")
            constraint_names.append(floop_name_pcs)
            # Add the Greens between measurements and coils
            for i_psu, power_supply in enumerate(rtgsfit_psus):
                coil_names = power_supply["coils"]
                for coil_name in coil_names:
                    g_measured_coil[i_constraint, i_psu] += flux_loops.get_f64([floop_name, "greens", "pf", coil_name])
            # Add the weight
            constraints_weight[i_constraint] = 2 * np.pi * flux_loops.get_f64([floop_name, "fit_settings", "weight"]) / flux_loops.get_f64(
                [floop_name, "fit_settings", "expected_value"]
            )
            # Add the Greens between measurements and degrees of freedom
            i_vessel_dof = 0
            for passive_name in passive_names:
                dof_names = passives.keys([passive_name, "dof"])
                for dof_name in dof_names:
                    g_dof_meas[i_constraint, n_plasma_dof + i_vessel_dof] = flux_loops.get_f64([floop_name, "greens", "passives", passive_name, dof_name])
                    i_vessel_dof += 1
            # Greens between the sensors and the plasma grid
            g_grid_meas[i_constraint, :] = flux_loops.get_array1([floop_name, "greens", "plasma"])
            # Set-up index for next sensor
            i_constraint += 1

    # bp_probes are the second set of constraints
    for i_bp_probe, bp_name in enumerate(bp_probes.keys()):
        if bp_probes_to_include[i_bp_probe]:
            # Add flux loop name in PCS (Plasma Control System) format
            bp_name_pcs = bp_name.replace("P", "B_BPPROBE_")
            constraint_names.append(bp_name_pcs)
            # Add the Greens between measurements and coils
            for i_psu, power_supply in enumerate(rtgsfit_psus):
                coil_names = power_supply["coils"]
                for coil_name in coil_names:
                    g_measured_coil[i_constraint, i_psu] += bp_probes.get_f64([bp_name, "greens", "pf", coil_name])
            # Add the weight
            constraints_weight[i_constraint] = bp_probes.get_f64([bp_name, "fit_settings", "weight"]) / bp_probes.get_f64(
                [bp_name, "fit_settings", "expected_value"]
            )
            # Add the Greens between measurements and degrees of freedom
            i_vessel_dof = 0
            for passive_name in passive_names:
                dof_names = passives.keys([passive_name, "dof"])
                for dof_name in dof_names:
                    g_dof_meas[i_constraint, n_plasma_dof + i_vessel_dof] = bp_probes.get_f64([bp_name, "greens", "passives", passive_name, dof_name])
                    i_vessel_dof += 1
            # Greens between the sensors and the plasma grid
            g_grid_meas[i_constraint, :] = bp_probes.get_array1([bp_name, "greens", "plasma"])
            # Set-up index for next sensor
            i_constraint += 1

    rogowski_coils_names_rtgsfit_order = [
        "INIVC000",
        "BVLT",
        "BVLB",
        "GASBFLT",
        "GASBFLB",
        "HFSPSRT",
        "HFSPSRB",
        "DIVPSRT",
        "DIVPSRB",
    ]

    # rogowski_coils are the third set of constraints
    # for i_rogowski_coil, rogowski_coil_name in enumerate(rogowski_coils.keys()):
    for i_rogowski_coil, rogowski_coil_name in enumerate(rogowski_coils_names_rtgsfit_order):
        # if rogowski_coils_to_include[i_rogowski_coil]:
        if rogowski_coils.get_vec_bool([rogowski_coil_name, "fit_settings", "include"]):
            # Add Rogowski coil name in PCS (Plasma Control System) format
            rogowski_coil_name_pcs = f"I_ROG_{rogowski_coil_name}"
            constraint_names.append(rogowski_coil_name_pcs)
            # Add the Greens between measurements and coils
            for i_psu, power_supply in enumerate(rtgsfit_psus):
                coil_names = power_supply["coils"]
                for coil_name in coil_names:
                    g_measured_coil[i_constraint, i_psu] += rogowski_coils.get_f64([rogowski_coil_name, "greens", "pf", coil_name])
            # Add the weight
            constraints_weight[i_constraint] = rogowski_coils.get_f64([rogowski_coil_name, "fit_settings", "weight"]) / rogowski_coils.get_f64(
                [rogowski_coil_name, "fit_settings", "expected_value"]
            )
            # Add the Greens between measurements and degrees of freedom
            i_vessel_dof = 0
            for passive_name in passive_names:
                dof_names = passives.keys([passive_name, "dof"])
                for dof_name in dof_names:
                    g_dof_meas[i_constraint, n_plasma_dof + i_vessel_dof] = rogowski_coils.get_f64(
                        [rogowski_coil_name, "greens", "passives", passive_name, dof_name]
                    )
                    i_vessel_dof += 1
            # Greens between the sensors and the plasma grid
            g_grid_meas[i_constraint, :] = rogowski_coils.get_array1([rogowski_coil_name, "greens", "plasma"])
            # Set-up index for next sensor
            i_constraint += 1

    # "passive regularisations" are the fourth set of constraints
    # Loop over all passives and add regularisations if they exist
    i_dof_start = n_plasma_dof
    for passive_name in passive_names:
        # Find the number of degrees of freedom for this passive
        n_dof_local = len(passives.keys([passive_name, "dof"]))

        passive_regularisation_local = passives.get_array2([passive_name, "regularisations"])
        [n_reg_local, _] = passive_regularisation_local.shape

        i_dof_end = i_dof_start + n_dof_local

        for i_reg in range(n_reg_local):
            g_dof_meas[i_constraint, i_dof_start:i_dof_end] = passive_regularisation_local[i_reg, :]
            regularisation_scaling = 0.001 * np.pi
            constraints_weight[i_constraint] = regularisation_scaling * passives.get_array1([passive_name, "regularisations_weight"])[i_reg]

            # Set-up index for next sensor / constraint
            i_constraint += 1

        # Set-up index for next passive
        i_dof_start += n_dof_local

    # Store in MDSplus
    results["PRESHOT"]["GREENS"]["MEAS_COIL"] = g_measured_coil.flatten()
    results["PRESHOT"]["WEIGHT"] = constraints_weight
    g_dof_meas_weight = np.dot(np.diag(constraints_weight), g_dof_meas).T
    results["PRESHOT"]["GREENS"]["COEF_MEAS_W"] = g_dof_meas_weight.flatten()  # .reshape((n_coef, n_constraints))
    g_grid_meas_weight = np.dot(np.diag(constraints_weight), g_grid_meas).T * d_r * d_z
    results["PRESHOT"]["GREENS"]["GRID_MEAS_W"] = g_grid_meas_weight.flatten()

    # Collect the greens for "grid-vessel"
    g_grid_vessel = np.zeros((n_z * n_r, n_passive_dofs))
    # Loop over all passives
    i_dof = 0
    for passive_name in passive_names:
        current_distribution_dof_names = passives.keys([passive_name, "dof"])
        # `current_distribution_dof_names` can be "constant_current_density", "eig_01", "eig_02", etc.
        for current_distribution_dof_name in current_distribution_dof_names:
            g_grid_vessel[:, i_dof] = plasma.get_array1(["greens", "passives", passive_name, current_distribution_dof_name, "psi"])
            i_dof += 1
    # Store in MDSplus
    results["PRESHOT"]["GREENS"]["GRID_VESSEL"] = g_grid_vessel.flatten()  # .reshape((n_z * n_r, n_passive_dofs))
    results["PRESHOT"]["N_VESS"] = n_passive_dofs

    # Store some settings
    rtgsfit_code_settings = gsfit_controller.settings["RTGSFIT_code_settings.json"]
    results["PRESHOT"]["N_XPT_MAX"] = np.int32(rtgsfit_code_settings["n_xpt_max"])
    results["PRESHOT"]["N_LCFS_MAX"] = np.int32(rtgsfit_code_settings["n_lcfs_max"])
    results["PRESHOT"]["N_INTRP"] = np.int32(rtgsfit_code_settings["n_intrp"])
    results["PRESHOT"]["THRESH"] = rtgsfit_code_settings["thresh"]
    results["PRESHOT"]["FRAC"] = rtgsfit_code_settings["frac"]

    # Add initial conditions
    flux_norm = gsfit_controller.settings["rtgsfit_initial_conditions.json"]["flux_norm"]
    mask = gsfit_controller.settings["rtgsfit_initial_conditions.json"]["mask"]
    psi_total = gsfit_controller.settings["rtgsfit_initial_conditions.json"]["psi_total"]
    results["PRESHOT"]["INITIAL_COND"]["FLUX_NORM"] = np.array(flux_norm).astype(np.float64)
    results["PRESHOT"]["INITIAL_COND"]["MASK"] = np.array(mask).astype(np.int32)
    results["PRESHOT"]["INITIAL_COND"]["PSI_TOTAL"] = np.array(psi_total).astype(np.float64)

    # Vessel
    vessel_r = plasma.get_array1(["vessel", "r"])
    vessel_z = plasma.get_array1(["vessel", "z"])
    vessel_polygon = shapely.geometry.Polygon(np.column_stack((vessel_r, vessel_z)))

    # Test if grid-points are inside the vessel polygon
    grid_points = []
    for i_z in range(n_z):
        for i_r in range(n_r):
            grid_points.append(shapely.geometry.Point(r[i_r], z[i_z]))
    mask_lim = vessel_polygon.contains(grid_points)

    # # Ensure that the (R,Z) grid cell nearest to the vessel is included in the mask
    # for r_v, z_v in zip(vessel_r, vessel_z):
    #     # Find the nearest grid-point to the vessel point
    #     i_r_nearest = np.argmin(np.abs(r - r_v))
    #     i_z_nearest = np.argmin(np.abs(z - z_v))
    #     mask_lim[i_z_nearest * n_r + i_r_nearest] = True

    # results["PRESHOT"]["MASK_LIM"] = mask_lim.astype(np.int32)

    def compute_limit_idx_and_weights(
        r: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        lim_r: npt.NDArray[np.float64],
        lim_z: npt.NDArray[np.float64],
        n_intrp: np.int32,
        n_lim: int,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        n_r = len(r)
        n_lim = len(lim_r)
        limit_idx = np.zeros(n_lim * n_intrp, dtype=int)
        limit_w = np.zeros(n_lim * n_intrp, dtype=float)

        for i, (lr, lz) in enumerate(zip(lim_r, lim_z)):
            if lr < r[0] or lr > r[-1] or lz < z[0] or lz > z[-1]:
                raise ValueError(f"Limiter point ({lr}, {lz}) is out of bounds of the grid.")
            r_idx = np.searchsorted(r, lr) - 1
            z_idx = np.searchsorted(z, lz) - 1

            limit_idx[n_intrp * i + 0] = n_r * z_idx + r_idx  # (r_idx, z_idx)
            limit_idx[n_intrp * i + 1] = n_r * z_idx + r_idx + 1  # (r_idx + 1, z_idx)
            limit_idx[n_intrp * i + 2] = n_r * (z_idx + 1) + r_idx  # (r_idx, z_idx + 1)
            limit_idx[n_intrp * i + 3] = n_r * (z_idx + 1) + r_idx + 1  # (r_idx + 1, z_idx + 1)

            r0, r1 = r[r_idx], r[r_idx + 1]
            z0, z1 = z[z_idx], z[z_idx + 1]
            dr = r1 - r0
            dz = z1 - z0
            limit_w[n_intrp * i + 0] = (r1 - lr) * (z1 - lz) / (dr * dz)  # (r_idx, z_idx)
            limit_w[n_intrp * i + 1] = (lr - r0) * (z1 - lz) / (dr * dz)  # (r_idx + 1, z_idx)
            limit_w[n_intrp * i + 2] = (r1 - lr) * (lz - z0) / (dr * dz)  # (r_idx, z_idx + 1)
            limit_w[n_intrp * i + 3] = (lr - r0) * (lz - z0) / (dr * dz)  # (r_idx + 1, z_idx + 1)

        return limit_idx, limit_w

    n_intrp = np.int32(rtgsfit_code_settings["n_intrp"])
    lim_r = plasma.get_array1(["limiter", "limit_pts", "r"])
    lim_z = plasma.get_array1(["limiter", "limit_pts", "z"])
    # Remove indices where |lim_z| > 0.9 m
    lim_r = lim_r[np.abs(lim_z) < 0.7]
    lim_z = lim_z[np.abs(lim_z) < 0.7]
    n_lim = len(lim_r)
    limit_idx, limit_w = compute_limit_idx_and_weights(r, z, lim_r, lim_z, n_intrp, n_lim)
    results["PRESHOT"]["N_LIMIT"] = np.int32(n_lim)
    results["PRESHOT"]["LIMIT_IDX"] = limit_idx.astype(np.int32)
    results["PRESHOT"]["LIMIT_W"] = limit_w.astype(np.float64)
    results["PRESHOT"]["LIMIT_R"] = lim_r.astype(np.float64)
    results["PRESHOT"]["LIMIT_Z"] = lim_z.astype(np.float64)

    # If grid point is within sqrt(d_r^2 + d_z^2) of
    # a limiter point then include it in the mask_lim
    d_rz = np.sqrt(d_r**2 + d_z**2)
    for i, (lr, lz) in enumerate(zip(lim_r, lim_z)):
        points_near_lim = np.sqrt((lr - r_flat) ** 2 + (lz - z_flat) ** 2) < d_rz
        mask_lim = np.logical_or(mask_lim, points_near_lim)

    results["PRESHOT"]["MASK_LIM"] = mask_lim.astype(np.int32)

    r_ltrb = np.concatenate(
        (
            [r[0]],  # (bottom, left)
            r[0] * np.ones(len(z[1:-1])),  # traverse (bottom, left) to (top, left) (excluding corners)
            [r[0]],  # (top, left)
            r[1:-1],  # traverse (top, left) to (top, right) (excluding corners)
            [r[-1]],  # (top, right)
            r[-1] * np.ones(len(z[1:-1])),  # traverse (top, right) to (bottom, right) (excluding corners)
            [r[-1]],  # (bottom, right)
            np.flip(r[1:-1]),  # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    inv_r_ltrb_mu0 = 1.0 / (r_ltrb * mu_0)
    results["PRESHOT"]["INV_R_L_MU0"] = inv_r_ltrb_mu0.astype(np.float64)

    lower_band, upper_band, perm_idx = compute_lup_bands(r, z)
    results["PRESHOT"]["LOWER_BAND"] = lower_band.astype(np.float64)
    results["PRESHOT"]["UPPER_BAND"] = upper_band.astype(np.float64)
    results["PRESHOT"]["PERM_IDX"] = perm_idx.astype(np.int32)

    # Greens with the boundary points
    g_ltrb = greens_with_boundary_points(plasma)
    results["PRESHOT"]["GREENS"]["LTRB"] = g_ltrb

    # Sometimes important sensors are broken, mainly Rogowski coils which measure current in
    # passive plates. As a work around we can use the nearby flux loop voltage (V = I * R) as
    # a replacement. This section replaces the damaged "bad" sensors with working "good" sensors.
    sensors_pcs_should_read = copy.deepcopy(constraint_names)
    if "sensor_replacement.json" in gsfit_controller.settings:
        sensor_replacements = gsfit_controller.settings["sensor_replacement.json"]
        # sensor_replacement is a dictonary containing the sensors which should be replaced, the data structure is:
        # {
        #     "sensor_name_to_replace": {
        #         "replacements": ["sensor_name_to_use_01", "sensor_name_to_use_02", ...],
        #         "coefficients": [1.0,                     0.5,                     ...]
        #     },
        #     ...
        # }

        # Remove the sensors that are being replaced
        for sensor_replacement_name in sensor_replacements.keys():
            if sensor_replacement_name in sensors_pcs_should_read:
                sensors_pcs_should_read.remove(sensor_replacement_name)

        # Add the sensors that are replacing the removed sensors
        for sensor_replacement_item in sensor_replacements.values():
            sensor_replacement_names = sensor_replacement_item["replacements"]
            for sensor_replacement_name in sensor_replacement_names:
                # Don't double add sensor names
                if sensor_replacement_name not in sensors_pcs_should_read:
                    sensors_pcs_should_read.append(sensor_replacement_name)

        # Construct a matrix where:
        # `sensors_rtgsfit_wants = sensor_replacement_matrix * sensors_pcs_should_read`
        sensor_replacement_matrix = np.zeros((len(constraint_names), len(sensors_pcs_should_read)), dtype=np.float64)
        for i_constraint, constraint_name in enumerate(constraint_names):
            # Find the index of the sensor in the sensors_pcs_should_read list
            if constraint_name in sensors_pcs_should_read:
                # A sensor that is not being replaced
                i_pcs_sensor = sensors_pcs_should_read.index(constraint_name)
                sensor_replacement_matrix[i_constraint, i_pcs_sensor] = 1.0
            else:
                # This is a sensor that is being replaced
                sensor_replacement = sensor_replacements[constraint_name]
                sensor_replacement_names = sensor_replacement["replacements"]
                sensor_replacement_coefficients = sensor_replacement["coefficients"]
                n_replacements = len(sensor_replacement_names)
                for i_replacement in range(0, n_replacements):
                    replacement_name = sensor_replacement_names[i_replacement]
                    i_pcs_sensor = sensors_pcs_should_read.index(replacement_name)
                    sensor_replacement_matrix[i_constraint, i_pcs_sensor] = sensor_replacement_coefficients[i_replacement]

    else:
        # No sensor replacement, so the matrix is just the identity matrix
        sensor_replacement_matrix = np.eye(len(constraint_names), len(sensors_pcs_should_read), dtype=np.float64)

    results["PRESHOT"]["SENS_NAMES"] = sensors_pcs_should_read
    results["PRESHOT"]["SENS_REP_MAT"] = sensor_replacement_matrix.flatten()
    results["PRESHOT"]["N_SENS_PCS"] = np.int32(len(sensors_pcs_should_read))

    # Save IVC geometry data
    results["PASSIVES"]["IVC"]["GEOMETRY"]["R"] = passives.get_array1(["IVC", "geometry", "r"])
    results["PASSIVES"]["IVC"]["GEOMETRY"]["Z"] = passives.get_array1(["IVC", "geometry", "z"])
    results["PASSIVES"]["IVC"]["GEOMETRY"]["D_R"] = passives.get_array1(["IVC", "geometry", "d_r"])
    results["PASSIVES"]["IVC"]["GEOMETRY"]["D_Z"] = passives.get_array1(["IVC", "geometry", "d_z"])
    results["PASSIVES"]["IVC"]["GEOMETRY"]["AREA"] = \
        results["PASSIVES"]["IVC"]["GEOMETRY"]["D_R"] * \
        results["PASSIVES"]["IVC"]["GEOMETRY"]["D_Z"]
    n_eigs = gsfit_controller.settings["passive_dof_regularisation.json"]["IVC"]["n_dof"]
    n_segs = len(passives.get_array1(["IVC", "dof", f"eig_01", "current_distribution"]))
    current_dofs = np.zeros((n_eigs, n_segs))
    for eig_num in range(n_eigs):
        current_dofs[eig_num, :] = \
            passives.get_array1(["IVC", "dof", f"eig_{eig_num + 1:02d}", "current_distribution"])
    results["PASSIVES"]["IVC"]["GEOMETRY"]["CURRENT_DOFS"] = current_dofs

    # Create coef names list and save it
    coef_names = ["pls0", "pls1", "pls2"]
    for passive_name in passives.keys():
        dof_names = passives.keys([passive_name, "dof"])
        for dof_name in dof_names:
            if dof_name == "constant_current_density":
                coef_names.append(passive_name)
            elif dof_name.startswith("eig_"):
                coef_names.append(dof_name)
            else:
                raise ValueError(f"Unknown DoF name: {dof_name}")
    results["PRESHOT"]["COEF_NAMES"] = coef_names

    # create meas_names list and save it
    meas_names = []
    for i_flux_loop, floop_name in enumerate(flux_loops.keys()):
        if flux_loops_to_include[i_flux_loop]:
            meas_names.append(floop_name)
    for i_bp_probe, bp_name in enumerate(bp_probes.keys()):
        if bp_probes_to_include[i_bp_probe]:
            meas_names.append(bp_name)
    for rog_name in rogowski_coils_names_rtgsfit_order:
        meas_names.append(rog_name)
    for passive_name in passive_names:
        passive_regularisation_local = passives.get_array2([passive_name, "regularisations"])
        [n_reg_local, _] = passive_regularisation_local.shape
        for i_reg in range(n_reg_local):
            meas_names.append(f"{passive_name}_reg_{i_reg}")
    results["PRESHOT"]["MEAS_NAMES"] = meas_names
