"""
Translate the GSFit results into an IMAS `equilibrium` IDS.

**Core principle**: this module performs *data retrieval only*.
No physics, no unit conversions and no derived quantities are calculated here.
Everything written into the IDS is read verbatim out of the Rust objects
(`plasma`, `bp_probes`, `flux_loops`, `rogowski_coils`, `dialoop`,
`pressure_sensors` and `coils`).

The only re-shaping performed is:
* slicing a `[n_time, ...]` array to a single time-slice,
* trimming the `f64::NAN` padding from the ragged boundary outline using the
  stored number of points,
* transposing the 2D maps from GSFit's `(n_z, n_r)` to IMAS' `(dim1=R, dim2=Z)`,
* casting `bool`/`usize` to the `int` expected by the IMAS data dictionary.

The following IMAS nodes are deliberately left **empty** because GSFit does not
store the quantity; they need to be added on the Rust side first
(see the `TODO (rust)` comments in-line):

| IMAS node                                                             | what is missing in GSFit                                     |
| -----------------------------------------------------                 | ------------------------------------------------------------ |
| `time_slice/global_quantities/psi_external_average`, `v_external`     | `global/v_loop` is `-d(psi_boundary)/dt`, which is a different quantity |
| `time_slice/global_quantities/plasma_inductance`, `plasma_resistance` | not calculated                                |
| `time_slice/profiles_1d/dpsi_drho_tor`                                | not calculated                                               |
| `time_slice/profiles_1d/j_phi`, `j_parallel`, `gm1..gm9`, `b_field_*`, `trapped_fraction`, `magnetic_shear`, `r_inboard`, `r_outboard`, `surface`, `rho_volume_norm` | not calculated |
| `time_slice/profiles_2d/j_parallel`, `phi`                            | not calculated                                               |
| `time_slice/constraints/*/chi_squared`                                | only the *total* `global/chi_mag` is stored, IMAS has no slot for it |
| `time_slice/constraints/b_field_tor_vacuum_r`                         | `r0 * b0`                                                    |
| `time_slice/constraints/pf_current/reconstructed`                     | `coils` only stores `i/measured`, not `i/calculated`          |
| `time_slice/contour_tree/node/psi`                                    | the poloidal flux at the x-points                            |

The following GSFit results have no home in the `equilibrium` IDS (DD 4.1.1)
and are therefore dropped: `global/li_1`, `global/li_2`, `global/beta_p_1`,
`global/beta_p_2`, `global/delta_z`, `global/i_rod`, `global/p`,
`global/chi_mag`, `profiles_1d/r_midplane/*`, `profiles_1d/psi_norm/rho_pol`,
`profiles_2d/r_z/mask`, `source_functions/*/coefficients`, `sol/*` and the
`fit_settings/include` / `fit_settings/expected_value` sensor flags.

Conventions to be aware of
--------------------------
1. **COCOS.** The IMAS data dictionary fixes its own convention: DD major
   version 4 is COCOS 17 (`<cocos>17</cocos>` in `IDSDef.xml`; DD 3 was
   COCOS 11). There is no field in which to declare a different one, so the
   data has to be *in* COCOS 17.

   GSFit is intended to be COCOS 13. Using the Sauter & Medvedev (2013) table,
   COCOS 13 and 17 are:

   |          | `e_Bp` | `sigma_Bp` | `sigma_RphiZ` | `sigma_rhothetaphi` |
   | -------- | ------ | ---------- | ------------- | ------------------- |
   | COCOS 13 | 1      | -1         | +1            | -1                  |
   | COCOS 17 | 1      | -1         | +1            | +1                  |

   They agree on everything except `sigma_rhothetaphi`, i.e. the direction of
   the poloidal angle. In particular both have `e_Bp = 1`, so `psi` is the
   *total* poloidal flux [weber] in both and there is **no factor of 2 * pi** to
   apply; and both have `sigma_Bp = -1`, so `psi`, `ip`, `f`, `phi` and the
   current densities need no sign change either.

   What does change is anything whose sign follows the poloidal angle direction,
   i.e. the DD's `q_like` (`cocos_transformation_expression = .fact_q`) and
   `pol_angle_like` (`.fact_dtheta`) groups. The affected leaves that this module
   writes are `time_slice/profiles_1d/q`, `time_slice/global_quantities/q_axis`,
   `time_slice/global_quantities/q_95` (`q_min/value` is not written) and
   `time_slice/profiles_2d/theta`.
   TODO (rust): flip the sign of `q` when going to COCOS 17, or store `q` in
   the IMAS convention directly. It is not done here, because negating a
   number is a calculation.

   `profiles_2d/theta` is the exception that is *already* stored in the IMAS
   convention: `epp_theta_2d` in `rust/gsfit_rs/src/plasma.rs` computes
   `theta = -atan2(z - z_mag, r - r_mag)`, which is the clockwise
   (`sigma_rhothetaphi = +1`) sense that COCOS 17 requires, so it is written
   through unchanged. Note that this makes `theta` inconsistent with GSFit's
   COCOS 13 intent for the rest of its results. If GSFit later settles on
   COCOS 13 throughout, `epp_theta_2d` is where the sense should be flipped,
   and this module would then need `q` and `theta` handled the same way.

   Note: `imas-python` only implements COCOS handling as part of its DD3 -> DD4
   conversion (a sign flip on the `psi_like` / `dodpsi_like` groups); there is
   no general "convert from COCOS N" entry point to lean on.
2. **Constraint ordering.** The IMAS constraint arrays are positional and are
   meant to line up with the `magnetics` / `pf_active` / `pf_passive` IDSs.
   Those IDSs are not written here, so the GSFit sensor name is stored in
   `source` to keep the mapping recoverable.
3. **Rogowski coils are not all passive.** GSFit keeps every Rogowski coil in a
   single `rogowski_coils` object, but on ST40 they measure two different
   things: the `*WIRE` coils (`BVLWIRE`, `BVUBWIRE`, `BVUTWIRE`, `DIVBWIRE`,
   `DIVTWIRE`, `MCWIRE`, `PSHBWIRE`, `PSHTWIRE`, `SOLWIRE`, `TFWIRE`) measure
   the current in the *active* coil windings, while the rest (coil cases,
   `INIVC*`, `HFSPSR*`, `DIVPSR*`, `GASBFL*`) measure *passive* structure
   currents. The whole family is written to `constraints/pf_passive_current`,
   because splitting it here would mean inferring the sensor type from its
   name. The `source` field holds the GSFit name, so the split stays possible.
   TODO (rust): tag each Rogowski coil with what it measures, then the `*WIRE`
   coils can be routed to `constraints/pf_current` instead.
4. **Missing sensor values are passed through as-is.** Sensors that GSFit did
   not fit carry `fit_settings/weight = NAN` (14 of the 27 ST40 Rogowski coils),
   un-measured channels carry `0.0` rather than `NAN` (e.g. flux loop `L003`),
   and `DIALOOP` has `measured = NAN` on some pulses. None of this is cleaned
   up here; it is written into the IDS exactly as GSFit stores it.
"""

import json
from typing import TYPE_CHECKING

import imas
import imas.ids_defs
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base import version_storage

if TYPE_CHECKING:
    from imas.ids_toplevel import IDSToplevel

    from ...gsfit import Gsfit
    from . import DatabaseWriterIMAS


def map_results_to_database(
    self: "DatabaseWriterIMAS",
    gsfit_controller: "Gsfit",
) -> "IDSToplevel":
    """
    Map the GSFit results into an IMAS `equilibrium` IDS.

    :param gsfit_controller: the `Gsfit` controller object, holding the solved Rust objects
    :return: a populated `equilibrium` IDS (not written to any backend)
    """

    # Take the class objects out of the `gsfit_controller` object
    settings = gsfit_controller.settings
    plasma = gsfit_controller.plasma
    bp_probes = gsfit_controller.bp_probes
    flux_loops = gsfit_controller.flux_loops
    dialoop = gsfit_controller.dialoop
    rogowski_coils = gsfit_controller.rogowski_coils
    coils = gsfit_controller.coils
    pressure_sensors = gsfit_controller.pressure_sensors

    time: npt.NDArray[np.float64] = np.asarray(gsfit_controller.results["TIME"], dtype=np.float64)  # [second]
    n_time: int = len(time)

    # ------------------------------------------------------------------
    # Retrieve everything from the Rust objects, once
    # ------------------------------------------------------------------

    # Global; shape = [n_time]
    global_current_centre_r: npt.NDArray[np.float64] = plasma.get_array1(["global", "current_centre", "r"])  # [metre]
    global_current_centre_z: npt.NDArray[np.float64] = plasma.get_array1(["global", "current_centre", "z"])  # [metre]
    global_magnetic_axis_b_field_phi: npt.NDArray[np.float64] = plasma.get_array1(["global", "magnetic_axis", "b_field_phi"])  # [tesla]
    global_magnetic_axis_r: npt.NDArray[np.float64] = plasma.get_array1(["global", "magnetic_axis", "r"])  # [metre]
    global_magnetic_axis_z: npt.NDArray[np.float64] = plasma.get_array1(["global", "magnetic_axis", "z"])  # [metre]
    global_area: npt.NDArray[np.float64] = plasma.get_array1(["global", "area"])  # [metre ** 2]
    global_beta_n: npt.NDArray[np.float64] = plasma.get_array1(["global", "beta_n"])  # [dimensionless]
    global_beta_p_3: npt.NDArray[np.float64] = plasma.get_array1(["global", "beta_p_3"])  # [dimensionless]
    global_beta_t: npt.NDArray[np.float64] = plasma.get_array1(["global", "beta_t"])  # [dimensionless]
    global_gs_error: npt.NDArray[np.float64] = plasma.get_array1(["global", "gs_error"])  # [dimensionless]
    global_ip: npt.NDArray[np.float64] = plasma.get_array1(["global", "ip"])  # [ampere]
    global_length_pol: npt.NDArray[np.float64] = plasma.get_array1(["global", "length_pol"])  # [metre]
    global_li_3: npt.NDArray[np.float64] = plasma.get_array1(["global", "li_3"])  # [dimensionless]
    global_surface: npt.NDArray[np.float64] = plasma.get_array1(["global", "surface"])  # [metre ** 2]
    global_psi_a: npt.NDArray[np.float64] = plasma.get_array1(["global", "psi_a"])  # [weber]
    global_q_95: npt.NDArray[np.float64] = plasma.get_array1(["global", "q_95"])  # [dimensionless]
    global_q_axis: npt.NDArray[np.float64] = plasma.get_array1(["global", "q_axis"])  # [dimensionless]
    global_volume: npt.NDArray[np.float64] = plasma.get_array1(["global", "volume"])  # [metre ** 3]
    global_w_mhd: npt.NDArray[np.float64] = plasma.get_array1(["global", "w_mhd"])  # [joule]
    global_n_iter: list[int] = plasma.get_vec_usize(["global", "n_iter"])  # [count]
    global_xpt_diverted: list[bool] = plasma.get_vec_bool(["global", "xpt_diverted"])  # [dimensionless]

    # Plasma boundary; shape = [n_time], except the outline which is [n_time, n_boundary_max]
    boundary_elongation: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "elongation"])  # [dimensionless]
    boundary_geometric_axis_r: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "geometric_axis", "r"])  # [metre]
    boundary_geometric_axis_z: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "geometric_axis", "z"])  # [metre]
    boundary_minor_radius: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "minor_radius"])  # [metre]
    boundary_psi: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "psi"])  # [weber]
    boundary_psi_norm: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "psi_norm"])  # [dimensionless]
    boundary_squareness_lower_inner: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "squareness_lower_inner"])  # [dimensionless]
    boundary_squareness_lower_outer: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "squareness_lower_outer"])  # [dimensionless]
    boundary_squareness_upper_inner: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "squareness_upper_inner"])  # [dimensionless]
    boundary_squareness_upper_outer: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "squareness_upper_outer"])  # [dimensionless]
    boundary_triangularity: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "triangularity"])  # [dimensionless]
    boundary_triangularity_lower: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "triangularity_lower"])  # [dimensionless]
    boundary_triangularity_upper: npt.NDArray[np.float64] = plasma.get_array1(["boundary", "triangularity_upper"])  # [dimensionless]
    boundary_outline_n: list[int] = plasma.get_vec_usize(["boundary", "outline", "n"])  # [count]
    boundary_outline_r: npt.NDArray[np.float64] = plasma.get_array2(["boundary", "outline", "r"])  # [metre]
    boundary_outline_z: npt.NDArray[np.float64] = plasma.get_array2(["boundary", "outline", "z"])  # [metre]

    # x-points; shape = [n_time]
    xpoint_lower_r: npt.NDArray[np.float64] = plasma.get_array1(["xpoints", "lower", "r"])  # [metre]
    xpoint_lower_z: npt.NDArray[np.float64] = plasma.get_array1(["xpoints", "lower", "z"])  # [metre]
    xpoint_upper_r: npt.NDArray[np.float64] = plasma.get_array1(["xpoints", "upper", "r"])  # [metre]
    xpoint_upper_z: npt.NDArray[np.float64] = plasma.get_array1(["xpoints", "upper", "z"])  # [metre]

    # Profiles vs normalised poloidal flux; shape = [n_time, n_psi_n]
    profiles_1d_psi_norm: npt.NDArray[np.float64] = plasma.get_array1(["profiles_1d", "psi_norm", "psi_norm"])  # [dimensionless]; shape = [n_psi_n]
    profiles_1d_area: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "area"])  # [metre ** 2]
    profiles_1d_area_prime: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "area_prime"])  # [metre ** 2 / weber]
    profiles_1d_f: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "f"])  # [metre * tesla]
    profiles_1d_ff_prime: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "ff_prime"])  # [metre ** 2 * tesla ** 2 / weber]
    profiles_1d_flux_tor: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "flux_tor"])  # [weber]
    profiles_1d_p: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "p"])  # [pascal]
    profiles_1d_p_prime: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "p_prime"])  # [pascal / weber]
    profiles_1d_psi: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "psi"])  # [weber]
    profiles_1d_q: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "q"])  # [dimensionless]
    profiles_1d_rho_tor: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "rho_tor"])  # [metre]
    profiles_1d_rho_tor_norm: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "rho_tor_norm"])  # [dimensionless]
    profiles_1d_vol: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "vol"])  # [metre ** 3]
    profiles_1d_vol_prime: npt.NDArray[np.float64] = plasma.get_array2(["profiles_1d", "psi_norm", "vol_prime"])  # [metre ** 3 / weber]

    # Reference vacuum toroidal field; `r0` is a fixed scalar, `b0` has shape = [n_time]
    vacuum_toroidal_field_r0: float = plasma.get_f64(["vacuum_toroidal_field", "r0"])  # [metre]
    vacuum_toroidal_field_b0: npt.NDArray[np.float64] = plasma.get_array1(["vacuum_toroidal_field", "b0"])  # [tesla]

    # Grid; shape = [n_r] and [n_z]
    grid_r: npt.NDArray[np.float64] = plasma.get_array1(["grid", "r"])  # [metre]
    grid_z: npt.NDArray[np.float64] = plasma.get_array1(["grid", "z"])  # [metre]

    # 2D maps; shape = [n_time, n_z, n_r]
    profiles_2d_br: npt.NDArray[np.float64] = plasma.get_array3(["profiles_2d", "r_z", "br"])  # [tesla]
    profiles_2d_bt: npt.NDArray[np.float64] = plasma.get_array3(["profiles_2d", "r_z", "bt"])  # [tesla]
    profiles_2d_bz: npt.NDArray[np.float64] = plasma.get_array3(["profiles_2d", "r_z", "bz"])  # [tesla]
    profiles_2d_j: npt.NDArray[np.float64] = plasma.get_array3(["profiles_2d", "r_z", "j"])  # [ampere / metre ** 2]
    profiles_2d_psi: npt.NDArray[np.float64] = plasma.get_array3(["profiles_2d", "r_z", "psi"])  # [weber]
    profiles_2d_theta: npt.NDArray[np.float64] = plasma.get_array3(["profiles_2d", "r_z", "theta"])  # [radian]

    # Magnetic sensors; shape = [n_sensor, n_time], apart from the weights which are [n_sensor]
    bp_probe_names: list[str] = list(bp_probes.keys())
    n_bp_probes: int = len(bp_probe_names)
    bp_probe_measured: npt.NDArray[np.float64] = np.full((n_bp_probes, n_time), np.nan)  # [tesla]
    bp_probe_calculated: npt.NDArray[np.float64] = np.full((n_bp_probes, n_time), np.nan)  # [tesla]
    bp_probe_time: npt.NDArray[np.float64] = np.full((n_bp_probes, n_time), np.nan)  # [second]
    bp_probe_weight: npt.NDArray[np.float64] = np.full(n_bp_probes, np.nan)  # [dimensionless]
    for i_bp_probe in range(n_bp_probes):
        sensor_name: str = bp_probe_names[i_bp_probe]
        bp_probe_measured[i_bp_probe, :] = bp_probes.get_array1([sensor_name, "b", "measured", "value"])
        bp_probe_calculated[i_bp_probe, :] = bp_probes.get_array1([sensor_name, "b", "calculated", "value"])
        bp_probe_time[i_bp_probe, :] = bp_probes.get_array1([sensor_name, "b", "measured", "time"])
        bp_probe_weight[i_bp_probe] = bp_probes.get_f64([sensor_name, "fit_settings", "weight"])

    flux_loop_names: list[str] = list(flux_loops.keys())
    n_flux_loops: int = len(flux_loop_names)
    flux_loop_measured: npt.NDArray[np.float64] = np.full((n_flux_loops, n_time), np.nan)  # [weber]
    flux_loop_calculated: npt.NDArray[np.float64] = np.full((n_flux_loops, n_time), np.nan)  # [weber]
    flux_loop_time: npt.NDArray[np.float64] = np.full((n_flux_loops, n_time), np.nan)  # [second]
    flux_loop_weight: npt.NDArray[np.float64] = np.full(n_flux_loops, np.nan)  # [dimensionless]
    for i_flux_loop in range(n_flux_loops):
        sensor_name = flux_loop_names[i_flux_loop]
        flux_loop_measured[i_flux_loop, :] = flux_loops.get_array1([sensor_name, "psi", "measured", "value"])
        flux_loop_calculated[i_flux_loop, :] = flux_loops.get_array1([sensor_name, "psi", "calculated", "value"])
        flux_loop_time[i_flux_loop, :] = flux_loops.get_array1([sensor_name, "psi", "measured", "time"])
        flux_loop_weight[i_flux_loop] = flux_loops.get_f64([sensor_name, "fit_settings", "weight"])

    # Most ST40 Rogowski coils measure the current flowing in the passive structures
    # (vessel, coil cases, ...), so the family is mapped onto `constraints/pf_passive_current`.
    # TODO (rust): the `*WIRE` Rogowski coils actually measure the active coil winding current
    # and belong in `constraints/pf_current`; tag the sensor type on the Rust side so that the
    # split does not have to be guessed from the sensor name here (see note 4 in the docstring)
    rogowski_coil_names: list[str] = list(rogowski_coils.keys())
    n_rogowski_coils: int = len(rogowski_coil_names)
    rogowski_coil_measured: npt.NDArray[np.float64] = np.full((n_rogowski_coils, n_time), np.nan)  # [ampere]
    rogowski_coil_calculated: npt.NDArray[np.float64] = np.full((n_rogowski_coils, n_time), np.nan)  # [ampere]
    rogowski_coil_time: npt.NDArray[np.float64] = np.full((n_rogowski_coils, n_time), np.nan)  # [second]
    rogowski_coil_weight: npt.NDArray[np.float64] = np.full(n_rogowski_coils, np.nan)  # [dimensionless]
    for i_rogowski_coil in range(n_rogowski_coils):
        sensor_name = rogowski_coil_names[i_rogowski_coil]
        rogowski_coil_measured[i_rogowski_coil, :] = rogowski_coils.get_array1([sensor_name, "i", "measured", "value"])
        rogowski_coil_calculated[i_rogowski_coil, :] = rogowski_coils.get_array1([sensor_name, "i", "calculated", "value"])
        rogowski_coil_time[i_rogowski_coil, :] = rogowski_coils.get_array1([sensor_name, "i", "measured", "time"])
        rogowski_coil_weight[i_rogowski_coil] = rogowski_coils.get_f64([sensor_name, "fit_settings", "weight"])

    # Diamagnetic loop; IMAS only has room for a single diamagnetic flux constraint,
    # so the last sensor in `dialoop` wins (matching `tokamak_energy_mdsplus_new`)
    dialoop_names: list[str] = list(dialoop.keys())
    n_dialoops: int = len(dialoop_names)
    dialoop_measured: npt.NDArray[np.float64] = np.full((n_dialoops, n_time), np.nan)  # [weber]
    dialoop_calculated: npt.NDArray[np.float64] = np.full((n_dialoops, n_time), np.nan)  # [weber]
    dialoop_time: npt.NDArray[np.float64] = np.full((n_dialoops, n_time), np.nan)  # [second]
    dialoop_weight: npt.NDArray[np.float64] = np.full(n_dialoops, np.nan)  # [dimensionless]
    for i_dialoop in range(n_dialoops):
        sensor_name = dialoop_names[i_dialoop]
        dialoop_measured[i_dialoop, :] = dialoop.get_array1([sensor_name, "b", "measured", "value"])
        dialoop_calculated[i_dialoop, :] = dialoop.get_array1([sensor_name, "b", "calculated", "value"])
        dialoop_time[i_dialoop, :] = dialoop.get_array1([sensor_name, "b", "measured", "time"])
        dialoop_weight[i_dialoop] = dialoop.get_f64([sensor_name, "fit_settings", "weight"])

    # Kinetic pressure constraints
    pressure_sensor_names: list[str] = list(pressure_sensors.keys())
    n_pressure_sensors: int = len(pressure_sensor_names)
    pressure_measured: npt.NDArray[np.float64] = np.full((n_pressure_sensors, n_time), np.nan)  # [pascal]
    pressure_calculated: npt.NDArray[np.float64] = np.full((n_pressure_sensors, n_time), np.nan)  # [pascal]
    pressure_psi: npt.NDArray[np.float64] = np.full((n_pressure_sensors, n_time), np.nan)  # [weber]
    pressure_time: npt.NDArray[np.float64] = np.full((n_pressure_sensors, n_time), np.nan)  # [second]
    pressure_weight: npt.NDArray[np.float64] = np.full(n_pressure_sensors, np.nan)  # [dimensionless]
    pressure_r: npt.NDArray[np.float64] = np.full(n_pressure_sensors, np.nan)  # [metre]
    pressure_z: npt.NDArray[np.float64] = np.full(n_pressure_sensors, np.nan)  # [metre]
    for i_pressure_sensor in range(n_pressure_sensors):
        sensor_name = pressure_sensor_names[i_pressure_sensor]
        pressure_measured[i_pressure_sensor, :] = pressure_sensors.get_array1([sensor_name, "pressure", "measured", "value"])
        pressure_calculated[i_pressure_sensor, :] = pressure_sensors.get_array1([sensor_name, "pressure", "calculated", "value"])
        pressure_psi[i_pressure_sensor, :] = pressure_sensors.get_array1([sensor_name, "pressure", "calculated", "psi"])
        pressure_time[i_pressure_sensor, :] = pressure_sensors.get_array1([sensor_name, "pressure", "measured", "time"])
        pressure_weight[i_pressure_sensor] = pressure_sensors.get_f64([sensor_name, "fit_settings", "weight"])
        pressure_r[i_pressure_sensor] = pressure_sensors.get_f64([sensor_name, "geometry", "r"])
        pressure_z[i_pressure_sensor] = pressure_sensors.get_f64([sensor_name, "geometry", "z"])

    # Poloidal field coil currents.
    # TODO (rust): `coils` only stores `i/measured`; there is no `i/calculated` to write into `reconstructed`
    pf_coil_names: list[str] = list(coils.keys(["pf"]))
    n_pf_coils: int = len(pf_coil_names)
    pf_coil_measured: npt.NDArray[np.float64] = np.full((n_pf_coils, n_time), np.nan)  # [ampere]
    for i_pf_coil in range(n_pf_coils):
        pf_name: str = pf_coil_names[i_pf_coil]
        pf_coil_measured[i_pf_coil, :] = coils.get_array1(["pf", pf_name, "i", "measured", "value"])

    # ------------------------------------------------------------------
    # Build the IDS
    # ------------------------------------------------------------------

    factory = imas.IDSFactory()
    equilibrium_ids = factory.equilibrium()

    equilibrium_ids.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    equilibrium_ids.ids_properties.comment = gsfit_controller.run_description
    equilibrium_ids.ids_properties.name = gsfit_controller.run_name
    equilibrium_ids.ids_properties.provider = version_storage.__user__
    equilibrium_ids.ids_properties.creation_date = version_storage.__datetime__

    equilibrium_ids.code.name = "GSFit"
    equilibrium_ids.code.description = "GSFit: Grad-Shafranov Fit"
    equilibrium_ids.code.repository = "https://github.com/tokamakenergy/gsfit"
    if version_storage.__version__ is not None:
        equilibrium_ids.code.version = version_storage.__version__
    if version_storage.__git_short_hash__ is not None:
        equilibrium_ids.code.commit = version_storage.__git_short_hash__
    equilibrium_ids.code.parameters = json.dumps(settings, default=str)

    # `r0` is the fixed reference radius set by `vacuum_toroidal_field_reference_radius` in
    # `GSFIT_code_settings.json`, and `b0` the vacuum toroidal field at that radius.
    # Note, `global/bt_vac_at_r_geo` is a different quantity: it is evaluated at the time varying `r_geo`.
    equilibrium_ids.vacuum_toroidal_field.r0 = vacuum_toroidal_field_r0
    equilibrium_ids.vacuum_toroidal_field.b0 = vacuum_toroidal_field_b0

    equilibrium_ids.time = time

    equilibrium_ids.time_slice.resize(n_time)
    for i_time in range(n_time):
        time_slice = equilibrium_ids.time_slice[i_time]
        time_slice.time = time[i_time]

        # Global quantities
        time_slice.global_quantities.area = global_area[i_time]
        # `beta_p_3 = 4 * integral(p dV) / (mu_0 * ip ** 2 * r_geo)`, which is the IMAS definition of
        # `beta_pol` with `R_0 = r_geo` (GSFit's `beta_p_1` and `beta_p_2` use other normalisations)
        time_slice.global_quantities.beta_pol = global_beta_p_3[i_time]
        time_slice.global_quantities.beta_tor = global_beta_t[i_time]
        time_slice.global_quantities.beta_tor_norm = global_beta_n[i_time]
        time_slice.global_quantities.energy_mhd = global_w_mhd[i_time]
        time_slice.global_quantities.ip = global_ip[i_time]
        time_slice.global_quantities.length_pol = global_length_pol[i_time]
        time_slice.global_quantities.li_3 = global_li_3[i_time]
        time_slice.global_quantities.surface = global_surface[i_time]
        time_slice.global_quantities.psi_axis = global_psi_a[i_time]
        time_slice.global_quantities.psi_boundary = boundary_psi[i_time]
        time_slice.global_quantities.psi_magnetic_axis = global_psi_a[i_time]
        time_slice.global_quantities.q_95 = global_q_95[i_time]
        time_slice.global_quantities.rho_tor_boundary = profiles_1d_rho_tor[i_time, -1]
        time_slice.global_quantities.q_axis = global_q_axis[i_time]
        time_slice.global_quantities.volume = global_volume[i_time]
        time_slice.global_quantities.current_centre.r = global_current_centre_r[i_time]
        time_slice.global_quantities.current_centre.z = global_current_centre_z[i_time]
        time_slice.global_quantities.magnetic_axis.b_field_phi = global_magnetic_axis_b_field_phi[i_time]
        time_slice.global_quantities.magnetic_axis.r = global_magnetic_axis_r[i_time]
        time_slice.global_quantities.magnetic_axis.z = global_magnetic_axis_z[i_time]

        # Convergence
        time_slice.convergence.iterations_n = int(global_n_iter[i_time])
        time_slice.convergence.grad_shafranov_deviation_value = global_gs_error[i_time]

        # Plasma boundary
        # `boundary/type`: 0 = limiter, 1 = diverted
        time_slice.boundary.type = int(global_xpt_diverted[i_time])
        time_slice.boundary.elongation = boundary_elongation[i_time]
        time_slice.boundary.minor_radius = boundary_minor_radius[i_time]
        time_slice.boundary.psi = boundary_psi[i_time]
        time_slice.boundary.psi_norm = boundary_psi_norm[i_time]
        time_slice.boundary.squareness_lower_inner = boundary_squareness_lower_inner[i_time]
        time_slice.boundary.squareness_lower_outer = boundary_squareness_lower_outer[i_time]
        time_slice.boundary.squareness_upper_inner = boundary_squareness_upper_inner[i_time]
        time_slice.boundary.squareness_upper_outer = boundary_squareness_upper_outer[i_time]
        time_slice.boundary.triangularity = boundary_triangularity[i_time]
        time_slice.boundary.triangularity_lower = boundary_triangularity_lower[i_time]
        time_slice.boundary.triangularity_upper = boundary_triangularity_upper[i_time]
        time_slice.boundary.geometric_axis.r = boundary_geometric_axis_r[i_time]
        time_slice.boundary.geometric_axis.z = boundary_geometric_axis_z[i_time]
        # The boundary outline is stored ragged, padded with `f64::NAN`; trim it using the stored length
        n_boundary: int = int(boundary_outline_n[i_time])
        time_slice.boundary.outline.r = boundary_outline_r[i_time, 0:n_boundary]
        time_slice.boundary.outline.z = boundary_outline_z[i_time, 0:n_boundary]

        # Contour tree; x-points are saddle points of the poloidal flux, so `critical_type = 1`
        xpoint_r: list[float] = []
        xpoint_z: list[float] = []
        if not np.isnan(xpoint_lower_r[i_time]):
            xpoint_r.append(xpoint_lower_r[i_time])
            xpoint_z.append(xpoint_lower_z[i_time])
        if not np.isnan(xpoint_upper_r[i_time]):
            xpoint_r.append(xpoint_upper_r[i_time])
            xpoint_z.append(xpoint_upper_z[i_time])
        n_xpoints: int = len(xpoint_r)
        time_slice.contour_tree.node.resize(n_xpoints)
        for i_xpoint in range(n_xpoints):
            time_slice.contour_tree.node[i_xpoint].critical_type = 1
            time_slice.contour_tree.node[i_xpoint].r = xpoint_r[i_xpoint]
            time_slice.contour_tree.node[i_xpoint].z = xpoint_z[i_xpoint]

        # Profiles vs normalised poloidal flux
        time_slice.profiles_1d.area = profiles_1d_area[i_time, :]
        time_slice.profiles_1d.darea_dpsi = profiles_1d_area_prime[i_time, :]
        time_slice.profiles_1d.dpressure_dpsi = profiles_1d_p_prime[i_time, :]
        time_slice.profiles_1d.dvolume_dpsi = profiles_1d_vol_prime[i_time, :]
        time_slice.profiles_1d.f = profiles_1d_f[i_time, :]
        time_slice.profiles_1d.f_df_dpsi = profiles_1d_ff_prime[i_time, :]
        time_slice.profiles_1d.phi = profiles_1d_flux_tor[i_time, :]
        time_slice.profiles_1d.pressure = profiles_1d_p[i_time, :]
        time_slice.profiles_1d.psi = profiles_1d_psi[i_time, :]
        time_slice.profiles_1d.psi_norm = profiles_1d_psi_norm
        time_slice.profiles_1d.q = profiles_1d_q[i_time, :]
        time_slice.profiles_1d.rho_tor = profiles_1d_rho_tor[i_time, :]
        time_slice.profiles_1d.rho_tor_norm = profiles_1d_rho_tor_norm[i_time, :]
        time_slice.profiles_1d.volume = profiles_1d_vol[i_time, :]

        # 2D maps.
        # GSFit stores these as `[n_time, n_z, n_r]`, IMAS wants `[dim1 = n_r, dim2 = n_z]`,
        # so the `(n_z, n_r)` time-slice is transposed. No values are changed.
        time_slice.profiles_2d.resize(1)
        profiles_2d = time_slice.profiles_2d[0]
        profiles_2d.type.index = 0
        profiles_2d.type.name = "total"
        profiles_2d.type.description = "Total fields"
        profiles_2d.grid_type.index = 1
        profiles_2d.grid_type.name = "rectangular"
        profiles_2d.grid_type.description = "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)"
        profiles_2d.grid.dim1 = grid_r
        profiles_2d.grid.dim2 = grid_z
        profiles_2d.b_field_r = profiles_2d_br[i_time, :, :].transpose()
        profiles_2d.b_field_phi = profiles_2d_bt[i_time, :, :].transpose()
        profiles_2d.b_field_z = profiles_2d_bz[i_time, :, :].transpose()
        profiles_2d.j_phi = profiles_2d_j[i_time, :, :].transpose()
        profiles_2d.psi = profiles_2d_psi[i_time, :, :].transpose()
        profiles_2d.theta = profiles_2d_theta[i_time, :, :].transpose()

        # Constraints: poloidal field probes
        time_slice.constraints.b_field_pol_probe.resize(n_bp_probes)
        for i_bp_probe in range(n_bp_probes):
            b_field_pol_probe = time_slice.constraints.b_field_pol_probe[i_bp_probe]
            b_field_pol_probe.source = bp_probe_names[i_bp_probe]
            b_field_pol_probe.measured = bp_probe_measured[i_bp_probe, i_time]
            b_field_pol_probe.reconstructed = bp_probe_calculated[i_bp_probe, i_time]
            b_field_pol_probe.time_measurement = bp_probe_time[i_bp_probe, i_time]
            b_field_pol_probe.weight = bp_probe_weight[i_bp_probe]

        # Constraints: flux loops
        time_slice.constraints.flux_loop.resize(n_flux_loops)
        for i_flux_loop in range(n_flux_loops):
            flux_loop = time_slice.constraints.flux_loop[i_flux_loop]
            flux_loop.source = flux_loop_names[i_flux_loop]
            flux_loop.measured = flux_loop_measured[i_flux_loop, i_time]
            flux_loop.reconstructed = flux_loop_calculated[i_flux_loop, i_time]
            flux_loop.time_measurement = flux_loop_time[i_flux_loop, i_time]
            flux_loop.weight = flux_loop_weight[i_flux_loop]

        # Constraints: Rogowski coils, measuring the passive structure currents
        time_slice.constraints.pf_passive_current.resize(n_rogowski_coils)
        for i_rogowski_coil in range(n_rogowski_coils):
            pf_passive_current = time_slice.constraints.pf_passive_current[i_rogowski_coil]
            pf_passive_current.source = rogowski_coil_names[i_rogowski_coil]
            pf_passive_current.measured = rogowski_coil_measured[i_rogowski_coil, i_time]
            pf_passive_current.reconstructed = rogowski_coil_calculated[i_rogowski_coil, i_time]
            pf_passive_current.time_measurement = rogowski_coil_time[i_rogowski_coil, i_time]
            pf_passive_current.weight = rogowski_coil_weight[i_rogowski_coil]

        # Constraints: poloidal field coil currents
        time_slice.constraints.pf_current.resize(n_pf_coils)
        for i_pf_coil in range(n_pf_coils):
            pf_current = time_slice.constraints.pf_current[i_pf_coil]
            pf_current.source = pf_coil_names[i_pf_coil]
            pf_current.measured = pf_coil_measured[i_pf_coil, i_time]

        # Constraints: diamagnetic flux
        for i_dialoop in range(n_dialoops):
            time_slice.constraints.diamagnetic_flux.source = dialoop_names[i_dialoop]
            time_slice.constraints.diamagnetic_flux.measured = dialoop_measured[i_dialoop, i_time]
            time_slice.constraints.diamagnetic_flux.reconstructed = dialoop_calculated[i_dialoop, i_time]
            time_slice.constraints.diamagnetic_flux.time_measurement = dialoop_time[i_dialoop, i_time]
            time_slice.constraints.diamagnetic_flux.weight = dialoop_weight[i_dialoop]

        # Constraints: kinetic pressure
        time_slice.constraints.pressure.resize(n_pressure_sensors)
        for i_pressure_sensor in range(n_pressure_sensors):
            pressure = time_slice.constraints.pressure[i_pressure_sensor]
            pressure.source = pressure_sensor_names[i_pressure_sensor]
            pressure.measured = pressure_measured[i_pressure_sensor, i_time]
            pressure.reconstructed = pressure_calculated[i_pressure_sensor, i_time]
            pressure.time_measurement = pressure_time[i_pressure_sensor, i_time]
            pressure.weight = pressure_weight[i_pressure_sensor]
            pressure.position.r = pressure_r[i_pressure_sensor]
            pressure.position.z = pressure_z[i_pressure_sensor]
            pressure.position.psi = pressure_psi[i_pressure_sensor, i_time]

    return equilibrium_ids
