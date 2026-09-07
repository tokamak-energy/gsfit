"""
Translate the GSFit results into an IMAS `equilibrium` IDS, using the official `imas-python`
package.

**How it works.** Since the IMAS migration, GSFit already stores its equilibrium results in the
IMAS data model: `plasma.equilibrium_ids` is an `equilibrium` IDS on the Rust side, and
`gsfit_rs.imas.equilibrium_paths` is a path builder over the same schema. So this module does not
hand-map several hundred leaves. It **walks the Rust schema and copies each leaf into the node of
the same name** in the official IDS.

That matters for maintenance: a leaf added on the Rust side appears in the output automatically,
and a leaf the official data dictionary does not have is skipped rather than crashing the run.
Which leaves those are depends on the data dictionary in use, so they are logged at `debug` level
at the end of every run rather than written down here.

**Data retrieval only.** No physics, no unit conversions, no derived quantities. The only
re-shaping is:

* a leaf that is entirely non-finite is left unset, rather than written as `NaN`, because an unset
  node is how IMAS represents "not calculated";
* the ragged 1D contours (the boundary outline) are trimmed of the `NaN` padding that the gather
  into a rectangle introduced;
* the 2D maps are transposed if needed so that `dim1` is R and `dim2` is Z, checked against the
  lengths of `grid/dim1` and `grid/dim2` rather than assumed;
* `bool` and `usize` are cast to the `int` the data dictionary expects.

**Nothing is written to a database.** The IDS is returned, and `Gsfit.write_results_to_database`
keeps it on `gsfit_controller.equilibrium_ids`. Selecting this writer also stops GSFit writing to
MDSplus at all, whatever `write_to_mds` is set to - see `Gsfit.writes_to_mdsplus`.

Conventions to be aware of
--------------------------
1. **COCOS.** The IMAS data dictionary fixes its own convention: DD major version 4 is COCOS 17
   (`<cocos>17</cocos>` in `IDSDef.xml`; DD 3 was COCOS 11). There is no field in which to declare
   a different one, so the data has to be *in* COCOS 17.

   GSFit is intended to be COCOS 13. Using the Sauter & Medvedev (2013) table:

   |          | `e_Bp` | `sigma_Bp` | `sigma_RphiZ` | `sigma_rhothetaphi` |
   | -------- | ------ | ---------- | ------------- | ------------------- |
   | COCOS 13 | 1      | -1         | +1            | -1                  |
   | COCOS 17 | 1      | -1         | +1            | +1                  |

   They agree on everything except `sigma_rhothetaphi`, the direction of the poloidal angle. Both
   have `e_Bp = 1`, so `psi` is the *total* poloidal flux [weber] in both and there is **no factor
   of 2 * pi**; both have `sigma_Bp = -1`, so `psi`, `ip`, `f`, `phi` and the current densities
   need no sign change either.

   What does change is anything whose sign follows the poloidal angle direction: the DD's `q_like`
   (`.fact_q`) and `pol_angle_like` (`.fact_dtheta`) groups. The affected leaves written here are
   `time_slice/profiles_1d/q`, `global_quantities/q_axis`, `global_quantities/q_95` and
   `time_slice/profiles_2d/theta`.
   TODO (rust): flip the sign of `q` when going to COCOS 17, or store `q` in the IMAS convention
   directly. It is not done here, because negating a number is a calculation.

   `profiles_2d/theta` is the exception that is *already* in the IMAS convention: `epp_theta_2d`
   in `rust/gsfit_rs/src/plasma.rs` computes `theta = -atan2(z - z_mag, r - r_mag)`, the clockwise
   (`sigma_rhothetaphi = +1`) sense COCOS 17 requires, so it passes through unchanged. That makes
   `theta` inconsistent with GSFit's COCOS 13 intent for everything else. If GSFit settles on
   COCOS 13 throughout, `epp_theta_2d` is where the sense should be flipped, and `q` and `theta`
   would then be handled the same way here.

   Note: `imas-python` only implements COCOS handling as part of its DD3 -> DD4 conversion (a sign
   flip on the `psi_like` / `dodpsi_like` groups); there is no general "convert from COCOS N" entry
   point to lean on.
2. **Constraint ordering.** The IMAS constraint arrays are positional and are meant to line up
   with the `magnetics` / `pf_active` / `pf_passive` IDSs. Those IDSs are not written here, so the
   GSFit sensor name is stored in `source` to keep the mapping recoverable.
3. **Rogowski coils are not all passive.** GSFit keeps every Rogowski coil in one `rogowski_coils`
   object, but on ST40 they measure two different things: the `*WIRE` coils (`BVLWIRE`, `BVUBWIRE`,
   `BVUTWIRE`, `DIVBWIRE`, `DIVTWIRE`, `MCWIRE`, `PSHBWIRE`, `PSHTWIRE`, `SOLWIRE`, `TFWIRE`)
   measure the current in the *active* coil windings, while the rest (coil cases, `INIVC*`,
   `HFSPSR*`, `DIVPSR*`, `GASBFL*`) measure *passive* structure currents. The whole family is
   written to `constraints/pf_passive_current`, because splitting it here would mean inferring the
   sensor type from its name. `source` holds the GSFit name, so the split stays possible.
   TODO (rust): tag each Rogowski coil with what it measures, then the `*WIRE` coils can be routed
   to `constraints/pf_current` instead.
4. **Missing sensor values are passed through as-is.** Sensors GSFit did not fit carry
   `fit_settings/weight = NAN` (14 of the 27 ST40 Rogowski coils), un-measured channels carry `0.0`
   rather than `NAN` (e.g. flux loop `L003`), and `DIALOOP` has `measured = NAN` on some pulses.
   None of it is cleaned up here.
"""

import typing
from typing import TYPE_CHECKING

import gsfit_rs.imas as gsfit_imas
import imas
import imas.ids_defs
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base import version_storage

if TYPE_CHECKING:
    from imas.ids_toplevel import IDSToplevel

    from ...gsfit import Gsfit
    from . import DatabaseWriterIMAS

equilibrium_paths = gsfit_imas.equilibrium_paths


def _is_leaf(path: typing.Any) -> bool:
    """A path node holding data, as opposed to a structure or an array of structures.

    Both a leaf and an array of structures have no children in `dir()`. They are told apart by
    `data_type`, which an array of structures reports as an empty string.
    """

    if [name for name in dir(path) if not name.startswith("_")]:
        return False
    try:
        return bool(path.data_type)
    except Exception:
        return False


def _is_array_of_structures(path: typing.Any) -> bool:
    """A path node which cannot be read until it is indexed."""

    return not [name for name in dir(path) if not name.startswith("_")] and not _is_leaf(path)


def _read_leaf(equilibrium_ids: typing.Any, path: typing.Any) -> typing.Any:
    """Read one leaf out of the Rust IDS, or `None` if GSFit never set it.

    A float GSFit did not set reads back as `NaN`, but an integer or a string has no such
    stand-in, so the Rust getter raises instead. Either way the answer is the same - the node stays
    unset in the output.
    """

    try:
        return equilibrium_ids.get(path)
    except (IndexError, ValueError, TypeError):
        return None


def _trim_trailing_nan(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Drop the `NaN` padding from a ragged 1D contour.

    The Rust getter gathers one contour per time-slice into a rectangle, padding the short rows.
    Only trailing padding is removed, so an interior `NaN` - which would be real data - survives.
    """

    finite = np.isfinite(values)
    if not finite.any():
        return values[:0]
    return values[: int(np.flatnonzero(finite)[-1]) + 1]


def _write_leaf(destination: typing.Any, name: str, values: typing.Any, data_type: str) -> None:
    """Write one leaf into the official IDS, doing the minimum reshaping the DD requires."""

    if data_type.startswith("STR"):
        if values is not None and str(values) != "":
            setattr(destination, name, str(values))
        return

    array = np.asarray(values)

    if data_type.startswith("INT"):
        # `EMPTY_INT` is how a time-slice which did not converge reads back
        if array.ndim == 0:
            if int(array) != imas.ids_defs.EMPTY_INT:
                setattr(destination, name, int(array))
        else:
            setattr(destination, name, array.astype(np.int32))
        return

    array = array.astype(np.float64)
    if not np.isfinite(array).any():
        # Nothing was calculated; an unset node says that, a node full of NaN does not
        return

    if array.ndim == 1:
        array = _trim_trailing_nan(array)
        if array.size == 0:
            return

    setattr(destination, name, array)


def _copy_time_slice_branch(
    source: typing.Any,
    destination: typing.Any,
    equilibrium_ids: typing.Any,
    i_time: int,
    path: str,
    skipped: set[str],
) -> None:
    """Copy one branch of `time_slice[i_time]` from the Rust IDS into the official IDS.

    `source` is a path over `time_slice[:]`, so every leaf read returns the whole time series at
    once and `i_time` selects the slice. Reading per-slice instead would copy the IDS out of Rust
    once per leaf per slice.
    """

    for name in sorted(n for n in dir(source) if not n.startswith("_")):
        source_child = getattr(source, name)
        if not isinstance(source_child, gsfit_imas.Path):
            continue

        child_path = f"{path}/{name}" if path else name

        if not hasattr(destination, name):
            skipped.add(child_path)
            continue
        destination_child = getattr(destination, name)

        if _is_array_of_structures(source_child):
            # `profiles_2d` is the only array of structures this copier fills, and always with a
            # single entry. The constraint arrays are filled from the sensor objects further down,
            # and the rest (`boundary/gap`, `contour_tree/node`, `ggd`) GSFit does not store. None
            # of them is a schema mismatch, so none is reported as one
            if name == "profiles_2d":
                _copy_profiles_2d(source_child, destination_child, equilibrium_ids, i_time, child_path, skipped)
            continue

        if _is_leaf(source_child):
            values = _read_leaf(equilibrium_ids, source_child)
            if values is not None:
                _write_leaf(destination, name, np.asarray(values)[i_time], source_child.data_type)
            continue

        _copy_time_slice_branch(source_child, destination_child, equilibrium_ids, i_time, child_path, skipped)


def _copy_profiles_2d(
    source: typing.Any,
    destination: typing.Any,
    equilibrium_ids: typing.Any,
    i_time: int,
    path: str,
    skipped: set[str],
) -> None:
    """Copy `profiles_2d[0]`, orienting the 2D maps as the data dictionary wants them."""

    destination.resize(1)
    source_entry = source[0]
    destination_entry = destination[0]

    dim1 = np.asarray(_read_leaf(equilibrium_ids, source_entry.grid.dim1))[i_time]
    dim2 = np.asarray(_read_leaf(equilibrium_ids, source_entry.grid.dim2))[i_time]
    n_dim1 = int(np.isfinite(dim1).sum())
    n_dim2 = int(np.isfinite(dim2).sum())

    for name in sorted(n for n in dir(source_entry) if not n.startswith("_")):
        source_child = getattr(source_entry, name)
        if not isinstance(source_child, gsfit_imas.Path):
            continue
        child_path = f"{path}/{name}"

        if not hasattr(destination_entry, name):
            skipped.add(child_path)
            continue

        if not _is_leaf(source_child):
            if _is_array_of_structures(source_child):
                continue
            _copy_time_slice_branch(
                source_child, getattr(destination_entry, name), equilibrium_ids, i_time, child_path, skipped
            )
            continue

        raw = _read_leaf(equilibrium_ids, source_child)
        if raw is None:
            continue
        values = np.asarray(raw)[i_time]

        # The DD requires (dim1, dim2) = (R, Z). Checked against the grid rather than assumed,
        # so that a change of storage order on the Rust side is caught instead of silently
        # transposing the wrong way
        if values.ndim == 2 and values.shape != (n_dim1, n_dim2):
            if values.shape == (n_dim2, n_dim1):
                values = values.T
            else:
                raise ValueError(
                    f"{child_path}: shape {values.shape} matches neither the grid "
                    f"({n_dim1}, {n_dim2}) nor its transpose"
                )

        _write_leaf(destination_entry, name, values, source_child.data_type)


def _copy_time_independent(
    source: typing.Any,
    destination: typing.Any,
    equilibrium_ids: typing.Any,
    path: str,
    skipped: set[str],
) -> None:
    """Copy a branch which is not under `time_slice`, such as `vacuum_toroidal_field` or `code`."""

    for name in sorted(n for n in dir(source) if not n.startswith("_")):
        source_child = getattr(source, name)
        if not isinstance(source_child, gsfit_imas.Path):
            continue
        child_path = f"{path}/{name}" if path else name

        if not hasattr(destination, name):
            skipped.add(child_path)
            continue

        if _is_array_of_structures(source_child):
            continue

        if _is_leaf(source_child):
            values = _read_leaf(equilibrium_ids, source_child)
            if values is not None:
                _write_leaf(destination, name, values, source_child.data_type)
            continue

        _copy_time_independent(source_child, getattr(destination, name), equilibrium_ids, child_path, skipped)


def _sensor_series(sensor_object: typing.Any, sensor_name: str, quantity: str) -> dict[str, typing.Any]:
    """The measured / reconstructed / time / weight series for one sensor.

    The sensor objects still carry the flat getters (`get_array1`, `get_f64`); only `plasma` moved
    to the IMAS path API.
    """

    return {
        "measured": sensor_object.get_array1([sensor_name, quantity, "measured", "value"]),
        "reconstructed": sensor_object.get_array1([sensor_name, quantity, "calculated", "value"]),
        "time_measurement": sensor_object.get_array1([sensor_name, quantity, "measured", "time"]),
        "weight": sensor_object.get_f64([sensor_name, "fit_settings", "weight"]),
    }


def _write_sensor_constraint(entry: typing.Any, series: dict[str, typing.Any], sensor_name: str, i_time: int) -> None:
    """Fill one `constraints/...` entry for one time-slice."""

    entry.source = sensor_name
    entry.measured = float(series["measured"][i_time])
    entry.reconstructed = float(series["reconstructed"][i_time])
    entry.time_measurement = float(series["time_measurement"][i_time])
    entry.weight = float(series["weight"])


def map_results_to_database(
    self: "DatabaseWriterIMAS",
    gsfit_controller: "Gsfit",
) -> "IDSToplevel":
    """
    Map the GSFit results into an IMAS `equilibrium` IDS.

    :param gsfit_controller: the `Gsfit` controller, holding the solved Rust objects
    :return: a populated `equilibrium` IDS. Nothing is written to any backend
    """

    logger = gsfit_controller.logger

    plasma = gsfit_controller.plasma
    bp_probes = gsfit_controller.bp_probes
    flux_loops = gsfit_controller.flux_loops
    rogowski_coils = gsfit_controller.rogowski_coils
    dialoop = gsfit_controller.dialoop
    pressure_sensors = gsfit_controller.pressure_sensors

    time: npt.NDArray[np.float64] = np.asarray(gsfit_controller.results["TIME"], dtype=np.float64)  # [second]
    n_time: int = len(time)

    # Read once: the `equilibrium_ids` getter copies the whole IDS out of Rust, so fetching it per
    # quantity would copy every 2D map on every time-slice, once each
    equilibrium_ids = plasma.equilibrium_ids

    equilibrium = imas.IDSFactory().equilibrium()

    # ------------------------------------------------------------------
    # ids_properties and code provenance
    # ------------------------------------------------------------------
    equilibrium.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.ids_properties.comment = (
        f"GSFit reconstruction of pulse {gsfit_controller.pulseNo}, run {gsfit_controller.run_name}: "
        f"{gsfit_controller.run_description}"
    )
    equilibrium.ids_properties.provider = version_storage.__user__
    equilibrium.ids_properties.creation_date = version_storage.__datetime__

    equilibrium.time = time
    equilibrium.time_slice.resize(n_time)

    # ------------------------------------------------------------------
    # The equilibrium itself, copied leaf by leaf from the Rust IDS
    # ------------------------------------------------------------------
    skipped: set[str] = set()

    for i_time in range(n_time):
        _copy_time_slice_branch(
            equilibrium_paths.time_slice[:],
            equilibrium.time_slice[i_time],
            equilibrium_ids,
            i_time,
            "",
            skipped,
        )

    _copy_time_independent(
        equilibrium_paths.vacuum_toroidal_field,
        equilibrium.vacuum_toroidal_field,
        equilibrium_ids,
        "vacuum_toroidal_field",
        skipped,
    )
    _copy_time_independent(equilibrium_paths.code, equilibrium.code, equilibrium_ids, "code", skipped)

    # GSFit stores some things the data dictionary has no home for. Which ones depends on the
    # dictionary in use, so they are reported from the run rather than listed in the source
    if skipped:
        logger.debug(f"IMAS writer: {len(skipped)} GSFit leaves have no node in the data dictionary: {sorted(skipped)}")

    # ------------------------------------------------------------------
    # Constraints, which live in the sensor objects rather than in the equilibrium IDS
    # ------------------------------------------------------------------
    bp_probe_names: list[str] = list(bp_probes.keys())
    flux_loop_names: list[str] = list(flux_loops.keys())
    rogowski_coil_names: list[str] = list(rogowski_coils.keys())
    dialoop_names: list[str] = list(dialoop.keys())
    pressure_sensor_names: list[str] = list(pressure_sensors.keys())

    bp_probe_series = {name: _sensor_series(bp_probes, name, "b") for name in bp_probe_names}
    flux_loop_series = {name: _sensor_series(flux_loops, name, "psi") for name in flux_loop_names}
    rogowski_coil_series = {name: _sensor_series(rogowski_coils, name, "i") for name in rogowski_coil_names}
    dialoop_series = {name: _sensor_series(dialoop, name, "b") for name in dialoop_names}

    pressure_series = {name: _sensor_series(pressure_sensors, name, "pressure") for name in pressure_sensor_names}
    pressure_psi = {
        name: pressure_sensors.get_array1([name, "pressure", "calculated", "psi"]) for name in pressure_sensor_names
    }
    pressure_r = {name: pressure_sensors.get_f64([name, "geometry", "r"]) for name in pressure_sensor_names}
    pressure_z = {name: pressure_sensors.get_f64([name, "geometry", "z"]) for name in pressure_sensor_names}

    for i_time in range(n_time):
        constraints = equilibrium.time_slice[i_time].constraints

        constraints.b_field_pol_probe.resize(len(bp_probe_names))
        for i_sensor, sensor_name in enumerate(bp_probe_names):
            _write_sensor_constraint(constraints.b_field_pol_probe[i_sensor], bp_probe_series[sensor_name], sensor_name, i_time)

        constraints.flux_loop.resize(len(flux_loop_names))
        for i_sensor, sensor_name in enumerate(flux_loop_names):
            _write_sensor_constraint(constraints.flux_loop[i_sensor], flux_loop_series[sensor_name], sensor_name, i_time)

        # See conventions note 3: the whole Rogowski family goes to `pf_passive_current`
        constraints.pf_passive_current.resize(len(rogowski_coil_names))
        for i_sensor, sensor_name in enumerate(rogowski_coil_names):
            _write_sensor_constraint(
                constraints.pf_passive_current[i_sensor], rogowski_coil_series[sensor_name], sensor_name, i_time
            )

        for sensor_name in dialoop_names:
            # The DD has a single diamagnetic flux constraint, not an array
            _write_sensor_constraint(constraints.diamagnetic_flux, dialoop_series[sensor_name], sensor_name, i_time)

        constraints.pressure.resize(len(pressure_sensor_names))
        for i_sensor, sensor_name in enumerate(pressure_sensor_names):
            entry = constraints.pressure[i_sensor]
            _write_sensor_constraint(entry, pressure_series[sensor_name], sensor_name, i_time)
            entry.position.r = float(pressure_r[sensor_name])
            entry.position.z = float(pressure_z[sensor_name])
            entry.position.psi = float(pressure_psi[sensor_name][i_time])

    return equilibrium
