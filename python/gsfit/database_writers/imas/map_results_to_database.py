"""
Translate the GSFit results into an IMAS `equilibrium` IDS, using the official `imas-python`
package.

**How it works.** Since the IMAS migration, GSFit already stores its equilibrium results in the
IMAS data model: `plasma` holds an `equilibrium` IDS on the Rust side, read in place with `plasma.get`,
and `gsfit_rs.imas.equilibrium_paths` is a path builder over the same schema. So this module does not
hand-map several hundred leaves. It **walks the Rust schema and copies each leaf into the node of
the same name** in the official IDS.

Each time-slice is traversed on its own, reading `time_slice[i_time]` rather than the accumulator
`time_slice[:]`. So every leaf comes back exactly as that time-slice stores it: a leaf whose length
differs between time-slices, such as the boundary outline, keeps its own length, and there is no
padding to remove.

That matters for maintenance: a leaf added on the Rust side appears in the output automatically,
and a leaf the official data dictionary does not have is skipped rather than crashing the run.
Which leaves those are depends on the data dictionary in use, so they are logged at `debug` level
at the end of every run rather than written down here.

**Data retrieval only.** No physics, no unit conversions, no derived quantities. The only
re-shaping is:

* a leaf that is entirely non-finite is left unset, rather than written as `NaN`, because an unset
  node is how IMAS represents "not calculated";
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
    import gsfit_rs
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


def _read_leaf(plasma: "gsfit_rs.Plasma", path: typing.Any) -> typing.Any:
    """Read one leaf out of the Rust IDS held by `plasma`, or `None` if its path does not exist.

    Every path read here selects a single element of each array of structures on it, e.g.
    `time_slice[i_time]` and `profiles_2d[0]`, so the leaf comes back exactly as it is stored in that
    element, at its own length. None of the accumulators (`time_slice[:]`) is used, because gathering
    elements of different lengths into one array has to pad them.

    A leaf GSFit never set reads back as its IMAS empty value - `NaN`, `EMPTY_INT`, an empty string
    or an empty array - and `_write_leaf` leaves such a node unset in the output. The read itself
    only fails when an index is beyond the end of an array of structures, e.g. `profiles_2d[0]` on
    a time-slice which has no `profiles_2d`.
    """

    try:
        return plasma.get(path)
    except IndexError:
        return None


def _write_leaf(destination: typing.Any, name: str, values: typing.Any, data_type: str) -> None:
    """Write one leaf into the official IDS, doing the minimum reshaping the DD requires."""

    if data_type.startswith("STR"):
        if values is not None and str(values) != "":
            setattr(destination, name, str(values))
        return

    array = np.asarray(values)

    if data_type.startswith("INT"):
        # `EMPTY_INT` is how an unset integer reads back, e.g. from a time-slice which did not
        # converge; an unset integer array is empty
        if array.ndim == 0:
            if int(array) != imas.ids_defs.EMPTY_INT:
                setattr(destination, name, int(array))
        elif array.size > 0:
            setattr(destination, name, array.astype(np.int32))
        return

    array = array.astype(np.float64)
    if not np.isfinite(array).any():
        # Nothing was calculated; an unset node says that, a node full of NaN does not
        return

    setattr(destination, name, array)


def _copy_branch(
    source: typing.Any,
    destination: typing.Any,
    plasma: "gsfit_rs.Plasma",
    path: str,
    skipped: set[str],
) -> None:
    """Traverse one branch of the Rust IDS, and fill the node of the same name in the official IDS.

    :param source: a path into the Rust IDS which selects single elements only, such as
        `equilibrium_paths.time_slice[i_time]` or `equilibrium_paths.vacuum_toroidal_field`
    :param destination: the matching node of the official IDS
    :param path: the IMAS path of `source`, used to report the leaves which are skipped
    :param skipped: the leaves the data dictionary has no node for, added to in place
    """

    for name in sorted(n for n in dir(source) if not n.startswith("_")):
        source_child = getattr(source, name)
        if not isinstance(source_child, gsfit_imas.Path):
            continue
        child_path = f"{path}/{name}"

        if not hasattr(destination, name):
            skipped.add(child_path)
            continue
        destination_child = getattr(destination, name)

        if _is_array_of_structures(source_child):
            # `profiles_2d` and `boundary/gap` are the only arrays of structures this copier fills.
            # The constraint arrays are filled from the sensor objects further down, and the rest
            # (`contour_tree/node`, `ggd`) GSFit does not store. None of them is a schema mismatch,
            # so none is reported as one
            if name == "profiles_2d":
                _copy_profiles_2d(source_child, destination_child, plasma, child_path, skipped)
            elif name == "gap":
                _copy_gaps(source_child, destination_child, plasma, child_path, skipped)
            continue

        if _is_leaf(source_child):
            values = _read_leaf(plasma, source_child)
            if values is not None:
                _write_leaf(destination, name, values, source_child.data_type)
            continue

        _copy_branch(source_child, destination_child, plasma, child_path, skipped)


def _copy_profiles_2d(
    source: typing.Any,
    destination: typing.Any,
    plasma: "gsfit_rs.Plasma",
    path: str,
    skipped: set[str],
) -> None:
    """Copy `profiles_2d[0]`, orienting the 2D maps as the data dictionary wants them."""

    source_entry = source[0]

    dim1 = _read_leaf(plasma, source_entry.grid.dim1)
    dim2 = _read_leaf(plasma, source_entry.grid.dim2)
    if dim1 is None or dim2 is None:
        # This time-slice has no `profiles_2d`
        return
    n_dim1: int = len(dim1)
    n_dim2: int = len(dim2)

    destination.resize(1)
    _copy_profiles_2d_branch(source_entry, destination[0], plasma, path, skipped, n_dim1, n_dim2)


def _copy_gaps(
    source: typing.Any,
    destination: typing.Any,
    plasma: "gsfit_rs.Plasma",
    path: str,
    skipped: set[str],
) -> None:
    """Copy `boundary/gap`, one element at a time.

    There are only gaps when `solve_grad_shafranov` was given a pulse schedule defining some, so an
    empty array is the usual case and leaves the node empty. Every time-slice carries the same gaps,
    whether or not it converged; one which did not has each `value` unset.
    """

    gap_names = _read_leaf(plasma, source[:].name)
    if gap_names is None:
        return
    n_gaps: int = len(gap_names)
    if n_gaps == 0:
        return

    destination.resize(n_gaps)
    for i_gap in range(n_gaps):
        _copy_branch(source[i_gap], destination[i_gap], plasma, path, skipped)


def _copy_profiles_2d_branch(
    source: typing.Any,
    destination: typing.Any,
    plasma: "gsfit_rs.Plasma",
    path: str,
    skipped: set[str],
    n_dim1: int,
    n_dim2: int,
) -> None:
    """Copy one branch of `profiles_2d[0]`, orienting every 2D map it holds against the grid.

    Recursive, so that the 2D maps in a sub-structure, such as `grid/volume_element`, are oriented
    too, not only the direct children of `profiles_2d`.

    :param n_dim1: the length of `grid/dim1`, the major radius [count]
    :param n_dim2: the length of `grid/dim2`, the height [count]
    """

    for name in sorted(n for n in dir(source) if not n.startswith("_")):
        source_child = getattr(source, name)
        if not isinstance(source_child, gsfit_imas.Path):
            continue
        child_path = f"{path}/{name}"

        if not hasattr(destination, name):
            skipped.add(child_path)
            continue

        if not _is_leaf(source_child):
            if _is_array_of_structures(source_child):
                continue
            _copy_profiles_2d_branch(source_child, getattr(destination, name), plasma, child_path, skipped, n_dim1, n_dim2)
            continue

        raw = _read_leaf(plasma, source_child)
        if raw is None:
            continue
        values = np.asarray(raw)

        # An unset map, e.g. on a time-slice which did not converge, is empty; there is nothing to
        # orient, and `_write_leaf` leaves it unset
        if values.size == 0:
            continue

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

        _write_leaf(destination, name, values, source_child.data_type)


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
        _copy_branch(equilibrium_paths.time_slice[i_time], equilibrium.time_slice[i_time], plasma, "time_slice", skipped)

    _copy_branch(equilibrium_paths.vacuum_toroidal_field, equilibrium.vacuum_toroidal_field, plasma, "vacuum_toroidal_field", skipped)
    _copy_branch(equilibrium_paths.code, equilibrium.code, plasma, "code", skipped)

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
