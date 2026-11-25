import typing
from typing import TYPE_CHECKING

import freegs
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Magnetics
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_magnetics(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
    time: npt.NDArray[np.float64],
    freegs_eqs: list[freegs.equilibrium.Equilibrium],
    isofluxes: list[dict[str, typing.Any]] | None = None,
    isoflux_boundaries: list[dict[str, typing.Any]] | None = None,
    stationary_points: list[dict[str, typing.Any]] | None = None,
    **kwargs: typing.Any,
) -> Magnetics:
    """
    This method initialises the Rust `Magnetics` class, which holds an IMAS `magnetics` IDS.

    :param pulseNo: Pulse number. Not used: the magnetic sensors are read from the FreeGS equilibria
    :param settings: `MagneticsSettings` object containing the XML settings read from the `settings` directory. Not used: the magnetic sensors
        are read from the FreeGS equilibria
    :param time: The timebase of `freegs_eqs` [second]
    :param freegs_eqs: The FreeGS equilibria the synthetic measurements are made from, one for each time
    :param isofluxes: (optional) The isoflux constraints, each a dictionary of the keyword arguments of `Magnetics.add_isoflux`
    :param isoflux_boundaries: (optional) The isoflux boundary constraints, each a dictionary of the keyword arguments of `Magnetics.add_isoflux_boundary`
    :param stationary_points: (optional) The stationary point constraints, each a dictionary of the keyword arguments of `Magnetics.add_stationary_point`
    :param kwargs: The other keyword arguments of `Gsfit.run`, which this method does not use

    **This method is specific to FreeGS.**

    The sensors are FreeGS's, `tokamak.sensors`, and their measurements are what `tokamak.takeMeasurements` stored on each equilibrium's sensors:
    1. `magnetics/b_field_pol_probe`: the `freegs.machine.PoloidalFieldSensor`s. FreeGS's `theta` is GSFit's angle, counter-clockwise from `grad(R)`
    2. `magnetics/flux_loop`: the `freegs.machine.FluxLoopSensor`s. FreeGS's flux is per radian, so it is multiplied by `2 * pi` to give GSFit's `psi`
    3. `magnetics/rogowski_coil`: one Rogowski coil, `plasma_current_rogowski`, goes around FreeGS's wall, measuring the plasma current

    There is no diamagnetic loop.
    The constraints at points which can move, `magnetics/isoflux`, `magnetics/isoflux_boundary` and `magnetics/stationary_point`, are not FreeGS's,
    so they are passed to `Gsfit.run`, e.g. `gsfit_controller.run(time=time, freegs_eqs=freegs_eqs, isofluxes=isofluxes)`. Each needs its fit
    settings in the equilibrium's settings, e.g. `equilibrium/constraints_isoflux.xml`, with the same `name`.

    Every signal is stored on the timebase of `freegs_eqs`.
    Every sensor is added, whether or not it is fitted: which sensors are fitted, and their weights, are the equilibrium's constraints.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the magnetic sensors record in `magnetics/code/parameters`: none, as the sensors are read from the FreeGS
    # equilibria
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=MagneticsSettings(),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Magnetics Rust class
    magnetics: Magnetics = Magnetics(settings_xml)

    # The machine does not change in time, so the sensors are read from the first equilibrium, and only their measurements from each one
    freegs_tokamak = freegs_eqs[0].tokamak
    n_time: int = len(time)
    n_sensor: int = len(freegs_tokamak.sensors)
    sensors_measurement: npt.NDArray[np.float64] = np.full((n_time, n_sensor), np.nan)
    for i_time in range(n_time):
        for i_sensor in range(n_sensor):
            sensors_measurement[i_time, i_sensor] = freegs_eqs[i_time].tokamak.sensors[i_sensor].measurement

    for i_sensor in range(n_sensor):
        sensor = freegs_tokamak.sensors[i_sensor]
        if isinstance(sensor, freegs.machine.PoloidalFieldSensor):
            magnetics.add_b_field_pol_probe(
                name=sensor.name,
                r=float(sensor.R),
                z=float(sensor.Z),
                poloidal_angle=float(sensor.theta),
                time=time,
                field=sensors_measurement[:, i_sensor],
            )
        elif isinstance(sensor, freegs.machine.FluxLoopSensor):
            magnetics.add_flux_loop(
                name=sensor.name,
                r=float(sensor.R),
                z=float(sensor.Z),
                time=time,
                # Converted from FreeGS's "weber / radian" to "weber"
                flux=2.0 * np.pi * sensors_measurement[:, i_sensor],
            )
        else:
            raise TypeError(f"FreeGS's sensor `{sensor.name}` is a `{type(sensor).__name__}`, which is not read")

    # One Rogowski coil goes around FreeGS's wall, measuring the plasma current
    plasma_current: npt.NDArray[np.float64] = np.full(n_time, np.nan)  # [ampere]
    for i_time in range(n_time):
        plasma_current[i_time] = freegs_eqs[i_time].plasmaCurrent()
    magnetics.add_rogowski_coil(
        name="plasma_current_rogowski",
        r=np.asarray(freegs_tokamak.wall.R, dtype=np.float64),
        z=np.asarray(freegs_tokamak.wall.Z, dtype=np.float64),
        time=time,
        current=plasma_current,
    )

    # The constraints at points which can move, passed to `Gsfit.run`
    if isofluxes is not None:
        for isoflux in isofluxes:
            magnetics.add_isoflux(**isoflux)
    if isoflux_boundaries is not None:
        for isoflux_boundary in isoflux_boundaries:
            magnetics.add_isoflux_boundary(**isoflux_boundary)
    if stationary_points is not None:
        for stationary_point in stationary_points:
            magnetics.add_stationary_point(**stationary_point)

    return magnetics
