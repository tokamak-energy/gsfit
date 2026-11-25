import typing
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from freegsnke.equilibrium_update import Equilibrium as FreeGsnkeEquilibrium
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
    freegsnke_eqs: list[FreeGsnkeEquilibrium],
    isofluxes: list[dict[str, typing.Any]] | None = None,
    isoflux_boundaries: list[dict[str, typing.Any]] | None = None,
    stationary_points: list[dict[str, typing.Any]] | None = None,
    **kwargs: typing.Any,
) -> Magnetics:
    """
    This method initialises the Rust `Magnetics` class, which holds an IMAS `magnetics` IDS.

    :param pulseNo: Pulse number. Not used: the magnetic sensors are read from the FreeGSNKE equilibria
    :param settings: `MagneticsSettings` object containing the XML settings read from the `settings` directory
    :param time: The timebase of `freegsnke_eqs` [second]
    :param freegsnke_eqs: The FreeGSNKE equilibria the synthetic measurements are made from, one for each time
    :param isofluxes: (optional) The isoflux constraints, each a dictionary of the keyword arguments of `Magnetics.add_isoflux`
    :param isoflux_boundaries: (optional) The isoflux boundary constraints, each a dictionary of the keyword arguments of `Magnetics.add_isoflux_boundary`
    :param stationary_points: (optional) The stationary point constraints, each a dictionary of the keyword arguments of `Magnetics.add_stationary_point`
    :param kwargs: The other keyword arguments of `Gsfit.run`, which this method does not use

    **This method is specific to FreeGSNKE.**

    The sensors are FreeGSNKE's, and their measurements are what FreeGSNKE calculates from each equilibrium:
    1. `magnetics/b_field_pol_probe`: the pickup coils measuring the field in the poloidal plane, those whose `orientation` is "PARALLEL"
    2. `magnetics/flux_loop`: the flux loops. FreeGSNKE's flux is per radian, so it is multiplied by `2 * pi` to give GSFit's `psi`
    3. `magnetics/rogowski_coil`: FreeGSNKE has no Rogowski coils, so one, `plasma_current_rogowski`, goes around the limiter, measuring the
       plasma current

    There is no diamagnetic loop.
    The constraints at points which can move, `magnetics/isoflux`, `magnetics/isoflux_boundary` and `magnetics/stationary_point`, are not FreeGSNKE's,
    so they are passed to `Gsfit.run`, e.g. `gsfit_controller.run(time=time, freegsnke_eqs=freegsnke_eqs, isofluxes=isofluxes)`. Each needs
    its fit settings in the equilibrium's settings, e.g. `equilibrium/constraints_isoflux.xml`, with the same `name`.

    Every signal is stored on the timebase of `freegsnke_eqs`.
    Every sensor is added, whether or not it is fitted: which sensors are fitted, and their weights, are the equilibrium's constraints.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Serialise the settings, which the magnetic sensors record in `magnetics/code/parameters`
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=settings,
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Magnetics Rust class
    magnetics: Magnetics = Magnetics(settings_xml)

    # The machine does not change in time, so the sensors are read from the first equilibrium, and only their measurements from each one
    freegsnke_tokamak = freegsnke_eqs[0].tokamak
    n_time: int = len(time)

    # Poloidal field probes: the pickup coils in the poloidal plane. `position` is (R, phi, Z), and `orientation_vector` is (R, phi, Z)
    pickups: list[dict[str, typing.Any]] = freegsnke_tokamak.probes.pickups
    n_pickup: int = len(pickups)
    pickups_field: npt.NDArray[np.float64] = np.full((n_time, n_pickup), np.nan)  # [tesla]
    for i_time in range(n_time):
        pickups_field[i_time, :] = freegsnke_tokamak.probes.calculate_pickup_value(freegsnke_eqs[i_time])
    for i_pickup in range(n_pickup):
        pickup: dict[str, typing.Any] = pickups[i_pickup]
        if pickup["orientation"] != "PARALLEL":
            continue
        orientation_r: float = float(pickup["orientation_vector"][0])
        orientation_z: float = float(pickup["orientation_vector"][2])
        if orientation_r == 0.0 and orientation_z == 0.0:
            raise ValueError(f"FreeGSNKE's pickup coil `{pickup['name']}` has no component in the poloidal plane")
        magnetics.add_b_field_pol_probe(
            name=pickup["name"],
            r=float(pickup["position"][0]),
            z=float(pickup["position"][2]),
            # Counter-clockwise from `grad(R)`, GSFit's COCOS 13
            poloidal_angle=float(np.arctan2(orientation_z, orientation_r)),
            time=time,
            field=pickups_field[:, i_pickup],
        )

    # Flux loops. `position` is (R, Z)
    floops: list[dict[str, typing.Any]] = freegsnke_tokamak.probes.floops
    n_floop: int = len(floops)
    floops_flux: npt.NDArray[np.float64] = np.full((n_time, n_floop), np.nan)  # [weber]
    for i_time in range(n_time):
        # Converted from FreeGSNKE's "weber / radian" to "weber"
        floops_flux[i_time, :] = 2.0 * np.pi * freegsnke_tokamak.probes.calculate_fluxloop_value(freegsnke_eqs[i_time])
    for i_floop in range(n_floop):
        floop: dict[str, typing.Any] = floops[i_floop]
        magnetics.add_flux_loop(
            name=floop["name"],
            r=float(floop["position"][0]),
            z=float(floop["position"][1]),
            time=time,
            flux=floops_flux[:, i_floop],
        )

    # FreeGSNKE has no Rogowski coils, so one goes around the limiter, measuring the plasma current
    plasma_current: npt.NDArray[np.float64] = np.full(n_time, np.nan)  # [ampere]
    for i_time in range(n_time):
        plasma_current[i_time] = freegsnke_eqs[i_time].plasmaCurrent()
    magnetics.add_rogowski_coil(
        name="plasma_current_rogowski",
        r=np.asarray(freegsnke_tokamak.limiter.R, dtype=np.float64),
        z=np.asarray(freegsnke_tokamak.limiter.Z, dtype=np.float64),
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
