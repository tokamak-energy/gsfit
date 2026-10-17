import typing
from typing import Protocol

import numpy as np
import numpy.typing as npt
from gsfit_rs import Equilibrium
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import Magnetics
from gsfit_rs import PfActive
from gsfit_rs import PfPassive
from gsfit_rs import Pressure
from gsfit_rs import PulseSchedule
from gsfit_rs import StationaryPoint
from gsfit_rs import Tf
from gsfit_rs import Wall

from ..settings_schema.python.equilibrium.equilibrium import EquilibriumSettings
from ..settings_schema.python.magnetics.magnetics import MagneticsSettings
from ..settings_schema.python.pf_active.pf_active import PfActiveSettings
from ..settings_schema.python.pf_passive.pf_passive import PfPassiveSettings
from ..settings_schema.python.pulse_schedule.pulse_schedule import PulseScheduleSettings
from ..settings_schema.python.tf.tf import TfSettings
from ..settings_schema.python.wall.wall import WallSettings


class DatabaseReaderProtocol(Protocol):
    """
    Protocol for reading experimental data.
    Each method is responsible for initialising one of the Rust objects:
    `equilibrium`, `isoflux`, `isoflux_boundary`, `magnetics`, `stationary_point`, `pf_active`, `pf_passive`, `tf`, and `wall`.

    The Protocol defines the inputs and outputs of each method.
    New database readers should be implemented **all** methods.

    The one exception is `setup_pulse_schedule`, which initialises the optional `pulse_schedule`.
    It has a default here, returning `None`, so a reader for a machine with no gaps to measure need not implement it.
    """

    def setup_equilibrium(self, pulseNo: int, settings: EquilibriumSettings, **kwargs: dict[str, typing.Any]) -> Equilibrium:
        """
        This method initialises the Rust `Equilibrium` class, which holds an IMAS `equilibrium` IDS.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `EquilibriumSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        The equilibrium is built from its settings alone. `equilibrium/code/parameters` holds them as XML: `equilibrium.xml`, with
        each file it links to written in its link's place. `Equilibrium` reads the XML into `equilibrium/code/parameters_parsed`,
        e.g. `equilibrium/code/parameters_parsed/grid/n_r`, whose structs are generated from the settings' XML schema, and makes
        everything else from it: one time-slice per time to reconstruct, each with the grid and the magnetic sensors' weights, the
        Greens tables between the grid points, `greens/grid/plasma`, and the p' and ff' source functions. So **record all the settings**: a reader which records
        only some leaves the rest of `code/parameters_parsed` unset.

        The output of this method must always be an `Equilibrium` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Equilibrium Rust class, with the settings it is built from, which it records in `equilibrium/code/parameters`,
        # and reads into `equilibrium/code/parameters_parsed`. See `serialise_python_dataclass_to_imas_parameters_input`
        equilibrium = Equilibrium(parameters_input)

        return equilibrium
        ```
        """
        ...

    def setup_pf_active(self, pulseNo: int, settings: PfActiveSettings, **kwargs: dict[str, typing.Any]) -> PfActive:
        """
        This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `PfActiveSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGSNKE object

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus, or FreeGSNKE object): Which contains the coil geometry and current measurements

        One `pf_active/coil(i1)` is added for each poloidal field (PF) coil, with one rectangular `element(i2)` per filament.
        Each filament is a single turn, so `coil(i1)/current` is the current per turn.

        **Each coil's `name` must be unique**: the sensors and the solver match up the PF coils by name.

        **Store the experimental signal**, on its own timebase, not one interpolated onto the reconstruction times.
        `Equilibrium.copy_from_pf_active_ids` interpolates it itself, so the timebase must cover every time being reconstructed.

        **`current` is signed**: positive means counter-clockwise viewed from above.

        **Each filament's `width` and `height` must be positive.**
        They give the coil's self-inductance, `coil(i1)/inductance`, which the circuit equations use; a reconstruction does not use them.

        Different machines will use different data stores for the coil geometry and currents.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `PfActive` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the PfActive Rust class, with the settings it is built from, which it records in `pf_active/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        pf_active = PfActive(parameters_input)

        # Add all of the PF coils
        for i_pf_coil in range(n_pf_coils):
            pf_active.add_pf_coil(
                name=...,     # read from a database
                r=...,        # read from a database, one value per filament
                z=...,        # read from a database, one value per filament
                width=...,    # read from a database, one value per filament
                height=...,   # read from a database, one value per filament
                time=...,     # read from a database
                current=...,  # read from a database, per turn
            )

        return pf_active
        ```
        """
        ...

    def setup_isoflux_sensors(
        self, pulseNo: int, settings: MagneticsSettings, times_to_reconstruct: npt.NDArray[np.float64], **kwargs: dict[str, typing.Any]
    ) -> Isoflux:
        """
        This method initialises the Rust `Isoflux` class (two locations which have equal poloidal flux).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param times_to_reconstruct: Times to reconstruct the equilibrium
        :param kwargs: Additional objects, such as FreeGNSKE object

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory

        Initialising requires reading data from two locations:
        1. `sensor_weights_isoflux.json`: Which contains "fitting parameters", e.g. if the constraint should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the coordinates where the isoflux constraint is applied

        Different machines will use different data stores for the isoflux coordiantes.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Isoflux` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Isoflux Rust class
        isoflux = Isoflux()

        # Add all of the isoflux constraints
        for i_isoflux_constraint in range(n_isoflux_constraints):
            isoflux.add_sensor(
                name=...,                  # need to be same in database and `sensor_weights_isoflux.json` file
                fit_settings_comment=...,  # read from `sensor_weights_isoflux.json` file
                fit_settings_include=...,  # read from `sensor_weights_isoflux.json` file
                fit_settings_weight=...,   # read from `sensor_weights_isoflux.json` file
                time=...,                  # read from a database
                location_1_r=...,          # read from a database
                location_1_z=...,          # read from a database
                location_2_r=...,          # read from a database
                location_2_z=...,          # read from a database
                times_to_reconstruct=...,  # read from `GSFIT_code_settings.json` file
            )

        return isoflux
        ```
        """
        ...

    def setup_isoflux_boundary_sensors(self, pulseNo: int, settings: MagneticsSettings, **kwargs: dict[str, typing.Any]) -> IsofluxBoundary:
        """
        This method initialises the Rust `IsofluxBoundary` class (a location which has the same poloidal flux as the plasma boundary).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_isoflux_boundary.json`: Which contains "fitting parameters", e.g. if the constraint should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the coordinates where the isoflux boundary constraint is applied

        Different machines will use different data stores for the isoflux coordiantes.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `IsofluxBoundary` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the IsofluxBoundary Rust class
        isoflux_boundary = IsofluxBoundary()

        # Add all of the isoflux_boundary constraints
        for i_isoflux_boundary_constraint in range(n_isoflux_boundary_constraints):
            isoflux_boundary.add_sensor(
                name=...,                         # need to be same in database and `sensor_weights_isoflux_boundary.json` file
                fit_settings_comment=...,         # read from `sensor_weights_isoflux_boundary.json` file
                fit_settings_include=...,         # read from `sensor_weights_isoflux_boundary.json` file
                fit_settings_weight=...,          # read from `sensor_weights_isoflux_boundary.json` file
                time=...,                         # read from a database
                location_1_r=...,                 # read from a database
                location_1_z=...,                 # read from a database
                location_2_r=...,                 # read from a database
                location_2_z=...,                 # read from a database
                times_to_reconstruct=...,         # read from `GSFIT_code_settings.json` file
            )

        return isoflux_boundary
        ```
        """
        ...

    def setup_stationary_point_sensors(self, pulseNo: int, settings: MagneticsSettings, **kwargs: dict[str, typing.Any]) -> StationaryPoint:
        """
        This method initialises the Rust `StationaryPoint` class (the magnetic axis, and x-point positions).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_magnetic_axis.json`: Which contains "fitting parameters", e.g. if a probe should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the measurements and the probe geometry

        Different machines will use different data stores for the measured signals.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `StationaryPoint` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the StationaryPoint Rust class
        stationary_point = StationaryPoint()

        # Add all of the magnetic axis sensors
        for i_magnetic_axis in range(n_magnetic_axis_sensors):
            stationary_point.add_sensor(
                name=...,                         # read from a database
                fit_settings_comment=...,         # read from `sensor_weights_magnetic_axis.json` file
                fit_settings_expected_value=...,  # read from `sensor_weights_magnetic_axis.json` file
                fit_settings_include=...,         # read from `sensor_weights_magnetic_axis.json` file
                fit_settings_weight=...,          # read from `sensor_weights_magnetic_axis.json` file
                time=...,                         # read from a database
                mag_axis_r=...,                   # read from a database
                mag_axis_z=...,                   # read from a database
                times_to_reconstruct=...,         # read from `GSFIT_code_settings.json` file
            )

        return stationary_point
        ```
        """
        ...

    def setup_pf_passive(
        self, pulseNo: int, settings: PfPassiveSettings, settings_equilibrium: EquilibriumSettings, **kwargs: dict[str, typing.Any]
    ) -> PfPassive:
        """
        This method initialises the Rust `PfPassive` class, which holds an IMAS `pf_passive` IDS (the passive toroidal conductors).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `PfPassiveSettings` object containing the XML settings read from the `settings` directory
        :param settings_equilibrium: `EquilibriumSettings` object, whose `passive_dof_regularisation` says which passive conductors to fit, and how
        :param kwargs: Additional objects, such as FreeGSNKE object

        Initialising requires reading data from two locations:
        1. `equilibrium/passive_dof_regularisation.xml`: Which specifies how the conductor should be represented (e.g. constant current density,
           or eigenmode decomposition)
        2. Database reading (e.g. MDSplus, or FreeGSNKE object): Which contains the coordinates where the passive conductors are located

        One `pf_passive/loop(i1)` is added for each passive conductor, with one `element(i2)` per filament.

        **Each loop's `name` must be unique**: the sensors and the solver match up the passive conductors by name.

        **Each filament is a parallelogram, described as the data dictionary's `oblique`**: from one of its corners, the reference point `r`, `z`,
        with the lengths of the two sides which meet there, `length_alpha` and `length_beta`, and their angles, `alpha` counter-clockwise from the
        R axis, and `beta` counter-clockwise from the Z axis. A database which describes filaments in another way, e.g. ST40's ELMAG, which uses
        EFIT's centre, widths and angles, must be converted.

        Different machines will use different data stores for the passive conductor coordinates.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `PfPassive` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the PfPassive Rust class, with the settings it is built from, which it records in `pf_passive/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        pf_passive = PfPassive(parameters_input)

        # Add all of the passive toroidal conductors
        for i_passive in range(n_passives):
            pf_passive.add_loop(
                name=...,                       # need to be same in database and `passive_dof_regularisation.xml` file
                r=...,                          # read from a database, one value per filament
                z=...,                          # read from a database, one value per filament
                length_alpha=...,               # read from a database, one value per filament
                length_beta=...,                # read from a database, one value per filament
                alpha=...,                      # read from a database, one value per filament
                beta=...,                       # read from a database, one value per filament
                resistivity=...,                # read from a database
                current_distribution_type=...,  # read from `passive_dof_regularisation.xml` file
                n_dof=...,                      # read from `passive_dof_regularisation.xml` file
                regularisations=...,            # read from `passive_dof_regularisation.xml` file
                regularisations_weight=...,     # read from `passive_dof_regularisation.xml` file
            )

        return pf_passive
        ```
        """
        ...

    def setup_tf(self, pulseNo: int, settings: TfSettings, **kwargs: dict[str, typing.Any]) -> Tf:
        """
        This method initialises the Rust `Tf` class, which holds an IMAS `tf` IDS.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `TfSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus, or FreeGSNKE object): Which contains the toroidal field measurement

        Two nodes are filled. `tf/r0` is the machine's reference major radius, which the solver
        copies onto `equilibrium/vacuum_toroidal_field/r0` so that the two IDSs cannot disagree.
        `tf/b_field_phi_vacuum_r` is the vacuum field times major radius, which is the vacuum
        poloidal-current function `f_vac = R0 * B_phi0 = mu_0 * i_rod / (2 * pi)`. The rod current
        is not stored separately: the solver recovers it as `i_rod = 2 * pi * f_vac / mu_0`.

        **Store the experimental signal**, on its own timebase, not one interpolated onto the
        reconstruction times. `Equilibrium.copy_from_tf_ids` interpolates it itself.

        **`b_field_phi_vacuum_r` is signed**: positive means counter-clockwise viewed from above.
        A reader which passes a magnitude will silently reverse the toroidal field, and with it
        `f`, `q`, `profiles_2d/b_field_phi` and the diamagnetic flux constraint.

        Different machines will use different data stores for the toroidal field.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Tf` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Tf Rust class, with the settings it is built from, which it records in `tf/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        tf = Tf(parameters_input)

        tf.set_r0(...)  # read from `GSFIT_code_settings.json` file

        # A machine holding a rod current converts it; one holding `R * B_phi` passes it straight
        tf.set_b_field_phi_vacuum_r(
            time=...,  # read from a database
            data=...,  # read from a database
        )

        return tf
        ```
        """
        ...

    def setup_wall(self, pulseNo: int, settings: WallSettings, **kwargs: dict[str, typing.Any]) -> Wall:
        """
        This method initialises the Rust `Wall` class, which holds an IMAS `wall` IDS.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `WallSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus, or FreeGSNKE object): Which contains the limiter geometry

        Only `wall/description_2d(0)/limiter` is filled. Each limiter unit is a plasma facing
        component, and every point of every unit is a candidate limit point.

        **The order the units are added in is part of the contract**: `unit(0)` is the vacuum
        vessel contour, and the solver uses that one, and only that one, as the region the
        plasma is allowed to occupy. Later units are components which protrude inside it, such
        as tiles.

        Different machines will use different data stores for the limiter geometry.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Wall` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Wall Rust class, with the settings it is built from, which it records in `wall/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        wall = Wall(parameters_input)

        # The vacuum vessel contour must be added first
        wall.add_limiter_unit(
            name="vacuum_vessel",  # a short identifier for the unit
            r=...,                 # read from a database
            z=...,                 # read from a database
        )

        # Any other plasma facing components follow
        for i_unit in range(n_units):
            wall.add_limiter_unit(
                name=...,  # a short identifier for the unit
                r=...,     # read from a database
                z=...,     # read from a database
            )

        return wall
        ```
        """
        ...

    def setup_magnetics(self, pulseNo: int, settings: MagneticsSettings, **kwargs: dict[str, typing.Any]) -> Magnetics:
        """
        This method initialises the Rust `Magnetics` class, which holds an IMAS `magnetics` IDS.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `MagneticsSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus, or FreeGSNKE object): Which contains the sensors' geometry and measurements

        Every sensor is added, whether or not it is fitted: which sensors are fitted, and their weights, are the equilibrium's constraints.

        **Store the experimental signals**, each on its own timebase, not interpolated onto the reconstruction times.

        **The probes' `poloidal_angle` keeps GSFit's present definition, COCOS 13**: counter-clockwise from `grad(R)`, in the usual plot with `R` to the
        right and `Z` upwards, so an angle of `pi / 2` points towards `+Z`.
        This is not yet the data dictionary's COCOS 17, which measures it clockwise; see the `Magnetics` documentation.

        **A flux loop's `flux` is GSFit's `psi`**, the total poloidal flux, with no factor of `2 * pi`.

        Different machines will use different data stores for the magnetic sensors.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Magnetics` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Magnetics Rust class, with the settings it is built from, which it records in `magnetics/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        magnetics = Magnetics(parameters_input)

        for i_b_field_pol_probe in range(n_b_field_pol_probes):
            magnetics.add_b_field_pol_probe(
                name=...,            # need to be same in database and `constraints_b_field_pol_probe.xml` file
                r=...,               # read from a database
                z=...,               # read from a database
                poloidal_angle=...,  # read from a database, counter-clockwise from `grad(R)`
                time=...,            # read from a database
                field=...,           # read from a database
            )

        for i_flux_loop in range(n_flux_loops):
            magnetics.add_flux_loop(
                name=...,  # need to be same in database and `constraints_flux_loop.xml` file
                r=...,     # read from a database
                z=...,     # read from a database
                time=...,  # read from a database
                flux=...,  # read from a database
            )

        for i_rogowski_coil in range(n_rogowski_coils):
            magnetics.add_rogowski_coil(
                name=...,          # need to be same in database and `constraints_rogowski_coil.xml` file
                r=...,             # read from a database
                z=...,             # read from a database
                time=...,          # read from a database
                current=...,       # read from a database
                gaps_name=...,     # (optional) read from a database
                gaps_r_start=...,  # (optional) read from a database
                gaps_r_end=...,    # (optional) read from a database
                gaps_z_start=...,  # (optional) read from a database
                gaps_z_end=...,    # (optional) read from a database
            )

        magnetics.add_diamagnetic_flux(
            method_name=...,  # need to be same in database and `constraints_diamagnetic_flux.xml` file
            time=...,         # read from a database
            data=...,         # read from a database
        )

        return magnetics
        ```
        """
        ...

    def setup_pressure_sensors(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> Pressure:
        """
        This method initialises the Rust `Pressure` class (pressure measurements at specific locations in the poloidal plane).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_pressure.json`: Which contains "fitting parameters", e.g. if a sensor should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the measurements and sensor geometry

        Different machines will use different data stores for the sensor geometry and measured signals.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Pressure` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Pressure Rust class
        pressure = Pressure()

        # Add all of the pressure sensors
        for i_pressure_sensor in range(n_pressure_sensors):
            pressure.add_sensor(
                name=...,                         # need to be same in database and `sensor_weights_pressure.json` file
                geometry_r=...,                   # read from a database
                geometry_z=...,                   # read from a database
                fit_settings_comment=...,         # read from `sensor_weights_pressure.json` file
                fit_settings_expected_value=...,  # read from `sensor_weights_pressure.json` file
                fit_settings_include=...,         # read from `sensor_weights_pressure.json` file
                fit_settings_weight=...,          # read from `sensor_weights_pressure.json` file
                time=...,                         # read from a database
                measured=...,                     # read from a database
            )

        return pressure
        ```
        """
        ...

    def setup_pulse_schedule(self, pulseNo: int, settings: PulseScheduleSettings, **kwargs: dict[str, typing.Any]) -> PulseSchedule | None:
        """
        This method initialises the Rust `PulseSchedule` class, which holds an IMAS `pulse_schedule` IDS.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `PulseScheduleSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        **This method is optional.**
        Unlike every other method it has a default, which returns `None`: `Equilibrium.copy_from_pulse_schedule_ids` is then not called, and the equilibrium has no gaps.

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus): Which contains the gap definitions

        Only the gap definitions are filled, `pulse_schedule/position_control/gap(i)`.
        A gap is a reference point and a direction, and its value is the distance from the reference point to the plasma boundary along that direction.
        `Equilibrium.copy_from_pulse_schedule_ids` copies the definitions onto every time-slice's `equilibrium/time_slice(itime)/boundary/gap`, and the equilibrium post-processor calculates each value.

        **The angle is clockwise from `grad(R)`**, in the usual plot with `R` to the right and `Z` upwards, which is the equilibrium IDS's convention.
        So an angle of `pi / 2` points towards `-Z`.
        A machine which stores the angle counter-clockwise has to negate it.

        Different machines will use different data stores for the gap definitions.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `PulseSchedule` object, or `None`.

        At a minimum this method should look like this:
        ```python
        # Initialise the PulseSchedule Rust class, with the settings it is built from, which it records in `pulse_schedule/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        pulse_schedule = PulseSchedule(parameters_input)

        # Add all of the gaps; a machine with no gaps adds none
        for i_gap in range(n_gaps):
            pulse_schedule.add_gap(
                name=...,   # read from a database
                r=...,      # read from a database
                z=...,      # read from a database
                angle=...,  # read from a database, clockwise from `grad(R)`
            )

        return pulse_schedule
        ```
        """

        return None
