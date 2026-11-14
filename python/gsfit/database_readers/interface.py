import typing
from typing import Protocol

from gsfit_rs import CoreProfiles
from gsfit_rs import Equilibrium
from gsfit_rs import Magnetics
from gsfit_rs import PfActive
from gsfit_rs import PfPassive
from gsfit_rs import PulseSchedule
from gsfit_rs import Tf
from gsfit_rs import Wall

from ..settings_schema.python.core_profiles.core_profiles import CoreProfilesSettings
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
    `equilibrium`, `magnetics`, `pf_active`, `pf_passive`, `tf`, and `wall`.

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
        each file it links to written in its link's place. `Equilibrium` validates the XML against its XML schema, reads it into structs
        generated from that XML schema, and makes everything else from them: one time-slice per time to reconstruct, each with the grid and
        the magnetic sensors' weights, the Greens tables between the grid points, `greens_tables/grid/plasma`, and the p' and ff' source functions.
        So **pass all the settings**: settings which leave out a required element do not follow their XML schema, and are refused.

        The output of this method must always be an `Equilibrium` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Equilibrium Rust class, with the settings it is built from, which it records in `equilibrium/code/parameters`,
        # and reads into structs. See `serialise_python_dataclass_to_imas_parameters_input`
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

    def setup_pf_passive(self, pulseNo: int, settings: PfPassiveSettings, **kwargs: dict[str, typing.Any]) -> PfPassive:
        """
        This method initialises the Rust `PfPassive` class, which holds an IMAS `pf_passive` IDS (the passive toroidal conductors).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `PfPassiveSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGSNKE object

        The passive conductors are machine data only: where they are and their resistivity, read from a database (e.g. MDSplus, or a
        FreeGSNKE object). **Add every passive conductor the database holds**: which of them to fit, and how (e.g. a constant current density,
        or an eigenmode decomposition), is the equilibrium's, `equilibrium/pf_passive_dof.xml`, which `Equilibrium.copy_from_pf_passive_ids`
        reads, finding each passive conductor by its `name`.

        One `pf_passive/loop(i1)` is added for each passive conductor, with one `element(i2)` per filament.

        **Each loop's `name` must be unique**, and is the name `pf_passive_dof.xml` gives it: the equilibrium, the sensors and the solver match up
        the passive conductors by name.

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
                name=...,                       # need to be same in database and `pf_passive_dof.xml` file
                r=...,                          # read from a database, one value per filament
                z=...,                          # read from a database, one value per filament
                length_alpha=...,               # read from a database, one value per filament
                length_beta=...,                # read from a database, one value per filament
                alpha=...,                      # read from a database, one value per filament
                beta=...,                       # read from a database, one value per filament
                resistivity=...,                # read from a database
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

        Two nodes are filled. `tf/r0` is the machine's reference major radius, which `Tf` sets from the settings'
        `vacuum_toroidal_field_reference_radius`, and which the solver copies onto
        `equilibrium/vacuum_toroidal_field/r0` so that the two IDSs cannot disagree.
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
        # Initialise the Tf Rust class, with the settings it is built from, which set `tf/r0` and which it records in `tf/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        tf = Tf(parameters_input)

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

        The constraints at points which can move are added here too, as custom keys of the magnetics IDS: the isoflux constraints, two points
        on the same flux surface, the isoflux boundary constraints, a point on the plasma boundary, and the stationary point constraints, an
        X-point or the magnetic axis. Each point is a time trace on the constraint's own timebase, which need not be the reconstruction
        times; NaN where it is not known. A reader with none of them adds none.

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

        # GSFit accepts only one diamagnetic flux, so the signals of several diamagnetic loops are averaged into one before it is added
        magnetics.add_diamagnetic_flux(
            method_name=...,  # how the diamagnetic flux is calculated, e.g. "DIALOOP"
            time=...,         # read from a database
            data=...,         # read from a database
        )

        for i_isoflux in range(n_isofluxes):
            magnetics.add_isoflux(
                name=...,          # need to be same in database and `constraints_isoflux.xml` file
                description=...,   # e.g. where the two points are read from
                time=...,          # read from a database
                position_1_r=...,  # read from a database, NaN where not known
                position_1_z=...,  # read from a database
                position_2_r=...,  # read from a database, NaN where not known
                position_2_z=...,  # read from a database
            )

        for i_isoflux_boundary in range(n_isoflux_boundaries):
            magnetics.add_isoflux_boundary(
                name=...,         # need to be same in database and `constraints_isoflux_boundary.xml` file
                description=...,  # e.g. where the point is read from
                time=...,         # read from a database
                position_r=...,   # read from a database, NaN where not known
                position_z=...,   # read from a database
            )

        for i_stationary_point in range(n_stationary_points):
            magnetics.add_stationary_point(
                name=...,         # need to be same in database and `constraints_stationary_point.xml` file, e.g. "MAGNETIC_AXIS"
                description=...,  # e.g. whether the point is the magnetic axis or an X-point, and where it is read from
                time=...,         # read from a database
                position_r=...,   # read from a database, NaN where not known
                position_z=...,   # read from a database
            )

        return magnetics
        ```
        """
        ...

    def setup_core_profiles(self, pulseNo: int, settings: CoreProfilesSettings, **kwargs: dict[str, typing.Any]) -> CoreProfiles | None:
        """
        This method initialises the Rust `CoreProfiles` class, which holds an IMAS `core_profiles` IDS: the kinetic profiles which the
        pressure constraints are made from.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: `CoreProfilesSettings` object containing the XML settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        **This method is optional.**
        Like `setup_pulse_schedule` it has a default, which returns `None`: the equilibrium then cannot fit pressure. `Gsfit` only calls
        it when the equilibrium fits pressure, with the `core_profiles.xml` which `GSFIT_code_settings.xml` links, and then calls
        `Equilibrium.copy_from_core_profiles_ids` with what it returns.

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus): Which contains the kinetic measurements, e.g. the Thomson scattering

        `core_profiles/profiles_2d(itime)` is filled at a list of points in the poloidal cross-section, the irregular (R, Z) grid
        `irregular_rz_na`, one time per measurement time: the electron pressure, `electrons/pressure`, and the thermal pressure,
        `pressure_thermal`, which `Equilibrium.copy_from_core_profiles_ids` makes one pressure constraint per point from. A value which was
        not measured, or is not physical, is NaN, which the equilibrium does not fit.

        **Store the experimental times**, not the reconstruction times. `Equilibrium.copy_from_core_profiles_ids` interpolates onto them
        itself.

        Different machines will use different data stores for the kinetic measurements.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `CoreProfiles` object, or `None`.

        At a minimum this method should look like this:
        ```python
        # Initialise the CoreProfiles Rust class, with the settings it is built from, which it records in `core_profiles/code/parameters`.
        # See `serialise_python_dataclass_to_imas_parameters_input`
        core_profiles = CoreProfiles(parameters_input)

        core_profiles.set_profiles_2d_irregular_rz(
            time=...,                # read from a database, shape = [n_time]
            r=...,                   # read from a database, shape = [n_point]
            z=...,                   # read from a database, shape = [n_point]
            electrons_pressure=...,  # read from a database, shape = [n_time, n_point]
            pressure_thermal=...,    # e.g. 2 * electrons_pressure, shape = [n_time, n_point]
        )

        return core_profiles
        ```
        """

        return None

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
