import typing
from typing import Protocol

import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes
from gsfit_rs import Coils
from gsfit_rs import Dialoop
from gsfit_rs import FluxLoops
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import Passives
from gsfit_rs import Plasma
from gsfit_rs import RogowskiCoils


class DatabaseReaderProtocol(Protocol):
    """
    Protocol for reading experimental data.
    Each method is responsible for initialising one of the Rust objects:
    `bp_probes`, `coils`, `dialoop`, `flux_loops`, `isoflux`, `isoflux_boundary`, `passives`, `plasma`, and `rogowski_coils`.

    The Protocol defines the inputs and outputs of each method.
    New database readers should be implemented **all** methods.
    """

    def setup_bp_probes(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> BpProbes:
        """
        This method initialises the Rust `BpProbes` class (Mirnov coils with a direction in the poloidal plane).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_bp_probe.json`: Which contains "fitting parameters", e.g. if a probe should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the measurements and probe geometry

        Different machines will use different data stores for the probe geometry and measured signals.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `BpProbes` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the BpProbes Rust class
        bp_probes = BpProbes()

        # Add all of the BP probes
        for i_bp_probe in range(n_bp_probes):
            bp_probes.add_sensor(
                name=...,                         # need to be same in database and `sensor_weights_bp_probe.json` file
                geometry_angle_pol=...,           # read from a database
                geometry_r=...,                   # read from a database
                geometry_z=...,                   # read from a database
                fit_settings_comment=...,         # read from `sensor_weights_bp_probe.json` file
                fit_settings_expected_value=...,  # read from `sensor_weights_bp_probe.json` file
                fit_settings_include=...,         # read from `sensor_weights_bp_probe.json` file
                fit_settings_weight=...,          # read from `sensor_weights_bp_probe.json` file
                time=...,                         # read from a database
                measured=...,                     # read from a database
            )

        return bp_probes
        ```
        """
        ...

    def setup_coils(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> Coils:
        """
        This method initialises the Rust `Coils` class, which contains both the poloidal fied (PF) and toroidal field (TF) coils.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGSNKE object

        Initialising requires reading data from:
        1. Database reading (e.g. MDSplus, or FreeGSNKE object): Which contains the coil geometry and current measurements

        Different machines will use different data stores for the coil locations and currents.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Coils` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Coils Rust class
        coils = Coils()

        # Add all of the PF coils
        for i_pf_coil in range(n_pf_coils):
            coils.add_pf_coil(
                coil_name=...,  # read from a database
                coil_r=...,     # read from a database
                coil_z=...,     # read from a database
                coil_d_r=...,   # read from a database
                coil_d_z=...,   # read from a database
                time=...,       # read from a database
                measured=...,   # read from a database
            )

        # Add TF coil
        coils.add_tf_coil(
            time=...,   # read from a database
            i_rod=...,  # read from a database
        )

        return coils
        ```
        """
        ...

    def setup_dialoop(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> Dialoop:
        """
        This method initialises the Rust `Dialoop` class (the plasma's diamagnetic flux, with the vaccuum toroidal field subtracted).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_dialoop.json`: Which contains "fitting parameters", e.g. if a probe should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the measurements and the probe geometry

        Different machines will use different data stores for the measured signals.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Dialoop` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Dialoop Rust class
        dialoop = Dialoop()

        # Add all of the diamagnetic loops
        for i_dialoop in range(n_dialoops):
            dialoop.add_sensor(
                name=...,                         # read from a database
                fit_settings_comment=...,         # read from `sensor_weights_dialoop.json` file
                fit_settings_expected_value=...,  # read from `sensor_weights_dialoop.json` file
                fit_settings_include=...,         # read from `sensor_weights_dialoop.json` file
                fit_settings_weight=...,          # read from `sensor_weights_dialoop.json` file
                time=...,                         # read from a database
                measured=...,                     # read from a database
            )

        return dialoop
        ```
        """
        ...

    def setup_flux_loops(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> FluxLoops:
        """
        This method initialises the Rust `FluxLoops` class (loops going in the toroidal direction measuring poloidal flux).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_flux_loops.json`: Which contains "fitting parameters", e.g. if a probe should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the measurements and the probe geometry

        Different machines will use different data stores for the probes geometry and measured signals.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `FluxLoops` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the FluxLoops Rust class
        flux_loops = FluxLoops()

        # Add all of the BP probes
        for i_flux_loop in range(n_flux_loops):
            flux_loops.add_sensor(
                name=...,                         # need to be same in database and `sensor_weights_flux_loops.json` file
                geometry_r=...,                   # read from a database
                geometry_z=...,                   # read from a database
                fit_settings_comment=...,         # read from `sensor_weights_flux_loops.json` file
                fit_settings_expected_value=...,  # read from `sensor_weights_flux_loops.json` file
                fit_settings_include=...,         # read from `sensor_weights_flux_loops.json` file
                fit_settings_weight=...,          # read from `sensor_weights_flux_loops.json` file
                time=...,                         # read from a database
                measured=...,                     # read from a database
            )

        return flux_loops
        ```
        """
        ...

    def setup_isoflux_sensors(
        self, pulseNo: int, settings: dict[str, typing.Any], times_to_reconstruct: npt.NDArray[np.float64], **kwargs: dict[str, typing.Any]
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

    def setup_isoflux_boundary_sensors(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> IsofluxBoundary:
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

    def setup_passives(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> Passives:
        """
        This method initialises the Rust `Passives` class (toroidal conductors).

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `passive_dof_regularisation.json`: Which specifies how the conductor should be represented (e.g. constant current density, or eigenmode decomposition)
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the coordinates where the passive conductors are located

        Different machines will use different data stores for the passive conductor coordiantes.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `Passives` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Passives Rust class
        passives = Passives()

        # Add all of the passive toroidal conductors
        for i_passives in range(n_passives):
            passives.add_passive(
                name=...,                       # need to be same in database and `passive_dof_regularisation.json` file
                r=...,                          # read from a database
                z=...,                          # read from a database
                d_r=...,                        # read from a database
                d_z=...,                        # read from a database
                angle_1=...,                    # read from a database
                angle_2=...,                    # read from a database
                resistivity=...,                # read from a database
                current_distribution_type=...,  # read from `passive_dof_regularisation.json` file
                n_dof=...,                      # read from `passive_dof_regularisation.json` file
                regularisations=...,            # read from `passive_dof_regularisation.json` file
                regularisations_weight=...,     # read from `passive_dof_regularisation.json` file
            )

        return passives
        ```
        """
        ...

    def setup_plasma(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> Plasma:
        """
        This method initialises the Rust `Plasma` class.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from three locations:
        1. `GSFIT_code_settings.json`: Which contains the plasma grid size and the maximum number of iterations
        2. `source_function_p_prime.json`: Which contains the number of degrees of freedom for p_prime, and regularisation
        3. `source_function_ff_prime.json`: Which contains the number of degrees of freedom for ff_prime, and regularisation

        Normally, this method will be the same for all machines.

        The output of this method must always be a `Plasma` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the Plasma Rust class
        p_prime_source_function = gsfit_rs.EfitPolynomial(
            n_dof=...,            # read from `source_function_p_prime.json` file
            regularisations=...,  # read from `source_function_p_prime.json` file
        )
        ff_prime_source_function = gsfit_rs.EfitPolynomial(
            n_dof=...,            # read from `source_function_ff_prime.json` file
            regularisations=...,  # read from `source_function_ff_prime.json` file
        )

        # Initialise the Plasma Rust class
        plasma = Plasma(
            n_r=...,                                            # read from `GSFIT_code_settings.json` file
            n_z=...,                                            # read from `GSFIT_code_settings.json` file
            r_min=...,                                          # read from `GSFIT_code_settings.json` file
            r_max=...,                                          # read from `GSFIT_code_settings.json` file
            z_min=...,                                          # read from `GSFIT_code_settings.json` file
            z_max=...,                                          # read from `GSFIT_code_settings.json` file
            psi_n=...,                                          # read from `GSFIT_code_settings.json` file
            limit_pts_r=...,                                    # read from `GSFIT_code_settings.json` file
            limit_pts_z=...,                                    # read from `GSFIT_code_settings.json` file
            vessel_r=...,                                       # read from `GSFIT_code_settings.json` file
            vessel_z=...,                                       # read from `GSFIT_code_settings.json` file
            p_prime_source_function=p_prime_source_function,    # built above
            ff_prime_source_function=ff_prime_source_function,  # built above
        )

        return plasma
        ```
        """
        ...

    def setup_rogowski_coils(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: dict[str, typing.Any]) -> RogowskiCoils:
        """
        This method initialises the Rust `RogowskiCoils` class.

        :param pulseNo: Pulse number, used to read from the database
        :param settings: Dictionary containing the JSON settings read from the `settings` directory
        :param kwargs: Additional objects, such as FreeGNSKE object

        Initialising requires reading data from two locations:
        1. `sensor_weights_rogowski_coils.json`: Which contains "fitting parameters", e.g. if a probe should be used in the fitting
        2. Database reading (e.g. MDSplus, or FreeGNSKE object): Which contains the measurements and the Rogowski coil path

        Different machines will use different data stores for the probes geometry and measured signals.
        This Protocol allows different database readers to be selected.
        The output of this method must always be a `RogowskiCoils` object.

        At a minimum this method should look like this:
        ```python
        # Initialise the RogowskiCoils Rust class
        rogowski_coils = RogowskiCoils()

        # Add all of the BP probes
        for i_rogowski_coil in range(n_rogowski_coils):
            rogowski_coils.add_sensor(
                sensor_name=...,              # need to be same in database and `sensor_weights_flux_loops.json` file
                path_r,                       # read from a database
                path_z,                       # read from a database
                fit_settings_comment,         # read from `sensor_weights_flux_loops.json` file
                fit_settings_expected_value,  # read from `sensor_weights_flux_loops.json` file
                fit_settings_include,         # read from `sensor_weights_flux_loops.json` file
                fit_settings_weight,          # read from `sensor_weights_flux_loops.json` file
                time,                         # read from a database
                measured,                     # read from a database
                gaps_r=gaps_r,                # read from a database
                gaps_z=gaps_z,                # read from a database
                gaps_d_r=gaps_d_r,            # read from a database
                gaps_d_z=gaps_d_z,            # read from a database
                gaps_name=gaps_name,          # read from a database
            )

        return rogowski_coils
        ```
        """
        ...
