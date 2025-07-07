import os
import time as time_py
import typing

import gsfit_rs
import numpy as np
from diagnostic_and_simulation_base import DiagnosticAndSimulationBase

from .database_readers import get_database_reader
from .database_writers import get_database_writer

np.set_printoptions(linewidth=200)


class Gsfit(DiagnosticAndSimulationBase):
    """
    GSFit: Grad-Shafranov Fit

    Example usage:
    ```python
    # Note, this example will only run inside Tokamak Energy's network
    # This is a reproducton of `examples/example_01_st40_with_experimental_data.py`

    from gsfit import Gsfit

    pulseNo = 12050  # A real experimental shot
    pulseNo_write = pulseNo + 11_000_000  # Write to a "million" modelling pulse number

    gsfit_controller = Gsfit(
        pulseNo=pulseNo,
        run_name="TEST01",
        run_description="Test run",
        write_to_mds=True,
        pulseNo_write=pulseNo_write,
    )

    gsfit_controller.run()
    ```
    """

    coils: gsfit_rs.Coils
    passives: gsfit_rs.Passives
    plasma: gsfit_rs.Plasma
    bp_probes: gsfit_rs.BpProbes
    flux_loops: gsfit_rs.FluxLoops
    rogowski_coils: gsfit_rs.RogowskiCoils
    isoflux: gsfit_rs.Isoflux
    isoflux_boundary: gsfit_rs.IsofluxBoundary

    # TODO: move to DiagnosticAndSimulationBase
    def __getitem__(self, key: str) -> typing.Any:
        return self.results[key]

    # TODO: move to DiagnosticAndSimulationBase
    def print_keys(self) -> None:
        """
        Print the keys of the `self.results` nested dictionary, including subkeys.
        `self.results` is a 1:1 mapping to the MDSplus database structure.
        """

        self.results.print_keys()

    # TODO: move to DiagnosticAndSimulationBase
    def keys(self, search: str | None = None) -> typing.KeysView[str]:
        """
        Return the keys of the `self.results` nested dictionary, only the top level keys.
        `self.results` is a 1:1 mapping to the MDSplus database structure.
        """

        return self.results.keys()

    def run(self, **kwargs: typing.Any) -> None:
        """
        Run all components of GSFit
        (reading databases & initialisation, solving GS equation, and writing to data store).

        :param kwargs: Additional arguments to be passed to the database_reader. This is for FreeGS and FreeGNSKE

        This will perform the following steps:
        1. Set the environment variables
        2. Setup the timeslices to reconstruct
        3. Read in all the machine settings and initalise the following Rust implementations:
            `coils`, `passives`, `plasma`, `bp_probes`, `flux_loops`, `rogowski_coils`, `isoflux`, and `isoflux_boundary`
        4. Initialise the Greens functions
        5. Solve the GS equation
        6. Map the results to the MDSplus database structure and store in `self.results`
        7. Write the results to MDSplus
        """

        self.logger.info(f"Running Gsfit, for pulseNo={self.pulseNo}")

        self.set_environment_variables()

        self.setup_timeslices()

        # Read in all the machine settings and initalise the following Rust implementations:
        # `coils`, `passives`, `plasma`, `bp_probes`, `flux_loops`, `rogowski_coils`, `isoflux`, and `isoflux_boundary`
        self.setup_objects(**kwargs)

        # Calculate the Greens functions for all permutations between current source objects and sensors.
        self.calculate_greens()

        # Solve the GS equation
        if self.settings["GSFIT_code_settings.json"]["type_of_run"]["inverse"]:
            self.inverse_solver_rust()
        elif self.settings["GSFIT_code_settings.json"]["type_of_run"]["forward"]:
            self.logger.warning("Forward GS solver not implemented yet. Skipping forward run!!")
        else:
            raise ValueError(f"Unknown type_of_run={self.settings['GSFIT_code_settings.json']['type_of_run']}")

        self.write_results_to_mdsplus()

    def write_results_to_mdsplus(self) -> None:
        """
        Write the results to MDSplus:
        1. Results are collected from the Rust objects and stored in `self.results`,which is similar
           to a nested dictionary, and has a 1:1 mapping to the MDSplus database structure.
        2. The results are then written to MDSplus.
        """

        # Map the results to MDSplus.
        # `self.results` is a 1:1 mapping to MDSplus
        database_writer_method = self.settings["GSFIT_code_settings.json"]["database_writer"]["method"]
        database_writer = get_database_writer(database_writer_method)
        database_writer.map_results_to_database(self)

        # Do the writing to MDSplus
        if self.write_to_mds:
            self.logger.info("Writing to MDSplus")
            self._write_to_mds()

    def setup_timeslices(self) -> None:
        """
        Calculates the timeslices to reconstruct, and stores them in `self.results["TIME"]`
        """

        # Extract the timeslice settings from the JSON file
        timeslices_settings = self.settings["GSFIT_code_settings.json"]["timeslices"]

        # Calculate the timeslices
        if timeslices_settings["method"] == "arange":
            time = np.arange(
                timeslices_settings["arange"]["time_start"],
                timeslices_settings["arange"]["time_end"],
                timeslices_settings["arange"]["dt"],
            )
        elif timeslices_settings["method"] == "linspace":
            time = np.linspace(
                timeslices_settings["linspace"]["time_start"],
                timeslices_settings["linspace"]["time_end"],
                timeslices_settings["linspace"]["n_time"],
            )
        elif timeslices_settings["method"] == "user_defined":
            time = np.array(timeslices_settings["user_defined"])

        # Store the times to reconstruct
        self.results["TIME"] = time

    def set_environment_variables(self) -> None:
        """
        Set the system environment variables before running.
        Presently, this defines the numer of cores for the Rust code
        """

        # Set the number of cores (for Rayon, Rust's parallelisation library)
        os.environ["RAYON_NUM_THREADS"] = str(self.settings["GSFIT_code_settings.json"]["RAYON_NUM_THREADS"])

    def inverse_solver_rust(self) -> None:
        """
        Solve the "inverse" problem, i.e. reconstruction
        """

        # Extract objects from class
        coils = self.coils
        passives = self.passives
        plasma = self.plasma
        bp_probes = self.bp_probes
        flux_loops = self.flux_loops
        rogowski_coils = self.rogowski_coils
        isoflux = self.isoflux
        isoflux_boundary = self.isoflux_boundary

        times_to_reconstruct = self.results["TIME"]

        self.logger.info(msg="About to call: `gsfit_rs.solve_inverse_problem`")
        # Note: the solution to the GS equation is stored inside: `plasma`, `passives`, `bp_probes`, `flux_loops`, and `rogowski_coils`
        t0 = time_py.time()
        gsfit_rs.solve_inverse_problem(
            plasma,
            coils,
            passives,
            bp_probes,
            flux_loops,
            rogowski_coils,
            isoflux,
            isoflux_boundary,
            times_to_reconstruct,
            self.settings["GSFIT_code_settings.json"]["numerics"]["n_iter_max"],
            self.settings["GSFIT_code_settings.json"]["numerics"]["n_iter_min"],
            self.settings["GSFIT_code_settings.json"]["numerics"]["n_iter_no_vertical_feedback"],
            self.settings["GSFIT_code_settings.json"]["numerics"]["gs_error"],
            self.settings["GSFIT_code_settings.json"]["numerics"]["anderson_mixing"]["use"],
            self.settings["GSFIT_code_settings.json"]["numerics"]["anderson_mixing"]["mixing_from_previous_iter"],
        )
        t1 = time_py.time()
        self.logger.info(msg=f"Finished: `gsfit_rs.solve_inverse_problem` time = {(t1 - t0) * 1e3}ms")

    def calculate_greens(self) -> None:
        """
        Calculates the Greens table for all permutations between current source objects and sensors.
        """

        # Get Rust classes out of self
        coils = self.coils
        passives = self.passives
        plasma = self.plasma
        bp_probes = self.bp_probes
        flux_loops = self.flux_loops
        rogowski_coils = self.rogowski_coils
        isoflux = self.isoflux
        isoflux_boundary = self.isoflux_boundary

        # Greens with coils
        plasma.greens_with_coils(coils)
        bp_probes.greens_with_coils(coils)
        flux_loops.greens_with_coils(coils)
        rogowski_coils.greens_with_coils(coils)
        isoflux.greens_with_coils(coils)
        isoflux_boundary.greens_with_coils(coils)
        self.logger.info("Finished Greens with coils")

        # Greens with passives
        plasma.greens_with_passives(passives)
        bp_probes.greens_with_passives(passives)
        flux_loops.greens_with_passives(passives)
        rogowski_coils.greens_with_passives(passives)
        isoflux.greens_with_passives(passives)
        isoflux_boundary.greens_with_passives(passives)
        self.logger.info("Finished Greens with passives")

        # Greens with plasma
        bp_probes.greens_with_plasma(plasma)
        flux_loops.greens_with_plasma(plasma)
        rogowski_coils.greens_with_plasma(plasma)
        isoflux.greens_with_plasma(plasma)
        isoflux_boundary.greens_with_plasma(plasma)
        self.logger.info("Finished Greens with plasma")

    def setup_objects(self, **kwargs: dict[str, typing.Any]) -> None:
        """
        Initialises the Rust objects needed to run the GSFit inverse solver:
        `coils`, `passives`, `plasma`, `bp_probes`, `flux_loops`, `rogowski_coils`, `isoflux`, and `isoflux_boundary`

        Different machines will use different data stores (e.g. MDSplus, or FreeGNSKE object).
        New readers for different devices / forward GS solvers can be added to:
        `python/gsfit/database_readers/__init__.py` and `python/gsfit/database_readers/<new_reader_name>`

        See: `gsfit/database_readers/interface.py` for a description of the interfaces.
        """

        # Get the database_reader
        database_reader_method = self.settings["GSFIT_code_settings.json"]["database_reader"]["method"]
        database_reader = get_database_reader(database_reader_method)

        # Initialise and store the Rust implementations
        self.coils = database_reader.setup_coils(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`coils` initialised")
        self.bp_probes = database_reader.setup_bp_probes(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`bp_probes` initialised")
        self.flux_loops = database_reader.setup_flux_loops(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`flux_loops` initialised")
        self.rogowski_coils = database_reader.setup_rogowski_coils(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`rogowski_coils` initialised")
        self.passives = database_reader.setup_passives(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`passives` initialised")
        self.plasma = database_reader.setup_plasma(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`plasma` initialised")
        times_to_reconstruct = self.results["TIME"]
        self.isoflux = database_reader.setup_isoflux_sensors(pulseNo=self.pulseNo, settings=self.settings, times_to_reconstruct=times_to_reconstruct, **kwargs)
        self.logger.info(msg="`isoflux` initialised")
        self.isoflux_boundary = database_reader.setup_isoflux_boundary_sensors(pulseNo=self.pulseNo, settings=self.settings, **kwargs)
        self.logger.info(msg="`isoflux_boundary` initialised")
