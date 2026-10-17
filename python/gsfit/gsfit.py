import os
import time as time_py
import typing

import gsfit_rs
import numpy as np
from diagnostic_and_simulation_base import DiagnosticAndSimulationBase

from .database_readers import get_database_reader
from .database_writers import get_database_writer
from .settings_schema.python.gsfit_code_settings import GsfitCodeSettings

if typing.TYPE_CHECKING:
    from imas.ids_toplevel import IDSToplevel

np.set_printoptions(linewidth=200)


class Gsfit(DiagnosticAndSimulationBase[GsfitCodeSettings]):
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

    # The dataclass which `GSFIT_code_settings.xml`, and the XML files it links to, are read into, as `self.settings_xml`
    settings_xml_class = GsfitCodeSettings

    equilibrium: gsfit_rs.Equilibrium
    pf_active: gsfit_rs.PfActive
    pf_passive: gsfit_rs.PfPassive
    isoflux: gsfit_rs.Isoflux
    isoflux_boundary: gsfit_rs.IsofluxBoundary
    pressure_sensors: gsfit_rs.Pressure
    stationary_point: gsfit_rs.StationaryPoint
    tf: gsfit_rs.Tf
    wall: gsfit_rs.Wall

    # Optional: only a `database_reader` which reads gap definitions builds one. `None` means
    # the equilibrium has no gaps
    pulse_schedule: gsfit_rs.PulseSchedule | None = None

    # Set by `write_results_to_database` when the `imas` database_writer is selected: the
    # populated IMAS `equilibrium` IDS. `None` for every other writer
    equilibrium_ids: "IDSToplevel | None" = None

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
        1. Start creating the MDSplus nodes, in a separate process
        2. Set the environment variables
        3. Setup the timeslices to reconstruct
        4. Read in all the machine settings and initalise the following Rust implementations:
            `equilibrium`, `pf_active`, `pf_passive`, `tf`, `wall`, `magnetics`, `isoflux`, `isoflux_boundary`, and `stationary_point`,
            and the optional `pulse_schedule`
        5. Initialise the Greens functions
        6. Solve the GS equation
        7. Map the results to the MDSplus database structure and store in `self.results`
        8. Wait for the MDSplus nodes, then write the results into them

        Steps 1 and 8 are skipped for the `imas` database_writer, which writes nothing.
        """

        self.logger.info(f"Running Gsfit, for pulseNo={self.pulseNo}")

        # Creating the MDSplus nodes needs nothing but the settings, which have already been read,
        # so start it now and let it run while we read the databases and solve the GS equation.
        # The settings are the user's to change up to this point, so the values in force now are
        # the ones the node creation is given; from here on they are fixed.
        # `write_results_to_database` waits for it, and checks that it worked, before writing
        if self.writes_to_mdsplus():
            self.logger.info("Creating the MDSplus nodes in a separate process")
            self.start_mds_node_creation(workflows=self.get_workflow_names())

        self.set_environment_variables()

        self.setup_timeslices()

        # Read in all the machine settings and initalise the following Rust implementations:
        # `equilibrium`, `pf_active`, `pf_passive`, `tf`, `wall`, `magnetics`, `isoflux`, `isoflux_boundary`, and `stationary_point`,
        # and the optional `pulse_schedule`
        self.setup_objects(**kwargs)

        # Calculate the Greens functions for all permutations between current source objects and sensors.
        self.calculate_greens()

        # Solve the GS equation
        self.inverse_solver_rust()

        self.write_results_to_database()

    def writes_to_mdsplus(self) -> bool:
        """
        Whether this run will write to MDSplus at all.

        `write_to_mds` is the user's switch, but the `imas` database_writer does not write to
        MDSplus whatever it is set to: it builds an IMAS `equilibrium` IDS instead. Asking this
        rather than `write_to_mds` keeps the MDSplus node creation, which starts before anything
        else in `run`, in step with the writing that happens at the end.
        """

        database_writer_method = self.settings_xml.database_reader.method

        return self.write_to_mds and database_writer_method != "imas"

    def write_results_to_database(self) -> None:
        """
        Write the results to the database:
        1. Results are collected from the Rust objects and stored in `self.results`, which is
           similar to a nested dictionary, and has a 1:1 mapping to the MDSplus database structure.
        2. We wait for the MDSplus nodes, which `run` started creating in a separate process, and
           check that the creation did not fail. If `run` was not used, the nodes are created here.
        3. The results are then written to MDSplus.

        The `imas` database_writer does neither 2 nor 3. It returns an IMAS `equilibrium` IDS,
        which is kept on `self.equilibrium_ids` for the caller to use, and nothing is written.
        """

        # Map the results to the database structure.
        # For the MDSplus writers `self.results` is a 1:1 mapping to MDSplus
        database_writer_methods = self.settings_xml.database_writer.method
        for database_writer_method in database_writer_methods:
            database_writer = get_database_writer(database_writer_method)
            if database_writer_method != "imas":
                database_writer.map_results_to_database(self)
            else:
                self.equilibrium_ids = database_writer.map_results_to_database(self)
                self.logger.info("IMAS IDS populated; access it via `self.equilibrium_ids`")

            # Do the writing to MDSplus
            self.logger.info(f"pulseNo = {self.pulseNo} pulseNo_write = {self.pulseNo_write} run_name = {self.run_name}")
            if self.writes_to_mdsplus():
                # The nodes must exist, and must have been created without error, before we write into them
                self.wait_for_mds_node_creation(workflows=self.get_workflow_names())

                self.logger.info("Writing to database")
                self._write_data_to_mds()
                if database_writer_method == "tokamak_energy_mdsplus_new":
                    from .database_writers.tokamak_energy_mdsplus_new.create_mdsplus_links import create_mdsplus_links

                    create_mdsplus_links(pulseNo_write=self.pulseNo_write, run_name=self.run_name, tree_name=self.analysis_name)

    def get_workflow_names(self) -> list[str]:
        """
        The names of the input codes which will be stored under `INPUT.WORKFLOW`,
        for example `["elmag", "mag", "psu2coil"]`.

        Which codes these are depends on the `database_reader`, and whether they are stored at all
        depends on the `database_writer`, so the selected `database_writer` is asked. The answer
        comes from the settings alone, which means the `INPUT.WORKFLOW` MDSplus nodes can be
        created before GSFit has read any data.
        """

        database_writer_methods = self.settings_xml.database_writer.method
        # TODO: need to fix this for multiple database writers...
        for database_writer_method in database_writer_methods:
            # import pdb; pdb.set_trace()
            database_writer = get_database_writer(database_writer_method)

        return database_writer.get_workflow_names(self)

    def setup_timeslices(self) -> None:
        """
        Calculates the timeslices to reconstruct, and stores them in `self.results["TIME"]`
        """

        # Extract the timeslice settings from the JSON file
        timeslices_settings = self.settings_xml.ids.equilibrium.timeslices

        # Calculate the timeslices
        match timeslices_settings.method:
            case "arange":
                time = np.arange(
                    timeslices_settings.arange.time_start,
                    timeslices_settings.arange.time_end,
                    timeslices_settings.arange.dt,
                )
            case "linspace":
                time = np.linspace(
                    timeslices_settings.linspace.time_start,
                    timeslices_settings.linspace.time_end,
                    timeslices_settings.linspace.n_time,
                )
            case "user_defined":
                time = np.array(timeslices_settings.user_defined)
            case "good_pressure_sensors":
                # TODO: needs implementing...
                raise ValueError("Needs implementing...")
                # # Reconstruct only the time-slices where the pressure sensors are "good".
                # # The `BAD_MA` reading lives in the `st40_mdsplus_with_digital_filter` reader.
                # from .database_readers.st40_mdsplus_with_digital_filter.setup_pressure_sensors import get_good_pressure_sensor_times

                # time = get_good_pressure_sensor_times(pulseNo=self.pulseNo, settings=self.settings)
            case _:
                raise ValueError(f"Unknown timeslices method: {timeslices_settings.method}")

        # Store the times to reconstruct
        self.results["TIME"] = time

    def set_environment_variables(self) -> None:
        """
        Set the system environment variables before running.
        Presently, this defines the numer of cores for the Rust code
        """

        # Set the number of cores (for Rayon, Rust's parallelisation library)
        n_rayon_threads: int = self.settings_xml.ids.equilibrium.numerics.RAYON_NUM_THREADS
        os.environ["RAYON_NUM_THREADS"] = str(n_rayon_threads)
        self.logger.info(msg=f"RAYON_NUM_THREADS = {n_rayon_threads}")

    def inverse_solver_rust(self) -> None:
        """
        Solve the "inverse" problem, i.e. reconstruction
        """

        # Extract objects from class
        equilibrium = self.equilibrium
        isoflux = self.isoflux
        isoflux_boundary = self.isoflux_boundary
        pressure_sensors = self.pressure_sensors
        stationary_point = self.stationary_point

        self.logger.info(msg="About to call: `gsfit_rs.solve_grad_shafranov`")
        # Note: the solution to the GS equation is stored inside `equilibrium`
        tic = time_py.time()
        gsfit_rs.solve_grad_shafranov(
            equilibrium,
            isoflux,
            isoflux_boundary,
            pressure_sensors,
            stationary_point,
        )
        toc = time_py.time()
        self.logger.info(msg=f"Finished: `gsfit_rs.solve_inverse_problem` time = {(toc - tic) * 1e3:,.2f}ms")

    def calculate_greens(self) -> None:
        """
        Calculates the Greens table for all permutations between current source objects and sensors.
        """

        # Get Rust classes out of self
        magnetics = self.magnetics
        pf_active = self.pf_active
        pf_passive = self.pf_passive
        equilibrium = self.equilibrium
        isoflux = self.isoflux
        isoflux_boundary = self.isoflux_boundary
        pressure_sensors = self.pressure_sensors
        stationary_point = self.stationary_point

        # Greens with the PF coils
        tic = time_py.time()
        equilibrium.greens_with_pf_active(pf_active)
        isoflux.greens_with_pf_active(pf_active)
        isoflux_boundary.greens_with_pf_active(pf_active)
        pressure_sensors.greens_with_pf_active(pf_active)
        stationary_point.greens_with_pf_active(pf_active)
        toc = time_py.time()
        self.logger.info(f"Finished Greens with the PF coils;  {(toc - tic) * 1e3:,.2f}ms")

        # Greens with the passive conductors
        tic = time_py.time()
        equilibrium.greens_with_pf_passive(pf_passive)
        isoflux.greens_with_pf_passive(pf_passive)
        isoflux_boundary.greens_with_pf_passive(pf_passive)
        pressure_sensors.greens_with_pf_passive(pf_passive)
        stationary_point.greens_with_pf_passive(pf_passive)
        toc = time_py.time()
        self.logger.info(f"Finished Greens with the passive conductors;  {(toc - tic) * 1e3:,.2f}ms")

        # Greens of the magnetic sensors, with the plasma, the PF coils and the passive conductors. The PF coils and the passive
        # conductors are in the order of the Greens tables above, so these come after them
        tic = time_py.time()
        equilibrium.greens_with_magnetics(magnetics, pf_active, pf_passive)
        toc = time_py.time()
        self.logger.info(f"Finished Greens of the magnetic sensors;  {(toc - tic) * 1e3:,.2f}ms")

        # Greens with plasma
        tic = time_py.time()
        isoflux.greens_with_plasma(equilibrium)
        isoflux_boundary.greens_with_plasma(equilibrium)
        pressure_sensors.greens_with_plasma(equilibrium)
        stationary_point.greens_with_plasma(equilibrium)
        toc = time_py.time()
        self.logger.info(f"Finished Greens with plasma;  {(toc - tic) * 1e3:,.2f}ms")

    def setup_objects(self, **kwargs: dict[str, typing.Any]) -> None:
        """
        Initialises the Rust objects needed to run the GSFit inverse solver:
        `equilibrium`, `pf_active`, `pf_passive`, `tf`, `wall`, `magnetics`, `isoflux`, `isoflux_boundary`, and `stationary_point`,
        and the optional `pulse_schedule`, which is `None` unless the `database_reader` implements `setup_pulse_schedule`

        Different machines will use different data stores (e.g. MDSplus, or FreeGNSKE object).
        New readers for different devices / forward GS solvers can be added to:
        `python/gsfit/database_readers/__init__.py` and `python/gsfit/database_readers/<new_reader_name>`

        See: `gsfit/database_readers/interface.py` for a description of the interfaces.
        """

        # Get the database_reader
        database_reader_method = self.settings_xml.database_reader.method
        database_reader = get_database_reader(database_reader_method)
        settings_xml = self.settings_xml

        # `equilibrium` is in IDS structure, built from its settings alone
        tic = time_py.time()
        self.equilibrium = database_reader.setup_equilibrium(pulseNo=self.pulseNo, settings=settings_xml.ids.equilibrium, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`equilibrium` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        # `wall` is in IDS structure
        tic = time_py.time()
        self.wall = database_reader.setup_wall(pulseNo=self.pulseNo, settings=settings_xml.ids.wall, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`wall` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        # `tf` is in IDS structure
        tic = time_py.time()
        self.tf = database_reader.setup_tf(pulseNo=self.pulseNo, settings=settings_xml.ids.tf, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`tf` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        # `pulse_schedule` is in IDS structure
        tic = time_py.time()
        self.pulse_schedule = database_reader.setup_pulse_schedule(pulseNo=self.pulseNo, settings=settings_xml.ids.pulse_schedule, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`pulse_schedule` initialised;  {(toc - tic) * 1e3:,.2f}ms")


        # `pf_passive` is in IDS structure
        tic = time_py.time()
        self.pf_passive = database_reader.setup_pf_passive(pulseNo=self.pulseNo, settings=self.settings_xml.ids.pf_passive, settings_equilibrium=self.settings_xml.ids.equilibrium, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`pf_passive` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        # `pf_active` is in IDS structure
        tic = time_py.time()
        self.pf_active = database_reader.setup_pf_active(pulseNo=self.pulseNo, settings=self.settings_xml.ids.pf_active, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`pf_active` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        times_to_reconstruct = self.results["TIME"]

        tic = time_py.time()
        self.isoflux = database_reader.setup_isoflux_sensors(pulseNo=self.pulseNo, settings=self.settings_xml.ids.magnetics, settings_isoflux_sensors=self.settings_xml.ids.equilibrium.constraints.isoflux, times_to_reconstruct=times_to_reconstruct, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`isoflux` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        tic = time_py.time()
        self.isoflux_boundary = database_reader.setup_isoflux_boundary_sensors(pulseNo=self.pulseNo, settings=self.settings_xml.ids.magnetics, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`isoflux_boundary` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        tic = time_py.time()
        self.pressure_sensors = database_reader.setup_pressure_sensors(pulseNo=self.pulseNo, settings=self.settings_xml.ids.thomson_scattering, settings_pressure_sensors=self.settings_xml.ids.equilibrium.constraints.pressure, **kwargs)
        toc = time_py.time()
        self.logger.info(msg=f"`pressure_sensors` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        tic = time_py.time()
        # TODO: temporary removal
        from gsfit_rs import StationaryPoint
        self.stationary_point = StationaryPoint()
        # self.stationary_point = database_reader.setup_stationary_point_sensors(
        #     pulseNo=self.pulseNo,
        #     settings=self.settings,
        #     times_to_reconstruct=times_to_reconstruct,
        #     **kwargs,
        # )
        toc = time_py.time()
        self.logger.info(msg=f"`stationary_point` initialised;  {(toc - tic) * 1e3:,.2f}ms")

        # Copy IDS data into the equilibrium
        self.equilibrium.copy_from_wall_ids(self.wall)
        self.equilibrium.copy_from_tf_ids(self.tf)
        self.equilibrium.copy_from_pf_active_ids(self.pf_active)
        self.equilibrium.copy_from_pf_passive_ids(self.pf_passive)
        # `pulse_schedule` is optional. Only copy it in if it is not None
        if self.pulse_schedule is not None:
            # is this ever None?
            self.equilibrium.copy_from_pulse_schedule_ids(self.pulse_schedule)

        self.magnetics = database_reader.setup_magnetics(
            pulseNo=self.pulseNo,
            settings=self.settings_xml.ids.magnetics,
        )

        self.equilibrium.copy_from_magnetics_ids(self.magnetics)

        # import pdb; pdb.set_trace()
