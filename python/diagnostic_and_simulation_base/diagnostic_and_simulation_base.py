import json
import logging
import multiprocessing
import sys
import textwrap
import traceback
import typing
from importlib import resources as resources_py  # BUXTON: this is deprecated and needs removing
from multiprocessing.connection import Connection
from pathlib import Path
from time import time as time_py

import f90nml

from . import version_storage
from .nested_dictionary import NestedDict
from .utility import logging as util_logging
from .utility.make_settings_json import make_settings_json


def _create_script_nodes(
    analysis_name: str,
    pulseNo_write: int,
    pulseNo_cal: int | None,
    run_name: str,
    run_description: str,
    workflows: list[str],
    link_best: bool,
) -> float:
    """Creates the MDSplus nodes. Returns how long the creation took [second].

    A module level function taking nothing but primitives, because it is also the body of the
    child process started by `start_mds_node_creation`, which must not reach back into the parent.
    """

    # Lazy loading of `standard_utility` because it's specific to Tokamak Energy.
    import standard_utility as util

    tic = time_py()
    util.create_script_nodes(
        script_name=analysis_name,
        pulseNo_write=pulseNo_write,
        pulseNo_cal=pulseNo_cal,
        run_name=run_name,
        run_info=run_description,
        workflows=workflows,
        link_best=link_best,
    )
    toc = time_py()

    return toc - tic


def _create_script_nodes_in_child(connection: Connection, **keyword_arguments: typing.Any) -> None:
    """Runs `_create_script_nodes` in the child process, and reports the outcome to the parent.

    The outcome goes back down `connection` because raising in a child process only prints to
    stderr and sets an exit code, which does not tell the parent what went wrong. The message is a
    few hundred bytes at most, so it always fits in the pipe buffer and the child never blocks
    sending it.
    """

    try:
        connection.send({"seconds": _create_script_nodes(**keyword_arguments)})
    except BaseException:
        connection.send({"traceback": traceback.format_exc()})
    finally:
        connection.close()


class DiagnosticAndSimulationBase:
    """Class for creating Diagnostic Analysis codes, such as: `Gas`, `Efit`, `Ppts`, `Zeff_pi`"""

    log_records: list[logging.LogRecord] = []

    @property
    def log_string(self) -> str:
        return util_logging.format_logs(self.log_records)

    def __init__(
        self,
        pulseNo: int,
        run_name: str,
        run_description: str = "Standard run with default settings",
        settings_path: str = "default",
        write_to_mds: bool = True,
        pulseNo_write: int | None = None,
        analysis_name: str | None = None,
        link_run_to_best: bool = False,
    ) -> None:
        """Class constructor

        :param pulseNo: pulse number
        :param run_name: run_name to save to MDSplus
        :param run_description: help string for MDSplus Tree
        :param settings_path: location where code inputs are stored
        :param write_to_mds: flag to turn on / off writing to MDSplus
        :param pulseNo_write: pulse number in which data is to be written, if different from the current pulse
        :param analysis_name: for example "EFITP" is analysis using the "EFIT code"
        :param link_run_to_best: BEST will be linked to the run
        """
        self.analysis_time_start = time_py()

        # code_name: is the package name from "pip list" in upper case (note, this includes "_")
        code_name = self.__class__.__module__.split(".")[0].upper()  # BUXTON: I don't like this!!

        # analysis_name
        if analysis_name is None:
            analysis_name = code_name

        # Create a logger with the module name
        # Get the logging_level
        logging_level = util_logging.logger.getEffectiveLevel()

        # Create logger
        self.logger = logging.getLogger(analysis_name)
        # Don't propagate messages to the root logger; `mdsthin` uses the root logger and this avoids double logging
        self.logger.propagate = False
        # Set the logging level
        self.logger.setLevel(level=logging_level)

        # Check if the logger already has handlers to avoid adding duplicates
        if not self.logger.handlers:
            # Create and add handler to emit the logs to "stadnard out",
            # i.e. printing to the terminal (standard behaviour)
            handler_to_standard_out = logging.StreamHandler(sys.stdout)
            handler_to_standard_out.setFormatter(util_logging.CustomFormatter())
            self.logger.addHandler(handler_to_standard_out)

            # Create and add handler to emit the log "records" into `self.log_records`
            handler_to_list = util_logging.EmitLogToListHandler(self.log_records)
            self.logger.addHandler(handler_to_list)

        # `log_records` is a class-level list we "clear" it when a new instance is created
        self.log_records.clear()

        # Store inputs
        self.pulseNo = pulseNo
        self.code_name = code_name
        self.analysis_name = analysis_name
        self.run_name = run_name.upper()
        self.run_description = run_description
        self.write_to_mds = write_to_mds
        self.link_run_to_best = link_run_to_best

        # Determine which pulseNo to write to and store in class object
        if pulseNo_write is None:
            pulseNo_write = pulseNo
        self.pulseNo_write = pulseNo_write

        # Test if "settings_path" is a directory.
        # If directory doesn't exist, then treat as "relative path"
        if not (Path(settings_path).is_dir()):
            python_module = self.code_name.lower()
            settings_path = f"{resources_py.files(python_module)}/settings/{settings_path}"

        # Test if settings directory exists
        if not Path(settings_path).is_dir():
            raise FileNotFoundError("settings directory not found")

        # Store the resolved settings path
        self.settings_path = settings_path

        # Create empty settings dictionary
        self.settings = {}  # type: dict[str, typing.Any]

        # State for the background MDSplus node creation, see `start_mds_node_creation`
        self._mds_node_creation_process: multiprocessing.process.BaseProcess | None = None
        self._mds_node_creation_connection: Connection | None = None
        self._mds_nodes_created = False

        self._load_settings_from_files()

        # Create results dictionary, with the stuff we already know
        # Important: results is a 1:1 mapping to MDSplus data-strucutre
        # TO-DO: Change this to a pre-populated dictionary from the *.csv file
        self.results = NestedDict()
        self.results["CODE_VERSION"]["COMPUTER"] = version_storage.__computer__
        self.results["CODE_VERSION"]["DATETIME"] = version_storage.__datetime__
        self.results["CODE_VERSION"]["GIT_ID"] = version_storage.__git_short_hash__
        self.results["CODE_VERSION"]["LIBRARY"] = version_storage.__python_library__
        self.results["CODE_VERSION"]["PYTHON"] = version_storage.__python__
        self.results["CODE_VERSION"]["USER"] = version_storage.__user__
        self.results["CODE_VERSION"]["VERSION"] = version_storage.__version__

    def __repr__(self) -> str:
        """Print to screen"""
        string_output = ""
        string_output += "╔═════════════════════════════════════════════════════════════════════════════╗\n"
        string_output += f"║ {f' <{self.__class__.__name__}>':<75} ║\n"
        string_output += f"║ {f' {version_storage.__version__}':<75} ║\n"
        string_output += f"║ {' ':<75} ║\n"
        string_output += f"║ {' pulseNo = ' + f'{self.pulseNo:_}':<75} ║\n"
        string_output += f"║ {' pulseNo_write = ' + f'{self.pulseNo_write:_}':<75} ║\n"
        string_output += f"║ {' run_name = ' + str(self.run_name):<75} ║\n"
        string_output += f"║ {' run_description = ' + str(self.run_description):<75} ║\n"
        string_output += f"║ {' settings_path = ...':<75} ║\n"
        return string_output

    def _load_settings_from_files(self) -> None:
        """Look in the "settings_path" directory and load all settings files.
        Will recursively load *.json and *.nml files

        TODO: add *.csv reader #SUNDAR - is ths even needed?
        """

        settings_path = self.settings_path

        # Load *.json settings, including sub-directories
        for file in Path(settings_path).glob("**/*.json"):
            with open(file, "r") as file_id:
                step_name = f'Loading settings from: "{file.name}"'
                try:
                    relative_path = str(file.relative_to(settings_path))
                    self.settings[relative_path] = json.load(file_id)
                    self.logger.info(msg=step_name)
                except Exception as exception_obj:
                    self.logger.exception(msg=step_name)
                    raise exception_obj

        # Load *.nml (namelist) settings, including sub-directories
        for file in Path(settings_path).glob("**/*.nml"):
            with open(file, "r") as file_id:
                step_name = f"Loading settings from {file.name}"
                try:
                    relative_path = str(file.relative_to(settings_path))
                    self.settings[relative_path] = f90nml.read(file_id)
                    self.logger.info(msg=step_name)
                except Exception as exception_obj:
                    self.logger.exception(msg=step_name)
                    raise exception_obj

    def _get_pulseNo_cal(self) -> int | None:
        """The calibration pulse number, taken from this code's MDSplus settings file.

        Returns `None` when the code has no MDSplus settings file, or the file does not name a
        calibration pulse.
        """

        mdsplus_settings_file: str = f"{self.analysis_name}_mdsplus_settings.json"
        pulseNo_cal: int | None
        if mdsplus_settings_file in self.settings:
            pulseNo_cal = self.settings[mdsplus_settings_file].get("calibration", {}).get("pulse", None)
        else:
            pulseNo_cal = None

        return pulseNo_cal

    def _mds_node_arguments(self, workflows: list[str] | None = None) -> dict[str, typing.Any]:
        """Everything the node creation needs, taken from the settings as they stand right now.

        This is deliberately a snapshot. The settings belong to the user right up until the
        analysis starts - which `MAG` or `PSU2COIL` run to read from, for example - so the node
        creation has to use the values that were in force when it was started, not whatever they
        become afterwards. Once the analysis is running they are fixed, which is what makes it
        safe to hand this to another process and not look at it again.

        :param workflows: names of the input codes to create the `INPUT.WORKFLOW` structure for.
            When `None` the names are read out of `self.results`, which means the results must
            already have been mapped to the database structure.
        """

        if workflows is None:
            workflows = list(self.results["INPUT"]["WORKFLOW"].keys())

        return {
            "analysis_name": self.analysis_name,
            "pulseNo_write": self.pulseNo_write,
            "pulseNo_cal": self._get_pulseNo_cal(),
            "run_name": self.run_name,
            "run_description": self.run_description,
            "workflows": workflows,
            "link_best": self.link_run_to_best,
        }

    def _create_mds_nodes(self, workflows: list[str] | None = None) -> None:
        """Creates the MDSplus nodes which `_write_data_to_mds` later writes the data into,
        here, in this process.

        Everything this needs comes from the settings, which are read in the constructor, so it
        does not have to wait for the analysis to finish. See `start_mds_node_creation`.

        :param workflows: see `_mds_node_arguments`.
        """

        seconds = _create_script_nodes(**self._mds_node_arguments(workflows))
        self._mds_nodes_created = True

        # Logged so that we can see how much of the node creation was hidden behind the analysis
        self.logger.info(msg=f"MDSplus nodes created;  {seconds * 1e3:,.2f}ms")

    def _write_data_to_mds(self) -> None:
        """Writes the data into the MDSplus nodes created by `_create_mds_nodes`.

        The nodes must already exist, so either `_create_mds_nodes` or `wait_for_mds_node_creation`
        has to have been called first.
        """

        # Lazy loading of `standard_utility` because it's specific to Tokamak Energy.
        import standard_utility as util

        # Add settings files to results.
        # We do this right at the end, as they can be programatially changed, e.g. for scans
        self.results["INPUT"]["SETTINGS"] = make_settings_json(data=self.settings, json_indent=2)

        util.write_script_data(
            script_name=self.analysis_name,
            pulseNo_write=self.pulseNo_write,
            data_to_write=self.results.to_dictionary(),
            pulseNo_cal=self._get_pulseNo_cal(),
            run_name=self.run_name,
            run_description=self.run_description,
            force_write=True,
        )

    def _write_to_mds(self) -> None:
        """Creates the MDSplus nodes and then writes the data into them, one after the other.

        A code which wants to overlap the node creation with its own computation should instead
        call `start_mds_node_creation` early on, and then `wait_for_mds_node_creation` followed by
        `_write_data_to_mds` at the end.
        """

        self._create_mds_nodes()
        self._write_data_to_mds()

    def start_mds_node_creation(self, workflows: list[str] | None = None) -> None:
        """Starts creating the MDSplus nodes in a separate process.

        Creating the nodes is talking to the MDSplus server, and needs none of the results, so it
        can run while the analysis is still computing. `wait_for_mds_node_creation` must be called
        before writing, both to make sure the nodes exist and to check that the creation worked.

        **A separate process, not a thread.** The analysis spends most of its wall time inside
        `gsfit_rs`, which holds the GIL for the whole call, so a thread makes no progress during
        exactly the part of the run we want to hide the node creation behind. Measured on a
        480 time-slice run: as a thread the node creation ran for 279 s of wall clock but only
        progressed for about 95 s of it, all of it during the database reads.

        **`fork`, not `spawn`.** `spawn` re-imports the main module in the child, and GSFit's entry
        scripts run the analysis at module level with no `if __name__ == "__main__"` guard, so the
        child would start a second reconstruction. `forkserver` re-imports it too, so it is no help
        either. `fork` does not re-import, and is safe here because nothing has started a thread yet
        at the point the analysis calls this.

        **Only where `fork` can be used.** Windows has no `fork` at all, and on macOS forking a
        process which has already initialised Apple's frameworks can abort the child. On those
        platforms the nodes are created here and now instead, which costs the overlap with the
        analysis but is correct everywhere.

        :param workflows: see `_mds_node_arguments`.
        """

        if "fork" not in multiprocessing.get_all_start_methods() or sys.platform == "darwin":
            self._create_mds_nodes(workflows)
            return

        context = multiprocessing.get_context("fork")
        self._mds_node_creation_connection, connection_child = context.Pipe(duplex=False)

        # `daemon=True`, so that an analysis which fails before it reaches the writing cannot leave
        # a process behind that nobody is going to wait for
        self._mds_node_creation_process = context.Process(
            target=_create_script_nodes_in_child,
            args=(connection_child,),
            kwargs=self._mds_node_arguments(workflows),
            name="mds_node_creation",
            daemon=True,
        )
        self._mds_node_creation_process.start()

        # The child now holds the only sending end. Closing ours means that if the child dies
        # without sending anything, the pipe reports end-of-file rather than waiting for a writer
        # which no longer exists
        connection_child.close()

    def wait_for_mds_node_creation(self, workflows: list[str] | None = None) -> None:
        """Waits for the MDSplus node creation to finish, and raises if it did not succeed.
        The data must not be written unless this returns without raising.

        If `start_mds_node_creation` was never called then the nodes are created here instead, so
        this is always safe to call before writing. The same applies when it was called but had to
        create them synchronously, in which case there is nothing left to wait for.

        :param workflows: only used when the nodes have to be created here; see
            `_mds_node_arguments`.
        """

        if self._mds_nodes_created:
            return

        process = self._mds_node_creation_process
        connection = self._mds_node_creation_connection
        self._mds_node_creation_process = None
        self._mds_node_creation_connection = None

        if process is None:
            self._create_mds_nodes(workflows=workflows)
            return

        process.join()

        # The child sends its result before exiting, so by now it is sitting in the pipe. `poll`
        # rather than a blocking `recv`, because a child which was killed sends nothing at all -
        # and for that same child `poll` reports readable but `recv` then raises `EOFError`, which
        # is the end of the pipe rather than an error worth propagating
        result: dict[str, typing.Any] = {}
        if connection is not None:
            try:
                if connection.poll():
                    result = connection.recv()
            except EOFError:
                pass
            connection.close()

        if "traceback" in result:
            self.logger.error(msg=f"Creating the MDSplus nodes failed; not writing\n{result['traceback']}")
            raise RuntimeError(f"Creating the MDSplus nodes failed:\n{result['traceback']}")

        if "seconds" not in result:
            # Killed, or died without managing to say why. Either way the nodes are not trustworthy
            self.logger.error(msg="Creating the MDSplus nodes failed; not writing")
            raise RuntimeError(
                f"Creating the MDSplus nodes failed: the process exited with code {process.exitcode} "
                "without reporting a result"
            )

        # Logged so that we can see how much of the node creation was hidden behind the analysis
        self.logger.info(msg=f"MDSplus nodes created;  {result['seconds'] * 1e3:,.2f}ms")

    def run_with_log(
        self,
        operator: typing.Callable[..., typing.Any],
        message: str | None,
        args: dict[typing.Any, typing.Any] | None = None,
    ) -> typing.Any:
        if message is None:
            step_name = ""
        else:
            step_name = message

        try:
            if args is None:
                ret_data = operator()
            else:
                ret_data = operator(**args)
            self.logger.info(msg=step_name)
            return ret_data
        except Exception as exception_obj:
            self.logger.exception(msg=step_name)
            raise exception_obj
