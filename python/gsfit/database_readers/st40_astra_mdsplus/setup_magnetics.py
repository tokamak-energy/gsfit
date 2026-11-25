from typing import TYPE_CHECKING
from typing import Any

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import Magnetics
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.magnetics.magnetics import MagneticsSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree
from .astra_rogowski_coils_reader import astra_rogowski_coils_reader

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_magnetics(
    self: "DatabaseReader",
    pulseNo: int,
    settings: MagneticsSettings,
) -> Magnetics:
    """
    This method initialises the Rust `Magnetics` class, which holds an IMAS `magnetics` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `MagneticsSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ASTRA
        run to read

    **This method is specific to ST40's ASTRA runs on MDSplus.**

    ASTRA's synthetic measurements, which its equilibrium, SPIDER, makes at ST40's magnetic sensors, each named as ST40's MAG names it, so that the
    equilibrium's settings find them:
    1. the poloidal field probes, `BPPROBE.ALL`, named "P" and their number, e.g. "BPPROBE_101" is "P101";
    2. the flux loops, `FLOOP.ALL`, named "L" and their number, e.g. "FLOOP_001" is "L001";
    3. the Rogowski coils, `ROG.ALL`, without "ROG_", e.g. "ROG_INIVC000" is "INIVC000", whose paths are ASTRA's `exp/equ/MCVC/rogpath.dat`, kept next
       to this file. A Rogowski coil with a path of 4 points or fewer is not a real one, e.g. the MC supports, and is not added. ASTRA's Rogowski
       coils have no gaps: each measures all the current its path encloses;
    4. the diamagnetic flux, `GLOBAL:DF`, as the method "ASTRA".

    No constraints at points which can move are added, so `magnetics/isoflux`, `magnetics/isoflux_boundary` and `magnetics/stationary_point`
    are left empty, and the equilibrium's fit settings must not list any.

    Their conventions are GSFit's, checked against ASTRA's own flux map, `PSI2D:PSI`:
    * the probes' angles, `BPPROBE.ALL:ANGLE`, are counter-clockwise from `grad(R)`, COCOS 13, as MAG's are;
    * the flux loops' `FLOOP.ALL:PSI` is GSFit's `psi`, the total poloidal flux, falling outwards for a positive plasma current, which is minus
      ASTRA's `PSI2D:PSI`.

    Every signal is stored on ASTRA's timebase. Every sensor is added, whether or not it is fitted: which sensors are fitted, and their weights, are
    the equilibrium's constraints.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ASTRA run to read, from the `workflow.xml` which `magnetics.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "ASTRA" not in workflow.tree:
        raise ValueError(
            "`magnetics.xml` must link to a `workflow.xml` holding the ASTRA `tree`, which says which ASTRA run to read the synthetic magnetic sensors from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    astra_settings: WorkflowTree = workflow.tree["ASTRA"]
    if astra_settings.pulseNo is None:
        astra_pulseNo: int = pulseNo
    else:
        astra_pulseNo = astra_settings.pulseNo

    # Serialise the settings, which the magnetic sensors record in `magnetics/code/parameters`: `magnetics.xml` holding only the run which built
    # the magnetic sensors
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=MagneticsSettings(workflow=Workflow(tree=KeyedList([astra_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the Magnetics Rust class
    magnetics: Magnetics = Magnetics(settings_xml)

    conn: mdsthin.Connection = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", astra_pulseNo)
    run_path: str = f"\\ASTRA::TOP.{astra_settings.run_name}"
    # ASTRA's `TIME` holds every time it was to run to, and each signal only the times it was run to, which are the first of them
    time_all: npt.NDArray[np.float64] = conn.get(f"{run_path}:TIME").data().astype(np.float64)  # [second]

    # 1. The poloidal field probes. ASTRA stores these as float32; GSFit uses float64
    probes_name: npt.NDArray[np.str_] = np.asarray(conn.get(f"{run_path}.BPPROBE.ALL:NAME").data())
    probes_r: npt.NDArray[np.float64] = conn.get(f"{run_path}.BPPROBE.ALL:R").data().astype(np.float64)  # [metre]
    probes_z: npt.NDArray[np.float64] = conn.get(f"{run_path}.BPPROBE.ALL:Z").data().astype(np.float64)  # [metre]
    probes_poloidal_angle: npt.NDArray[np.float64] = conn.get(f"{run_path}.BPPROBE.ALL:ANGLE").data().astype(np.float64)  # [radian]
    probes_field: npt.NDArray[np.float64] = conn.get(f"{run_path}.BPPROBE.ALL:B").data().astype(np.float64)  # [n_time, n_probe] [tesla]
    probes_time: npt.NDArray[np.float64] = time_all[: probes_field.shape[0]]  # [second]
    n_probe: int = len(probes_name)
    for i_probe in range(n_probe):
        magnetics.add_b_field_pol_probe(
            name=str(probes_name[i_probe]).strip().replace("BPPROBE_", "P"),
            r=float(probes_r[i_probe]),
            z=float(probes_z[i_probe]),
            poloidal_angle=float(probes_poloidal_angle[i_probe]),
            time=probes_time,
            field=probes_field[:, i_probe],
        )

    # 2. The flux loops
    flux_loops_name: npt.NDArray[np.str_] = np.asarray(conn.get(f"{run_path}.FLOOP.ALL:NAME").data())
    flux_loops_r: npt.NDArray[np.float64] = conn.get(f"{run_path}.FLOOP.ALL:R").data().astype(np.float64)  # [metre]
    flux_loops_z: npt.NDArray[np.float64] = conn.get(f"{run_path}.FLOOP.ALL:Z").data().astype(np.float64)  # [metre]
    flux_loops_flux: npt.NDArray[np.float64] = conn.get(f"{run_path}.FLOOP.ALL:PSI").data().astype(np.float64)  # [n_time, n_flux_loop] [weber]
    flux_loops_time: npt.NDArray[np.float64] = time_all[: flux_loops_flux.shape[0]]  # [second]
    n_flux_loop: int = len(flux_loops_name)
    for i_flux_loop in range(n_flux_loop):
        magnetics.add_flux_loop(
            name=str(flux_loops_name[i_flux_loop]).strip().replace("FLOOP_", "L"),
            r=float(flux_loops_r[i_flux_loop]),
            z=float(flux_loops_z[i_flux_loop]),
            time=flux_loops_time,
            flux=flux_loops_flux[:, i_flux_loop],
        )

    # 3. The Rogowski coils, whose paths are in `rogpath.dat`
    rogowski_coils_name: npt.NDArray[np.str_] = np.asarray(conn.get(f"{run_path}.ROG.ALL:NAME").data())
    rogowski_coils_current: npt.NDArray[np.float64] = conn.get(f"{run_path}.ROG.ALL:I").data().astype(np.float64)  # [n_time, n_rogowski_coil] [ampere]
    rogowski_coils_time: npt.NDArray[np.float64] = time_all[: rogowski_coils_current.shape[0]]  # [second]
    rogowski_coils_path: dict[str, dict[str, Any]] = astra_rogowski_coils_reader()
    n_rogowski_coil: int = len(rogowski_coils_name)
    for i_rogowski_coil in range(n_rogowski_coil):
        rogowski_coil_name_long: str = str(rogowski_coils_name[i_rogowski_coil]).strip()
        if rogowski_coil_name_long not in rogowski_coils_path:
            raise ValueError(f"`{run_path}.ROG.ALL:NAME` has {rogowski_coil_name_long}, whose path `rogpath.dat` does not have")
        path_r: npt.NDArray[np.float64] = rogowski_coils_path[rogowski_coil_name_long]["r"]  # [metre]
        path_z: npt.NDArray[np.float64] = rogowski_coils_path[rogowski_coil_name_long]["z"]  # [metre]

        # Don't add the "fake" Rogowski coils (e.g. the MC supports)
        if len(path_r) <= 4:
            continue

        magnetics.add_rogowski_coil(
            name=rogowski_coil_name_long.replace("ROG_", ""),
            r=path_r,
            z=path_z,
            time=rogowski_coils_time,
            current=rogowski_coils_current[:, i_rogowski_coil],
            gaps_name=[],
            gaps_r_start=np.array([]),
            gaps_r_end=np.array([]),
            gaps_z_start=np.array([]),
            gaps_z_end=np.array([]),
        )

    # 4. The diamagnetic flux
    diamagnetic_flux: npt.NDArray[np.float64] = conn.get(f"{run_path}.GLOBAL:DF").data().astype(np.float64)  # [weber]
    magnetics.add_diamagnetic_flux(
        method_name="ASTRA",
        time=time_all[: len(diamagnetic_flux)],
        data=diamagnetic_flux,
    )

    return magnetics
