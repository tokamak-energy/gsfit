from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import CoreProfiles
from gsfit_rs.dd_support import ParametersInput
from scipy.constants import elementary_charge
from scipy.interpolate import RectBivariateSpline

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.core_profiles.core_profiles import CoreProfilesSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree

if TYPE_CHECKING:
    from . import DatabaseReader


def setup_core_profiles(
    self: "DatabaseReader",
    pulseNo: int,
    settings: CoreProfilesSettings,
) -> CoreProfiles:
    """
    This method initialises the Rust `CoreProfiles` class, which holds an IMAS `core_profiles` IDS, with ASTRA's synthetic pressure at the
    positions of ST40's Thomson scattering (TS) channels.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `CoreProfilesSettings` object, read from `core_profiles/core_profiles.xml`, whose `workflow` says which ASTRA run to read
        the pressure from, and which TS run to read the channels' positions from

    **This method is specific to ST40's ASTRA runs on MDSplus.**

    ASTRA's run has no TS of its own, so the channels' positions are read from the TS tree of a real pulse: set the TS `tree`'s `pulseNo` to
    the pulse ASTRA's run models, e.g. 13349 for 13013349; a nil `pulseNo` reads the pulse being reconstructed, as for every tree.

    `core_profiles/profiles_2d(itime)` is filled at each TS channel's position, the irregular (R, Z) grid `irregular_rz_na`, at every one of
    ASTRA's times:
    1. `electrons/pressure`, ASTRA's electron pressure, `n_e * T_e`
    2. `pressure_thermal`, **ASTRA's total pressure**, thermal and fast, `PROFILES.PSI_NORM:P`, the pressure its equilibrium was made with.
       This is not the thermal pressure the node is named for: it is the pressure the equilibrium fits, as if the total pressure were measured
       at the TS channels. The settings' `thomson_scattering`, which turns a measured electron pressure into the thermal pressure, is not used.

    Each channel is put on ASTRA's flux surfaces with ASTRA's own flux map, `PSI2D:PSI`, and its normalised poloidal flux, with the flux on the
    magnetic axis and on the boundary of `PROFILES.PSI_NORM:PSI`. A channel outside the plasma, `psi_norm > 1`, is NaN, which the equilibrium
    does not fit; the TS channels are along the midplane, away from any X-point, so `psi_norm <= 1` is inside the plasma.

    Both are on ASTRA's timebase. `Equilibrium.copy_from_core_profiles_ids` makes the pressure constraints from `pressure_thermal`, interpolated
    onto the reconstruction times itself, so do not interpolate here.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ASTRA and TS runs to read, from the `workflow.xml` which `core_profiles.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "ASTRA" not in workflow.tree or "TS" not in workflow.tree:
        raise ValueError(
            "`core_profiles.xml` must link to a `workflow.xml` holding the ASTRA and TS `tree`s, which say which ASTRA run to read the pressure "
            "from, and which TS run to read the channels' positions from"
        )
    # A nil `pulseNo` reads the pulse being reconstructed
    astra_settings: WorkflowTree = workflow.tree["ASTRA"]
    if astra_settings.pulseNo is None:
        astra_pulseNo: int = pulseNo
    else:
        astra_pulseNo = astra_settings.pulseNo
    ts_settings: WorkflowTree = workflow.tree["TS"]
    if ts_settings.pulseNo is None:
        ts_pulseNo: int = pulseNo
    else:
        ts_pulseNo = ts_settings.pulseNo

    # Serialise the settings to be stored inside the IDS: `core_profiles.xml` holding only the settings which built the pressure
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=CoreProfilesSettings(workflow=Workflow(tree=KeyedList([astra_settings, ts_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the CoreProfiles Rust class
    core_profiles: CoreProfiles = CoreProfiles(settings_xml)

    # The TS channels' positions
    with mdsthin.Connection("smaug") as conn:
        conn.openTree("TS", ts_pulseNo)
        channel_r: npt.NDArray[np.float64] = np.asarray(conn.get(f"\\TS::TOP.{ts_settings.run_name}:R").data(), dtype=np.float64)  # [metre]
        channel_z: npt.NDArray[np.float64] = np.asarray(conn.get(f"\\TS::TOP.{ts_settings.run_name}:Z").data(), dtype=np.float64)  # [metre]
    n_channel: int = len(channel_r)

    # ASTRA stores these as float32; GSFit uses float64
    with mdsthin.Connection("smaug") as conn:
        conn.openTree("ASTRA", astra_pulseNo)
        run_path: str = f"\\ASTRA::TOP.{astra_settings.run_name}"
        # ASTRA's flux map, with ASTRA's flux, which is minus GSFit's, [n_time, n_z, n_r]
        psi: npt.NDArray[np.float64] = conn.get(f"{run_path}.PSI2D:PSI").data().astype(np.float64)  # [weber]
        grid_r: npt.NDArray[np.float64] = conn.get(f"{run_path}.PSI2D:RGRID").data().astype(np.float64)  # [metre]
        grid_z: npt.NDArray[np.float64] = conn.get(f"{run_path}.PSI2D:ZGRID").data().astype(np.float64)  # [metre]
        # On ASTRA's flux surfaces, from the magnetic axis to the boundary, [n_time, n_surface]: the flux, and the total pressure
        surfaces_psi: npt.NDArray[np.float64] = conn.get(f"{run_path}.PROFILES.PSI_NORM:PSI").data().astype(np.float64)  # [weber]
        surfaces_pressure: npt.NDArray[np.float64] = conn.get(f"{run_path}.PROFILES.PSI_NORM:P").data().astype(np.float64)  # [pascal]
        # On ASTRA's radial grid, [n_time, n_astra_grid]: the normalised poloidal flux, and the electrons' density and temperature
        astra_grid_psi_norm: npt.NDArray[np.float64] = conn.get(f"{run_path}.PROFILES.ASTRA:PSIN").data().astype(np.float64)  # [dimensionless]
        astra_grid_electrons_density: npt.NDArray[np.float64] = conn.get(f"{run_path}.PROFILES.ASTRA:NE").data().astype(np.float64) * 1.0e19  # [1 / metre ** 3]
        astra_grid_electrons_temperature: npt.NDArray[np.float64] = (
            conn.get(f"{run_path}.PROFILES.ASTRA:TE").data().astype(np.float64) * 1.0e3
        )  # [electron_volt]
        n_time: int = psi.shape[0]
        # ASTRA's `TIME` holds every time it was to run to, and each signal only the times it was run to, which are the first of them
        time: npt.NDArray[np.float64] = conn.get(f"{run_path}:TIME").data().astype(np.float64)[:n_time]  # [second]

    # ASTRA's pressures at the channels, NaN outside the plasma
    electrons_pressure: npt.NDArray[np.float64] = np.full((n_time, n_channel), np.nan)  # [pascal]
    pressure_thermal: npt.NDArray[np.float64] = np.full((n_time, n_channel), np.nan)  # [pascal]
    for i_time in range(n_time):
        # The normalised poloidal flux of each channel, and of ASTRA's flux surfaces
        psi_interpolator: RectBivariateSpline = RectBivariateSpline(grid_z, grid_r, psi[i_time, :, :])
        channel_psi: npt.NDArray[np.float64] = psi_interpolator(channel_z, channel_r, grid=False)  # [weber]
        psi_axis: float = surfaces_psi[i_time, 0]  # [weber]
        psi_boundary: float = surfaces_psi[i_time, -1]  # [weber]
        channel_psi_norm: npt.NDArray[np.float64] = (channel_psi - psi_axis) / (psi_boundary - psi_axis)  # [dimensionless]
        surfaces_psi_norm: npt.NDArray[np.float64] = (surfaces_psi[i_time, :] - psi_axis) / (psi_boundary - psi_axis)  # [dimensionless]
        inside_plasma: npt.NDArray[np.bool_] = channel_psi_norm <= 1.0

        pressure_thermal[i_time, inside_plasma] = np.interp(channel_psi_norm[inside_plasma], surfaces_psi_norm, surfaces_pressure[i_time, :])
        electrons_pressure[i_time, inside_plasma] = np.interp(
            channel_psi_norm[inside_plasma],
            astra_grid_psi_norm[i_time, :],
            astra_grid_electrons_density[i_time, :] * astra_grid_electrons_temperature[i_time, :] * elementary_charge,
        )

    core_profiles.set_profiles_2d_irregular_rz(
        time=time,
        r=channel_r,
        z=channel_z,
        electrons_pressure=electrons_pressure,
        pressure_thermal=pressure_thermal,
    )

    return core_profiles
