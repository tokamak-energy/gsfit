# mypy: ignore-errors
# TODO: need to fix mypy errors

import typing
from typing import TYPE_CHECKING

import mdsthin  # type: ignore
import numpy as np
import numpy.typing as npt
from gsfit_rs import Coils
from st40_database import GetData  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from . import DatabaseReaderST40AstraMDSplus


def setup_coils(
    self: "DatabaseReaderST40AstraMDSplus",
    settings: dict[str, typing.Any],
    pulseNo: int,
) -> Coils:
    """
    This method initialises the Rust `Coils` class.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: Dictionary containing the JSON settings read from the `settings` directory

    **This method is specific to ST40's ASTRA stored on MDSplus.**

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # Initialise the Coils Rust class
    coils = Coils()

    # Extract the astra_run_name from settings
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["astra_run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # Looks like I could read in the PF coil geometry from the ASTRA MDSplus run. But different format!!
    astra_coils_r = conn.get(f"\\ASTRA::TOP.{astra_run_name}.COILS:R").data().astype(np.float64)
    astra_coils_z = conn.get(f"\\ASTRA::TOP.{astra_run_name}.COILS:Z").data().astype(np.float64)
    astra_coils_name = conn.get(f"\\ASTRA::TOP.{astra_run_name}.COILS:PF_NAMES").data()

    # Read PF coil geometry from MDSplus
    # FIXME: using a fixed shot is not a good idea!
    elmag = GetData(12050, "ELMAG#RUN16")

    coils_r = elmag.get("COILS.R")
    coils_z = elmag.get("COILS.Z")
    coil_names = elmag.get("COILS.COIL_NAMES")
    fils2coils = elmag.get("COILS.FILS2COILS") == 1.0

    # ASTRA time
    print(astra_run_name)
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)

    # Read in BVL PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.BVL:I").data().astype(np.float64) * 1.0e6
    # Add BVLB PF coil
    coil_name = "BVLB"
    shift_r = astra_coils_r[10] - np.mean(coils_r[i_filaments])
    shift_z = astra_coils_z[10] - np.mean(coils_r[i_filaments])
    print(f"BVLB coil shift_r={shift_r * 100.0}cm, shift_z={shift_z * 100.0}cm")
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments] + shift_r
    coil_z = coils_z[i_filaments] + shift_z
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )
    # Add BVLT PF coil
    coil_name = "BVLT"
    shift_r = astra_coils_r[9] - np.mean(coils_r[i_filaments])
    shift_z = astra_coils_z[9] - np.mean(coils_r[i_filaments])
    print(f"BVLT coil shift_r={shift_r * 100.0}cm, shift_z={shift_z * 100.0}cm")
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments] + shift_r
    coil_z = coils_z[i_filaments] + shift_z
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Read in BVUB PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.BVUB:I").data().astype(np.float64) * 1.0e6
    shift_r = astra_coils_r[12] - np.mean(coils_r[i_filaments])
    shift_z = astra_coils_z[12] - np.mean(coils_r[i_filaments])
    print(f"BVUB coil shift_r={shift_r * 100.0}cm, shift_z={shift_z * 100.0}cm")
    # Add BVUB PF coil
    coil_name = "BVUB"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments] + shift_r
    coil_z = coils_z[i_filaments] + shift_z
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Read in BVUT PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.BVUT:I").data().astype(np.float64) * 1.0e6
    shift_r = astra_coils_r[11] - np.mean(coils_r[i_filaments])
    shift_z = astra_coils_z[11] - np.mean(coils_r[i_filaments])
    print(f"BVUT coil shift_r={shift_r * 100.0}cm, shift_z={shift_z * 100.0}cm")
    # Add BVUT PF coil
    coil_name = "BVUT"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments] + shift_r
    coil_z = coils_z[i_filaments] + shift_z
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Read in SOL PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.CS:I").data().astype(np.float64) * 1.0e6
    # Add SOL PF coil
    coil_name = "SOL"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments]
    coil_z = coils_z[i_filaments]
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Read in DIV PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.DIV:I").data().astype(np.float64) * 1.0e6
    # Add DIVT PF coil
    shift_r = astra_coils_r[7] - np.mean(coils_r[i_filaments])
    shift_z = astra_coils_z[7] - np.mean(coils_r[i_filaments])
    print(f"DIVT coil shift_r={shift_r * 100.0}cm, shift_z={shift_z * 100.0}cm")
    coil_name = "DIVT"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments] + shift_r
    coil_z = coils_z[i_filaments] + shift_z
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )
    # Add DIVB PF coil
    shift_r = astra_coils_r[8] - np.mean(coils_r[i_filaments])
    shift_z = astra_coils_z[8] - np.mean(coils_r[i_filaments])
    print(f"DIVB coil shift_r={shift_r * 100.0}cm, shift_z={shift_z * 100.0}cm")
    coil_name = "DIVB"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments] + shift_r
    coil_z = coils_z[i_filaments] + shift_z
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Read in MC PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.MC:I").data().astype(np.float64) * 1.0e6
    # Add MCT PF coil
    coil_name = "MCT"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments]
    coil_z = coils_z[i_filaments]
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )
    # Add MCB PF coil
    coil_name = "MCB"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments]
    coil_z = coils_z[i_filaments]
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Read in PSH PSU current
    current = conn.get(f"\\ASTRA::TOP.{astra_run_name}.PSU.PSH:I").data().astype(np.float64) * 1.0e6
    # Add PSHT PF coil
    coil_name = "PSHT"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments]
    coil_z = coils_z[i_filaments]
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )
    # Add PSHB PF coil
    coil_name = "PSHB"
    i_pf = coil_names.index(coil_name)
    i_filaments = fils2coils[:, i_pf]
    coil_r = coils_r[i_filaments]
    coil_z = coils_z[i_filaments]
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=current,
    )

    # Add TF coil
    i_rod = conn.get(f"\\ASTRA::TOP.{astra_run_name}.GLOBAL:BTVAC").data().astype(np.float64)
    coils.add_tf_coil(
        time=time,
        measured=i_rod,
    )

    return coils
