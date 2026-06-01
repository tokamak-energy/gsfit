# mypy: ignore-errors
# TODO: need to fix mypy errors

import typing
from typing import TYPE_CHECKING

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Coils
from scipy.constants import mu_0

from .astra_coils_reader import astra_coils_reader

if TYPE_CHECKING:
    from . import DatabaseReader

# Read coil geometry from coil.dat
astra_coil_description = astra_coils_reader()


def dividing_parallograms(coil_dictionary) -> tuple[int, int]:
    """
    The PF coils are described as parallograms, and discretized in `ASTRA/SRC/Trecur_2.f`, in the `equil_astra_tepm` GitLab repository.

    ```
       SW    = SQRT( NDIVA*WSIZE/HSIZE )
       SH    = SQRT( NDIVA*HSIZE/WSIZE )
       SW    = SW + 0.5
       SH    = SH + 0.5
    C
       NDIVW = IDINT(SW)
       NDIVH = IDINT(SH)
    ```
    """
    ndi = coil_dictionary["ndi"]
    wc = coil_dictionary["wc"]
    hc = coil_dictionary["hc"]

    sw = np.sqrt(ndi * wc / hc)
    sh = np.sqrt(ndi * hc / wc)
    sw = sw + 0.5
    sh = sh + 0.5
    ndivw = int(sw)
    ndivh = int(sh)
    return ndivw, ndivh


def calculate_coil_filament_positions(coil_dictionary) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
        Calculate the coil filament positions from the coil description dictionary.

        :param coil_dictionary: Dictionary containing the coil description
        :return: Tuple of numpy arrays containing the r and z positions of the coil filaments

        ```
    C**************************************************************
    C        SUBROUTINE FOR DIVIDING OF PARALLELOGRAM
    C                ( FOR PFC CROSS-SECTION )
    C**************************************************************
    C  INPUT DATE:
    C  ----------
    C   RC,ZC - CILINDER COORDINATES OF CENTER OF PARALLELOGRAM
    C   WC    - PROJECTION OF THE FIRST SIDE OF PARALLELOGRAM
    C           ON AXIS "R" (IN METER)
    C   HC    - PROJECTION OF THE SECOND SIDE OF PARALLELOGRAM
    C           ON AXIS "Z" (IN METER)
    C   AWC   - ANGLE BETWEEN THE FIRST SIDE OF PARAL. AND
    C           AXIS "R" (IN DEGREES, IT MUST NOT BE EQUAL 90 ,
    C                                 IT MUST BE :  -90 < AWC < 90 )
    C   AHC   - ANGLE BETWEEN THE SECOND SIDE OF PARAL. AND
    C           AXIS "R" (IN DEGREES, IT MUST NOT BE EQUAL 0 ,
    C                                 IT MUST BE :  0 < AHC < 180 )
    C   CURC  - CURRENT OF PARALLELOGRAM (IN MA)
    C   NDIV  - APPROXIMATE NUMBER OF CELLS OF DIVIDING:
    C           NDIV=0 - A SPECIAL CASE: AUTOMATICALLY NDIVRE=1,
    C                    RS(1)=RC, ZS(1)=ZC, PS=CURC
    C           IF NDIV > 0 THEN WE HAVE THE MOST TOTAL ALGORITHM OF
    C                            DIVIDING
    C           IF NDIV < 0 THEN NDIVW AND NDIVH ARE CUT OFF BY ABS(NDIV)
    C
    C  OUTPUT DATE:
    C  ----------
    C   NDIVRE      - REAL NUMBER OF CELLS OF DIVIDING ( = NDIVW*NDIVH )
    C   NDIVW       - NUMBER OF DIVIDING OF THE FIRST  SIDE OF PARAL.
    C   NDIVH       - NUMBER OF DIVIDING OF THE SECOND SIDE OF PARAL.
    C   RS(L),ZS(L) - CILINDER COORDINATES OF CENTERS OF CELLS OF DIVIDING
    C                 L = 1,2,...,NDIVRE ! (IN METER)
    C   PS          - CURRENT OF EVERY CELL OF DIVIDING  (IN MA)
    C   VERS - VERTICAL (OR LINEAR, OR RADIUS) SIZE OF CELL CROSS-S.
    C   HORS - HORIZONTAL SIZE OF CELL CROSS-SECTION
    C
    C**************************************************************
    C
            SUBROUTINE DIVPAR( RC, ZC, WC, HC, AWC, AHC, CURC, NDIV,
         *                     NDIVRE, NDIVW, NDIVH, RS, ZS, PS,
         *                     VERS, HORS )
    C
    C
           include 'double.inc'
    C
            DIMENSION  RS(1), ZS(1)
    C
    C**************************************************************
                    SIN(X) = DSIN(X)
                    COS(X) = DCOS(X)
                   ATAN(X) = DATAN(X)
                   SQRT(X) = DSQRT(X)
    C**************************************************************
    C
           IF(NDIV.EQ.0) THEN
              NDIVW  = 1
              NDIVH  = 1
              NDIVRE = 1
              RS(1)  = RC
              ZS(1)  = ZC
              PS     = CURC
              VERS   = HC
              HORS   = WC
              RETURN
           END IF
    C***************************************
    C
           IF(AHC.LT.0) AHC = AHC + 180.
    C
           IF(NDIV.GE.0) THEN
              NDIVA =  NDIV
           ELSE
              NDIVA = -NDIV
           END IF
    C***************************************
    C   PARAMETERS OF PARALLELOGRAM
    C
           XX    = 1.
           PI    = 4.*ATAN(XX)
    C
           AWCR  = AWC * PI /180.
           AHCR  = AHC * PI /180.
    C
           R0    = RC - 0.5*( WC + HC * COS(AHCR)/SIN(AHCR) )
           Z0    = ZC - 0.5*( HC + WC * SIN(AWCR)/COS(AWCR) )
    C
           WSIZE = WC / COS(AWCR)
           HSIZE = HC / SIN(AHCR)
    C
           WR    = WC
           WZ    = WC * SIN(AWCR) / COS(AWCR)
           HR    = HC * COS(AHCR) / SIN(AHCR)
           HZ    = HC
    C***************************************
    C   CALCULATION  NDIVW, NDIVH, NDIVRE, PS
    C
           SW    = SQRT( NDIVA*WSIZE/HSIZE )
           SH    = SQRT( NDIVA*HSIZE/WSIZE )
           SW    = SW + 0.5
           SH    = SH + 0.5
    C
           NDIVW = IDINT(SW)
           NDIVH = IDINT(SH)
    C
           IF(NDIVW.EQ.0)  NDIVW = 1
           IF(NDIVH.EQ.0)  NDIVH = 1
    C
           IF((NDIV.LT.0).AND.(NDIVW.GT.NDIVA)) NDIVW = NDIVA
           IF((NDIV.LT.0).AND.(NDIVH.GT.NDIVA)) NDIVH = NDIVA
    C
           NDIVRE = NDIVW * NDIVH
           PS     = CURC / NDIVRE
    C***************************************
    C   CALCULATION  RS(L), ZS(L) : L = 1,2,...,NDIVRE
    C
           WR  = WR / NDIVW
           WZ  = WZ / NDIVW
           HR  = HR / NDIVH
           HZ  = HZ / NDIVH
    C
           HORS = WR
           VERS = HZ
    C
           RS(1) = R0 + 0.5*(WR + HR)
           ZS(1) = Z0 + 0.5*(WZ + HZ)
    C
          DO 1 I=1,NDIVW
          DO 1 J=1,NDIVH
             L     = (I-1)*NDIVH + J
             RS(L) = RS(1) + (I-1)*WR + (J-1)*HR
             ZS(L) = ZS(1) + (I-1)*WZ + (J-1)*HZ
        1 CONTINUE
    C***************************************
    C
            RETURN
            END
        ```
    """

    # Extract required parameters
    rc = float(coil_dictionary["rc"])  # center R
    zc = float(coil_dictionary["zc"])  # center Z
    wc = float(coil_dictionary["wc"])  # projection of first side on R
    hc = float(coil_dictionary["hc"])  # projection of second side on Z
    # Optional angles (degrees). Default to AWC=0, AHC=90 if not provided,
    # matching the most common cases in the coil table.
    awc = float(coil_dictionary.get("awc", 0.0))
    ahc = float(coil_dictionary.get("ahc", 90.0))
    # Approximate number of cells (NDIV/NDIVA)
    ndia = int(coil_dictionary.get("ndi", 1))

    # Handle the special case NDIV == 0
    if ndia == 0:
        rs = np.array([rc], dtype=np.float64)
        zs = np.array([zc], dtype=np.float64)
        return rs, zs

    # Convert angles to radians, applying the Fortran adjustment for AHC < 0
    if ahc < 0.0:
        ahc = ahc + 180.0

    pi = np.pi
    awcr = awc * pi / 180.0
    ahcr = ahc * pi / 180.0

    # Guard against invalid angles (avoid division by zero)
    # In Fortran code: it assumes -90 < AWC < 90 and 0 < AHC < 180
    # If cos(awcr) or sin(ahcr) are too close to zero, nudge slightly.
    eps = 1e-12
    c_aw = np.cos(awcr)
    s_aw = np.sin(awcr)
    s_ah = np.sin(ahcr)
    c_ah = np.cos(ahcr)
    if abs(c_aw) < eps:
        c_aw = eps if c_aw >= 0 else -eps
    if abs(s_ah) < eps:
        s_ah = eps if s_ah >= 0 else -eps

    # Fortran mapping
    r0 = rc - 0.5 * (wc + hc * c_ah / s_ah)
    z0 = zc - 0.5 * (hc + wc * s_aw / c_aw)

    wsize = wc / c_aw
    hsize = hc / s_ah

    wr = wc
    wz = wc * s_aw / c_aw
    hr = hc * c_ah / s_ah
    hz = hc

    # Calculate NDIVW, NDIVH per Fortran's DIVPAR (NDIV >= 0 path)
    sw = np.sqrt(ndia * wsize / hsize) + 0.5
    sh = np.sqrt(ndia * hsize / wsize) + 0.5
    ndivw = int(sw)
    ndivh = int(sh)
    if ndivw == 0:
        ndivw = 1
    if ndivh == 0:
        ndivh = 1

    # Real number of divided cells
    ndivre = ndivw * ndivh

    # Cell sizes
    wr_cell = wr / ndivw
    wz_cell = wz / ndivw
    hr_cell = hr / ndivh
    hz_cell = hz / ndivh

    # First cell center
    r1 = r0 + 0.5 * (wr_cell + hr_cell)
    z1 = z0 + 0.5 * (wz_cell + hz_cell)

    # Generate centers for all cells (Fortran order L = (I-1)*NDIVH + J)
    rs = np.empty(ndivre, dtype=np.float64)
    zs = np.empty(ndivre, dtype=np.float64)
    idx = 0
    for i in range(1, ndivw + 1):
        for j in range(1, ndivh + 1):
            r = r1 + (i - 1) * wr_cell + (j - 1) * hr_cell
            z = z1 + (i - 1) * wz_cell + (j - 1) * hz_cell
            rs[idx] = r
            zs[idx] = z
            idx += 1

    return rs, zs


def setup_coils(
    self: "DatabaseReader",
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
    astra_run_name = settings["GSFIT_code_settings.json"]["database_reader"]["st40_astra_mdsplus"]["workflow"]["astra"]["run_name"]

    # Connect to MDSplus
    conn = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", pulseNo)

    # ASTRA time
    time = conn.get(f"\\ASTRA::TOP.{astra_run_name}:TIME").data().astype(np.float64)

    # PF coil currents
    # ["IPL", "MC", "PSH", "DIV", "BVL", "BVUT", "BVUB", "CS", "MCVC"]
    # [ 0,     1,    2,     3,     4,     5,      6,      7     8]
    currents = conn.get(f"\\ASTRA::TOP.{astra_run_name}.COILS.PSU2PF:I").data().astype(np.float64) * 1.0e6

    # Add BVLB PF coil
    coil_name = "BVLB"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    # Read in BVL PSU current, and normalise by the number of filaments
    currents_local = currents[:, 4] * 4.0 * 4.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )
    # Add BVLT PF coil
    coil_name = "BVLT"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Read in BVUB PSU current, and normalise by the number of filaments
    coil_name = "BVUB"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = currents[:, 6] * 4.0 * 6.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Read in BVUT PSU current, and normalise by the number of filaments
    coil_name = "BVUT"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = currents[:, 5] * 4.0 * 6.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Read in SOL PSU current, and normalise by the number of filaments
    coil_name = "SOL"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = currents[:, 7] * 2.0 * 95.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Add DIVT PF coil, and normalise by the number of filaments
    coil_name = "DIVT"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = currents[:, 3] * 4.0 * 7.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )
    # Add DIVB PF coil
    coil_name = "DIVB"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Add MCT PF coil, and normalise by the number of filaments
    coil_name = "MCT"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = -currents[:, 8] * 11.0 / len(coil_r) / 2.0 + currents[:, 1] * 11.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )
    # Add MCB PF coil
    coil_name = "MCB"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = currents[:, 8] * 11.0 / len(coil_r) / 2.0 + currents[:, 1] * 11.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Add PSHT PF coil
    coil_name = "PSHT1"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    currents_local = currents[:, 2] * 4.0 / len(coil_r)
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )
    # Add PSHB PF coil
    coil_name = "PSHB1"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )
    # Add PSHT2 PF coil
    coil_name = "PSHT2"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )
    # Add PSHB2 PF coil
    coil_name = "PSHB2"
    coil_r, coil_z = calculate_coil_filament_positions(astra_coil_description[coil_name])
    coils.add_pf_coil(
        coil_name,
        coil_r,
        coil_z,
        d_r=0.0 * coil_r,
        d_z=0.0 * coil_z,
        time=time,
        measured=currents_local,
    )

    # Add TF coil
    bt_vac = conn.get(f"\\ASTRA::TOP.{astra_run_name}.GLOBAL:BTVAC").data().astype(np.float64)  # time-dependent
    r_reference = 0.5
    i_rod = bt_vac * (2.0 * np.pi * r_reference) / mu_0  # time-dependent
    coils.add_tf_coil(
        time=time,
        measured=i_rod,
    )

    return coils
