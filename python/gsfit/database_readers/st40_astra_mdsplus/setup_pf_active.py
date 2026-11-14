from typing import TYPE_CHECKING
from typing import Any

import mdsthin
import numpy as np
import numpy.typing as npt
from diagnostic_and_simulation_base.xml_reader_and_writer import KeyedList
from diagnostic_and_simulation_base.xml_reader_and_writer import serialise_python_dataclass_to_imas_parameters_input
from gsfit_rs import PfActive
from gsfit_rs.dd_support import ParametersInput

from ...settings_schema import XSD_DIRECTORY_PATH
from ...settings_schema.python.pf_active.pf_active import PfActiveSettings
from ...settings_schema.python.workflow import Workflow
from ...settings_schema.python.workflow import WorkflowTree
from .astra_coils_reader import astra_coils_reader

if TYPE_CHECKING:
    from . import DatabaseReader

# ST40's coils as its own machine description, ELMAG, names them, in ELMAG's order, which the settings and the Tokamak Energy writers are made
# for, with ASTRA's windings which make each of them up
ELMAG_COILS: dict[str, list[str]] = {
    "SOL": ["SOL"],
    "MCT": ["MCT", "MCVCT"],
    "MCB": ["MCB", "MCVCB"],
    "DIVT": ["DIVT"],
    "DIVB": ["DIVB"],
    "BVLT": ["BVLT"],
    "BVLB": ["BVLB"],
    "BVUT": ["BVUT"],
    "BVUB": ["BVUB"],
    "PSHT": ["PSHT1", "PSHT2"],
    "PSHB": ["PSHB1", "PSHB2"],
}
# How far apart two windings' cells can be and still be taken as the same filaments [metre]: the vertical-control windings, e.g. MCVCT, are 1 mm
# outboard of the merging-compression coils, e.g. MCT
WINDINGS_SAME_PLACE_DISTANCE: float = 2.0e-3  # [metre]


def divide_parallelogram(
    coil: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Divide a coil's parallelogram cross-section into cells, as ASTRA's `DIVPAR` does, in `ASTRA/SRC/Trecur_2.f` of the `equil_astra_tepm` GitLab
    repository, which SPIDER's coils are made from.

    :param coil: one coil of `coil.dat`, from `astra_coils_reader`: its centre `rc` and `zc`, the projections of its sides on R and on Z, `wc` and
        `hc`, the angles of its sides from the R axis, `awc` and `ahc` [degree], and the approximate number of cells, `ndi`
    :return: each cell's centre, R and Z [metre], and its width and height [metre]: the projections of the cell's two sides on R and Z, which
        describe it as a rectangle, so are exact for a rectangular coil and approximate for a sheared one

    ```
    C   RC,ZC - CILINDER COORDINATES OF CENTER OF PARALLELOGRAM
    C   WC    - PROJECTION OF THE FIRST SIDE OF PARALLELOGRAM ON AXIS "R" (IN METER)
    C   HC    - PROJECTION OF THE SECOND SIDE OF PARALLELOGRAM ON AXIS "Z" (IN METER)
    C   AWC   - ANGLE BETWEEN THE FIRST SIDE OF PARAL. AND AXIS "R" (IN DEGREES, -90 < AWC < 90)
    C   AHC   - ANGLE BETWEEN THE SECOND SIDE OF PARAL. AND AXIS "R" (IN DEGREES, 0 < AHC < 180)
    C   NDIV  - APPROXIMATE NUMBER OF CELLS OF DIVIDING (NDIV=0: ONE CELL)
           R0    = RC - 0.5*( WC + HC * COS(AHCR)/SIN(AHCR) )
           Z0    = ZC - 0.5*( HC + WC * SIN(AWCR)/COS(AWCR) )
           WSIZE = WC / COS(AWCR)
           HSIZE = HC / SIN(AHCR)
           SW    = SQRT( NDIVA*WSIZE/HSIZE ) + 0.5
           SH    = SQRT( NDIVA*HSIZE/WSIZE ) + 0.5
           NDIVW = IDINT(SW)
           NDIVH = IDINT(SH)
           RS(L) = RS(1) + (I-1)*WR + (J-1)*HR,  L = (I-1)*NDIVH + J
           ZS(L) = ZS(1) + (I-1)*WZ + (J-1)*HZ
    ```
    """

    coil_r: float = float(coil["rc"])  # [metre]
    coil_z: float = float(coil["zc"])  # [metre]
    coil_width: float = float(coil["wc"])  # [metre]
    coil_height: float = float(coil["hc"])  # [metre]
    angle_width: float = np.deg2rad(float(coil["awc"]))  # [radian]
    angle_height: float = float(coil["ahc"])  # [degree]
    n_division_approximate: int = int(coil["ndi"])

    # One cell, at the centre
    if n_division_approximate == 0:
        return np.array([coil_r]), np.array([coil_z]), np.array([coil_width]), np.array([coil_height])

    if angle_height < 0.0:
        angle_height = angle_height + 180.0
    angle_height = np.deg2rad(angle_height)  # [radian]

    # The corner the cells are counted from, the lengths of the two sides, and their projections on R and Z
    corner_r: float = coil_r - 0.5 * (coil_width + coil_height * np.cos(angle_height) / np.sin(angle_height))  # [metre]
    corner_z: float = coil_z - 0.5 * (coil_height + coil_width * np.sin(angle_width) / np.cos(angle_width))  # [metre]
    side_width_length: float = coil_width / np.cos(angle_width)  # [metre]
    side_height_length: float = coil_height / np.sin(angle_height)  # [metre]
    side_width_r: float = coil_width  # [metre]
    side_width_z: float = coil_width * np.sin(angle_width) / np.cos(angle_width)  # [metre]
    side_height_r: float = coil_height * np.cos(angle_height) / np.sin(angle_height)  # [metre]
    side_height_z: float = coil_height  # [metre]

    # The number of cells along each side
    n_division_width: int = max(1, int(np.sqrt(n_division_approximate * side_width_length / side_height_length) + 0.5))
    n_division_height: int = max(1, int(np.sqrt(n_division_approximate * side_height_length / side_width_length) + 0.5))

    # One cell's sides
    cell_width_r: float = side_width_r / n_division_width  # [metre]
    cell_width_z: float = side_width_z / n_division_width  # [metre]
    cell_height_r: float = side_height_r / n_division_height  # [metre]
    cell_height_z: float = side_height_z / n_division_height  # [metre]

    # The cells' centres, in DIVPAR's order
    n_cell: int = n_division_width * n_division_height
    cells_r: npt.NDArray[np.float64] = np.full(n_cell, np.nan)  # [metre]
    cells_z: npt.NDArray[np.float64] = np.full(n_cell, np.nan)  # [metre]
    for i_division_width in range(n_division_width):
        for i_division_height in range(n_division_height):
            i_cell: int = i_division_width * n_division_height + i_division_height
            cells_r[i_cell] = corner_r + (i_division_width + 0.5) * cell_width_r + (i_division_height + 0.5) * cell_height_r
            cells_z[i_cell] = corner_z + (i_division_width + 0.5) * cell_width_z + (i_division_height + 0.5) * cell_height_z

    cells_width: npt.NDArray[np.float64] = np.full(n_cell, abs(cell_width_r))  # [metre]
    cells_height: npt.NDArray[np.float64] = np.full(n_cell, abs(cell_height_z))  # [metre]

    return cells_r, cells_z, cells_width, cells_height


def setup_pf_active(
    self: "DatabaseReader",
    pulseNo: int,
    settings: PfActiveSettings,
) -> PfActive:
    """
    This method initialises the Rust `PfActive` class, which holds an IMAS `pf_active` IDS.

    :param pulseNo: Pulse number, used to read from the database
    :param settings: `PfActiveSettings` object containing the XML settings read from the `settings` directory, whose `workflow` says which ASTRA run to read

    **This method is specific to ST40's ASTRA runs on MDSplus.**

    The coils' windings are ASTRA's, `exp/equ/MCVC/coil.dat`, kept next to this file: each winding's cross-section, number of turns, and the power
    supply which feeds it. Each winding is divided into cells as ASTRA does (`divide_parallelogram`), one filament each.
    The power supplies' currents are ASTRA's `COILS.PSU2PF:I`, in the order of `COILS.PSU2PF:PSU_NAMES` (IPL, MC, PSH, DIV, BVL, BVUT, BVUB, CS,
    MCVC), which `coil.dat` numbers from 1; IPL is the plasma, which feeds no coil.
    A winding of `turns` turns, fed by a current `I`, carries `turns * I`, shared equally by its filaments; a negative `turns`, as in MCVCT, is a
    winding wound against its power supply's current. The filaments are not the turns, so `coil(i1)/current`, the current of each filament, is
    `turns * I / n_filament`, not `I`, and the coil's self-inductance, which only the circuit equations use, is not the coil's.

    The coils are named, and ordered, as in ST40's own machine description, ELMAG, which the settings and the Tokamak Energy writers are made for
    (`ELMAG_COILS`), so some are made of two of ASTRA's windings:
    * the two windings of each poloidal shaping coil, e.g. PSHT1 and PSHT2, are one coil, PSHT, in two places, whose filaments all carry the same
      current, as both windings have the same turns, cells and power supply;
    * each vertical-control winding, e.g. MCVCT, is wound 1 mm outboard of a merging-compression coil, e.g. MCT, with the same cross-section, so it
      is taken on MCT's filaments, whose current is then that of both windings.

    See `python/gsfit/database_readers/interface.py` for more details on how a new database_reader should be implemented.
    """

    # The ASTRA run to read, from the `workflow.xml` which `pf_active.xml` links to
    workflow: Workflow | None = settings.workflow
    if workflow is None or "ASTRA" not in workflow.tree:
        raise ValueError("`pf_active.xml` must link to a `workflow.xml` holding the ASTRA `tree`, which says which ASTRA run to read the PF coil currents from")
    # A nil `pulseNo` reads the pulse being reconstructed
    astra_settings: WorkflowTree = workflow.tree["ASTRA"]
    if astra_settings.pulseNo is None:
        astra_pulseNo: int = pulseNo
    else:
        astra_pulseNo = astra_settings.pulseNo

    # Serialise the settings, which the PF coils record in `pf_active/code/parameters`: `pf_active.xml` holding only the run which built the PF coils
    settings_xml: ParametersInput = serialise_python_dataclass_to_imas_parameters_input(
        settings=PfActiveSettings(workflow=Workflow(tree=KeyedList([astra_settings]))),
        xsd_directory_path=XSD_DIRECTORY_PATH,
    )

    # Initialise the PfActive Rust class
    pf_active: PfActive = PfActive(settings_xml)

    # The power supplies' currents, [n_time, n_psu]
    conn: mdsthin.Connection = mdsthin.Connection("smaug")
    conn.openTree("ASTRA", astra_pulseNo)
    run_path: str = f"\\ASTRA::TOP.{astra_settings.run_name}"
    # ASTRA stores these as float32, in mega-ampere; GSFit uses float64, in ampere
    psus_current: npt.NDArray[np.float64] = conn.get(f"{run_path}.COILS.PSU2PF:I").data().astype(np.float64) * 1.0e6  # [ampere]
    n_time: int = psus_current.shape[0]
    # ASTRA's `TIME` holds every time it was to run to, and each signal only the times it was run to, which are the first of them
    time: npt.NDArray[np.float64] = conn.get(f"{run_path}:TIME").data().astype(np.float64)[:n_time]  # [second]

    windings: dict[str, dict[str, Any]] = astra_coils_reader()
    for coil_name, winding_names in ELMAG_COILS.items():
        # The coil's filaments, and the current of each, from each of its windings in turn
        coil_r: npt.NDArray[np.float64] = np.array([])  # [metre]
        coil_z: npt.NDArray[np.float64] = np.array([])  # [metre]
        coil_width: npt.NDArray[np.float64] = np.array([])  # [metre]
        coil_height: npt.NDArray[np.float64] = np.array([])  # [metre]
        coil_current: npt.NDArray[np.float64] = np.zeros(n_time)  # [ampere]
        for winding_name in winding_names:
            winding: dict[str, Any] = windings[winding_name]
            cells_r: npt.NDArray[np.float64]
            cells_z: npt.NDArray[np.float64]
            cells_width: npt.NDArray[np.float64]
            cells_height: npt.NDArray[np.float64]
            cells_r, cells_z, cells_width, cells_height = divide_parallelogram(winding)
            n_cell: int = len(cells_r)
            cells_current: npt.NDArray[np.float64] = psus_current[:, winding["psu_number"] - 1] * winding["turns"] / n_cell  # [ampere]

            if len(coil_r) == 0:
                # The coil's first winding
                coil_r = cells_r
                coil_z = cells_z
                coil_width = cells_width
                coil_height = cells_height
                coil_current = cells_current
            elif n_cell == len(coil_r) and np.max(np.hypot(cells_r - coil_r, cells_z - coil_z)) < WINDINGS_SAME_PLACE_DISTANCE:
                # A winding on the coil's filaments, whose current adds to theirs
                coil_current = coil_current + cells_current
            elif np.array_equal(cells_current, coil_current):
                # A winding in another place, whose filaments carry the coil's current
                coil_r = np.concatenate((coil_r, cells_r))
                coil_z = np.concatenate((coil_z, cells_z))
                coil_width = np.concatenate((coil_width, cells_width))
                coil_height = np.concatenate((coil_height, cells_height))
            else:
                raise ValueError(f"ASTRA's winding {winding_name} is neither on {coil_name}'s filaments, nor carries the current of each of them")

        pf_active.add_pf_coil(
            name=coil_name,
            r=coil_r,
            z=coil_z,
            width=coil_width,
            height=coil_height,
            time=time,
            current=coil_current,
        )

    return pf_active
