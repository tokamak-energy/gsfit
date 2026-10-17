import typing

import mdsthin
import numpy as np
import numpy.typing as npt
from gsfit_rs import Magnetics
from st40_database import GetData


def add_to_magnetics_rogowski_coil(
    magnetics: Magnetics,
    mag: GetData,
) -> None:
    """
    Add every Rogowski coil to `magnetics/rogowski_coil`.

    :param magnetics: the `Magnetics` the Rogowski coils are added to
    :param mag: the MAG run the Rogowski coils are read from, opened with `GetData`

    The rogowski coil measures ∮ B·dl.
    A current into the page (counter clockwise when viewed from above) means the poloidal field
    rotates clockwise in the poloidal plane:
       →  →  →
     ↑         ↓
    ↑     ⊗    ↓       ⊗ = Ip into the page
     ↑         ↓       B circulates clockwise
       ←  ←  ←
    So:
    * Clockwise integral: ∮ B·dl = +μ0 Ip
    * Anti-clockwise integral: ∮ B·dl = -μ0 Ip
    Therefore on ST40 we are performing a **CLOCKWISE** integral.

    If we represent the gaps as, gap = ∫ B·dl, with these signs:
    Z
    ↑                    INIVC000
    │                   ┌──────────────────────────┐
    │ inboard_upper_gap ↑                          ↓ outboard_upper_gap
    │                   │                          │
    │                   │          plasma          │
    │                   │                          │
    │ inboard_lower_gap ↑                          ↓ outboard_lower_gap
    │                   └──────────────────────────┘
    └───────────────────────────────────────────────────────────→ R

    Then:
    `I_INIVC000 = Ip - ∑gaps`
    """

    names_long: list[str] = typing.cast(list[str], mag.get("ROG.ALL.NAMES"))
    sensors_name: npt.NDArray[np.str_] = np.char.replace(names_long, "ROG_", "")
    paths_r: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("ROG.ALL.R_PATH")).astype(np.float64)  # [metre]
    paths_z: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("ROG.ALL.Z_PATH")).astype(np.float64)  # [metre]

    # Every Rogowski coil is measured on MAG's timebase
    time: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get("TIME")).astype(np.float64)  # [second]

    n_sensors: int = len(sensors_name)
    for i_sensor in range(n_sensors):
        sensor_name: str = str(sensors_name[i_sensor])

        # MDSplus "ALL" does not allow jagged arrays, so the shorter contours are padded with NaN
        path_r: npt.NDArray[np.float64] = paths_r[i_sensor, :]
        path_z: npt.NDArray[np.float64] = paths_z[i_sensor, :]
        path_r = path_r[~np.isnan(path_r)]
        path_z = path_z[~np.isnan(path_z)]

        # Don't add the "fake" Rogowski coils (e.g. the MC supports)
        if len(path_r) <= 4:
            continue

        current: npt.NDArray[np.float64] = typing.cast(npt.NDArray[np.float32], mag.get(f"ROG.{sensor_name}.I_PROCESSED")).astype(np.float64)  # [ampere]

        # Only INIVC000 has gaps
        gaps_name: list[str] = []
        gaps_r_start: npt.NDArray[np.float64] = np.array([])  # [metre]
        gaps_r_end: npt.NDArray[np.float64] = np.array([])  # [metre]
        gaps_z_start: npt.NDArray[np.float64] = np.array([])  # [metre]
        gaps_z_end: npt.NDArray[np.float64] = np.array([])  # [metre]
        if sensor_name == "INIVC000":
            conn: mdsthin.Connection = mdsthin.Connection("smaug")
            conn.openTree("st40", 11010605)
            gaps_centre_r: npt.NDArray[np.float64] = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:R").data().astype(np.float64)  # [metre]
            gaps_centre_z: npt.NDArray[np.float64] = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:Z").data().astype(np.float64)  # [metre]
            gaps_half_width: npt.NDArray[np.float64] = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:DR").data().astype(np.float64)  # [metre]
            gaps_half_height: npt.NDArray[np.float64] = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:DZ").data().astype(np.float64)  # [metre]
            gaps_name_raw: npt.NDArray[np.str_] = conn.get("\\MAG::TOP.RUN14C.ROG.INIVC000.GAPS:NAME").data()

            gaps_r_start = gaps_centre_r - gaps_half_width
            gaps_r_end = gaps_centre_r + gaps_half_width
            gaps_z_start = gaps_centre_z - gaps_half_height
            gaps_z_end = gaps_centre_z + gaps_half_height
            n_gaps: int = len(gaps_name_raw)
            for i_gap in range(n_gaps):
                # mdsthin returns the names as strings, e.g. "INIVC000_gap_outboard_upper"; `strip` removes any padding
                gaps_name.append(str(gaps_name_raw[i_gap]).strip())

        magnetics.add_rogowski_coil(
            name=sensor_name,
            r=path_r,
            z=path_z,
            time=time,
            current=current,
            gaps_name=gaps_name,
            gaps_r_start=gaps_r_start,
            gaps_r_end=gaps_r_end,
            gaps_z_start=gaps_z_start,
            gaps_z_end=gaps_z_end,
        )
