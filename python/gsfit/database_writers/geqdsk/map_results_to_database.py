"""
Write the GSFit results to G EQDSK files, EFIT's equilibrium file format.

One file is written for each time-slice, named e.g. `gsfit_shot=14928_time=120ms.geqdsk`, into the current working directory.
Which time-slices are written is set by `database_writer/geqdsk/timeslices` in the settings.
A time-slice which did not converge is skipped, with a warning.

Everything is read from GSFit's Equilibrium IDS, using the IMAS paths in `gsfit_rs.imas.equilibrium_paths`.

File format
-----------
The layout is EFIT's, which every G EQDSK reader expects:
1. A 48 character description, then three integers in `3i4`: a dummy, `nw` (the number of R grid points) and `nh` (the number of Z grid points).
2. Four lines of five numbers each, in `5e16.9`:
   `rdim, zdim, rcentr, rleft, zmid`;
   `rmaxis, zmaxis, simag, sibry, bcentr`;
   `current, simag, xdum, rmaxis, xdum`;
   `zmaxis, xdum, sibry, xdum, xdum`.
3. `fpol`, `pres`, `ffprim` and `pprime`, each on `nw` points equally spaced in psi from the magnetic axis to the boundary.
4. `psirz`, the poloidal flux on the (R, Z) grid, with R varying fastest.
5. `qpsi`, on the same `nw` points as `fpol`.
6. `nbbbs` and `limitr`, the number of boundary and limiter points, in `2i5`.
7. The boundary, then the limiter, each as (R, Z) pairs.

Every array starts on a new line, with five numbers to a line.

COCOS
-----
GSFit is COCOS 13. G EQDSK files are written in COCOS 1, the convention G EQDSK readers assume.
The two differ in three ways:
* `e_Bp`: COCOS 13 uses the total poloidal flux, COCOS 1 the flux per radian, so psi is divided by 2 * pi;
* `sigma_Bp`: -1 in COCOS 13 and +1 in COCOS 1, so psi changes sign.
  Together, psi becomes -psi / (2 * pi), and the derivatives with respect to psi, `ffprim` and `pprime`, are multiplied by -2 * pi;
* `sigma_rhothetaphi`: in COCOS 1, q has the sign of ip * f. GSFit's q has the sign of f whatever the sign of ip, so q is multiplied by the sign of ip.

f, the pressure, ip, `rcentr` and `bcentr` are the same in both.
"""

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gsfit_rs.imas import equilibrium_paths

if TYPE_CHECKING:
    from ...gsfit import Gsfit
    from . import DatabaseWriterGeqdsk


def map_results_to_database(
    self: "DatabaseWriterGeqdsk",
    gsfit_controller: "Gsfit",
) -> None:
    """
    Write one G EQDSK file for each selected time-slice, into the current working directory.

    :param gsfit_controller: the `Gsfit` controller, holding the solved Rust objects
    """

    logger = gsfit_controller.logger
    equilibrium = gsfit_controller.equilibrium

    settings_geqdsk = gsfit_controller.settings_xml.database_writer.geqdsk
    if settings_geqdsk is None:
        raise ValueError("`database_writer/geqdsk` is needed when `database_writer/method` includes geqdsk")
    settings_timeslices = settings_geqdsk.timeslices

    time: npt.NDArray[np.float64] = np.asarray(equilibrium.get(equilibrium_paths.time_slice[:].time), dtype=np.float64)  # [second]
    n_time: int = len(time)

    # The indices of the time-slices to write
    time_indices: list[int] = []
    match settings_timeslices.method:
        case "all":
            time_indices = list(range(n_time))
        case "user_defined":
            # Each requested time is written at the nearest time-slice, but only when it lies within the reconstructed times.
            # The range is widened by 1 nanosecond so that a time which lands on the first or last time-slice, e.g. on the only
            # time-slice of a single time-slice reconstruction, is written despite rounding, e.g. 0.12 and the 0.12000000000000001
            # which `numpy.arange` can produce
            for time_requested in settings_timeslices.user_defined:
                if time_requested < np.min(time) - 1.0e-9 or time_requested > np.max(time) + 1.0e-9:
                    logger.warning(f"G EQDSK: time = {time_requested} second is outside the reconstructed times, so no file is written")
                    continue
                i_time_nearest: int = int(np.argmin(np.abs(time - time_requested)))
                # Two requested times can share their nearest time-slice, which is written once
                if i_time_nearest not in time_indices:
                    time_indices.append(i_time_nearest)
        case _:
            raise ValueError(f'`database_writer/geqdsk/timeslices/method` = "{settings_timeslices.method}" is not recognized')

    # The limiter is the same for every time-slice. `unit[0]` is the closed vessel outline; the other units only add limit points
    limiter_r: npt.NDArray[np.float64] = np.asarray(equilibrium.get(equilibrium_paths.limiter.unit[0].outline.r), dtype=np.float64)  # [metre]
    limiter_z: npt.NDArray[np.float64] = np.asarray(equilibrium.get(equilibrium_paths.limiter.unit[0].outline.z), dtype=np.float64)  # [metre]
    n_limiter: int = len(limiter_r)

    r0: float = float(equilibrium.get(equilibrium_paths.vacuum_toroidal_field.r0))  # [metre]
    b0: npt.NDArray[np.float64] = np.asarray(equilibrium.get(equilibrium_paths.vacuum_toroidal_field.b0), dtype=np.float64)  # [tesla]

    date_today: str = datetime.date.today().strftime("%d/%m/%Y")

    for i_time in time_indices:
        time_slice = equilibrium_paths.time_slice[i_time]

        # e.g. 0.12 second -> "120", 0.0805 second -> "80.5"; rounded first so that 0.12000000000000001 does not become "120.00000000000001"
        time_ms_text: str = np.format_float_positional(np.round(time[i_time] * 1.0e3, 6), trim="-")

        convergence_name: str = str(equilibrium.get(time_slice.convergence.result.name))
        if convergence_name != "converged":
            logger.warning(f'G EQDSK: time = {time_ms_text}ms did not converge ("{convergence_name}"), so no file is written')
            continue

        # The (R, Z) grid
        grid_r: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.profiles_2d[0].grid.dim1), dtype=np.float64)  # [metre]
        grid_z: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.profiles_2d[0].grid.dim2), dtype=np.float64)  # [metre]
        n_r: int = len(grid_r)
        n_z: int = len(grid_z)

        # G EQDSK writes psi with R varying fastest, i.e. a [n_z, n_r] array written row by row.
        # Which axis of the stored map is R is checked against the stored R of every grid point, rather than assumed
        psi_2d: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.profiles_2d[0].psi), dtype=np.float64)  # [weber]
        r_2d: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.profiles_2d[0].r), dtype=np.float64)  # [metre]
        if r_2d.shape == (n_z, n_r) and np.array_equal(r_2d[0, :], grid_r):
            psi_z_r: npt.NDArray[np.float64] = psi_2d
        elif r_2d.shape == (n_r, n_z) and np.array_equal(r_2d[:, 0], grid_r):
            psi_z_r = psi_2d.T
        else:
            raise ValueError(f"G EQDSK: time = {time_ms_text}ms, `profiles_2d[0]/psi` with shape {psi_2d.shape} does not match the (R, Z) grid")

        # Scalars
        ip: float = float(equilibrium.get(time_slice.global_quantities.ip))  # [ampere]
        mag_r: float = float(equilibrium.get(time_slice.global_quantities.magnetic_axis.r))  # [metre]
        mag_z: float = float(equilibrium.get(time_slice.global_quantities.magnetic_axis.z))  # [metre]
        psi_axis: float = float(equilibrium.get(time_slice.global_quantities.psi_magnetic_axis))  # [weber]
        psi_boundary: float = float(equilibrium.get(time_slice.boundary.psi))  # [weber]

        # The 1D profiles, interpolated onto `n_r` points equally spaced in psi_norm, as G EQDSK requires
        psi_norm: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.profiles_1d.psi_norm), dtype=np.float64)  # [dimensionless]
        psi_norm_geqdsk: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_r)  # [dimensionless]
        f: npt.NDArray[np.float64] = np.interp(psi_norm_geqdsk, psi_norm, equilibrium.get(time_slice.profiles_1d.f))  # [tesla * metre]
        pressure: npt.NDArray[np.float64] = np.interp(psi_norm_geqdsk, psi_norm, equilibrium.get(time_slice.profiles_1d.pressure))  # [pascal]
        f_df_dpsi: npt.NDArray[np.float64] = np.interp(
            psi_norm_geqdsk, psi_norm, equilibrium.get(time_slice.profiles_1d.f_df_dpsi)
        )  # [tesla ** 2 * metre ** 2 / weber]
        dpressure_dpsi: npt.NDArray[np.float64] = np.interp(
            psi_norm_geqdsk, psi_norm, equilibrium.get(time_slice.profiles_1d.dpressure_dpsi)
        )  # [pascal / weber]
        q: npt.NDArray[np.float64] = np.interp(psi_norm_geqdsk, psi_norm, equilibrium.get(time_slice.profiles_1d.q))  # [dimensionless]

        boundary_r: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.boundary.outline.r), dtype=np.float64)  # [metre]
        boundary_z: npt.NDArray[np.float64] = np.asarray(equilibrium.get(time_slice.boundary.outline.z), dtype=np.float64)  # [metre]
        n_boundary: int = len(boundary_r)

        # From GSFit's COCOS 13 to G EQDSK's COCOS 1, see the module docstring
        psi_axis_cocos_1: float = -psi_axis / (2.0 * np.pi)  # [weber / radian]
        psi_boundary_cocos_1: float = -psi_boundary / (2.0 * np.pi)  # [weber / radian]
        psi_z_r_cocos_1: npt.NDArray[np.float64] = -psi_z_r / (2.0 * np.pi)  # [weber / radian]
        f_df_dpsi_cocos_1: npt.NDArray[np.float64] = -2.0 * np.pi * f_df_dpsi  # [tesla ** 2 * metre ** 2 * radian / weber]
        dpressure_dpsi_cocos_1: npt.NDArray[np.float64] = -2.0 * np.pi * dpressure_dpsi  # [pascal * radian / weber]
        q_cocos_1: npt.NDArray[np.float64] = np.sign(ip) * q  # [dimensionless]

        # The boundary and limiter as (R, Z) pairs
        boundary_r_z: npt.NDArray[np.float64] = np.full(2 * n_boundary, np.nan)  # [metre]
        boundary_r_z[0::2] = boundary_r
        boundary_r_z[1::2] = boundary_z
        limiter_r_z: npt.NDArray[np.float64] = np.full(2 * n_limiter, np.nan)  # [metre]
        limiter_r_z[0::2] = limiter_r
        limiter_r_z[1::2] = limiter_z

        # The 48 character description, then `3i4`
        description: str = f"GSFit {gsfit_controller.run_name} {date_today} #{gsfit_controller.pulseNo} {time_ms_text}ms"
        content: str = f"{description[:48]:<48}{3:4d}{n_r:4d}{n_z:4d}\n"

        r_dim: float = grid_r[-1] - grid_r[0]  # [metre]
        z_dim: float = grid_z[-1] - grid_z[0]  # [metre]
        z_mid: float = 0.5 * (grid_z[0] + grid_z[-1])  # [metre]
        content += _format_values(np.array([r_dim, z_dim, r0, grid_r[0], z_mid]))
        content += _format_values(np.array([mag_r, mag_z, psi_axis_cocos_1, psi_boundary_cocos_1, b0[i_time]]))
        content += _format_values(np.array([ip, psi_axis_cocos_1, 0.0, mag_r, 0.0]))
        content += _format_values(np.array([mag_z, 0.0, psi_boundary_cocos_1, 0.0, 0.0]))
        content += _format_values(f)
        content += _format_values(pressure)
        content += _format_values(f_df_dpsi_cocos_1)
        content += _format_values(dpressure_dpsi_cocos_1)
        content += _format_values(psi_z_r_cocos_1.ravel())
        content += _format_values(q_cocos_1)
        content += f"{n_boundary:5d}{n_limiter:5d}\n"
        content += _format_values(boundary_r_z)
        content += _format_values(limiter_r_z)

        file_path: Path = Path.cwd() / f"gsfit_shot={gsfit_controller.pulseNo}_time={time_ms_text}ms.geqdsk"
        file_path.write_text(content)
        logger.info(f"G EQDSK written: {file_path}")


def _format_values(values: npt.NDArray[np.float64]) -> str:
    """
    Values in Fortran's `5e16.9` format: each 16 characters wide, five to a line.
    A new line is started at the end, so the next array starts on a line of its own.

    :param values: the values to write
    """

    content: str = ""
    n_value: int = len(values)
    for i_value in range(n_value):
        content += f"{values[i_value]:16.9E}"
        if (i_value + 1) % 5 == 0 or i_value == n_value - 1:
            content += "\n"
    return content
