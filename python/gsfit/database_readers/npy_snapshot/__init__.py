"""
`npy_snapshot` database reader.

Rebuilds the GSFit Rust objects from a frozen single-time-slice snapshot of ST40
data (`snapshot.npz` + `snapshot.json`) instead of a live MDSplus database. This
lets a reconstruction run in CI with no database, network, or `st40_database`
dependency.

The build logic mirrors the `st40_mdsplus` reader, but every `GetData(...).get()`
is replaced by a lookup into the frozen snapshot via `FrozenGetData`. The snapshot
is produced by `investigation/capture_snapshot.py`.

The snapshot directory is taken from:
    settings["GSFIT_code_settings.json"]["database_reader"]["npy_snapshot"]["snapshot_dir"]
"""

import json
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt
from gsfit_rs import BpProbes
from gsfit_rs import Coils
from gsfit_rs import Dialoop
from gsfit_rs import EfitPolynomial
from gsfit_rs import FluxLoops
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import Passives
from gsfit_rs import Plasma
from gsfit_rs import Pressure
from gsfit_rs import RogowskiCoils
from gsfit_rs import StationaryPoint
from gsfit_rs import TensionedCubicBSpline

from ..interface import DatabaseReaderProtocol

# ELMAG coil geometry is stored under a fixed "machine description" pulse (mirrors st40_mdsplus)
ELMAG_COILS_PULSE_NO = 11012050
# The INIVC000 Rogowski gaps were captured from this pulse
ROG_GAPS_PULSE_NO = 11010605


class FrozenGetData:
    """Drop-in replacement for `st40_database.GetData`, backed by a frozen snapshot."""

    _cache: dict[str, tuple[typing.Any, dict[str, str], dict[str, typing.Any]]] = {}

    def __init__(self, snapshot_dir: str, pulseNo: int, tree_run: str) -> None:
        self.pulseNo = pulseNo
        self.tree_name = tree_run.split("#")[0]
        self._arrays, self._array_index, self._values = FrozenGetData._load(snapshot_dir)

    @classmethod
    def _load(cls, snapshot_dir: str) -> tuple[typing.Any, dict[str, str], dict[str, typing.Any]]:
        if snapshot_dir not in cls._cache:
            arrays = np.load(Path(snapshot_dir) / "snapshot.npz")
            with open(Path(snapshot_dir) / "snapshot.json") as file_handle:
                meta = json.load(file_handle)
            cls._cache[snapshot_dir] = (arrays, meta["arrays"], meta["values"])
        return cls._cache[snapshot_dir]

    def get(self, node: str) -> typing.Any:
        key = f"{self.pulseNo}|{self.tree_name}|{node}"
        if key in self._array_index:
            return self._arrays[self._array_index[key]]
        if key in self._values:
            return self._values[key]
        raise KeyError(f"npy_snapshot: node not found in frozen snapshot: {key}")


def _snapshot_dir(settings: dict[str, typing.Any]) -> str:
    return str(settings["GSFIT_code_settings.json"]["database_reader"]["npy_snapshot"]["snapshot_dir"])


class DatabaseReader(DatabaseReaderProtocol):
    """Reads a frozen ST40 snapshot from `.npz`/`.json` files (see module docstring)."""

    def setup_coils(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> Coils:
        snapshot_dir = _snapshot_dir(settings)
        coils = Coils()

        elmag = FrozenGetData(snapshot_dir, ELMAG_COILS_PULSE_NO, "ELMAG")
        coils_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.R"))
        coils_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.Z"))
        coils_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DR"))
        coils_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("COILS.DZ"))
        coil_names = typing.cast(list[str], elmag.get("COILS.COIL_NAMES"))
        fils2coils = typing.cast(npt.NDArray[np.bool_], np.asarray(elmag.get("COILS.FILS2COILS")) == 1.0)

        psu2coil = FrozenGetData(snapshot_dir, pulseNo, "PSU2COIL")
        time = typing.cast(npt.NDArray[np.float64], psu2coil.get("TIME"))
        pf_i = typing.cast(npt.NDArray[np.float64], psu2coil.get("PF.ALL.I"))
        coils_connected_to_psus = typing.cast(list[list[str]], psu2coil.get("PF.ALL.COILS"))

        n_time, n_psu = pf_i.shape
        for i_psu in range(0, n_psu):
            current_this_psu = pf_i[:, i_psu]
            coils_connected_to_this_psu = coils_connected_to_psus[i_psu]

            for coil_name in coils_connected_to_this_psu:
                if coil_name != "":
                    i_pf = coil_names.index(coil_name)
                    i_filaments = fils2coils[:, i_pf]
                    coils.add_pf_coil(
                        coil_name,
                        coils_r[i_filaments],
                        coils_z[i_filaments],
                        coils_d_r[i_filaments],
                        coils_d_z[i_filaments],
                        time=time,
                        measured=current_this_psu,
                    )

        i_rod = typing.cast(npt.NDArray[np.float64], psu2coil.get("TF.I_ROD"))
        coils.add_tf_coil(time, i_rod)

        return coils

    def setup_bp_probes(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> BpProbes:
        snapshot_dir = _snapshot_dir(settings)
        bp_probes = BpProbes()

        mag = FrozenGetData(snapshot_dir, pulseNo, "MAG")
        names_long = typing.cast(list[str], mag.get("BPPROBE.ALL.NAMES"))
        sensors_names = np.char.replace(names_long, "BPPROBE_", "P")
        sensors_r = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.R"))
        sensors_z = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.Z"))
        sensors_angle_pol = typing.cast(npt.NDArray[np.float64], mag.get("BPPROBE.ALL.THETA"))
        time = typing.cast(npt.NDArray[np.float64], mag.get("TIME")).astype(np.float64)

        n_sensors = len(sensors_names)
        for i_sensor in range(0, n_sensors):
            sensor_name = sensors_names[i_sensor]
            fit_settings = _bp_probe_fit_settings(settings, sensor_name)
            measured = typing.cast(npt.NDArray[np.float64], mag.get(f"BPPROBE.{sensor_name}.B")).astype(np.float64)

            bp_probes.add_sensor(
                name=sensor_name,
                geometry_angle_pol=sensors_angle_pol[i_sensor],
                geometry_r=sensors_r[i_sensor],
                geometry_z=sensors_z[i_sensor],
                fit_settings_comment=fit_settings["comment"],
                fit_settings_expected_value=fit_settings["expected_value"],
                fit_settings_include=fit_settings["include"],
                fit_settings_weight=fit_settings["weight"],
                time=time,
                measured=measured,
            )

        return bp_probes

    def setup_flux_loops(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> FluxLoops:
        snapshot_dir = _snapshot_dir(settings)
        flux_loops = FluxLoops()

        mag = FrozenGetData(snapshot_dir, pulseNo, "MAG")
        names_long = typing.cast(list[str], mag.get("FLOOP.ALL.NAMES"))
        sensors_names = np.char.replace(names_long, "FLOOP_", "L")
        sensors_r = typing.cast(npt.NDArray[np.float64], mag.get("FLOOP.ALL.R"))
        sensors_z = typing.cast(npt.NDArray[np.float64], mag.get("FLOOP.ALL.Z"))
        time = typing.cast(npt.NDArray[np.float64], mag.get("TIME")).astype(np.float64)

        n_sensors = len(sensors_names)
        for i_sensor in range(0, n_sensors):
            sensor_name = sensors_names[i_sensor]
            fit_settings = _flux_loop_fit_settings(settings, sensor_name)
            measured = typing.cast(npt.NDArray[np.float64], mag.get(f"FLOOP.{sensor_name}.PSI")).astype(np.float64)

            flux_loops.add_sensor(
                name=sensor_name,
                geometry_r=sensors_r[i_sensor],
                geometry_z=sensors_z[i_sensor],
                fit_settings_comment=fit_settings["comment"],
                fit_settings_expected_value=fit_settings["expected_value"],
                fit_settings_include=fit_settings["include"],
                fit_settings_weight=fit_settings["weight"],
                time=time,
                measured=measured,
            )

        return flux_loops

    def setup_rogowski_coils(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> RogowskiCoils:
        snapshot_dir = _snapshot_dir(settings)
        rogowski_coils = RogowskiCoils()

        mag = FrozenGetData(snapshot_dir, pulseNo, "MAG")
        names_long = typing.cast(list[str], mag.get("ROG.ALL.NAMES"))
        sensors_names = np.char.replace(names_long, "ROG_", "")
        paths_r = typing.cast(npt.NDArray[np.float64], mag.get("ROG.ALL.R_PATH")).astype(np.float64)
        paths_z = typing.cast(npt.NDArray[np.float64], mag.get("ROG.ALL.Z_PATH")).astype(np.float64)

        n_sensors = len(sensors_names)
        for i_sensor in range(0, n_sensors):
            sensor_name = sensors_names[i_sensor]
            path_r = paths_r[i_sensor, :]
            path_z = paths_z[i_sensor, :]
            path_r = path_r[~np.isnan(path_r)]
            path_z = path_z[~np.isnan(path_z)]

            # Don't store the "fake" Rogowski coils (e.g. the MC supports)
            if len(path_r) > 4:
                fit_settings = _rogowski_fit_settings(settings, sensor_name)
                measured = typing.cast(npt.NDArray[np.float64], mag.get(f"ROG.{sensor_name}.I")).astype(np.float64)
                time = typing.cast(npt.NDArray[np.float64], mag.get("TIME")).astype(np.float64)

                gaps_r: npt.NDArray[np.float64] = np.array([])
                gaps_z: npt.NDArray[np.float64] = np.array([])
                gaps_d_r: npt.NDArray[np.float64] = np.array([])
                gaps_d_z: npt.NDArray[np.float64] = np.array([])
                gaps_name: list[str] = []

                if sensor_name == "INIVC000":
                    gaps = FrozenGetData(snapshot_dir, ROG_GAPS_PULSE_NO, "ROG_GAPS")
                    gaps_r = typing.cast(npt.NDArray[np.float64], gaps.get("INIVC000.R"))
                    gaps_z = typing.cast(npt.NDArray[np.float64], gaps.get("INIVC000.Z"))
                    gaps_d_r = typing.cast(npt.NDArray[np.float64], gaps.get("INIVC000.DR"))
                    gaps_d_z = typing.cast(npt.NDArray[np.float64], gaps.get("INIVC000.DZ"))
                    gaps_name = typing.cast(list[str], gaps.get("INIVC000.NAME"))

                rogowski_coils.add_sensor(
                    sensor_name,
                    path_r,
                    path_z,
                    fit_settings["comment"],
                    fit_settings["expected_value"],
                    fit_settings["include"],
                    fit_settings["weight"],
                    time,
                    measured,
                    gaps_r=gaps_r,
                    gaps_z=gaps_z,
                    gaps_d_r=gaps_d_r,
                    gaps_d_z=gaps_d_z,
                    gaps_name=gaps_name,
                )

        return rogowski_coils

    def setup_passives(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> Passives:
        snapshot_dir = _snapshot_dir(settings)
        passives = Passives()

        elmag = FrozenGetData(snapshot_dir, pulseNo, "ELMAG")
        vessel_r = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.R"))
        vessel_z = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.Z"))
        vessel_d_r = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.DR"))
        vessel_d_z = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.DZ"))
        vessel_angle_1 = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.ANGLE1"))
        vessel_angle_2 = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.ANGLE2"))
        vessel_resistivity = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.RESISTIVITY"))
        vessel_fillaments_to_passives = typing.cast(npt.NDArray[np.float64], elmag.get("VESSEL.FILS2PASSIVE"))
        [n_filaments, n_passives] = vessel_fillaments_to_passives.shape
        passive_names = typing.cast(list[str], elmag.get("VESSEL.PASSIVE_NAME"))

        for i_passive in range(0, n_passives):
            passive_name = passive_names[i_passive]
            i_filaments = vessel_fillaments_to_passives[:, i_passive].astype(bool)

            if passive_name == "IVC":
                current_distribution_type = "eig"
                n_dof = settings["passive_dof_regularisation.json"]["IVC"]["n_dof"]
                regularisations = np.array(settings["passive_dof_regularisation.json"]["IVC"]["regularisations"])
                regularisations_weight = np.array(settings["passive_dof_regularisation.json"]["IVC"]["regularisations_weight"])
            elif passive_name == "OVC":
                current_distribution_type = "constant_current_density"
                n_dof = 1
                regularisations = np.array([[1.0]])
                regularisations_weight = np.array([0.1])
            else:
                current_distribution_type = "constant_current_density"
                n_dof = 1
                regularisations = np.empty((0, 0))
                regularisations_weight = np.empty(0)

            passives.add_passive(
                name=passive_name,
                r=vessel_r[i_filaments],
                z=vessel_z[i_filaments],
                d_r=vessel_d_r[i_filaments],
                d_z=vessel_d_z[i_filaments],
                angle_1=vessel_angle_1[i_filaments],
                angle_2=vessel_angle_2[i_filaments],
                resistivity=vessel_resistivity[i_passive],
                current_distribution_type=current_distribution_type,
                n_dof=n_dof,
                regularisations=regularisations,
                regularisations_weight=regularisations_weight,
            )

        return passives

    def setup_plasma(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> Plasma:
        snapshot_dir = _snapshot_dir(settings)

        initial_ip = settings["GSFIT_code_settings.json"]["initial_guess"]["ip"]
        initial_cur_r = settings["GSFIT_code_settings.json"]["initial_guess"]["r_cur"]
        initial_cur_z = settings["GSFIT_code_settings.json"]["initial_guess"]["z_cur"]

        p_prime_source_function = _build_source_function(settings["source_function_p_prime.json"])
        ff_prime_source_function = _build_source_function(settings["source_function_ff_prime.json"])

        n_r = settings["GSFIT_code_settings.json"]["grid"]["n_r"]
        n_z = settings["GSFIT_code_settings.json"]["grid"]["n_z"]
        r_min = settings["GSFIT_code_settings.json"]["grid"]["r_min"]
        r_max = settings["GSFIT_code_settings.json"]["grid"]["r_max"]
        z_min = settings["GSFIT_code_settings.json"]["grid"]["z_min"]
        z_max = settings["GSFIT_code_settings.json"]["grid"]["z_max"]

        n_psi_n = settings["GSFIT_code_settings.json"]["n_psi_n"]
        psi_n = np.linspace(0.0, 1.0, n_psi_n).astype(np.float64)

        elmag = FrozenGetData(snapshot_dir, pulseNo, "ELMAG")
        limit_pts_r = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.R"))
        limit_pts_z = typing.cast(npt.NDArray[np.float64], elmag.get("LIMITER.LIMIT_PTS.Z"))

        vessel_r = limit_pts_r
        vessel_z = limit_pts_z

        # Add lower MC tiles
        limit_pts_r = np.append(limit_pts_r, 0.7103)
        limit_pts_z = np.append(limit_pts_z, -0.3131)
        # Add upper MC tiles
        limit_pts_r = np.append(limit_pts_r, 0.7103)
        limit_pts_z = np.append(limit_pts_z, 0.3031)

        plasma = Plasma(
            n_r,
            n_z,
            r_min,
            r_max,
            z_min,
            z_max,
            psi_n,
            limit_pts_r,
            limit_pts_z,
            vessel_r,
            vessel_z,
            p_prime_source_function,
            ff_prime_source_function,
            initial_ip,
            initial_cur_r,
            initial_cur_z,
        )

        return plasma

    # The following sensors are not used for the magnetics-only snapshot; return them empty
    def setup_isoflux_sensors(
        self, pulseNo: int, settings: dict[str, typing.Any], times_to_reconstruct: npt.NDArray[np.float64], **kwargs: typing.Any
    ) -> Isoflux:
        return Isoflux()

    def setup_isoflux_boundary_sensors(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> IsofluxBoundary:
        return IsofluxBoundary()

    def setup_pressure_sensors(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> Pressure:
        return Pressure()

    def setup_stationary_point_sensors(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> StationaryPoint:
        return StationaryPoint()

    def setup_dialoop(self, pulseNo: int, settings: dict[str, typing.Any], **kwargs: typing.Any) -> Dialoop:
        return Dialoop()


def _build_source_function(source_settings: dict[str, typing.Any]) -> EfitPolynomial | TensionedCubicBSpline:
    if source_settings["method"] == "efit_polynomial":
        n_dof = source_settings["efit_polynomial"]["n_dof"]
        regularisations = np.array(source_settings["efit_polynomial"]["regularizations"])
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        return EfitPolynomial(n_dof, regularisations)
    if source_settings["method"] == "tensioned_cubic_b_spline":
        regularisations = np.array(source_settings["tensioned_cubic_b_spline"]["regularizations"])
        interior_knots = np.array(source_settings["tensioned_cubic_b_spline"]["interior_knots"])
        n_dof = len(interior_knots) + 4
        if regularisations.shape == (1, 0):
            regularisations = np.zeros((0, n_dof), dtype=np.float64)
        interval_tensions = np.array(source_settings["tensioned_cubic_b_spline"]["interval_tensions"])
        return TensionedCubicBSpline(regularisations, interior_knots, interval_tensions)
    raise ValueError(f"Unknown source function method: {source_settings['method']}")


def _bp_probe_fit_settings(settings: dict[str, typing.Any], sensor_name: str) -> dict[str, typing.Any]:
    weights = settings["sensor_weights_bp_probe.json"]
    if sensor_name in weights:
        fit = weights[sensor_name]["fit_settings"]
        return {"comment": fit["comment"], "expected_value": fit["expected_value"], "include": fit["include"], "weight": fit["weight"]}
    return {"comment": "", "expected_value": np.nan, "include": False, "weight": np.nan}


def _flux_loop_fit_settings(settings: dict[str, typing.Any], sensor_name: str) -> dict[str, typing.Any]:
    weights = settings["sensor_weights_flux_loops.json"]
    if sensor_name in weights:
        fit = weights[sensor_name]["fit_settings"]
        return {"comment": fit["comment"], "expected_value": fit["expected_value"], "include": fit["include"], "weight": fit["weight"] / (2.0 * np.pi)}
    return {"comment": "", "expected_value": np.nan, "include": False, "weight": np.nan}


def _rogowski_fit_settings(settings: dict[str, typing.Any], sensor_name: str) -> dict[str, typing.Any]:
    weights = settings["sensor_weights_rogowski_coils.json"]
    if sensor_name in weights:
        fit = weights[sensor_name]["fit_settings"]
        return {"comment": fit["comment"], "expected_value": fit["expected_value"], "include": fit["include"], "weight": fit["weight"]}
    return {"comment": "", "expected_value": np.nan, "include": False, "weight": np.nan}
