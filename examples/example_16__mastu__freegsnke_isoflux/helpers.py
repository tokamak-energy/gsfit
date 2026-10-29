"""Setup and numerical details used by the Example 16 walkthrough."""

import pickle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from freegsnke import GSstaticsolver
from freegsnke import build_machine
from freegsnke.equilibrium_update import Equilibrium
from freegsnke.jtor_update import ConstrainPaxisIp
from gsfit import Gsfit
from gsfit_rs.imas import equilibrium_paths as ep
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


class ReconstructionError(RuntimeError):
    """A reconstruction completed without a converged equilibrium."""


def build_reference(freegsnke_root: Path, alpha_m: float = 1.8, alpha_n: float = 1.2) -> Equilibrium:
    machine_dir = freegsnke_root / "machine_configs" / "MAST-U"
    machine = build_machine.tokamak(
        active_coils_path=str(machine_dir / "MAST-U_like_active_coils.pickle"),
        passive_coils_path=str(machine_dir / "MAST-U_like_passive_coils.pickle"),
        limiter_path=str(machine_dir / "MAST-U_like_limiter.pickle"),
        wall_path=str(machine_dir / "MAST-U_like_wall.pickle"),
        magnetic_probe_path=str(machine_dir / "MAST-U_like_magnetic_probes.pickle"),
    )
    reference = Equilibrium(tokamak=machine, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129)
    profiles = ConstrainPaxisIp(eq=reference, paxis=8e3, Ip=600e3, fvac=0.5, alpha_m=alpha_m, alpha_n=alpha_n)
    # Only load trusted FreeGSNKE example data: pickle files can execute code.
    with (freegsnke_root / "examples" / "data" / "simple_diverted_currents_PaxisIp.pk").open("rb") as stream:
        currents = pickle.load(stream)
    for name, current in currents.items():
        machine[name].current = current
    GSstaticsolver.NKGSsolver(reference).solve(
        eq=reference, profiles=profiles, constrain=None, target_relative_tolerance=1e-9,
    )
    reference._updatePlasmaPsi(reference.plasma_psi)
    machine.probes.initialise_setup(reference)
    return reference


def make_pairs(reference: Equilibrium) -> npt.NDArray[np.float64]:
    """Return horizontal pairs followed by vertical pairs, at twelve shared flux levels."""
    axis_r, axis_z, axis_psi = reference.opt[0]
    pairs: list[list[float]] = []
    for horizontal in (True, False):
        centre: float = float(axis_r if horizontal else axis_z)
        coordinates = reference.R_1D if horizontal else reference.Z_1D
        for fraction in np.linspace(0.01, 1.0, 12):
            target = axis_psi + fraction * (reference.psi_bndry - axis_psi)
            endpoints: list[float] = []
            for boundary in (coordinates[0], coordinates[-1]):
                scan = np.linspace(centre, boundary, 1000)
                radii = scan if horizontal else np.full_like(scan, axis_r)
                heights = np.full_like(scan, axis_z) if horizontal else scan
                residual = np.asarray(reference.psiRZ(radii, heights)).reshape(-1) - target
                crossings = np.flatnonzero(residual[:-1] * residual[1:] <= 0)
                if crossings.size == 0:
                    raise ValueError(f"No {'horizontal' if horizontal else 'vertical'} crossing for psi_norm={fraction}")
                i: int = int(crossings[0])
                lo, hi = sorted((scan[i], scan[i + 1]))
                point = brentq(
                    lambda position, is_horizontal, level: float(
                        reference.psiRZ(position, axis_z) if is_horizontal else reference.psiRZ(axis_r, position)
                    ) - level,
                    lo, hi, args=(horizontal, target),
                )
                endpoints.extend((point, float(axis_z)) if horizontal else (float(axis_r), point))
            pairs.append(endpoints)
    return np.asarray(pairs, dtype=np.float64)


def reconstruct(
    reference: Equilibrium, pairs: npt.NDArray[np.float64], use_isoflux: bool,
    profile_method: Literal["efit_polynomial", "tensioned_cubic_b_spline"] = "efit_polynomial",
    use_pressure: bool = False,
    ff_prime_method: Literal["efit_polynomial", "tensioned_cubic_b_spline"] | None = None,
    spline_n_interior: int | None = None,
) -> Gsfit:
    if spline_n_interior is not None and (
        isinstance(spline_n_interior, bool) or not isinstance(spline_n_interior, int) or spline_n_interior < 0
    ):
        raise ValueError("spline_n_interior must be a nonnegative integer")
    controller = Gsfit(
        pulseNo=0,
        run_name="EX16_PRESSURE" if use_pressure else ("EX16_ISOFLUX" if use_isoflux else "EX16_MAGNETICS"),
        settings_path="mastu_with_synthetic_data_from_freegsnke",
        write_to_mds=False,
    )
    code = controller.settings["GSFIT_code_settings.json"]
    code["database_writer"]["method"] = "imas"
    code["timeslices"]["method"] = "user_defined"
    code["timeslices"]["user_defined"] = [0.5]
    code["initial_guess"]["cur_z"] = 0.0
    code["numerics"]["n_iter_no_vertical_feedback"] = 1
    controller.settings["sensor_weights_pressure.json"]["include"] = False
    for source in ("p_prime", "ff_prime"):
        source_settings = controller.settings[f"source_function_{source}.json"]
        source_method = ff_prime_method if source == "ff_prime" and ff_prime_method is not None else profile_method
        source_settings["method"] = source_method
        if source_method == "tensioned_cubic_b_spline":
            # A modest exploratory model: penalize coefficient curvature, with weak ridge damping.
            n_interior: int = 5 if use_pressure and source == "p_prime" else 3
            if spline_n_interior is not None:
                n_interior = spline_n_interior
            n_dof: int = n_interior + 4
            scale: float = 1e-5 if source == "p_prime" else 1.0
            regularisations = np.vstack((np.diff(np.eye(n_dof), n=2, axis=0), 0.01 * np.eye(n_dof))) * scale
            edge_row = np.zeros((1, n_dof))
            edge_row[0, -1] = scale * 100
            source_settings["tensioned_cubic_b_spline"] = {
                "interior_knots": np.linspace(0.0, 1.0, n_interior + 2)[1:-1].tolist(),
                "interval_tensions": [1.0] * (n_interior + 1),
                "regularizations": np.vstack((regularisations, edge_row)).tolist(),
            }
    time = np.array([0.0, 1.0], dtype=np.float64)

    # Follow Gsfit.run's stages, inserting the sensors before their Green's functions are built.
    controller.set_environment_variables()
    controller.setup_timeslices()
    # The existing **kwargs annotation incorrectly requires dict values rather than reader-specific arrays/lists.
    controller.setup_objects(time=time, freegsnke_eqs=[reference, reference])  # type: ignore[arg-type]
    if use_isoflux:
        for i, (r1, z1, r2, z2) in enumerate(pairs):
            controller.isoflux.add_sensor(
                name=f"ISOFLUX{i + 1:02d}",
                fit_settings_comment="Known FreeGSNKE flux-surface pair",
                fit_settings_include=True,
                fit_settings_weight=10.0,
                time=time,
                location_1_r=np.full(2, r1),
                location_1_z=np.full(2, z1),
                location_2_r=np.full(2, r2),
                location_2_z=np.full(2, z2),
                times_to_reconstruct=controller.results["TIME"],
            )
    if use_pressure:
        positions, measured = pressure_samples(reference, pairs)
        pressure_scale: float = float(reference.pressure(np.array([0.0]))[0])
        for i_sensor, ((sensor_r, sensor_z), pressure) in enumerate(zip(positions, measured, strict=True)):
            controller.pressure_sensors.add_sensor(
                name=f"PRESSURE{i_sensor + 1:02d}",
                geometry_r=float(sensor_r),
                geometry_z=float(sensor_z),
                fit_settings_comment="Known FreeGSNKE pressure at an isoflux sample location",
                fit_settings_expected_value=pressure_scale,
                fit_settings_include=True,
                fit_settings_weight=0.1,
                time=time,
                measured=np.full(time.shape, pressure),
            )
    controller.calculate_greens()
    controller.inverse_solver_rust()
    status = int(controller.plasma.get(ep.time_slice[0].convergence.result.index))
    if status != 1:
        description = controller.plasma.get(ep.time_slice[0].convergence.result.description)
        raise ReconstructionError(f"{controller.run_name} did not converge (status {status}): {description}")
    return controller


def pressure_samples(
    reference: Equilibrium, pairs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return all pair sample locations [metre] and their reference pressures [pascal]."""
    positions: npt.NDArray[np.float64] = pairs.reshape(-1, 2)
    psi_n = (reference.psiRZ(positions[:, 0], positions[:, 1]) - reference.opt[0][2]) / (
        reference.psi_bndry - reference.opt[0][2]
    )
    if not np.all(np.isfinite(psi_n)) or np.any(psi_n < -1e-9) or np.any(psi_n > 1.0 + 1e-9):
        raise ValueError("Isoflux sample locations must lie in the reference closed-flux interval")
    # Root-finding roundoff can put a boundary crossing just above psi_N = 1.
    measured = np.asarray(reference.pressure(np.clip(psi_n, 0.0, 1.0)), dtype=np.float64)
    if not np.all(np.isfinite(measured)) or np.any(measured < 0):
        raise ValueError("Isoflux sample location pressures must be finite and nonnegative")
    return positions, measured


def compare(
    reference: Equilibrium, controller: Gsfit, pairs: npt.NDArray[np.float64],
) -> tuple[dict[str, float | int], npt.NDArray[np.float64]]:
    ts = ep.time_slice[0]
    grid = ts.profiles_2d[0]
    r = controller.plasma.get(grid.grid.dim1)
    z = controller.plasma.get(grid.grid.dim2)
    psi = controller.plasma.get(grid.psi)
    interpolator = RectBivariateSpline(z, r, psi)
    pair_residuals = interpolator.ev(pairs[:, 1], pairs[:, 0]) - interpolator.ev(pairs[:, 3], pairs[:, 2])
    axis_r = float(controller.plasma.get(ts.global_quantities.magnetic_axis.r))
    axis_z = float(controller.plasma.get(ts.global_quantities.magnetic_axis.z))
    reference_r, reference_z, _ = reference.opt[0]
    reference_psi = reference.psi() * (2 * np.pi)
    reconstructed_psi = interpolator(reference.Z_1D, reference.R_1D).T
    # Flux has an arbitrary additive constant; remove it before comparing fields.
    difference = reconstructed_psi - reference_psi
    difference -= np.mean(difference)
    flux_span = abs(float(reference.psi_bndry - reference.opt[0][2])) * 2 * np.pi
    positions, pressure_reference = pressure_samples(reference, pairs)
    pressure_interpolator = RegularGridInterpolator((z, r), controller.plasma.get(grid.pressure))
    pressure_reconstructed = pressure_interpolator(positions[:, ::-1])
    if not np.all(np.isfinite(pressure_reconstructed)):
        raise ValueError("Reconstruction returned nonfinite pressure at the isoflux sample locations")
    pressure_rms: float = float(np.sqrt(np.mean((pressure_reconstructed - pressure_reference)**2)))
    pressure_axis: float = float(reference.pressure(np.array([0.0]))[0])
    metrics: dict[str, float | int] = {
        "convergence_status": int(controller.plasma.get(ts.convergence.result.index)),
        "iterations": int(controller.plasma.get(ts.convergence.iterations_n)),
        "final_delta_z_m": float(controller.plasma.get(ts.convergence.delta_z)),
        "gs_deviation": float(controller.plasma.get(ts.convergence.grad_shafranov_deviation_value)),
        "axis_error_m": float(np.hypot(axis_r - reference_r, axis_z - reference_z)),
        "max_pair_residual_Wb": float(np.max(np.abs(pair_residuals))),
        "max_pair_residual_over_flux_span": float(np.max(np.abs(pair_residuals)) / flux_span),
        "grid_flux_rms_over_flux_span": float(np.sqrt(np.mean(difference**2)) / flux_span),
        "sample_pressure_rms_Pa": pressure_rms,
        "sample_pressure_rms_over_axis_pressure": pressure_rms / pressure_axis,
    }
    psi_n = controller.plasma.get(ts.profiles_1d.psi_norm)
    for field, reference_values, name in (
        (ts.profiles_1d.dpressure_dpsi, reference.pprime(psi_n) / (2 * np.pi), "pprime"),
        (ts.profiles_1d.f_df_dpsi, reference.ffprime(psi_n) / (2 * np.pi), "ffprime"),
    ):
        values = controller.plasma.get(field)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(reference_values)):
            raise ValueError(f"Nonfinite {name} profile in reference comparison")
        metrics[f"{name}_rms_over_reference_peak"] = float(
            np.sqrt(np.mean((values - reference_values)**2)) / np.max(np.abs(reference_values))
        )
    return metrics, pair_residuals


def plot_source_profiles(reference: Equilibrium, controllers: list[Gsfit], labels: list[str], output_dir: Path) -> None:
    """Compare derivatives with respect to flux in Wb, displayed versus sqrt(psi_N)."""
    psi_n = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, reference_values, field, ylabel in (
        (axes[0], reference.pprime(psi_n) / (2 * np.pi), ep.time_slice[0].profiles_1d.dpressure_dpsi, "p' [Pa/Wb]"),
        (axes[1], reference.ffprime(psi_n) / (2 * np.pi), ep.time_slice[0].profiles_1d.f_df_dpsi, "FF' [T² m²/Wb]"),
    ):
        ax.plot(np.sqrt(psi_n), reference_values, color="black", linewidth=2, label="FreeGSNKE reference")
        for controller, label in zip(controllers, labels, strict=True):
            x = controller.plasma.get(ep.time_slice[0].profiles_1d.psi_norm)
            values = controller.plasma.get(field)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{label} returned nonfinite source profiles")
            ax.plot(np.sqrt(x), values, linestyle="--", label=label)
        ax.set(xlabel="sqrt(psi_N)", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "source_profiles.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_sensor_map(reference: Equilibrium, controller: Gsfit, output_dir: Path) -> None:
    """Plot only diagnostics included by the FreeGSNKE reader and active settings."""
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    psi_n = (reference.psi() - reference.opt[0][2]) / (reference.psi_bndry - reference.opt[0][2])
    ax.contour(reference.R, reference.Z, psi_n, levels=np.linspace(0.1, 0.95, 10), colors="0.75", linewidths=0.6)
    limiter = reference.tokamak.limiter
    ax.plot(limiter.R, limiter.Z, color="black", linewidth=1, label="Limiter / synthetic Ip contour")
    for kind, sensors, settings_file, marker, color in (
        ("Flux loop", reference.tokamak.probes.floops, "sensor_weights_flux_loops.json", "s", "tab:blue"),
        ("Mirnov probe", reference.tokamak.probes.pickups, "sensor_weights_bp_probe.json", "^", "tab:orange"),
    ):
        count = 0
        for sensor in sensors:
            if kind == "Mirnov probe" and sensor["orientation"] != "PARALLEL":
                continue
            fit = controller.settings[settings_file].get(sensor["name"])
            if fit is None or not fit["fit_settings"]["include"]:
                continue
            r = float(sensor["position"][0])
            z = float(sensor["position"][1 if kind == "Flux loop" else 2])
            count += 1
            identifier = f"{'F' if kind == 'Flux loop' else 'P'}{count}"
            ax.scatter(r, z, marker=marker, color=color, s=22)
            if count == 1 or count % 10 == 0:
                ax.annotate(identifier, (r, z), xytext=(3, 3), textcoords="offset points", fontsize=6)
            if kind == "Mirnov probe":
                direction = np.asarray(sensor["orientation_vector"])[[0, 2]]
                direction = direction / np.linalg.norm(direction)
                ax.quiver(r, z, direction[0], direction[1], angles="xy", scale_units="xy", scale=12, color=color, width=0.002)
        ax.scatter([], [], marker=marker, color=color, label=f"{kind}s included: {count}")
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="Synthetic magnetic diagnostics used in this reconstruction", aspect="equal")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "magnetic_sensors.png", dpi=180)
    plt.show()
    plt.close(fig)


def plot_midplane_pressure(
    reference: Equilibrium, controllers: list[Gsfit], labels: list[str], output_dir: Path,
) -> None:
    """Compare pressure [pascal] along the common geometric midplane Z = 0 [metre]."""
    radii = np.linspace(reference.R_1D[0], reference.R_1D[-1], 501)
    heights = np.zeros_like(radii)
    psi_n = (reference.psiRZ(radii, heights) - reference.opt[0][2]) / (reference.psi_bndry - reference.opt[0][2])
    reference_radii = midplane_plasma_radii(radii, psi_n, float(reference.opt[0][0]))
    reference_psi_n = (reference.psiRZ(reference_radii, np.zeros_like(reference_radii)) - reference.opt[0][2]) / (
        reference.psi_bndry - reference.opt[0][2]
    )
    pressure_reference = reference.pressure(np.clip(reference_psi_n, 0.0, 1.0))
    if not np.all(np.isfinite(pressure_reference)):
        raise ValueError("FreeGSNKE returned nonfinite midplane pressure")
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(reference_radii, pressure_reference / 1e3, color="black", linewidth=2, label="FreeGSNKE reference")
    for controller, label in zip(controllers, labels, strict=True):
        ts = ep.time_slice[0]
        grid = ep.time_slice[0].profiles_2d[0]
        coordinates = (controller.plasma.get(grid.grid.dim2), controller.plasma.get(grid.grid.dim1))
        psi_interpolator = RegularGridInterpolator(coordinates, controller.plasma.get(grid.psi))
        psi_axis = float(controller.plasma.get(ts.global_quantities.psi_magnetic_axis))
        psi_boundary = float(controller.plasma.get(ts.boundary.psi))
        psi_n = (psi_interpolator(np.column_stack((heights, radii))) - psi_axis) / (psi_boundary - psi_axis)
        plasma_radii = midplane_plasma_radii(
            radii, psi_n, float(controller.plasma.get(ts.global_quantities.magnetic_axis.r)),
        )
        interpolator = RegularGridInterpolator(
            coordinates, controller.plasma.get(grid.pressure),
        )
        pressure = interpolator(np.column_stack((np.zeros_like(plasma_radii), plasma_radii)))
        if not np.all(np.isfinite(pressure)):
            raise ValueError(f"{label} returned nonfinite midplane pressure")
        ax.plot(plasma_radii, pressure / 1e3, linestyle="--", label=label)
    ax.set(xlabel="R [m]", ylabel="Pressure [kPa]", title="Pressure on the common midplane Z = 0 m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "midplane_pressure.png", dpi=150)
    plt.show()
    plt.close(fig)


def midplane_plasma_radii(
    radii: npt.NDArray[np.float64], psi_n: npt.NDArray[np.float64], axis_r: float,
) -> npt.NDArray[np.float64]:
    """Return radii [metre] between the nearest psi_N = 1 crossings bracketing the axis."""
    if not np.all(np.isfinite(psi_n)):
        raise ValueError("Nonfinite normalized flux on the midplane")
    indices = np.flatnonzero((psi_n[:-1] - 1.0) * (psi_n[1:] - 1.0) <= 0.0)
    crossings: list[float] = []
    for i_r in indices:
        if psi_n[i_r + 1] == psi_n[i_r]:
            raise ValueError("Cannot locate an LCFS crossing on a flat psi_N = 1 interval")
        crossing: float = float(
            radii[i_r] + (1.0 - psi_n[i_r]) * (radii[i_r + 1] - radii[i_r]) / (psi_n[i_r + 1] - psi_n[i_r])
        )
        crossings.append(crossing)
    inboard = [value for value in crossings if value < axis_r]
    outboard = [value for value in crossings if value > axis_r]
    if not inboard or not outboard:
        raise ValueError("No midplane LCFS crossings bracket the magnetic-axis radius")
    return np.linspace(max(inboard), min(outboard), 501)


def save_plots(
    reference: Equilibrium, controllers: list[Gsfit], pairs: npt.NDArray[np.float64],
    residuals: list[npt.NDArray[np.float64]], labels: list[str], output_dir: Path, show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, len(controllers), figsize=(5 * len(controllers), 9), squeeze=False, constrained_layout=True)
    levels = np.linspace(0.1, 0.95, 10)
    reference_psi_n = (reference.psi() - reference.opt[0][2]) / (reference.psi_bndry - reference.opt[0][2])
    for ax, controller, label in zip(axes[0], controllers, labels, strict=True):
        ts = ep.time_slice[0]
        grid = ts.profiles_2d[0]
        ax.contour(reference.R, reference.Z, reference_psi_n, levels=levels, colors="tab:blue")
        ax.contour(reference.R, reference.Z, reference_psi_n, levels=[1.0], colors="tab:blue", linewidths=2.5)
        psi = controller.plasma.get(grid.psi)
        psi_axis = controller.plasma.get(ts.global_quantities.psi_magnetic_axis)
        psi_boundary = controller.plasma.get(ts.boundary.psi)
        ax.contour(
            controller.plasma.get(grid.grid.dim1), controller.plasma.get(grid.grid.dim2),
            (psi - psi_axis) / (psi_boundary - psi_axis),
            levels=levels, colors="tab:orange", linestyles="dashed",
        )
        ax.contour(
            controller.plasma.get(grid.grid.dim1), controller.plasma.get(grid.grid.dim2),
            (psi - psi_axis) / (psi_boundary - psi_axis),
            levels=[1.0], colors="tab:orange", linestyles="dashed", linewidths=2.5,
        )
        for i, (r1, z1, r2, z2) in enumerate(pairs):
            ax.plot([r1, r2], [z1, z2], "o", color="tab:blue" if i < len(pairs) // 2 else "tab:orange", markersize=3)
        ax.plot([], [], color="tab:blue", label="FreeGSNKE")
        ax.plot([], [], color="tab:orange", linestyle="--", label="GSFit")
        ax.plot([], [], color="tab:blue", linewidth=2.5, label="FreeGSNKE LCFS (psi_N = 1)")
        ax.plot([], [], color="tab:orange", linestyle="--", linewidth=2.5, label="GSFit LCFS (psi_N = 1)")
        ax.plot([], [], "o", color="tab:blue", label="Horizontal-chord pairs")
        ax.plot([], [], "o", color="tab:orange", label="Vertical-chord pairs")
        ax.set(title=label, xlabel="R [m]", ylabel="Z [m]", aspect="equal")
        ax.legend(fontsize=8)
        ax.grid()
    fig.savefig(output_dir / "flux_surfaces.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for values, label in zip(residuals, labels, strict=True):
        ax.plot(np.arange(1, len(pairs) + 1), values * 1e3, "o-", label=label)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(len(pairs) // 2 + 0.5, color="0.5", linestyle=":", label="Horizontal / vertical split")
    pair_labels: list[str] = [
        f"{chord}{i_pair + 1}"
        for chord in ("H", "V")
        for i_pair in range(len(pairs) // 2)
    ]
    ax.set_xticks(
        np.arange(1, len(pairs) + 1),
        [label if i_pair % 2 == 0 else f"\n{label}" for i_pair, label in enumerate(pair_labels)],
        fontsize=8,
    )
    ax.set(xlabel="Isoflux pair", ylabel="Flux difference [mWb]", title="Residuals evaluated on the returned flux grid")
    ax.legend()
    ax.grid()
    fig.savefig(output_dir / "pair_residuals.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
