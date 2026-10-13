"""Isoflux near-source regularisation, without database access or an equilibrium solve."""

import gsfit_rs
import numpy as np
import pytest
from gsfit_rs.imas import equilibrium_paths as ep


def make_isoflux(offset: float) -> gsfit_rs.Isoflux:
    sensor = gsfit_rs.Isoflux()
    sensor.add_sensor(
        name="pair",
        fit_settings_comment="near-source test",
        fit_settings_include=True,
        fit_settings_weight=100.0,
        time=np.array([0.0, 1.0]),
        location_1_r=np.array([0.4125 + offset, np.nan]),
        location_1_z=np.array([0.001, 0.001]),
        location_2_r=np.array([0.7, np.nan]),
        location_2_z=np.array([0.03, 0.03]),
        times_to_reconstruct=np.array([0.0, 1.0]),
    )
    return sensor


def make_plasma() -> gsfit_rs.Plasma:
    source = gsfit_rs.EfitPolynomial(1, np.zeros((0, 1)))
    return gsfit_rs.Plasma(
        n_r=3,
        n_z=3,
        r_min=0.4,
        r_max=0.425,
        z_min=-0.025,
        z_max=0.025,
        psi_norm=np.linspace(0.0, 1.0, 3),
        p_prime_source_function=source,
        ff_prime_source_function=source,
        initial_guess_ip=1.0e4,
        initial_guess_cur_r=0.4125,
        initial_guess_cur_z=0.0,
        initial_guess_minor_radius=0.01,
        initial_guess_elongation=1.0,
        n_iter_max=5,
        n_iter_min=1,
        n_iter_no_vertical_feedback=0,
        grad_shafranov_deviation_tolerance=1.0e-5,
        use_anderson_mixing=False,
        anderson_mixing_from_previous_iter=0.0,
        times_to_reconstruct=np.array([0.0, 1.0]),
    )


def expected_difference(
    offset: float, source_r: np.ndarray, source_z: np.ndarray, d_r: np.ndarray, d_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.array([0.4125 + offset, 0.7])
    z = np.array([0.001, 0.03])
    psi = gsfit_rs.greens_py(r, z, source_r, source_z, d_r, d_z)
    derivative = gsfit_rs.greens_d_psi_d_z(r, z, source_r, source_z, d_r, d_z)
    return psi[0] - psi[1], derivative[0] - derivative[1]


@pytest.mark.parametrize("offset", [0.0, 1.1e-7, 0.0049, 0.005, 0.0051, 0.006])
def test_plasma_uses_actual_cell_widths(offset: float) -> None:
    sensor = make_isoflux(offset)
    sensor.greens_with_plasma(make_plasma())
    results = sensor.__getstate__()["results"]["pair"]
    greens = results["greens"]
    r, z = np.meshgrid(np.linspace(0.4, 0.425, 3), np.linspace(-0.025, 0.025, 3))
    source_r, source_z = r.ravel(), z.ravel()
    d_r, d_z = np.full(9, 0.0125), np.full(9, 0.025)
    psi, derivative = expected_difference(offset, source_r, source_z, d_r, d_z)
    assert np.isfinite(greens["plasma"]).all()
    assert np.isfinite(greens["d_plasma_d_z"]).all()
    np.testing.assert_allclose(greens["plasma"][0], psi, rtol=1e-12)
    np.testing.assert_allclose(greens["d_plasma_d_z"][0], derivative, rtol=1e-12)
    np.testing.assert_array_equal(greens["plasma"][1], np.zeros(9))
    assert not results["fit_settings"]["include_dynamic"][1]

    evaluation_r = np.array([0.4125 + offset])
    evaluation_z = np.array([0.001])
    if offset < 0.005:
        # The centre approximation uses the evaluation radius in the self-term formula.
        reference = gsfit_rs.greens_py(evaluation_r, evaluation_z, evaluation_r, evaluation_z, d_r[:1], d_z[:1])
    else:
        reference = gsfit_rs.greens_py(evaluation_r, evaluation_z, source_r[4:5], source_z[4:5], np.zeros(1), np.zeros(1))
    other_point = gsfit_rs.greens_py(np.array([0.7]), np.array([0.03]), source_r[4:5], source_z[4:5], d_r[:1], d_z[:1])
    assert greens["plasma"][0, 4] == pytest.approx(reference[0, 0] - other_point[0, 0], rel=1e-12)


@pytest.mark.parametrize("offset", [1.1e-7, 0.006])
@pytest.mark.parametrize("source_kind", ["coil", "passive"])
def test_conductor_widths_and_far_field_are_preserved(offset: float, source_kind: str) -> None:
    sensor = make_isoflux(offset)
    r, z = np.array([0.4125]), np.array([0.0])
    d_r, d_z = np.array([0.01]), np.array([0.02])
    if source_kind == "coil":
        coils = gsfit_rs.Coils()
        coils.add_pf_coil("source", r, z, d_r, d_z, np.array([0.0, 1.0]), np.ones(2))
        sensor.greens_with_coils(coils)
        result = sensor.__getstate__()["results"]["pair"]["greens"]["pf"]["source"]
        scale = 1.0
    else:
        passives = gsfit_rs.Passives()
        passives.add_passive(
            "source", r, z, d_r, d_z, np.zeros(1), np.zeros(1),
            1e-6, "constant_current_density", 1, np.zeros((0, 1)), np.zeros(0),
        )
        sensor.greens_with_passives(passives)
        result = sensor.__getstate__()["results"]["pair"]["greens"]["passives"]["source"]["constant_current_density"]
        scale = d_r[0] * d_z[0]
    expected, _ = expected_difference(offset, r, z, d_r, d_z)
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(expected[0] * scale, rel=1e-12)
    assert result[1] == 0.0
    if offset > 0.005:
        filament, _ = expected_difference(offset, r, z, np.zeros(1), np.zeros(1))
        assert result[0] == pytest.approx(filament[0] * scale, rel=1e-12)


@pytest.mark.parametrize("offset", [1.1e-7, 0.006])
@pytest.mark.parametrize("sensor_kind", ["bp_probe", "flux_loop"])
@pytest.mark.parametrize("source_kind", ["plasma", "coil", "passive"])
def test_magnetic_sensors_use_source_widths(offset: float, sensor_kind: str, source_kind: str) -> None:
    sensor = gsfit_rs.BpProbes() if sensor_kind == "bp_probe" else gsfit_rs.FluxLoops()
    settings = dict(
        name="sensor", geometry_r=0.4125 + offset, geometry_z=0.001,
        fit_settings_comment="near-source test", fit_settings_expected_value=1.0,
        fit_settings_include=True, fit_settings_weight=1.0,
        time=np.array([0.0, 1.0]), measured=np.zeros(2),
    )
    if sensor_kind == "bp_probe":
        settings["geometry_angle_pol"] = np.pi / 2.0
    sensor.add_sensor(**settings)
    if source_kind == "plasma":
        sensor.greens_with_plasma(make_plasma())
        r, z = np.meshgrid(np.linspace(0.4, 0.425, 3), np.linspace(-0.025, 0.025, 3))
        source_r, source_z = r.ravel(), z.ravel()
        d_r, d_z = np.full(9, 0.0125), np.full(9, 0.025)
        scale = 1.0
    else:
        source_r, source_z = np.array([0.4125]), np.array([0.0])
        d_r, d_z = np.array([0.01]), np.array([0.02])
        if source_kind == "coil":
            coils = gsfit_rs.Coils()
            coils.add_pf_coil("source", source_r, source_z, d_r, d_z, np.array([0.0, 1.0]), np.ones(2))
            sensor.greens_with_coils(coils)
            scale = 1.0
        else:
            passives = gsfit_rs.Passives()
            passives.add_passive(
                "source", source_r, source_z, d_r, d_z, np.zeros(1), np.zeros(1),
                1e-6, "constant_current_density", 1, np.zeros((0, 1)), np.zeros(0),
            )
            sensor.greens_with_passives(passives)
            scale = d_r[0] * d_z[0]
    greens = sensor.__getstate__()["results"]["sensor"]["greens"]
    if source_kind == "plasma":
        result = greens["plasma"]
        assert np.isfinite(greens["d_plasma_d_z"]).all()
    elif source_kind == "coil":
        result = greens["pf"]["source"]
    else:
        result = greens["passives"]["source"]["constant_current_density"]
    assert np.isfinite(result).all()
    r, z = np.array([0.4125 + offset]), np.array([0.001])
    if sensor_kind == "bp_probe":
        kernel = gsfit_rs.greens_d_psi_d_r(r, z, source_r, source_z, d_r, d_z) / (2.0 * np.pi * r[0])
    else:
        kernel = gsfit_rs.greens_py(r, z, source_r, source_z, d_r, d_z)
    expected = kernel[0] if source_kind == "plasma" else kernel.sum() * scale
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)


def test_grid_to_conductors_uses_source_widths() -> None:
    plasma = make_plasma()
    r, z = np.array([0.4125 + 1.1e-7]), np.array([0.001])
    d_r, d_z = np.array([0.01]), np.array([0.02])
    coils = gsfit_rs.Coils()
    coils.add_pf_coil("source", r, z, d_r, d_z, np.array([0.0, 1.0]), np.ones(2))
    plasma.greens_with_coils(coils)
    passives = gsfit_rs.Passives()
    passives.add_passive(
        "source", r, z, d_r, d_z, np.zeros(1), np.zeros(1),
        1e-6, "constant_current_density", 1, np.zeros((0, 1)), np.zeros(0),
    )
    plasma.greens_with_passives(passives)
    for field in ["psi", "d_psi_d_r", "d_psi_d_z", "d2_psi_d_r2", "d2_psi_d_r_d_z", "d2_psi_d_z2"]:
        assert np.isfinite(plasma.get(getattr(ep.greens.pf_active[0], field))).all()
        assert np.isfinite(plasma.get(getattr(ep.greens.pf_passive[0].dof[0], field))).all()


@pytest.mark.parametrize("offset", [0.0, 0.0049, 0.0051, 0.006])
@pytest.mark.parametrize("sensor_kind", ["isoflux_boundary", "stationary_point"])
@pytest.mark.parametrize("source_kind", ["plasma", "coil", "passive"])
def test_boundary_and_stationary_sensors_use_source_widths(offset: float, sensor_kind: str, source_kind: str) -> None:
    sensor = gsfit_rs.IsofluxBoundary() if sensor_kind == "isoflux_boundary" else gsfit_rs.StationaryPoint()
    r, z = np.array([0.4125 + offset]), np.array([0.001])
    settings = dict(
        name="sensor", fit_settings_comment="finite source geometry",
        fit_settings_include=True, fit_settings_weight=1.0,
        time=np.array([0.0, 1.0]), times_to_reconstruct=np.array([0.0, 1.0]),
    )
    if sensor_kind == "isoflux_boundary":
        settings.update(location_1_r=np.repeat(r, 2), location_1_z=np.repeat(z, 2))
    else:
        settings.update(mag_axis_r=np.repeat(r, 2), mag_axis_z=np.repeat(z, 2), fit_settings_expected_value=0.0)
    sensor.add_sensor(**settings)

    if source_kind == "plasma":
        sensor.greens_with_plasma(make_plasma())
        mesh_r, mesh_z = np.meshgrid(np.linspace(0.4, 0.425, 3), np.linspace(-0.025, 0.025, 3))
        source_r, source_z = mesh_r.ravel(), mesh_z.ravel()
        d_r, d_z = np.full(9, 0.0125), np.full(9, 0.025)
        scale = 1.0
    else:
        source_r, source_z = np.array([0.4125]), np.array([0.0])
        d_r, d_z = np.array([0.01]), np.array([0.02])
        if source_kind == "coil":
            coils = gsfit_rs.Coils()
            coils.add_pf_coil("source", source_r, source_z, d_r, d_z, np.array([0.0, 1.0]), np.ones(2))
            sensor.greens_with_coils(coils)
            scale = 1.0
        else:
            passives = gsfit_rs.Passives()
            passives.add_passive(
                "source", source_r, source_z, d_r, d_z, np.zeros(1), np.zeros(1),
                1e-6, "constant_current_density", 1, np.zeros((0, 1)), np.zeros(0),
            )
            sensor.greens_with_passives(passives)
            scale = d_r[0] * d_z[0]

    greens = sensor.__getstate__()["results"]["sensor"]["greens"]
    if source_kind == "plasma":
        result = greens["plasma"]
    elif source_kind == "coil":
        result = greens["pf"]["source"]
    else:
        result = greens["passives"]["source"]["constant_current_density"]

    if sensor_kind == "isoflux_boundary":
        kernel = gsfit_rs.greens_py(r, z, source_r, source_z, d_r, d_z)
        derivative = gsfit_rs.greens_d_psi_d_z(r, z, source_r, source_z, d_r, d_z)
    else:
        kernel = -gsfit_rs.greens_d_psi_d_z(r, z, source_r, source_z, d_r, d_z) / (2.0 * np.pi * r[0])
        derivative = -gsfit_rs.greens_d2_psi_d_z2(r, z, source_r, source_z, d_r, d_z) / (2.0 * np.pi * r[0])
    expected = np.repeat(kernel, 2, axis=0) if source_kind == "plasma" else np.full(2, kernel.sum() * scale)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)
    if source_kind == "plasma":
        assert np.isfinite(greens["d_plasma_d_z"]).all()
        np.testing.assert_allclose(greens["d_plasma_d_z"], np.repeat(derivative, 2, axis=0), rtol=1e-12, atol=1e-15)
    elif offset > 0.005:
        if sensor_kind == "isoflux_boundary":
            filament = gsfit_rs.greens_py(r, z, source_r, source_z, np.zeros(1), np.zeros(1))
        else:
            filament = -gsfit_rs.greens_d_psi_d_z(r, z, source_r, source_z, np.zeros(1), np.zeros(1)) / (2.0 * np.pi * r[0])
        np.testing.assert_allclose(result, np.full(2, filament.sum() * scale), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("widths", [(None, None), (0.0, 0.0), (0.0, 0.02), (0.01, 0.0)])
def test_near_filament_flux_does_not_use_self_term(widths: tuple[float | None, float | None]) -> None:
    r, z = np.array([0.5]), np.array([0.001])
    source_r, source_z = np.array([0.5]), np.array([0.0])
    d_r = None if widths[0] is None else np.array([widths[0]])
    d_z = None if widths[1] is None else np.array([widths[1]])
    flux = gsfit_rs.greens_py(r, z, source_r, source_z, d_r, d_z)
    assert np.isfinite(flux).all()
    assert flux[0, 0] == pytest.approx(3.95467178e-6, rel=1e-8)


@pytest.mark.parametrize("width", [-0.01, np.nan, np.inf])
@pytest.mark.parametrize("dimension", ["d_r", "d_z"])
def test_invalid_green_widths_raise_value_error(width: float, dimension: str) -> None:
    widths = {"d_r": np.array([0.01]), "d_z": np.array([0.02])}
    widths[dimension] = np.array([width])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        gsfit_rs.greens_py(np.array([0.5]), np.array([0.001]), np.array([0.5]), np.array([0.0]), **widths)


@pytest.mark.parametrize("widths", [(None, None), (0.0, 0.0), (0.0, 0.02), (0.01, 0.0)])
def test_coincident_filament_flux_is_rejected(widths: tuple[float | None, float | None]) -> None:
    r, z = np.array([0.5]), np.array([0.0])
    d_r = None if widths[0] is None else np.array([widths[0]])
    d_z = None if widths[1] is None else np.array([widths[1]])
    with pytest.raises(ValueError, match="coincident filament evaluation is singular"):
        gsfit_rs.greens_py(r, z, r, z, d_r, d_z)


def test_numerically_singular_filament_flux_is_rejected() -> None:
    with pytest.raises(ValueError, match="filament separation is numerically singular"):
        gsfit_rs.greens_py(np.array([0.5]), np.array([1e-12]), np.array([0.5]), np.array([0.0]))


@pytest.mark.parametrize("sensor_kind", ["bp_probe", "flux_loop"])
def test_vacuum_sensor_values_preserve_passive_widths(sensor_kind: str) -> None:
    r, z = np.array([0.4125]), np.array([0.0])
    d_r, d_z = np.array([0.01]), np.array([0.02])
    time = np.array([0.0, 1.0])
    coils = gsfit_rs.Coils()
    coils.add_pf_coil("source", np.array([0.8]), np.array([0.1]), d_r, d_z, time, np.zeros(2))
    state = coils.__getstate__()
    state["results"]["pf"]["source"]["i"]["simulated"] = {"time": time, "value": np.zeros(2)}
    coils.__setstate__(state)
    passives = gsfit_rs.Passives()
    passives.add_passive(
        "source", r, z, d_r, d_z, np.zeros(1), np.zeros(1),
        1e-6, "constant_current_density", 1, np.zeros((0, 1)), np.zeros(0),
    )
    state = passives.__getstate__()
    state["results"]["source"]["i_filaments"] = {"simulated": {"time": time, "value": np.ones((2, 1))}}
    passives.__setstate__(state)
    sensor = gsfit_rs.BpProbes() if sensor_kind == "bp_probe" else gsfit_rs.FluxLoops()
    settings = dict(
        name="sensor", geometry_r=0.4125, geometry_z=0.001,
        fit_settings_comment="vacuum near-source test", fit_settings_expected_value=0.0,
        fit_settings_include=True, fit_settings_weight=1.0, time=time, measured=np.zeros(2),
    )
    if sensor_kind == "bp_probe":
        settings["geometry_angle_pol"] = np.pi / 2.0
    sensor.add_sensor(**settings)
    sensor.greens_with_coils(coils)
    sensor.calculate_sensor_values_vacuum(coils, passives)
    result = sensor.__getstate__()["results"]["sensor"]
    sensor_r, sensor_z = np.array([0.4125]), np.array([0.001])
    if sensor_kind == "bp_probe":
        expected = gsfit_rs.greens_d_psi_d_r(sensor_r, sensor_z, r, z, d_r, d_z)[0, 0] / (2.0 * np.pi * sensor_r[0])
        values = result["b"]["simulated"]["value"]
    else:
        expected = gsfit_rs.greens_py(sensor_r, sensor_z, r, z, d_r, d_z)[0, 0]
        values = result["psi"]["simulated"]["value"]
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, np.full(2, expected), rtol=1e-12, atol=1e-15)
