import numpy as np
from scipy.constants import mu_0

from gsfit_rs import BpProbes
from gsfit_rs import Dialoop
from gsfit_rs import EfitPolynomial
from gsfit_rs import FluxLoops
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import PfActive
from gsfit_rs import PfPassive
from gsfit_rs import Plasma
from gsfit_rs import Pressure
from gsfit_rs import RogowskiCoils
from gsfit_rs import StationaryPoint
from gsfit_rs import Tf
from gsfit_rs import Wall
from gsfit_rs import solve_grad_shafranov
from gsfit_rs.dd_support import ParametersInput
from gsfit_rs.imas import equilibrium_paths as ep

# DOF: 3 dof's
# 1. p_prime; 1 profile shape
# 2. ff_prime; 1 profile shape
# 3. delta_z
# Constraints:
# 1. magnetic_axis location
# 2. rogowski_coil = plasma current
# 3. pressure[rho=0] ?
# 4. ? BT at given R; f=R*BT ?


def run() -> None:
    # Make a Helmholtz PF coil. Its filaments are 0.1 m squares: `PfActive` needs a positive size, for the coil's self-inductance.
    # The self test reads no settings, so has none to record in `pf_active/code/parameters`
    pf_active = PfActive(ParametersInput())
    pf_active.add_pf_coil(
        name="helmholtz_01",
        r=np.array([15.123, 15.223]),
        z=np.array([-15.123, 15.223]),
        width=np.array([0.1, 0.1]),
        height=np.array([0.1, 0.1]),
        time=np.array([0.0, 1.0]),
        current=np.array([100.0e3, 100.0e3]),
    )

    limit_pts_r = np.array([10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.0, 10.0, 10.0])
    limit_pts_z = np.array([0.0, 0.25, 1.0, 1.0, 1.0, 0.25, 0.0, -0.25, -1.0, -1.0, -1.0, -0.25])

    ip_guess = 54.750e3  # ampere

    # Create sensors
    bp_probes = BpProbes()
    # bp_probes.add_sensor(
    #     name='bp_01',
    #     geometry_angle_pol=np.pi/2,
    #     geometry_r=10.5,
    #     geometry_z=0.0,
    #     fit_settings_comment="",
    #     fit_settings_expected_value=1.0,
    #     fit_settings_include=True,
    #     fit_settings_weight=1.0,
    #     time=np.array([0.0, 1.0]),
    #     measured=np.array([0.0, 0.0]),
    # )
    flux_loops = FluxLoops()
    rogowski_coils = RogowskiCoils()
    rogowski_coils.add_sensor(
        name="rogowski_01",
        r=limit_pts_r,
        z=limit_pts_z,
        fit_settings_comment="Rogowski coil 01",
        fit_settings_expected_value=1.0,
        fit_settings_include=True,
        fit_settings_weight=1.0,
        time=np.array([0.0, 1.0]),
        measured=np.array([ip_guess, ip_guess]),
        gaps_r=np.array([]),
        gaps_z=np.array([]),
        gaps_d_r=np.array([]),
        gaps_d_z=np.array([]),
        gaps_name=[],
    )
    isoflux = Isoflux()
    isoflux_boundary = IsofluxBoundary()
    dialoop = Dialoop()
    stationary_point = StationaryPoint()
    stationary_point.add_sensor(
        name="magnetic_axis",
        fit_settings_comment="Magnetic axis",
        fit_settings_expected_value=1.0e-3,
        fit_settings_include=True,
        fit_settings_weight=100.0,
        time=np.array([0.0, 1.0]),
        mag_axis_r=np.array([10.5, 10.5]),
        mag_axis_z=np.array([0.0, 0.0]),
        times_to_reconstruct=np.array([0.5]),
    )

    p_prime_source_function = EfitPolynomial(
        n_dof=1,
        regularisations=np.zeros((0, 1)),
    )
    ff_prime_source_function = EfitPolynomial(
        n_dof=1,
        regularisations=np.zeros((0, 1)),
    )

    # Plasma
    plasma = Plasma(
        n_r=100,
        n_z=201,
        r_min=9.0,
        r_max=11.5,
        z_min=-1.5,
        z_max=1.5,
        psi_norm=np.linspace(0.0, 1.0, 100),
        p_prime_source_function=p_prime_source_function,
        ff_prime_source_function=ff_prime_source_function,
        initial_guess_ip=ip_guess,
        initial_guess_cur_r=10.5,
        initial_guess_cur_z=0.0,
        initial_guess_minor_radius=0.5,
        initial_guess_elongation=2.0,
        n_iter_max=30,
        n_iter_min=1,
        grad_shafranov_deviation_tolerance=1.0e5,
        nonlinear_solver_method="picard",
        picard_n_iter_no_vertical_feedback=100,
        picard_apply_anderson_mixing=False,
        picard_anderson_n_history=5,
        picard_anderson_mixing=1.0,
        newton_krylov_picard_handover=1.0e-2,
        newton_krylov_n_krylov_max=10,
        newton_krylov_krylov_tolerance=0.1,
        newton_krylov_finite_difference_step=1.0,
        newton_krylov_verbose=False,
        newton_picard_n_basis_max=10,
        newton_picard_contraction_threshold=0.5,
        newton_picard_finite_difference_step=1.0,
        newton_picard_verbose=False,
        times_to_reconstruct=np.array([0.5]),
    )

    # Toroidal field. `b_field_phi_vacuum_r` is the vacuum poloidal-current function
    # `f_vac = mu_0 * i_rod / (2 * pi)`, here for a 2 kA rod current. `gsfit_rs` converts it back
    # to a rod current using its own `mu_0`, which is a different CODATA revision to
    # `scipy.constants`, so the value it recovers is ~7e-10 different from `i_rod` below. Compare
    # the two with a tolerance rather than exactly
    i_rod = 2.0e3  # [ampere]
    f_vac = mu_0 * i_rod / (2.0 * np.pi)  # [tesla * metre]
    # The self test reads no settings, so has none to record in `tf/code/parameters`
    tf = Tf(ParametersInput())
    tf.set_r0(10.5)
    tf.set_b_field_phi_vacuum_r(
        time=np.array([0.0, 1.0]),
        data=np.array([f_vac, f_vac]),
    )

    # The solver reads the vacuum toroidal field from the equilibrium IDS, so it is copied in from the toroidal field
    plasma.copy_from_tf_ids(tf)

    # Wall. `unit(0)` is the vacuum vessel contour; here it is the only limiter unit
    # The self test reads no settings, so has none to record in `wall/code/parameters`
    wall = Wall(ParametersInput())
    wall.add_limiter_unit(name="vacuum_vessel", r=limit_pts_r, z=limit_pts_z)

    # The solver reads the limiter from the equilibrium IDS, so it is copied in from the wall
    plasma.copy_from_wall_ids(wall)

    # The solver reads the PF coil current from the equilibrium IDS, so it is copied in from the PF coils
    plasma.copy_from_pf_active_ids(pf_active)

    # No passive conductors. This test reads no settings, so has none to record in `pf_passive/code/parameters`
    pf_passive = PfPassive(ParametersInput())

    # The solver reads the passive loops' regularisations from the equilibrium IDS, so they are copied in from the passive conductors
    plasma.copy_from_pf_passive_ids(pf_passive)

    pressure_sensors = Pressure()

    # Greens with the PF coils
    plasma.greens_with_pf_active(pf_active)
    bp_probes.greens_with_pf_active(pf_active)
    flux_loops.greens_with_pf_active(pf_active)
    rogowski_coils.greens_with_pf_active(pf_active)
    isoflux.greens_with_pf_active(pf_active)
    isoflux_boundary.greens_with_pf_active(pf_active)
    pressure_sensors.greens_with_pf_active(pf_active)
    stationary_point.greens_with_pf_active(pf_active)

    # Greens with the passive conductors
    plasma.greens_with_pf_passive(pf_passive)
    bp_probes.greens_with_pf_passive(pf_passive)
    flux_loops.greens_with_pf_passive(pf_passive)
    rogowski_coils.greens_with_pf_passive(pf_passive)
    isoflux.greens_with_pf_passive(pf_passive)
    isoflux_boundary.greens_with_pf_passive(pf_passive)
    pressure_sensors.greens_with_pf_passive(pf_passive)
    stationary_point.greens_with_pf_passive(pf_passive)

    # Greens with plasma
    bp_probes.greens_with_plasma(plasma)
    flux_loops.greens_with_plasma(plasma)
    rogowski_coils.greens_with_plasma(plasma)
    isoflux.greens_with_plasma(plasma)
    isoflux_boundary.greens_with_plasma(plasma)
    pressure_sensors.greens_with_plasma(plasma)
    stationary_point.greens_with_plasma(plasma)

    solve_grad_shafranov(
        plasma=plasma,
        bp_probes=bp_probes,
        flux_loops=flux_loops,
        rogowski_coils=rogowski_coils,
        isoflux=isoflux,
        isoflux_boundary=isoflux_boundary,
        pressure_sensors=pressure_sensors,
        stationary_point=stationary_point,
        dialoop=dialoop,
    )

    # `profiles_2d(0)` because GSFit solves on a single rectangular (R, Z) grid
    r = plasma.get(ep.time_slice[0].profiles_2d[0].grid.dim1)
    z = plasma.get(ep.time_slice[0].profiles_2d[0].grid.dim2)
    psi_2d = plasma.get(ep.time_slice[:].profiles_2d[0].psi)  # shape = (n_time, n_z, n_r)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.contour(r, z, psi_2d[0, :, :], 100)
    plt.axis("equal")
    plt.plot(limit_pts_r, limit_pts_z, marker="x", color="black")
    bounding_r = plasma.get(ep.time_slice[:].boundary.bounding.r)
    bounding_z = plasma.get(ep.time_slice[:].boundary.bounding.z)
    boundary_r = plasma.get(ep.time_slice[0].boundary.outline.r)
    boundary_z = plasma.get(ep.time_slice[0].boundary.outline.z)
    plt.plot(bounding_r, bounding_z, marker="o", color="red")
    plt.plot(boundary_r, boundary_z, color="red")
    plt.plot()
    plt.savefig("self_test_output.png")


if __name__ == "__main__":
    run()
