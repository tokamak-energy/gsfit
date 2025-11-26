import numpy as np

from gsfit_rs import BpProbes
from gsfit_rs import Coils
from gsfit_rs import EfitPolynomial
from gsfit_rs import FluxLoops
from gsfit_rs import Isoflux
from gsfit_rs import IsofluxBoundary
from gsfit_rs import StationaryPoint
from gsfit_rs import Passives
from gsfit_rs import Plasma
from gsfit_rs import RogowskiCoils
from gsfit_rs import solve_grad_shafranov

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
    # Make a Helmholtz PF coil
    coils = Coils()
    coils.add_pf_coil(
        name="helmholtz_01",
        r=np.array([15.123, 15.223]),
        z=np.array([-15.123, 15.223]),
        d_r=np.array([0.0, 0.0]),
        d_z=np.array([0.0, 0.0]),
        time=np.array([0.0, 1.0]),
        measured=np.array([100.0e3, 100.0e3]),
    )
    coils.add_tf_coil(
        time=np.array([0.0, 1.0]),
        measured=np.array([2.0e3, 2.0e3]),
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
        psi_n=np.linspace(0.0, 1.0, 100),
        limit_pts_r=limit_pts_r,
        limit_pts_z=limit_pts_z,
        vessel_r=limit_pts_r,
        vessel_z=limit_pts_z,
        p_prime_source_function=p_prime_source_function,
        ff_prime_source_function=ff_prime_source_function,
        initial_ip=ip_guess,
        initial_cur_r=10.5,
        initial_cur_z=0.0,
    )

    passives = Passives()

    # Greens with coils
    plasma.greens_with_coils(coils)
    bp_probes.greens_with_coils(coils)
    flux_loops.greens_with_coils(coils)
    rogowski_coils.greens_with_coils(coils)
    isoflux.greens_with_coils(coils)
    isoflux_boundary.greens_with_coils(coils)
    stationary_point.greens_with_coils(coils)

    # Greens with passives
    plasma.greens_with_passives(passives)
    bp_probes.greens_with_passives(passives)
    flux_loops.greens_with_passives(passives)
    rogowski_coils.greens_with_passives(passives)
    isoflux.greens_with_passives(passives)
    isoflux_boundary.greens_with_passives(passives)
    stationary_point.greens_with_passives(passives)

    # Greens with plasma
    bp_probes.greens_with_plasma(plasma)
    flux_loops.greens_with_plasma(plasma)
    rogowski_coils.greens_with_plasma(plasma)
    isoflux.greens_with_plasma(plasma)
    isoflux_boundary.greens_with_plasma(plasma)
    stationary_point.greens_with_plasma(plasma)

    solve_grad_shafranov(
        plasma=plasma,
        coils=coils,
        passives=passives,
        bp_probes=bp_probes,
        flux_loops=flux_loops,
        rogowski_coils=rogowski_coils,
        isoflux=isoflux,
        isoflux_boundary=isoflux_boundary,
        stationary_point=stationary_point,
        times_to_reconstruct=np.array([0.5]),
        n_iter_max=30,
        n_iter_min=1,
        n_iter_no_vertical_feedback=100,
        gs_error=1.0e5,
        use_anderson_mixing=False,
        anderson_mixing_from_previous_iter=0.0,
    )

    r = plasma.get_array1(["grid", "r"])
    z = plasma.get_array1(["grid", "z"])
    psi_2d = plasma.get_array3(["two_d", "psi"])
    import matplotlib.pyplot as plt

    plt.figure()
    plt.contour(r, z, psi_2d[0, :, :], 100)
    plt.axis("equal")
    plt.plot(limit_pts_r, limit_pts_z, marker="x", color="black")
    bounding_r = plasma.get_array1(["p_boundary", "bounding_r"])
    bounding_z = plasma.get_array1(["p_boundary", "bounding_z"])
    boundary_r = plasma.get_array2(["p_boundary", "rbnd"])[0, :]
    boundary_z = plasma.get_array2(["p_boundary", "zbnd"])[0, :]
    plt.plot(bounding_r, bounding_z, marker="o", color="red")
    plt.plot(boundary_r, boundary_z, color="red")
    plt.plot()
    plt.savefig("self_test_output.png")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    run()
