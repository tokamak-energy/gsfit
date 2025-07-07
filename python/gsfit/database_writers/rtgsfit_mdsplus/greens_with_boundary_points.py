import gsfit_rs
import numpy as np
import numpy.typing as npt


def greens_with_boundary_points(plasma: gsfit_rs.Plasma) -> npt.NDArray[np.float64]:
    """
    Calculate the Greens table with the boundary points.
    In this context "boundary" means the 2D (R, Z) grid, not the plasma boundary.

    :param plasma: Plasma object containing the grid information.

    Boundary points are ordered clockwise from the bottom left, i.e.
    * (bottom, left) = (r_min, z_min)
    * (top, left) = (r_min, z_max)
    * (top, right) = (r_max, z_max)
    * (bottom, right) = (r_max, z_min)

    Note: need to be careful not to double count the corner points
    """

    r = plasma.get_array1(["grid", "r"])
    z = plasma.get_array1(["grid", "z"])
    r_min = np.min(r)
    r_max = np.max(r)
    z_min = np.min(z)
    z_max = np.max(z)
    r_inner = r[1:-1]  # Exclude the first and last points to avoid double counting corners
    z_inner = z[1:-1]  # Exclude the first and last points to avoid double counting corners
    ones_len_r_inner = np.ones_like(r_inner)
    ones_len_z_inner = np.ones_like(z_inner)

    d_r = np.mean(np.diff(r))
    d_z = np.mean(np.diff(z))

    r_ltrb = np.concatenate(
        (
            [r_min],  # (bottom, left)
            r_min * ones_len_z_inner,  # traverse (bottom, left)-delta to (top, left)-delta
            [r_min],  # (top, left)
            r_inner,  # traverse (top, left)-delta to (top, right)-delta
            [r_max],  # (top, right)
            r_max * ones_len_z_inner,  # traverse (top, right)-delta to (bottom, right)-delta
            [r_max],  # (bottom, right)
            np.flip(r_inner),  # traverse (bottom, right)-delta to (bottom, left)-delta
        )
    )
    z_ltrb = np.concatenate(
        (
            [z_min],  # (bottom, left)
            z_inner,  # traverse (bottom, left)-delta to (top, left)-delta
            [z_max],  # (top, left)
            z_max * ones_len_r_inner,  # traverse (top, left)-delta to (top, right)-delta
            [z_max],  # (top, right)
            np.flip(z_inner),  # traverse (top, right)-delta to (bottom, right)-delta
            [z_min],  # (bottom, right)
            z_min * ones_len_r_inner,  # traverse (bottom, right)-delta to (bottom, left)-delta
        )
    )

    n_ltrb = len(r_ltrb)

    d_r_vec = np.ones(n_ltrb) * d_r
    d_z_vec = np.ones(n_ltrb) * d_z

    # Calculate the inductance matrix between boundary points
    g_ltrb = gsfit_rs.greens_py(
        r_ltrb,
        z_ltrb,
        r_ltrb,
        z_ltrb,
        d_r_vec,  # Needed for self-inductance
        d_z_vec,  # Needed for self-inductance
    )

    # Check diagonal values for self-inductance are g_ltrb[i, i] = self_inductance_rectangle_cross_section(r_ltrb[i], d_r, d_z)
    for i in range(n_ltrb):
        expected_self_inductance = self_inductance_rectangle_cross_section(r_ltrb[i], d_r_vec[i], d_z_vec[i])
        expected_self_inductance *= 2 * np.pi
        if not np.isclose(g_ltrb[i, i], expected_self_inductance):
            raise ValueError(f"Diagonal value at index {i} does not match expected self-inductance: {g_ltrb[i, i]} != {expected_self_inductance}")

    # Need to overwrite the diagonals of the matrix with the self-inductance for a surface current on
    # the computational boundary. Excluding the corners, which are not used by RTGSFIT
    # Note the corners are at the following indicdes:
    # 0, len(z) - 1, len(z) + len(r) - 2, 2 * len(z) + len(r) - 3, 2 * len(z) + 2 * len(r) - 4
    tl_corner_idx = len(z) - 1
    tr_corner_idx = len(z) + len(r) - 2
    br_corner_idx = 2 * len(z) + len(r) - 3
    bl_corner_idx = 2 * len(z) + 2 * len(r) - 4
    # Left side
    for i in range(tl_corner_idx):
        g_ltrb[i, i] = self_inductance_rectangle_cross_section(r_ltrb[i], 0, d_z)
    # Top side
    for i in range(tl_corner_idx + 1, tr_corner_idx):
        g_ltrb[i, i] = self_inductance_rectangle_cross_section(r_ltrb[i], d_r, 0)
    # Right side
    for i in range(tr_corner_idx + 1, br_corner_idx):
        g_ltrb[i, i] = self_inductance_rectangle_cross_section(r_ltrb[i], 0, d_z)
    # Bottom side
    for i in range(br_corner_idx + 1, bl_corner_idx):
        g_ltrb[i, i] = self_inductance_rectangle_cross_section(r_ltrb[i], d_r, 0)
    # Note: g_ltrb.shape = (n_ltrb, n_ltrb) = (2 * n_r + 2 * n_z - 4, 2 * n_r + 2 * n_z - 4)
    # The -4 is so that we don't double count corners

    # Divide through by 2pi
    g_ltrb /= 2 * np.pi

    return g_ltrb.flatten()


def self_inductance_rectangle_cross_section(r: float, delta_r: float, delta_z: float) -> float:
    """
    Calculate the self-inductance of an axisymmetric wire  with a rectangular cross-section
    of height delta_z and width delta_r, centered at (r, z).

    :param r: The R coordinate of the center of the rectangle.
    :param z: The Z coordinate of the center of the rectangle.
    :param delta_r: The width of the rectangle in the R direction.
    :param delta_z: The height of the rectangle in the Z direction.
    :return: The self-inductance of the rectangle in H (Henries).
    """

    mu_0 = 4e-7 * np.pi  # Vacuum permeability in H/m
    return (
        mu_0
        * r
        * ((1 + 2 * (delta_z / (8 * r)) ** 2 + 2 / 3 * (delta_r / (8 * r)) ** 2) * np.log(8 * r / (delta_r + delta_z)) - 0.5 + 0.5 * (delta_z / (8 * r)) ** 2)
    )
