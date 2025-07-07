"""
This module contains routines for discretizing the left-hand side of the Grad-Shafranov equation,
which involves applying a modified Laplacian operator to the poloidal flux function.

The resulting matrix is then factorized into a product of lower and upper triangular matrices (LU decomposition),
which accelerates the Gaussian elimination process used in the RTGSFIT code.
"""

import numpy as np
from scipy.linalg import lu


def poisson_matrix(
    r_vec: np.ndarray,
    z_vec: np.ndarray,
) -> np.ndarray:
    """
    Construct the Poisson matrix used in the RTGSFIT code for discretizing the Grad-Shafranov equation.

    This implementation follows the finite difference scheme described in Equation (45) of:
    Moret, J.-M., et al. "Tokamak equilibrium reconstruction code LIUQE and its real time implementation."
    Fusion Engineering and Design 91 (2015): 1–15.

    The discretized equation has the form:

        ψ_{i+1,j} + ψ_{i-1,j} + a_j * ψ_{i,j+1} + b_j * ψ_{i,j-1} - c_j * ψ_{i,j}

    where:
        - ψ is the poloidal flux function,
        - a_j = (dz / dr)^2 * r_j / (r_j - dr / 2),
        - b_j = (dz / dr)^2 * r_j / (r_j + dr / 2),
        - c_j = 2 + a_j + b_j

    Let:
        - n_r = length of `r_vec` (number of radial grid points),
        - n_z = length of `z_vec` (number of vertical grid points).

    The resulting Poisson matrix has dimensions (n_r * n_z) × (n_r * n_z), corresponding to the total
    number of grid points in the 2D domain.

    Parameters:
        r_vec (np.ndarray): 1D array of radial grid coordinates.
        z_vec (np.ndarray): 1D array of vertical grid coordinates.

    Returns:
        np.ndarray: The assembled Poisson matrix representing the discretized operator.
    """

    n_r = len(r_vec)
    n_z = len(z_vec)
    n_grid = n_r * n_z
    dr = r_vec[1] - r_vec[0]
    dz = z_vec[1] - z_vec[0]

    a = (dz / dr) ** 2 * r_vec / (r_vec + dr / 2)
    b = (dz / dr) ** 2 * r_vec / (r_vec - dr / 2)
    c = 2 + a + b

    poiss_matrix = np.zeros((n_grid, n_grid), dtype=np.float64)

    idx = -1
    for i in range(n_z):
        for j in range(n_r):
            idx += 1
            if i == 0 or i == n_z - 1 or j == 0 or j == n_r - 1:
                poiss_matrix[idx, idx] = 1  # Dirichlet BC
            else:
                poiss_matrix[idx, idx] = -c[j]  # coefficient of ψ_{i,j}
                poiss_matrix[idx, idx + 1] = a[j]  # coefficient of ψ_{i,j+1}
                poiss_matrix[idx, idx - 1] = b[j]  # coefficient of ψ_{i,j-1}
                poiss_matrix[idx, idx + n_r] = 1  # coefficient of ψ_{i+1,j}
                poiss_matrix[idx, idx - n_r] = 1  # coefficient of ψ_{i-1,j}

    return poiss_matrix


def poisson_matrix_lup_bands(poiss_matrix: np.ndarray, n_r: int) -> tuple:
    """
    Perform LU decomposition of the Poisson matrix with banded storage.

    This function uses the `scipy.linalg.lu` function to decompose the Poisson matrix
    into a lower triangular matrix L and an upper triangular matrix U, along with a
    permutation matrix P.

    We then extract the non-zero bands of L and U, which are used in the RTGSFIT code
    for efficient Gaussian elimination.
    We don't store the main diagonal of L as it is always 1.

    Note that it's not garunteed that L with have a bandwidth of n_r and U will have a bandwidth of n_r + 2,
    so we should check this if we use a new grid.

    Parameters:
        poisson_matrix (np.ndarray): The Poisson matrix to be decomposed.

    Returns:
        tuple: A tuple containing the permutation matrix P, lower triangular matrix L,
               and upper triangular matrix U.
    """
    perm, lower, upper = lu(poiss_matrix)
    # Invert the permutation matrix P to get it in the same form
    # as the Matlab lu function.
    perm = perm.T
    # Pad the matrices so we can extract the bands more easily.
    lower_pad = np.pad(lower, ((0, 0), (n_r, 0)), mode="constant", constant_values=0)
    upper_pad = np.pad(upper, ((0, 0), (0, n_r + 1)), mode="constant", constant_values=0)
    n_grid = np.shape(poiss_matrix)[0]
    lower_band = np.zeros((n_grid, n_r), dtype=np.float64)
    upper_band = np.zeros((n_grid, n_r + 2), dtype=np.float64)
    for ii in range(n_grid):
        lower_band[ii, :] = lower_pad[ii, ii : ii + n_r]
        upper_band[ii, :] = upper_pad[ii, ii : ii + n_r + 2]
    # We precompute the inverse of the main diagonal of U to avoid repeated division later.
    # In Gaussian elimination, the final step involves dividing by the diagonal elements of U.
    # By inverting them here, we save computation time during back-substitution.
    # For reference, see:
    # Jardin, S. (2010). Computational Methods in Plasma Physics, CRC Press, p. 54.
    upper_band[:, 0] = 1 / upper_band[:, 0]
    _, perm_idx = np.nonzero(perm)
    return perm_idx, lower_band.flatten(), upper_band.flatten()


def compute_lup_bands(r_vec: np.ndarray, z_vec: np.ndarray) -> tuple:
    """
    Compute the permutation indices and lower/upper bands of the Poisson matrix.

    This function constructs the Poisson matrix using the provided radial and vertical grid vectors,
    performs LU decomposition, and extracts the permutation indices, lower band, and upper band.

    Parameters:
        r_vec (np.ndarray): 1D array of radial grid coordinates.
        z_vec (np.ndarray): 1D array of vertical grid coordinates.

    Returns:
        tuple: A tuple containing the permutation indices, lower band, and upper band.
    """
    poisson_mat = poisson_matrix(r_vec, z_vec)
    perm_idx, lower_band, upper_band = poisson_matrix_lup_bands(poisson_mat, n_r=len(r_vec))
    return lower_band, upper_band, perm_idx


if __name__ == "__main__":
    # Example usage
    r_vec = np.linspace(110.000e-3, 1.070e0, 33)
    z_vec = np.linspace(-960.000e-3, 960.000e-3, 65)
    poisson_mat = poisson_matrix(r_vec, z_vec)
    perm_idx, lower_band, upper_band = poisson_matrix_lup_bands(poisson_mat, n_r=len(r_vec))
