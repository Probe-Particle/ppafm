import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve # Using scipy's wrapper for better handling
# from scipy.linalg import lu_factor, lu_solve # Alternative for explicit factorization

# --- Helper Functions ---

def wendland_c2(r, R_basis):
    """Wendland C2 compactly supported RBF."""
    r     = np.abs(r)
    mask  = r < R_basis
    t     = r[mask] / R_basis
    t1    = 1.0 - t
    t2 = t1 * t1
    t4 = t2 * t2
    out = np.zeros_like(r)
    out[mask] = t4 * (4.0 * t + 1.0)
    return out

def compact_c2_covariance(r, R_basis):
     """Compactly supported C2 Wendland function used as a covariance model C(r)."""
     # For Wendland C2, C(0)=1. The function itself can serve as the covariance kernel.
     # We use the same implementation as wendland_c2
     return wendland_c2(r, R_basis)

def compact_c2_variogram(r, R_basis):
    """Variogram gamma(r) = C(0) - C(r) derived from compact C2 covariance."""
    # C(0) for the wendland_c2 function is wendland_c2(0, R_basis), which is 1.0
    C0 = 1.0 # wendland_c2(0, R_basis) is 1.0 if R_basis > 0
    return C0 - compact_c2_covariance(r, R_basis)

def pairwise_distances(points1, points2):
    """Compute distances between all pairs of points from two arrays."""
    # Using broadcasting for efficiency in numpy
    # points1 shape (N, D), points2 shape (M, D) -> (N, 1, D) - (1, M, D) -> (N, M, D)
    # Sum squares over last dimension -> (N, M)
    # Take sqrt -> (N, M)
    return np.sqrt(np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :])**2, axis=-1))
