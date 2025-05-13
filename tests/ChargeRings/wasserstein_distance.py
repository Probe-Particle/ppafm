# wasserstein_distance.py
import numpy as np

def wasserstein_1d_general(
    xs1: np.ndarray,  # x-coordinates for distribution 1
    ys1: np.ndarray,  # y-values (intensities/masses) for distribution 1
    xs2: np.ndarray,  # x-coordinates for distribution 2
    ys2: np.ndarray   # y-values (intensities/masses) for distribution 2
) -> float:
    """
    Computes the 1st Wasserstein distance (Earth Mover's Distance) between two
    1D distributions, where each distribution is defined by (x_coordinates, y_intensities).

    Args:
        xs1: 1D array of x-coordinates for the first distribution.
        ys1: 1D array of y-values (intensities/masses) for the first distribution.
        xs2: 1D array of x-coordinates for the second distribution.
        ys2: 1D array of y-values (intensities/masses) for the second distribution.

    Returns:
        The 1st Wasserstein distance. Returns np.nan if inputs are invalid.
    """
    xs1 = np.asarray(xs1, dtype=float)
    ys1 = np.asarray(ys1, dtype=float)
    xs2 = np.asarray(xs2, dtype=float)
    ys2 = np.asarray(ys2, dtype=float)

    if xs1.ndim != 1 or ys1.ndim != 1 or xs2.ndim != 1 or ys2.ndim != 1:
        return np.nan
    if xs1.shape != ys1.shape or xs2.shape != ys2.shape:
        return np.nan

    # Normalize y-values (treat them as masses to be moved)
    sum_ys1 = np.sum(ys1)
    norm_ys1 = ys1 / sum_ys1 if sum_ys1 > 1e-9 else np.zeros_like(ys1)
    
    sum_ys2 = np.sum(ys2)
    norm_ys2 = ys2 / sum_ys2 if sum_ys2 > 1e-9 else np.zeros_like(ys2)

    if np.all(norm_ys1 < 1e-12) and np.all(norm_ys2 < 1e-12): # Both effectively empty
        return 0.0

    # Filter out points with negligible intensity (mass)
    # Keep at least one point if all ys are zero to define the domain for CDF eval
    if xs1.size > 0:
        valid1_mask = norm_ys1 > 1e-12
        if np.any(valid1_mask):
            xs1_f = xs1[valid1_mask]
            ys1_f_norm = norm_ys1[valid1_mask] # Use already normalized, filtered ys
        else: # All ys are zero or close to zero for dist1
            xs1_f = xs1[0:1] if xs1.size > 0 else np.array([]) 
            ys1_f_norm = np.zeros_like(xs1_f)
    else:
        xs1_f, ys1_f_norm = np.array([]), np.array([])

    if xs2.size > 0:
        valid2_mask = norm_ys2 > 1e-12
        if np.any(valid2_mask):
            xs2_f = xs2[valid2_mask]
            ys2_f_norm = norm_ys2[valid2_mask]
        else: # All ys are zero or close to zero for dist2
            xs2_f = xs2[0:1] if xs2.size > 0 else np.array([])
            ys2_f_norm = np.zeros_like(xs2_f)
    else:
        xs2_f, ys2_f_norm = np.array([]), np.array([])

    if xs1_f.size == 0 and xs2_f.size == 0: # Both effectively empty after filtering
        return 0.0
    
    # Sort points by x-coordinate
    sorter1 = np.argsort(xs1_f)
    xs1_sorted = xs1_f[sorter1]
    ys1_sorted_norm = ys1_f_norm[sorter1]

    sorter2 = np.argsort(xs2_f)
    xs2_sorted = xs2_f[sorter2]
    ys2_sorted_norm = ys2_f_norm[sorter2]

    # Create a set of all unique x-coordinates where CDFs might change.
    # Include original domain endpoints from non-empty original xs to ensure full range.
    all_x_points_for_cdf_eval = []
    if xs1.size > 0: all_x_points_for_cdf_eval.extend([np.min(xs1), np.max(xs1)])
    if xs2.size > 0: all_x_points_for_cdf_eval.extend([np.min(xs2), np.max(xs2)])
    if xs1_sorted.size > 0: all_x_points_for_cdf_eval.extend(xs1_sorted)
    if xs2_sorted.size > 0: all_x_points_for_cdf_eval.extend(xs2_sorted)
    
    if not all_x_points_for_cdf_eval: return 0.0
    
    all_xs_unique = np.unique(all_x_points_for_cdf_eval)
    
    if all_xs_unique.size <= 1: # Not enough points to form intervals
        return 0.0

    # Calculate CDFs at these unique x-points
    cdf1_at_all_xs = np.zeros_like(all_xs_unique, dtype=float)
    if xs1_sorted.size > 0: # Check if there's anything to cumsum
        cum_ys1_norm = np.cumsum(ys1_sorted_norm)
        # Find where each unique x would be inserted into xs1_sorted
        indices1 = np.searchsorted(xs1_sorted, all_xs_unique, side='right')
        # Get CDF values for points that are actually found or past in xs1_sorted
        valid_indices1_mask = indices1 > 0
        cdf1_at_all_xs[valid_indices1_mask] = cum_ys1_norm[indices1[valid_indices1_mask] - 1]

    cdf2_at_all_xs = np.zeros_like(all_xs_unique, dtype=float)
    if xs2_sorted.size > 0:
        cum_ys2_norm = np.cumsum(ys2_sorted_norm)
        indices2 = np.searchsorted(xs2_sorted, all_xs_unique, side='right')
        valid_indices2_mask = indices2 > 0
        cdf2_at_all_xs[valid_indices2_mask] = cum_ys2_norm[indices2[valid_indices2_mask] - 1]
    
    # Calculate the integral sum(|CDF1(x) - CDF2(x)| * dx_interval)
    cdf_diffs = np.abs(cdf1_at_all_xs[:-1] - cdf2_at_all_xs[:-1])
    interval_widths = np.diff(all_xs_unique)

    distance = np.sum(cdf_diffs * interval_widths)
    return distance


def wasserstein_1d_grid(
    ys1: np.ndarray,  # y-values (intensities/histogram heights) for dist 1 on grid
    ys2: np.ndarray,  # y-values (intensities/histogram heights) for dist 2 on grid
    dx: float         # Constant step size of the regular grid
) -> float:
    """
    Computes 1D Wasserstein distance for distributions represented by y-values (intensities)
    on a regular grid. Assumes y-values are proportional to mass in each bin.

    Args:
        ys1: y-values (intensities/histogram heights) of the first distribution.
        ys2: y-values (intensities/histogram heights) of the second distribution.
        dx: Bin width (constant step size of the regular grid).

    Returns:
        The 1D Wasserstein distance. Returns np.nan if inputs are invalid.
    """
    ys1 = np.asarray(ys1, dtype=float)
    ys2 = np.asarray(ys2, dtype=float)

    if ys1.shape != ys2.shape: return np.nan
    if ys1.ndim != 1: return np.nan
    if dx <= 0: return np.nan

    # Normalize y-values (treat them as masses to be moved)
    sum_ys1 = np.sum(ys1)
    norm_ys1 = ys1 / sum_ys1 if sum_ys1 > 1e-9 else np.zeros_like(ys1)
    
    sum_ys2 = np.sum(ys2)
    norm_ys2 = ys2 / sum_ys2 if sum_ys2 > 1e-9 else np.zeros_like(ys2)

    if np.all(norm_ys1 < 1e-12) and np.all(norm_ys2 < 1e-12): # Both effectively empty
        return 0.0

    cdf1 = np.cumsum(norm_ys1)
    cdf2 = np.cumsum(norm_ys2)
    
    distance = np.sum(np.abs(cdf1 - cdf2)) * dx
    return distance