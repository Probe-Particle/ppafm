import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve # Using scipy's wrapper for better handling
# from scipy.linalg import lu_factor, lu_solve # Alternative for explicit factorization

from .interpy import compact_c2_covariance, pairwise_distances, wendland_c2_varR

class InterpolatorKriging:
    def __init__(self, data_points, R_basis, nugget=0.0, global_eval=False):
        """
        Setup phase: Builds and factorizes the Kriging matrix [C 1; 1^T 0].
        data_points: (N, 2) numpy array of data point locations.
        R_basis: Support radius for the covariance function. Can be either:
            - float: global support radius (legacy behavior)
            - array-like shape (N,): per-point support radii
        """
        self.data_points = np.asarray(data_points, dtype=float)
        self.ndata       = self.data_points.shape[0]
        self.R_basis     = R_basis
        self.nugget      = float(nugget)
        self.global_eval = bool(global_eval)
        self.R_i         = None
        self.R_max       = None
        if np.ndim(R_basis) == 0:
            self.R_basis = float(R_basis)
        else:
            self.R_i = np.asarray(R_basis, dtype=float).reshape(-1)
            if self.R_i.shape[0] != self.ndata:
                raise ValueError(f"InterpolatorKriging: per-point R_basis must have shape (N,), got {self.R_i.shape} for N={self.ndata}")
            self.R_max = float(np.max(self.R_i)) if self.ndata > 0 else 0.0
        if self.ndata == 0:
            print("WARRNING: GlobalKrigingInterpolator initialized with no data points.")
            self.kriging_matrix = None
            # No factorization needed
            return
        if self.R_i is None:
            print(f"GlobalKrigingInterpolator.init(): Building {self.ndata+1}x{self.ndata+1} Kriging matrix (R_basis={self.R_basis})")
        else:
            print(f"GlobalKrigingInterpolator.init(): Building {self.ndata+1}x{self.ndata+1} Kriging matrix (R_i: min={self.R_i.min():.3g} max={self.R_i.max():.3g})")

        # Build the Kriging matrix K = [C 1; 1^T 0]
        # C_ij = compact_c2_covariance(||p_i - p_j||)
        # Use pairwise_distances to get all distances efficiently
        distances = pairwise_distances(self.data_points, self.data_points)
        if self.R_i is None:
            covariance_matrix = compact_c2_covariance(distances, self.R_basis)
        else:
            # Symmetric pair radius to keep the covariance matrix symmetric.
            # Using min(R_i, R_j) enforces compact support and tends to improve overlap.
            R_pair = np.minimum(self.R_i[:, None], self.R_i[None, :])
            covariance_matrix = wendland_c2_varR(distances, R_pair)

        # Create the full Kriging matrix
        # K size is (ndata + 1) x (ndata + 1)
        self.kriging_matrix = np.zeros((self.ndata + 1, self.ndata + 1), dtype=float)

        # Fill C block
        self.kriging_matrix[:self.ndata, :self.ndata] = covariance_matrix
        if self.nugget > 0.0:
            self.kriging_matrix[:self.ndata, :self.ndata][np.diag_indices(self.ndata)] += self.nugget

        # Fill the constraint row and column with 1s
        self.kriging_matrix[:self.ndata,  self.ndata] = 1.0 # Last column (top N rows)
        self.kriging_matrix[ self.ndata, :self.ndata] = 1.0 # Last row (left N cols)
        self.kriging_matrix[ self.ndata,  self.ndata] = 0.0 # Bottom-right element (already 0)
        # Factorize the matrix (solve handles internally, or use lu_factor)
        # try:
        #     self.lu_factors = lu_factor(self.kriging_matrix)
        # except np.linalg.LinAlgError:
        #     print("Error: Kriging matrix is singular or ill-conditioned.")
        #     self.lu_factors = None
        self.coefficients = None # Coefficients are updated per data_vals set


    def update_weights(self, data_vals):
        """
        Update phase: Solves for Kriging coefficients given new data values.
        Uses the system [C 1; 1^T 0] * [c; mu] = [z; 0].
        data_vals: (N,) numpy array of data values z.
        Returns: True on success, False on failure.
                 The coefficients c are stored in self.coefficients[:-1] and mu is self.coefficients[-1].
        """
        if self.kriging_matrix is None or self.ndata == 0:
            print("ERROR in InterpolatorKriging.update_coefficients(): Cannot update coefficients, setup failed or no data.")
            self.coefficients = None
            return False

        z = np.asarray(data_vals, dtype=float)
        if z.shape[0] != self.ndata:
             print(f"ERROR in InterpolatorKriging.update_coefficients(): data_vals size ({z.shape[0]}) does not match data_points size ({self.ndata}).")
             self.coefficients = None
             return False

        print(f"InterpolatorKriging.update_coefficients(): Solving for {self.ndata+1} coefficients (c and mu)...")

        # Build the RHS vector [z; 0]
        rhs = np.zeros(self.ndata + 1, dtype=float)
        rhs[:self.ndata] = z # The data values z go in the first N elements
        rhs[self.ndata] = 0.0 # The last element is 0 for Ordinary Kriging (zero constraint on the trend)

        # Solve K * [c; mu] = [z; 0]
        try:
            # Using solve directly
            self.coefficients = solve(self.kriging_matrix, rhs)

            # Using explicit factor+solve
            # if self.lu_factors is not None:
            #     self.coefficients = lu_solve(self.lu_factors, rhs)
            # else:
            #     print("Error: Matrix factorization failed previously.")
            #     self.coefficients = None
            #     return False # Indicate failure
            #print("Global Kriging Update: Coefficients solved.")
            return True

        except np.linalg.LinAlgError:
            print("ERROR in InterpolatorKriging.update_coefficients(): Kriging system is singular or ill-conditioned. Cannot solve for coefficients.")
            self.coefficients = None
            return False


    def evaluate(self, query_points):
        """
        Evaluation phase: Computes interpolated values at query points.
        Uses the form z*(p) = sum c_i * C(||p - p_i||) + mu.
        query_points: (M, 2) numpy array of query point locations.
        Returns: (M,) numpy array of interpolated values, or None if coefficients not set.
        """
        if self.coefficients is None:
            print("ERROR in InterpolatorKriging.evaluate(): Coefficients not computed. Call update_coefficients first.")
            return None
        if self.ndata == 0:
             # The coefficients would be [mu] = [0] in this case if ndata=0 solve was possible
             # But let's return 0 as sum c_i * C is empty
             return np.zeros(query_points.shape[0], dtype=float)

        query_points = np.asarray(query_points, dtype=float)
        nqps         = query_points.shape[0]
        if nqps == 0: return np.array([], dtype=float)

        print(f"InterpolatorKriging.evaluate(): Interpolating at {nqps} points...")

        # Coefficients c are the first N elements, mu is the last element
        c_coeffs = self.coefficients[:self.ndata]
        mu = self.coefficients[self.ndata]

        if self.global_eval:
            # Dense/global evaluation: use all points for every query (no locality cut-off)
            # Still uses the same compact covariance; for equivalence with fully global kriging you must
            # choose R_basis large enough such that C(d) is non-zero over the whole domain.
            neighbor_indices_list = [np.arange(self.ndata, dtype=int) for _ in range(nqps)]
        else:
            # Use KD-tree for efficient neighbor search (evaluation is local due to compact support)
            data_kdtree = KDTree(self.data_points)

            # Evaluate each query point
            # Use query_ball_point to get candidate neighbors; with per-point supports we query with R_max
            if self.R_i is None:
                neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=self.R_basis)
            else:
                neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=self.R_max)

        # Prepare output array
        interpolated_values = np.zeros(nqps, dtype=float)

        # Evaluate using the form z*(p) = sum c_i * C(||p - p_i||) + mu
        for i in range(nqps):
            q = query_points[i]
            neighbors_q_indices = neighbor_indices_list[i]

            # The 'mu' term is added regardless of neighbors (it's the global trend)
            val = mu

            if neighbors_q_indices is None or len(neighbors_q_indices) == 0:
                interpolated_values[i] = val
                continue

            # Get relevant data points and coefficients
            neighbor_pts = self.data_points[neighbors_q_indices, :]
            neighbor_c_coeffs = c_coeffs[neighbors_q_indices]

            # Compute distances from q to its neighbors
            dists = np.linalg.norm(neighbor_pts - q, axis=1)

            # Evaluate covariance; for per-point supports each basis function i has its own cutoff R_i
            if self.R_i is None:
                cov_vals = compact_c2_covariance(dists, self.R_basis)
            else:
                Ri = self.R_i[neighbors_q_indices]
                mask = dists < Ri
                if not np.any(mask):
                    interpolated_values[i] = val
                    continue
                cov_vals = wendland_c2_varR(dists[mask], Ri[mask])
                neighbor_c_coeffs = neighbor_c_coeffs[mask]

            # Sum weighted contributions (sum c_i * C(||p - p_i||))
            val += np.sum(neighbor_c_coeffs * cov_vals)

            interpolated_values[i] = val


        #print("Global Kriging Evaluate: Done.")
        return interpolated_values




