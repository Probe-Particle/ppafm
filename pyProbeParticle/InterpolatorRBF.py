import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve # Using scipy's wrapper for better handling
# from scipy.linalg import lu_factor, lu_solve # Alternative for explicit factorization

from .interpy import wendland_c2, pairwise_distances

class InterpolatorRBF:
    def __init__(self, data_points, R_basis):
        """
        Setup phase: Builds and factorizes the RBF matrix.
        data_points: (N, 2) numpy array of data point locations.
        R_basis: Support radius for the RBF.
        """
        self.data_points = np.asarray(data_points, dtype=float)
        self.ndata       = self.data_points.shape[0]
        self.R_basis     = float(R_basis)

        if self.ndata == 0:
            print("WARRNING: GlobalRbfInterpolator initialized with no data points.")
            self.phi_matrix = None
            # No factorization needed
            return

        print(f"GlobalRbfInterpolator.init(): Building {self.ndata}x{self.ndata} Phi matrix (R_basis={R_basis})")

        # Build the Phi matrix: Phi_ij = phi(||p_i - p_j||)
        # Use pairwise_distances to get all distances efficiently
        distances = pairwise_distances(self.data_points, self.data_points)
        self.phi_matrix = wendland_c2(distances, self.R_basis)

        # Factorize the matrix (e.g., LU decomposition) for efficient solves later
        # solve() handles factorization internally for repeat calls if the matrix doesn't change.
        # For explicit factor+solve:
        # try:
        #     self.lu_factors = lu_factor(self.phi_matrix)
        # except np.linalg.LinAlgError:
        #     print("Error: RBF matrix is singular or ill-conditioned.")
        #     self.lu_factors = None

        # For using solve directly (simpler):
        # solve() is typically optimized to reuse computations if matrix object is the same
        # when called multiple times with different b vectors.
        #print("Global RBF Setup: Matrix built.")
        self.weights = None # Weights are updated per data_vals set


    def update_weights(self, data_vals):
        """
        Update phase: Solves for RBF weights given new data values.
        data_vals: (N,) numpy array of data values.
        Returns: True on success, False on failure.
        """
        if self.phi_matrix is None or self.ndata == 0:
            print("ERROR in InterpolatorRBF.update_weights(): Cannot update weights, setup failed or no data.")
            self.weights = None
            return False

        z = np.asarray(data_vals, dtype=float)
        if z.shape[0] != self.ndata:
             print(f"ERROR in InterpolatorRBF.update_weights(): data_vals size ({z.shape[0]}) does not match data_points size ({self.ndata}).")
             self.weights = None
             return False

        print(f"InterpolatorRBF.update_weights(): Solving for {self.ndata} weights...")

        # Solve Phi * w = z
        try:
            # Using solve directly
            self.weights = solve(self.phi_matrix, z)

            # Using explicit factor+solve (more explicit about factorization step)
            # if self.lu_factors is not None:
            #     self.weights = lu_solve(self.lu_factors, z)
            # else:
            #      print("Error: Matrix factorization failed previously.")
            #      self.weights = None
            #      return False # Indicate failure
            #print("Global RBF Update: Weights solved.")
            return True

        except np.linalg.LinAlgError:
            print("Error: RBF system is singular or ill-conditioned. Cannot solve for weights.")
            self.weights = None
            return False


    def evaluate(self, query_points):
        """
        Evaluation phase: Computes interpolated values at query points.
        query_points: (M, 2) numpy array of query point locations.
        Returns: (M,) numpy array of interpolated values, or None if weights not set.
        """
        if self.weights is None:
            print("ERROR in InterpolatorRBF.evaluate(): Weights not computed. Call update_weights first.")
            return None
        if self.ndata == 0:
            return np.zeros(query_points.shape[0], dtype=float) # Or handle as NaN/error


        query_points = np.asarray(query_points, dtype=float)
        nqps = query_points.shape[0]
        if nqps == 0: return np.array([], dtype=float)


        print(f"InterpolatorRBF.evaluate(): Interpolating at {nqps} points...")

        # Use KD-tree for efficient neighbor search during evaluation
        # Build the KD-tree once on data_points (could be done in __init__)
        # For simplicity, rebuild here, but in practice, pass/store the tree.
        data_kdtree = KDTree(self.data_points)

        # Prepare output array
        interpolated_values = np.zeros(nqps, dtype=float)

        # Evaluate each query point
        # For large M and N, vectorizing this loop is key.
        # Instead of loop + KDTree query per point, query neighbors for *all* query points
        # within R_basis using query_ball_tree
        # This gives a list of lists: indices[i] is list of neighbors of query_points[i]
        neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=self.R_basis)

        # Efficiently compute contributions using broadcasting and fancy indexing
        # This is more complex than simple loop but faster in numpy for large arrays
        # A simpler loop for clarity:
        # for i in range(nqps):
        #     q = query_points[i]
        #     neighbors_q_indices = neighbor_indices_list[i] # Indices of data_points near q
        #     if not neighbors_q_indices:
        #         # If no neighbors within R_basis, sum is 0
        #         interpolated_values[i] = 0.0
        #         continue
        #
        #     # Get relevant data points and weights
        #     neighbor_pts = self.data_points[neighbors_q_indices, :]
        #     neighbor_weights = self.weights[neighbors_q_indices]
        #
        #     # Compute distances from q to its neighbors
        #     dists = np.linalg.norm(neighbor_pts - q, axis=1)
        #
        #     # Evaluate RBF for these distances
        #     phi_vals = wendland_c2(dists, self.R_basis)
        #
        #     # Sum weighted contributions
        #     interpolated_values[i] = np.sum(neighbor_weights * phi_vals)

        # More vectorized approach using numpy.concatenate for indices and distances
        # This avoids the Python loop over query points but is slightly less direct
        # Let's stick to the loop for clarity first, as KDTree already provides neighbor lists efficiently.
        # For absolute maximum performance, look into numba or more advanced vectorization strategies.

        # Let's make the loop version efficient by using numpy within the loop
        for i in range(nqps):
            q = query_points[i]
            neighbors_q_indices = neighbor_indices_list[i]

            if not neighbors_q_indices:
                interpolated_values[i] = 0.0
                continue

            # Get distances from q to its neighbors using vectorized subtraction and norm
            # q is (1, 2), neighbor_pts is (k, 2). (k, 2) - (1, 2) broadcasts correctly.
            neighbor_pts = self.data_points[neighbors_q_indices, :]
            dists = np.linalg.norm(neighbor_pts - q, axis=1)

            phi_vals = wendland_c2(dists, self.R_basis)
            neighbor_weights = self.weights[neighbors_q_indices]

            interpolated_values[i] = np.sum(neighbor_weights * phi_vals)
        #print("Global RBF Evaluate: Done.")
        return interpolated_values

