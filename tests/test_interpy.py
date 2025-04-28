import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
#import pyProbeParticle.interpy as interpy
from pyProbeParticle.InterpolatorRBF     import InterpolatorRBF
from pyProbeParticle.InterpolatorKriging import InterpolatorKriging

def check_interp( interp, data_vals, points ):    
    if interp.update_weights(data_vals):
        y_samp = interp.evaluate(points)
        print("Original | Interpolated | Difference")
        for i in range(ndata):
            print(f"{data_vals[i]:<9.4f}| {y_samp[i]:<13.4f}| {np.abs(data_vals[i] - y_samp[i]):.4e}")
    else:
        print("Failed to update weights for RBF.")

if __name__ == "__main__":
    # Example Data Points (Geometry)
    data_points = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5],
        [2, 0], [2, 1], [0, 2], [1, 2], [2, 2],
        [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]
    ])
    ndata = data_points.shape[0]

    # Define RBF/Covariance support radius
    # Make it large enough for some points to be neighbors, but smaller than the domain extent
    R_basis = 1.1 # e.g., slightly larger than diagonal of unit square cells

    data_vals1 = np.sin(   data_points[:, 0] * np.pi) * np.cos(data_points[:, 1] * np.pi) # Example values 1
    data_vals2 = np.exp(-((data_points[:, 0]-1.0)**2 + (data_points[:, 1]-1.0)**2) / 0.5) # Example values 2 (Gaussian peak)

    rbf     = InterpolatorRBF(data_points, R_basis)
    kriging = InterpolatorKriging(data_points, R_basis)

    #nx,ny = 10,10
    #x_grid, y_grid = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 2, ny))
    #query_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    check_interp( rbf, data_vals1,     data_points )
    check_interp( rbf, data_vals2,     data_points )

    check_interp( kriging, data_vals1, data_points )
    check_interp( kriging, data_vals2, data_points )








   



