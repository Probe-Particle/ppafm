import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../../')
import pauli_scan as ps
import orbital_utils as ou

# Define parameters for the scan
# params = {
#     'VBias': 0.6,
#     'Rtip': 3.0,
#     'z_tip': 5.0,
#     'W': 0.03,
#     'GammaS': 0.01,
#     'GammaT': 0.01,
#     'Temp': 0.224,
#     'onSiteCoulomb': 3.0,
#     'zV0': -3.3,
#     'zQd': 0.0,
#     'zVd': 2.0,
#     'nsite': 3,
#     'radius': 5.2,
#     'phiRot': np.pi * 0.5 + 0.2,
#     'Esite': -0.150,
#     'Q0': 1.0,
#     'Qzz': 0.0,
#     'L': 30.0,
#     'npix': 150,
#     'dQ': 0.001,
#     'decay': 0.3,
# }

params = {
    'nsite':   3,
    'radius':  5.2,
    'phiRot':  1.3,
    'phi0_ax': 0.2,
    'VBias':   0.70,
    'Rtip':    3.0,
    'z_tip':   5.0,
    'zV0':     -0.4,
    'zVd':     8.0,
    'zQd':     0.0,
    'Q0':      1.0,
    'Qzz':     0.0,
    'Esite':   -0.070,
    'W':       0.02,
    'decay':   0.3,
    'GammaS':  0.01,
    'GammaT':  0.01,
    'Temp':    0.224,
    'p1_x':    9.72,
    'p1_y':    -9.96,
    'p2_x':    -11.0,
    'p2_y':    12.0,
    'npix':    150,
    'L':       20.0,
    'dQ':      0.03,
}

# Define the scan parameters (e.g., sweeping 'z_tip')
# The values for p1_x, p1_y, p2_x, p2_y are now sweepable
# scan_params = [
#     ('p1_x', np.linspace(-5.0, 5.0, 5)),
#     ('p1_y', np.linspace(-5.0, 5.0, 5)),
#     ('p2_x', np.linspace(-5.0, 5.0, 5)),
#     ('p2_y', np.linspace(-5.0, 5.0, 5)),
# ]

scan_params = [
    #('Rtip',  [2.0,2.5, 3.0, 3.5,4.0]),
    #('p1_y',  [-11.0,-10.5,-10.0,-9.5, -9.0]),
    ('Qzz',  [-20.,-15., -10.0, -5.0, 0.0, 5.0, 10.0, 15.,20.]),
    #('z_tip', [4.0, 4.5, 5.0, 5.5, 6.0])
    #('Esite', [ -0.080,-0.100, -0.120, -0.140 ]),
    #('zVd',  [ 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0 ]),
    #('zV0', [ -0.5, -0.75, -1.0, -1.5, -2.0 ]),
    #('zV0', [ -0.5, -0.5, -0.5, -0.5, -0.5 ]),
    #('Esite', [ -0.060,-0.070, -0.080, -0.090, -0.100 ]),
    #('decay', [0.1, 0.2, 0.3, 0.4])
]

# Define parameters to display in the plot title
view_params = ['VBias', 'z_tip', 'W', 'Esite']

# Optional: Load experimental reference data
# ExpRef = {
#     'STM': np.load('exp_stm.npy'),
#     'dIdV': np.load('exp_didv.npy'),
#     'x': np.linspace(0, 10, 100),
#     'y': np.linspace(0, 10, 100),
# }
ExpRef = None

# Define orbital file to load
#orb_file = 'QD.cub'
orb_file = None

# Define result directory
result_dir = 'results_xy_scan'

# Run the sweep
fig, all_results = ps.sweep_scan_param_pauli_xy_orb(
    params,
    scan_params,
    view_params=view_params,
    orbital_file=orb_file,
    ExpRef=ExpRef,
    result_dir=result_dir
)

# Display the plot
plt.show()

print(f"Results saved to {result_dir}")
