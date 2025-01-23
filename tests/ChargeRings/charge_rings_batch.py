#!/usr/bin/python

import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from charge_rings_core     import calculate_tip_potential, calculate_qdot_system
from charge_rings_plotting import plot_tip_potential, plot_qdot_system

# Default parameters (same as GUI)
default_params = {
    'VBias': 1.0,
    'Rtip': 1.0,
    'z_tip': 2.0,
    'cCouling': 0.02,
    'temperature': 10.0,
    'onSiteCoulomb': 3.0,
    'zV0': -2.5,
    'zQd': 0.0,
    'nsite': 3,
    'radius': 5.0,
    'phiRot': -1.0,
    'Esite': -0.1,
    'Q0': 1.0,
    'Qzz': 0.0,
    'L': 20.0,
    'npix': 100,
    'decay': 0.3,
    'dQ': 0.02
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Charge Rings Batch Processing')

# Add arguments for each parameter
for param, value in default_params.items():
    parser.add_argument(f'--{param}', type=type(value), default=value,  help=f'Value for {param} (default: {value})')

# Add output file argument
parser.add_argument('--output', type=str, default='charge_rings_output.png',  help='Output file name (default: charge_rings_output.png)')

args = parser.parse_args()

# Convert args to params dictionary
params = {k: v for k, v in vars(args).items() if k != 'output'}

# Create figure
#fig = Figure(figsize=(15, 10))
fig = plt.figure(figsize=(15, 10))

# Create subplots
ax1 = fig.add_subplot(231)  # 1D Potential
ax2 = fig.add_subplot(232)  # Tip Potential
ax3 = fig.add_subplot(233)  # Site Potential
ax4 = fig.add_subplot(234)  # Energies
ax5 = fig.add_subplot(235)  # Total Charge
ax6 = fig.add_subplot(236)  # STM

# Calculate and plot
tip_data = calculate_tip_potential(**params)
qdot_data = calculate_qdot_system(**params)

plot_tip_potential(ax1, ax2, ax3, **tip_data, **params)
plot_qdot_system(ax4, ax5, ax6, **qdot_data, **params)

# Save figure
fig.tight_layout()
fig.savefig(args.output, dpi=300, bbox_inches='tight')
plt.show()
print(f"Results saved to {args.output}")
