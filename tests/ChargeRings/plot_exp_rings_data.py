#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the data
data = np.load('exp_rings_data.npz')
X = data['X']
Y = data['Y']
dIdV = data['dIdV']
I = data['I']
biases = data['biases']
center_x = data['center_x']
center_y = data['center_y']

# lets make the images ploted with respect to the center
X -= center_x
Y -= center_y

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)  # Make room for slider

# Initial voltage index
init_idx = len(biases) // 2

# Plot initial images
xmin, xmax = np.min(X[0]), np.max(X[0])
ymin, ymax = np.min(Y[0]), np.max(Y[0])



# Get maxval for dI/dV scaling
maxval1 = np.max(np.abs(dIdV[init_idx]))

im1 = ax1.imshow(dIdV[init_idx], aspect='equal', cmap='seismic', 
                 vmin=-maxval1, vmax=maxval1, extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im1, ax=ax1)
ax1.set_title(f'dI/dV at {biases[init_idx]:.3f} V')
ax1.set_xlabel('X (nm)')
ax1.set_ylabel('Y (nm)')

im2 = ax2.imshow(I[init_idx], aspect='equal', cmap='inferno',  vmin=0.0, vmax=600.0, extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im2, ax=ax2)
ax2.set_title(f'Current at {biases[init_idx]:.3f} V')
ax2.set_xlabel('X (nm)')
ax2.set_ylabel('Y (nm)')

# Mark the center position
#ax1.scatter(center_x, center_y, color='white', marker='x', s=100, label='Center')
#ax2.scatter(center_x, center_y, color='white', marker='x', s=100, label='Center')
ax1.legend()
ax2.legend()

# Add slider for voltage
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
voltage_slider = Slider(
    ax=ax_slider,
    label='Bias Voltage (V)',
    valmin=np.min(biases),
    valmax=np.max(biases),
    valinit=biases[init_idx],
)

# Update function for slider
def update(val):
    # Find closest bias value
    idx = np.abs(biases - val).argmin()
    
    # Update images with dynamic scaling for dI/dV
    maxval1 = np.max(np.abs(dIdV[idx]))
    im1.set_clim(vmin=-maxval1, vmax=maxval1)
    im1.set_array(dIdV[idx])
    im2.set_array(I[idx])
    
    # Update titles
    ax1.set_title(f'dI/dV at {biases[idx]:.3f} V')
    ax2.set_title(f'Current at {biases[idx]:.3f} V')
    
    fig.canvas.draw_idle()

voltage_slider.on_changed(update)

plt.show()
