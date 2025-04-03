import numpy as np
import matplotlib.pyplot as plt

def plot_results(positions, bias_voltages, eps_max_grid, current_grid, didv_grid ):
    """
    Create a three-panel plot of simulation results
    """
    plt.figure(figsize=(15, 5))
    extent = [positions.min(), positions.max(), bias_voltages[0], bias_voltages[-1]]

    # Plot 1: Maximum epsilon
    plt.subplot(131)
    vmax_eps = np.max(np.abs(eps_max_grid))
    eps_plot = plt.imshow(eps_max_grid, aspect='auto', origin='lower', extent=extent, cmap='bwr', vmin=-vmax_eps, vmax=vmax_eps)
    plt.colorbar(eps_plot, label='Max(ε1, ε2, ε3) [meV]')
    plt.title('Maximum Site Energy')
    plt.xlabel('Position [Å]')
    plt.ylabel('Bias Voltage [V]')

    # Plot 2: Current
    plt.subplot(132)
    vmax_current = np.max(np.abs(current_grid))
    current_plot = plt.imshow(current_grid, aspect='auto', origin='lower', extent=extent, cmap='bwr', vmin=-vmax_current, vmax=vmax_current)
    plt.colorbar(current_plot, label='Current [A]')
    plt.title('Current')
    plt.xlabel('Position [Å]')
    plt.ylabel('Bias Voltage [V]')

    # Plot 3: dI/dV
    plt.subplot(133)
    vmax_didv = np.max(np.abs(didv_grid))
    didv_plot = plt.imshow(didv_grid, aspect='auto', origin='lower', extent=extent, cmap='bwr', vmin=-vmax_didv, vmax=vmax_didv)
    plt.colorbar(didv_plot, label='dI/dV [S]')
    plt.title('Differential Conductance')
    plt.xlabel('Position [Å]')
    plt.ylabel('Bias Voltage [V]')

    plt.tight_layout()



