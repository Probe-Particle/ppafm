#!/usr/bin/env python3
"""
Plotting utilities for optimization results.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

def plot_optimization_progress(optimizer, figsize=(12, 5)):
    """
    Plot optimization progress.
    
    Args:
        optimizer: Optimizer instance with history
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    history = optimizer.history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot distance vs iteration
    ax1.plot(history['iterations'], history['distances'], 'b-')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance (log scale)')
    ax1.set_title('Optimization Progress')
    ax1.grid(True)
    
    # Plot parameter evolution
    param_names = list(optimizer.param_ranges.keys())
    n_params = len(param_names)
    
    # Only plot up to 10 parameters to avoid clutter
    if n_params > 10:
        print(f"Warning: Only showing first 10 of {n_params} parameters")
        param_names = param_names[:10]
        n_params = 10
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_params))
    
    for i, (param, color) in enumerate(zip(param_names, colors)):
        param_values = [p[param] for p in history['parameters']]
        ax2.plot(history['iterations'], param_values, 
                color=color, label=param)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_comparison(exp_data, sim_data, exp_voltages, exp_x, sim_voltages, sim_x, figsize=(15, 10), titles=None):
    """
    Plot comparison between experimental and simulated data.
    
    Args:
        exp_data: 2D array of experimental data
        sim_data: 2D array of simulated data
        exp_voltages: 1D array of experimental voltage values
        exp_x: 1D array of experimental x positions
        sim_voltages: 1D array of simulated voltage values
        sim_x: 1D array of simulated x positions
        figsize: Figure size
        titles: Tuple of (exp_title, sim_title, diff_title)
        
    Returns:
        matplotlib Figure
    """
    if titles is None:
        titles = ('Experimental Data', 'Simulation', 'Difference (Sim - Exp)')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot experimental data
    im1 = axes[0, 0].imshow(exp_data, aspect='auto', origin='lower',
                          extent=[exp_x[0], exp_x[-1], exp_voltages[0], exp_voltages[-1]])
    axes[0, 0].set_title(titles[0])
    axes[0, 0].set_xlabel('Position (Å)')
    axes[0, 0].set_ylabel('Voltage (V)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot simulation
    im2 = axes[0, 1].imshow(sim_data, aspect='auto', origin='lower',
                          extent=[sim_x[0], sim_x[-1], sim_voltages[0], sim_voltages[-1]])
    axes[0, 1].set_title(titles[1])
    axes[0, 1].set_xlabel('Position (Å)')
    axes[0, 1].set_ylabel('Voltage (V)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Interpolate simulation to experimental grid for difference
    from scipy.interpolate import RectBivariateSpline
    interp = RectBivariateSpline(sim_voltages, sim_x, sim_data)
    sim_interp = interp(exp_voltages, exp_x, grid=True)
    
    # Plot difference
    diff = sim_interp - exp_data
    vmax = max(np.abs(diff.min()), diff.max())
    im3 = axes[1, 0].imshow(diff, aspect='auto', origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax, extent=[exp_x[0], exp_x[-1], exp_voltages[0], exp_voltages[-1]])
    axes[1, 0].set_title(titles[2])
    axes[1, 0].set_xlabel('Position (Å)')
    axes[1, 0].set_ylabel('Voltage (V)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot linecuts at different voltages
    voltage_indices = [int(len(exp_voltages) * p) for p in [0.25, 0.5, 0.75]]
    for i, v_idx in enumerate(voltage_indices):
        v_val = exp_voltages[v_idx]
        axes[1, 1].plot(exp_x, exp_data[v_idx, :], '--',  label=f'Exp V={v_val:.2f}V')
        axes[1, 1].plot(exp_x, sim_interp[v_idx, :], '-',  label=f'Sim V={v_val:.2f}V')
    
    axes[1, 1].set_title('Linecuts at Different Voltages')
    axes[1, 1].set_xlabel('Position (Å)')
    axes[1, 1].set_ylabel('Signal (a.u.)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_parameter_correlations(optimizer, figsize=(10, 8)):
    """
    Plot parameter correlations from optimization history using pure numpy and matplotlib.
    Args:
        optimizer: Optimizer instance with history property
        figsize: Figure size
    Returns:
        matplotlib Figure
    """
    hist = optimizer.history
    params_list = hist.get('parameters', [])
    if not params_list:
        raise ValueError("No parameters history to plot.")
    keys = list(params_list[0].keys())
    data = np.array([[p[k] for k in keys] for p in params_list])
    # Compute correlation matrix (shape: n_params x n_params)
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)
    ax.set_yticklabels(keys)
    ax.set_title('Parameter Correlations')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
