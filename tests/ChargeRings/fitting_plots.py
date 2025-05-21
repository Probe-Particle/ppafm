#!/usr/bin/env python3
"""
Plotting utilities for optimization results.
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from scipy.interpolate import RectBivariateSpline

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

def plot_comparison(exp_STM, exp_dIdV,  sim_STM, sim_dIdV, exp_extent, sim_extent, ylim, figsize=(15, 12), scale_dIdV=1.0):
    """
    Generates a 2x2 comparison plot of experimental vs simulated STM/dIdV data
    with independent x-extents and shared voltage range.
    
    Parameters:
        exp_STM, exp_dIdV: Experimental 2D arrays
        exp_voltages: 1D array of experimental bias voltages
        exp_x: 1D array of experimental x positions
        sim_STM, sim_dIdV: Simulated 2D arrays
        sim_voltages: 1D array of simulated voltages
        sim_x: 1D array of simulated x positions
        exp_extent: [xmin, xmax, vmin, vmax] for experimental data
        sim_extent: [xmin, xmax, vmin, vmax] for simulated data
        vmin, vmax: Global voltage range for y-axis
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Determine symmetric color limits for dIdV
    # dv_max = max(np.abs(exp_dIdV).max(), np.abs(sim_dIdV).max())
    # dv_min = -dv_max
    
    # STM Plots
    im0 = axs[0,0].imshow(exp_STM, extent=exp_extent, aspect='auto', cmap='hot', origin='lower')
    axs[0,0].set(title='Experimental STM', xlabel='Distance (Å)', ylabel='Voltage (V)')
    fig.colorbar(im0, ax=axs[0,0], label='Current (nA)')
    
    im1 = axs[0,1].imshow(sim_STM, extent=sim_extent, aspect='auto', cmap='hot', origin='lower')
    axs[0,1].set(title='Simulated STM', xlabel='Distance (Å)')
    fig.colorbar(im1, ax=axs[0,1], label='Current (a.u.)')
    
    # dIdV Plots
    vmax = np.max(np.abs(exp_dIdV))
    im2 = axs[1,0].imshow(exp_dIdV, extent=exp_extent, aspect='auto', cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
    axs[1,0].set(title='Experimental dIdV', xlabel='Distance (Å)', ylabel='Voltage (V)')
    fig.colorbar(im2, ax=axs[1,0], label='dI/dV (nS)')
    
    vmax = np.max(np.abs(sim_dIdV))
    im3 = axs[1,1].imshow(sim_dIdV*scale_dIdV, extent=sim_extent, aspect='auto', cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
    axs[1,1].set(title='Simulated dIdV', xlabel='Distance (Å)')
    fig.colorbar(im3, ax=axs[1,1], label='dI/dV (a.u.)')

    # Enforce consistent voltage ranges across all plots
    for ax in axs.flatten():
        ax.set_ylim(ylim[0], ylim[1])
    
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

def plot_comparison_with_optimizer(optimizer, exp_data, exp_voltages, exp_x, sim_distance, vmin, vmax, figsize=(15, 10)):
    """
    Plot comparison between experimental data and optimized simulation.
    Args:
        optimizer: MonteCarloOptimizer instance with best_sim_results
        exp_data: 2D array of experimental data
        exp_voltages: 1D array of experimental voltage values
        exp_x: 1D array of experimental x positions
        sim_distance: Total distance of simulation line scan
        vmin: Minimum voltage for plot limits
        vmax: Maximum voltage for plot limits
        figsize: Figure size
    Returns:
        matplotlib Figure
    """
    STM, dIdV, voltages, x = optimizer.best_sim_results
    
    sim_extent = [0, sim_distance, vmin, vmax]
    exp_extent = [0, max(exp_x), vmin, vmax]

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Experimental data
    im0 = axs[0,0].imshow(exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0,0].set(title='Experimental STM', xlabel='Distance (Å)', ylabel='Voltage (V)')
    fig.colorbar(im0, ax=axs[0,0])
    
    # Simulation data
    im1 = axs[0,1].imshow(STM, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0,1].set(title='Best Simulation STM', xlabel='Distance (Å)', ylabel='Voltage (V)')
    fig.colorbar(im1, ax=axs[0,1])
    
    # # Difference plot
    # interp = optimizer.interpolate_func(STM, voltages, x)
    # diff = interp - exp_data
    # im2 = axs[1,0].imshow(diff, extent=exp_extent, aspect='auto', origin='lower', cmap='bwr')
    # axs[1,0].set(title='Difference (Sim - Exp)', xlabel='Distance (Å)', ylabel='Voltage (V)')
    # fig.colorbar(im2, ax=axs[1,0])
    
    # Progress plot
    axs[1,1].plot(optimizer.distance_history, 'b-')
    axs[1,1].set(title='Optimization Progress', xlabel='Iteration', ylabel='Distance')
    axs[1,1].grid(True)
    
    # Set consistent voltage limits
    for ax in axs[:, :-1].flatten():
        ax.set_ylim(vmin, vmax)
    
    plt.tight_layout()
    return fig


def plot_highres_comparison(exp_data, highres_results, exp_voltages, exp_x, sim_distance, vmin, vmax, figsize=(15, 12)):
    """
    Plot high-resolution comparison between experimental and simulated data.
    Args:
        exp_data: 2D experimental data
        highres_results: tuple (STM_highres, dIdV_highres, voltages_highres, x_highres)
        exp_voltages: 1D experimental voltages
        exp_x: 1D experimental x positions
        sim_distance: Total distance of simulation line scan
        vmin: Minimum voltage for plot limits
        vmax: Maximum voltage for plot limits
        figsize: Figure size
    Returns:
        matplotlib Figure
    """
    STM_h, dIdV_h, volt_h, x_h = highres_results
    
    sim_extent = [0, sim_distance, vmin, vmax]
    exp_extent = [0, max(exp_x), vmin, vmax]

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Experimental data
    im0 = axs[0,0].imshow(exp_data, extent=exp_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0,0].set(title='Experimental STM', xlabel='Distance (Å)', ylabel='Voltage (V)')
    fig.colorbar(im0, ax=axs[0,0])
    
    # High-res simulation
    im1 = axs[0,1].imshow(STM_h, extent=sim_extent, aspect='auto', origin='lower', cmap='hot')
    axs[0,1].set(title='High-Resolution Simulation', xlabel='Distance (Å)', ylabel='Voltage (V)')
    fig.colorbar(im1, ax=axs[0,1])
    
    # Linecut plots
    idxs = [int(len(volt_h)*p) for p in [0.25, 0.5, 0.75]]
    for v_idx in idxs:
        v = volt_h[v_idx]
        axs[1,1].plot(x_h, STM_h[v_idx], label=f'V={v:.2f}V')
    axs[1,1].set(title='Voltage Linecuts', xlabel='Distance (Å)', ylabel='STM Signal')
    axs[1,1].legend()
    axs[1,1].grid(True)
    
    # Set consistent voltage limits
    for ax in axs[:, :-1].flatten():
        ax.set_ylim(vmin, vmax)
    
    plt.tight_layout()
    return fig
