#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import argparse

def load_npz_data(filename):
    """Load data from .npz file and return as dictionary"""
    data = np.load(filename)
    # Convert to dictionary to make it easier to work with
    data_dict = {}
    for key in data.keys():
        data_dict[key] = data[key]
    
    # Also load parameters from the embedded JSON
    if 'params_json' in data_dict:
        params_json = data_dict['params_json']
        # Handle case where params_json is a numpy array instead of string
        if isinstance(params_json, np.ndarray):
            params_json = params_json.item() if params_json.size == 1 else str(params_json)
        params = json.loads(params_json)
        data_dict['params'] = params
    
    return data_dict


def print_min_max(data, name=""):
    """Print min and max of a numpy array"""
    print(f"{name}: shape {data.shape},  Min: {data.min()}, Max: {data.max()}")
    #print(f"{name}: Shape: {data.shape}")
    

def plot_data(data_dict):
    """Plot different channels from the data"""
    # Determine number of sites/channels
    n_Es = data_dict['Es'].shape[2] if 'Es' in data_dict and len(data_dict['Es'].shape) == 3 else 0
    n_Ts = data_dict['Ts'].shape[2] if 'Ts' in data_dict and len(data_dict['Ts'].shape) == 3 else 0
    
    # Calculate number of columns needed
    ncols = max(4, n_Es + 1, n_Ts + 1)  # +1 for max projection
    
    # Create figure with fixed 3 rows
    fig = plt.figure(figsize=(5 * ncols, 15))  # 5" per column, 15" total height
    
    # Row 1: Site Energies (Emax + individual sites)
    if 'Es' in data_dict:
        print_min_max(data_dict['Es'], "Es")
        Es = data_dict['Es']
        
        # Max projection
        ax = plt.subplot2grid((3, ncols), (0, 0))
        Es_max = np.max(Es, axis=2)
        im = ax.imshow(Es_max, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title('Es (Max)')
        
        # Individual sites
        for i in range(min(ncols-1, Es.shape[2])):
            ax = plt.subplot2grid((3, ncols), (0, i+1))
            im = ax.imshow(Es[:,:,i], origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Es Site {i}')
    
    # Row 2: Tunneling Rates (Tmax + individual sites)
    if 'Ts' in data_dict:
        print_min_max(data_dict['Ts'], "Ts")
        Ts = data_dict['Ts']
        
        # Max projection
        ax = plt.subplot2grid((3, ncols), (1, 0))
        Ts_max = np.max(Ts, axis=2)
        im = ax.imshow(Ts_max, origin='lower', cmap='plasma')
        plt.colorbar(im, ax=ax)
        ax.set_title('Ts (Max)')
        
        # Individual sites
        for i in range(min(ncols-1, Ts.shape[2])):
            ax = plt.subplot2grid((3, ncols), (1, i+1))
            im = ax.imshow(Ts[:,:,i], origin='lower', cmap='plasma')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Ts Site {i}')
    
    # Row 3: STM current + other data
    if 'STM' in data_dict:
        print_min_max(data_dict['STM'], "STM")
        ax = plt.subplot2grid((3, ncols), (2, 0))
        im = ax.imshow(data_dict['STM'], origin='lower', cmap='hot')
        plt.colorbar(im, ax=ax)
        ax.set_title('STM Current')
        
        # Add site positions
        if 'spos' in data_dict:
            spos = data_dict['spos']
            if spos.shape[1] >= 2:
                stm_shape = data_dict['STM'].shape
                x_scaled = (spos[:, 0] - spos[:, 0].min()) / (spos[:, 0].max() - spos[:, 0].min()) * (stm_shape[1] - 1)
                y_scaled = (spos[:, 1] - spos[:, 1].min()) / (spos[:, 1].max() - spos[:, 1].min()) * (stm_shape[0] - 1)
                ax.scatter(x_scaled, y_scaled, color='cyan', s=50, marker='x')
    
    # Add probability plots if space available
    if 'probs' in data_dict and data_dict['probs'] is not None and ncols > 1:
        print_min_max(data_dict['probs'], "probs")
        probs = data_dict['probs']
        
        if len(probs.shape) == 3:
            # Most probable state
            ax = plt.subplot2grid((3, ncols), (2, 1))
            prob_max = np.max(probs, axis=2)
            im = ax.imshow(prob_max, origin='lower', cmap='Blues')
            plt.colorbar(im, ax=ax)
            ax.set_title('Max Probability')
            
            if ncols > 2:  # State variance
                ax = plt.subplot2grid((3, ncols), (2, 2))
                prob_var = np.var(probs, axis=2)
                im = ax.imshow(prob_var, origin='lower', cmap='Reds')
                plt.colorbar(im, ax=ax)
                ax.set_title('Probability Variance')
    
    plt.tight_layout()
    return fig


def print_params(data_dict):
    """Print parameters from the data"""
    if 'params' in data_dict:
        print("\nParameters:")
        params = data_dict['params']
        for key, value in sorted(params.items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and plot data from .npz files saved by ChargeRingsGUI')
    parser.add_argument('filename', help='Path to .npz file')
    parser.add_argument('--save', help='Save plots to this filename (default: auto-generated from input filename)')
    parser.add_argument('--show', action='store_true', help='Show interactive plot window')
    parser.add_argument('--info', action='store_true', help='Print detailed information about the data structure')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.filename):
        print(f"Error: File {args.filename} not found")
        exit(1)

    
    # Load data
    data_dict = load_npz_data(args.filename)
    
    # Print info about data structure if requested
    if args.info:
        print(f"\nData structure in {args.filename}:")
        for key, value in data_dict.items():
            if key != 'params_json' and key != 'params':
                shape_info = f"shape={value.shape}, dtype={value.dtype}"
                print(f"  {key}: {shape_info}")
    
    # Always print basic shape info for main arrays
    print("\nArray Shapes:")
    for key in ['STM', 'Es', 'Ts', 'probs', 'spos']:
        if key in data_dict:
            print(f"  {key}: {data_dict[key].shape}")
    
    # Print parameters
    print_params(data_dict)
    
    # Plot data
    fig = plot_data(data_dict)
    
    # Determine output filename
    output_file = args.save
    if not output_file:
        # Auto-generate output filename based on input if not specified
        base = os.path.splitext(args.filename)[0]
        output_file = f"{base}_plot.png"
    
    # Save the figure
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    # Only show interactive plot if --show flag is set
    if args.show:
        plt.show()
