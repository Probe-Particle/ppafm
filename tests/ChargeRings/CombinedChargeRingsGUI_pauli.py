#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple, Union
from enum import Enum, auto
from scipy.interpolate import LinearNDInterpolator

# Add path to pyProbeParticle for Pauli solver
from sys import path
path.insert(0, '../../pyProbeParticle')
import pauli as psl
from pauli import PauliSolver

from GUITemplate import GUITemplate, PlotConfig, PlotManager
from charge_rings_core import calculate_tip_potential, calculate_qdot_system, makeCircle, compute_site_energies, compute_site_tunelling, occupancy_FermiDirac
#from charge_rings_plotting import plot_tip_potential, plot_qdot_system

from charge_rings_plotting import plot_ellipses

class ApplicationWindow(GUITemplate):
    def __init__(self):
        super().__init__("Combined Charge Rings GUI v6")
        
        # Parameter specifications
        self.param_specs = {
            # Tip Parameters
            'VBias':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.0, 2.0),   'value': 1.0, 'step': 0.1},
            'Rtip':          {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 5.0),   'value': 1.0, 'step': 0.5},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 20.0),  'value': 2.0, 'step': 0.5},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.01, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 10.0, 'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            
            # Mirror Parameters
            'zV0':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': -2.5, 'step': 0.1},
            'zQd':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': 0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            
            # Ellipse Parameters
            'R_major':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 3.0, 'step': 0.1},
            'R_minor':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (1.0, 10.0),   'value': 2.0, 'step': 0.1},
            'phi0_ax':       {'group': 'Ellipse Parameters','widget': 'double', 'range': (-3.14, 3.14), 'value': 0.0, 'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.1,'step': 0.01},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Pauli Solver Parameters
            'W':             {'group': 'Pauli Solver',      'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            'GammaS':        {'group': 'Pauli Solver',      'widget': 'double', 'range': (0.01, 1.0),  'value': 0.20, 'step': 0.01, 'decimals': 2},
            'GammaT':        {'group': 'Pauli Solver',      'widget': 'double', 'range': (0.01, 1.0),  'value': 0.05, 'step': 0.01, 'decimals': 2},
            'muS':           {'group': 'Pauli Solver',      'widget': 'double', 'range': (-1.0, 1.0),  'value': 0.0,  'step': 0.1},
            'coeffT':        {'group': 'Pauli Solver',      'widget': 'double', 'range': (0.0, 1.0),   'value': 0.3,  'step': 0.1,   'decimals': 1},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 100,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.3,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},
            
            # Experimental Data
            'exp_slice':     {'group': 'Experimental Data', 'widget': 'int',    'range': (0, 13),     'value': 0,    'step': 1},
        }
        
        self.create_gui()
        
        # Setup matplotlib figure with 3x3 layout
        self.fig = Figure(figsize=(15, 15))
        self.canvas = FigureCanvas(self.fig)
        self.main_widget.layout().insertWidget(0, self.canvas)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(331)  # 1D Potential
        self.ax2 = self.fig.add_subplot(332)  # Tip Potential
        self.ax3 = self.fig.add_subplot(333)  # Site Potential
        self.ax4 = self.fig.add_subplot(334)  # Energies
        self.ax5 = self.fig.add_subplot(335)  # Total Charge
        self.ax6 = self.fig.add_subplot(336)  # STM
        self.ax7 = self.fig.add_subplot(338)  # Experimental dI/dV
        self.ax8 = self.fig.add_subplot(339)  # Experimental Current
        self.ax9 = self.fig.add_subplot(337)  # Additional plot if needed
        
        # Initialize plot manager
        self.plot_manager = PlotManager(self.fig)
        
        # Configure plots with explicit extents
        L = self.param_specs['L']['value']
        extent = (-L, L, -L, L)
        self.plot_manager.add_plot('potential_1d',   PlotConfig(ax=self.ax1, title="1D Potential (z=0)",   xlabel="x [Å]", ylabel="V [V]", grid=True, styles=['-b', '--r', ':g']))
        self.plot_manager.add_plot('tip_potential',  PlotConfig(ax=self.ax2, title="Tip Potential",        xlabel="x [Å]", ylabel="z [Å]", cmap='bwr',     grid=True, clim=(-1.0, 1.0)))
        self.plot_manager.add_plot('site_potential', PlotConfig(ax=self.ax3, title="Site Potential",       xlabel="x [Å]", ylabel="z [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0)))
        self.plot_manager.add_plot('energies',       PlotConfig(ax=self.ax4, title="Site Energies",        xlabel="x [Å]", ylabel="y [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0)))
        self.plot_manager.add_plot('Qtot',           PlotConfig(ax=self.ax5, title="Total Charge",         xlabel="x [Å]", ylabel="y [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0)))
        self.plot_manager.add_plot('stm',            PlotConfig(ax=self.ax6, title="STM",                  xlabel="x [Å]", ylabel="y [Å]", cmap='inferno', grid=True, clim=(0.0, 1.0)))
        self.plot_manager.add_plot('exp_didv',       PlotConfig(ax=self.ax7, title="Experimental dI/dV",   xlabel="X [Å]", ylabel="Y [Å]", cmap='seismic', grid=True, clim=(-1.0, 1.0)))
        self.plot_manager.add_plot('exp_current',    PlotConfig(ax=self.ax8, title="Experimental Current", xlabel="X [Å]", ylabel="Y [Å]", cmap='inferno', grid=True, clim=(0.0, 600.0)))
        self.plot_manager.add_plot('overlay',        PlotConfig(ax=self.ax9, title="Overlay (Red: Exp, Green: Sim)", xlabel="X [Å]", ylabel="Y [Å]", cmap='viridis', grid=True, clim=(0.0, 1.0)))
        
        # Set initial axis limits for image plots
        for name, cfg in self.plot_manager.plots.items():
            if cfg.cmap:  # Check if it's an image plot by presence of cmap
                cfg.ax.set_xlim(extent[0], extent[1])
                cfg.ax.set_ylim(extent[2], extent[3])
        
        # Initialize all plots
        self.plot_manager.initialize_plots()
        
        # Load experimental data
        self.load_experimental_data()
        
        # Initialize click-and-drag variables
        self.clicking = False
        self.start_point = None
        self.line_artist = None
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Initialize Pauli solver
        self.NSingle = 3  # Number of single-particle states (same as nsite)
        self.NLeads = 2   # Number of leads (substrate and tip)
        self.verbosity = 0  # No verbose output
        self.state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)  # Standard state ordering
        
        # Initialize the Pauli solver
        self.pauli = PauliSolver(self.NSingle, self.NLeads, verbosity=self.verbosity)
        
        self.run()  # Call run() method
        self.plot_manager.bRestoreBackground = True
        QtCore.QTimer.singleShot(100, self.run)
        #self.plot_manager.bRestoreBackground = False
    
    def prepare_leads_cpp(self, params):
        """Prepare static inputs for Pauli solver that don't change with site energies"""
        # Leads chemical potentials (substrate and tip)
        lead_mu = np.array([params['muS'], params['muS'] + params['VBias']])
        # Lead temperatures
        lead_temp = np.array([params['temperature'], params['temperature']])
        # Lead coupling strengths
        lead_gamma = np.array([params['GammaS'], params['GammaT']])
        
        # Calculate tunneling coefficients from gamma values
        VS = np.sqrt(params['GammaS']/np.pi)  # substrate
        VT = np.sqrt(params['GammaT']/np.pi)  # tip
        
        # Lead Tunneling matrix - shape (NLeads, NSingle)
        # For tip (second lead), apply position-dependent coefficient to 2nd and 3rd sites
        TLeads = np.array([
            [VS, VS, VS],  # Substrate coupling (same for all sites)
            [VT, VT*params['coeffT'], VT*params['coeffT']]  # Tip coupling (varies by site)
        ])
        
        return TLeads, lead_mu, lead_temp, lead_gamma
    
    def prepare_hsingle_cpp(self, site_energies):
        """Prepare single-particle Hamiltonian with site energies and hopping"""
        # Get number of sites
        nsite = len(site_energies)
        
        # Create single-particle Hamiltonian matrix
        Hsingle = np.zeros((nsite, nsite))
        
        # Set diagonal elements (site energies)
        # Convert from eV to meV (multiply by 1000.0) - this matches pauli_1D_load.py implementation
        for i in range(nsite):
            Hsingle[i, i] = site_energies[i] * 1000.0
        
        # Set off-diagonal elements (hopping between adjacent sites only)
        # For a ring, we connect the last site to the first one too
        t = 0.0  # Direct hopping parameter (typically set to 0)
        for i in range(nsite):
            next_i = (i + 1) % nsite  # Next site (with wrap-around)
            Hsingle[i, next_i] = t
            Hsingle[next_i, i] = t  # Hermitian conjugate
        
        return Hsingle
    
    def calculate_pauli_current(self, site_energies, params):
        """Calculate current using Pauli solver for given site energies
        
        Parameters:
        -----------
        site_energies : ndarray
            Either 2D array of shape (npoints, nsite) for 1D scan
            or 3D array of shape (npix, npix, nsite) for 2D image
        params : dict
            Parameter dictionary with all necessary parameters
            
        Returns:
        --------
        ndarray
            Current values with same shape as input (minus the last dimension)
        """
        # Get original shape to determine dimensionality
        original_shape = site_energies.shape
        nsite = original_shape[-1]  # Last dimension is always nsite
        
        # Make sure the Pauli solver is updated with correct number of sites
        if nsite != self.NSingle:
            self.NSingle = nsite
            self.pauli = PauliSolver(self.NSingle, self.NLeads, verbosity=self.verbosity)
            # Update state order for new number of sites
            NStates = 2**self.NSingle
            if NStates <= 8:  # Standard 3 sites case with 8 states
                self.state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)
            else:  # For more than 3 sites, use default ordering
                self.state_order = np.arange(NStates, dtype=np.int32)
        
        # Prepare lead parameters that don't change with position
        TLeads, lead_mu, lead_temp, lead_gamma = self.prepare_leads_cpp(params)
        self.pauli.set_lead(0, lead_mu[0], lead_temp[0])
        self.pauli.set_lead(1, lead_mu[1], lead_temp[1])
        self.pauli.set_tunneling(TLeads)
        
        # Reshape to 2D array (npoints, nsite) regardless of original shape
        if len(original_shape) == 3:  # 3D array for 2D image
            npix = original_shape[0]
            npoints = npix * npix
            reshaped_energies = site_energies.reshape(npoints, nsite)
        else:  # Already 2D array for 1D scan
            npoints = original_shape[0]
            reshaped_energies = site_energies
        
        # Calculate current for each point
        current = np.zeros(npoints)
        for i in range(npoints):
            # Get site energies for this point
            epsi = reshaped_energies[i]
            
            # Create Hamiltonian for this point
            Hsingle = self.prepare_hsingle_cpp(epsi)
            
            # Solve current for this point (get current in the tip, which is lead 1)
            current[i] = self.pauli.solve_hsingle(Hsingle, params['W'], 1, self.state_order)
        
        # Reshape back to original shape (minus the last dimension)
        if len(original_shape) == 3:  # 3D array for 2D image
            current = current.reshape(original_shape[0], original_shape[1])
        
        return current

    def load_experimental_data(self):
        """Load experimental data from npz file"""
        data_path = os.path.join(os.path.dirname(__file__), 'exp_rings_data.npz')
        try:
            data = np.load(data_path)
            # Convert from nm to Å (1 nm = 10 Å)
            self.exp_X = data['X'] * 10
            self.exp_Y = data['Y'] * 10
            self.exp_dIdV = data['dIdV']
            self.exp_I = data['I']
            self.exp_biases = data['biases']
            center_x = data['center_x'] * 10  # Convert to Å
            center_y = data['center_y'] * 10  # Convert to Å
            
            # Center the coordinates
            self.exp_X -= center_x
            self.exp_Y -= center_y
            
            # Calculate experimental data extent
            self.exp_extent = [
                np.min(self.exp_X), np.max(self.exp_X),
                np.min(self.exp_Y), np.max(self.exp_Y)
            ]
            
            # Update exp_slice range based on actual data size
            self.param_specs['exp_slice']['range'] = (0, len(self.exp_biases) - 1)
            self.param_specs['exp_slice']['value'] = len(self.exp_biases) // 2
            
            # Set initial voltage index to middle
            self.exp_idx = len(self.exp_biases) // 2
            
        except FileNotFoundError:
            print(f"Warning: Experimental data file not found at {data_path}")
            self.exp_X = None
            self.exp_Y = None
            self.exp_dIdV = None
            self.exp_I = None
            self.exp_biases = None

    def resample_to_simulation_grid(self, data, src_extent, target_size=100, target_extent=(-20, 20, -20, 20)):
        """Resample data to match simulation grid and extent"""
        # Create source coordinate grids
        x_src = np.linspace(src_extent[0], src_extent[1], data.shape[1])
        y_src = np.linspace(src_extent[2], src_extent[3], data.shape[0])
        
        # Create target coordinate grid
        x_target = np.linspace(target_extent[0], target_extent[1], target_size)
        y_target = np.linspace(target_extent[2], target_extent[3], target_size)
        
        # Create interpolator
        from scipy.interpolate import interp2d
        interpolator = interp2d(x_src, y_src, data)
        
        # Resample data to target grid
        resampled = interpolator(x_target, y_target)
        
        # Create mask for points outside source extent
        xx, yy = np.meshgrid(x_target, y_target)
        mask = ((xx < src_extent[0]) | (xx > src_extent[1]) | 
                (yy < src_extent[2]) | (yy > src_extent[3]))
        
        # Set points outside source extent to zero
        resampled[mask] = 0
        
        return resampled

    def create_overlay_image(self, exp_data, sim_data, exp_extent, sim_extent):
        """Create RGB overlay of experimental and simulation data"""
        # Resample experimental data to simulation grid
        exp_resampled = self.resample_to_simulation_grid(
            exp_data, 
            exp_extent,
            target_size=sim_data.shape[0],
            target_extent=sim_extent
        )
        
        # Normalize experimental data to [0,1] range for red channel
        exp_norm = (exp_resampled - np.min(exp_resampled)) / (np.max(exp_resampled) - np.min(exp_resampled))
        
        # Normalize simulation data to [0,1] range for green channel
        sim_norm = (sim_data - np.min(sim_data)) / (np.max(sim_data) - np.min(sim_data))
        
        # Create RGB image (Red: experimental, Green: simulation, Blue: zeros)
        rgb_image = np.zeros((*sim_data.shape, 3))
        rgb_image[..., 0] = exp_norm  # Red channel: experimental
        rgb_image[..., 1] = sim_norm  # Green channel: simulation
        
        return rgb_image, sim_extent

    def run(self):
        """Main calculation and plot update routine"""
        params = self.get_param_values()
        
        # Calculate tip potential
        tip_data = calculate_tip_potential(**params)
        Vtip    = tip_data['Vtip']
        Esites  = tip_data['Esites']
        V1d     = tip_data['V1d']
        
        # Calculate quantum dot system
        qdot_data = calculate_qdot_system(**params)
        Es        = np.max(qdot_data['Es'], axis=2)  # Take maximum across all layers for 2D plot
        Qtot      = qdot_data['Qtot']
        
        # Calculate current using the Pauli solver
        Current = self.calculate_pauli_current(qdot_data['Es'], params)
        
        # Scale current for visualization
        max_current = np.max(np.abs(Current))
        if max_current > 0:
            STM = Current / max_current  # Normalize for plotting
        else:
            STM = Current  # If current is zero everywhere
        
        # Update plots with new data
        self.plot_manager.restore_backgrounds()

        L = params['L']
        extent = (-L, L, -L, L)
        
        # Update 1D potential plot with all three lines
        x_coords = np.linspace(-L, L, len(V1d))
        self.plot_manager.update_plot('potential_1d', [
            (x_coords, V1d),
            (x_coords, (V1d + params['Esite']) ),
            (x_coords, np.full_like(x_coords, params['VBias']))
        ])
        
        # Update image plots
        self.plot_manager.update_plot('tip_potential',  Vtip,         extent=extent)
        self.plot_manager.update_plot('site_potential', Esites,       extent=extent)
        
        # Add horizontal lines to tip and site potential plots
        # Clear previous overlays
        self.plot_manager.clear_overlays('tip_potential')
        self.plot_manager.clear_overlays('site_potential')
        
        # Add horizontal lines to tip potential plot
        ax2 = self.plot_manager.plots['tip_potential'].ax
        mirror_line = ax2.axhline(y=params['zV0'], color='k', linestyle='--', linewidth=1, label='mirror surface')
        qdot_line = ax2.axhline(y=params['zQd'], color='g', linestyle='--', linewidth=1, label='Qdot height')
        tip_line = ax2.axhline(y=params['z_tip'], color='r', linestyle='--', linewidth=1, label='Tip Height')
        
        # Add a legend
        if not hasattr(ax2, 'legend_added'):
            ax2.legend(loc='upper right', fontsize='small')
            ax2.legend_added = True
        
        # Register lines as overlays for the plot manager
        self.plot_manager.add_overlay('tip_potential', mirror_line)
        self.plot_manager.add_overlay('tip_potential', qdot_line)
        self.plot_manager.add_overlay('tip_potential', tip_line)
        
        # Add horizontal lines to site potential plot
        ax3 = self.plot_manager.plots['site_potential'].ax
        mirror_line2 = ax3.axhline(y=params['zV0'], color='k', linestyle='--', linewidth=1, label='mirror surface')
        qdot_line2 = ax3.axhline(y=params['zQd'], color='g', linestyle='--', linewidth=1, label='Qdot height')
        
        # Register lines as overlays for the plot manager
        self.plot_manager.add_overlay('site_potential', mirror_line2)
        self.plot_manager.add_overlay('site_potential', qdot_line2)
        self.plot_manager.update_plot('energies',       Es,           extent=extent)
        # Calculate and use appropriate color limits for Qtot
        charge_max = np.abs(Qtot).max()
        self.plot_manager.update_plot('Qtot',   Qtot, extent=extent, clim=(-charge_max, charge_max))
        self.plot_manager.update_plot('stm',            STM,          extent=extent)
        
        # Update experimental plots if data is available
        if hasattr(self, 'exp_dIdV') and self.exp_dIdV is not None:
            self.exp_idx = params['exp_slice']
            
            # Update experimental plots
            I    = self.exp_I[self.exp_idx]
            dIdV = self.exp_dIdV[self.exp_idx]
            vmax = np.abs( dIdV ).max()
            self.plot_manager.update_plot('exp_didv',    self.exp_dIdV[self.exp_idx], extent=self.exp_extent, clim=(-vmax,vmax))
            self.plot_manager.update_plot('exp_current', self.exp_I[self.exp_idx],    extent=self.exp_extent, clim=(I.min(),I.max()))
            
            # Create and update overlay
            rgb_overlay, overlay_extent = self.create_overlay_image( self.exp_dIdV[self.exp_idx], Qtot, self.exp_extent, extent )
            self.plot_manager.update_plot('overlay', rgb_overlay, extent=overlay_extent)
        
        # Plot ellipses on relevant plots
        #for plot_name, ax in [('Qtot', self.ax5), ('exp_didv', self.ax7)]:
        for plot_name in ['Qtot', 'exp_didv','energies' ]:
            # Clear previous overlays
            self.plot_manager.clear_overlays(plot_name)
            ax = self.plot_manager.plots[plot_name].ax
            # Create new ellipse artists
            artists = plot_ellipses(ax, **params)
            # Add each artist as an overlay
            for artist in artists:
                self.plot_manager.add_overlay(plot_name, artist)
        
        # Perform final blitting update
        self.plot_manager.blit()

    def on_mouse_press(self, event):
        """Handle mouse button press event"""
        if event.inaxes in [self.ax4, self.ax7]:  # Energy plot or experimental dI/dV
            self.clicking = True
            self.start_point = (event.xdata, event.ydata)
            print(f"Mouse press in {'energy plot' if event.inaxes == self.ax4 else 'experimental plot'} at {self.start_point}")

    def on_mouse_motion(self, event):
        """Handle mouse motion event"""
        if self.clicking and event.inaxes in [self.ax4, self.ax7]:
            # Determine which plot we're working with
            # First remove old line
            if self.line_artist:
                try:
                    self.line_artist.remove()
                except:
                    pass
            
            # Fully restore the background first
            self.plot_manager.restore_backgrounds()
            
            # Now draw a new line - this needs to be visible
            self.line_artist, = event.inaxes.plot([self.start_point[0], event.xdata],
                                               [self.start_point[1], event.ydata], 'r-', linewidth=2, zorder=100)
            
            # Update the display to make the line visible
            self.canvas.draw_idle()  # Use draw_idle for smooth updates


    def on_mouse_release(self, event):
        """Handle mouse button release event"""
        if not self.clicking or event.inaxes is None:
            return
            
        end_point = (event.xdata, event.ydata)
        self.clicking = False
        
        print(f"Mouse release in subplot {event.inaxes}")
        print(f"Start point: {self.start_point}")
        print(f"End point: {end_point}")
        
        if event.inaxes == self.ax4:  # Energy plot
            print("Processing energy plot line scan")
            self.calculate_1d_scan(self.start_point, end_point)
        elif event.inaxes == self.ax7:  # Experimental dI/dV plot
            print("Processing experimental plot voltage scan")
            try:
                self.plot_voltage_line_scan(self.start_point, end_point)
            except Exception as e:
                print(f"Error in plot_voltage_line_scan: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Clean up by forcing a complete redraw
        if self.line_artist:
            try:
                self.line_artist.remove()
            except:
                pass
            self.line_artist = None
        
        # Force a clean redraw
        self.plot_manager.restore_backgrounds()
        self.plot_manager.blit()
        
        self.start_point = None

    def calculate_1d_scan(self, start_point, end_point, pointPerAngstrom=5):
        """Calculate and plot 1D scan between two points"""
        params = self.get_param_values()
        L = params['L']
        nsite = params['nsite']
        
        # Create line coordinates in real space (no rounding)
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate number of points based on distance
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        npoints = max(100, int(dist * pointPerAngstrom))
        
        # Create line coordinates
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        distance = np.sqrt((x - x[0])**2 + (y - y[0])**2)
        
        # Create positions array for calculations
        pTips = np.zeros((npoints, 3))
        pTips[:,0] = x
        pTips[:,1] = y
        pTips[:,2] = params['z_tip'] + params['Rtip']
        
        # Calculate site positions
        spos, phis = makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
        spos[:,2] = params['zQd']
        
        # Calculate energies and charges for each site
        Esite_arr = np.full(nsite, params['Esite'])
        Es = compute_site_energies(pTips, spos, VBias=params['VBias'], Rtip=params['Rtip'], zV0=params['zV0'], E0s=Esite_arr)
        
        # Calculate tunneling and charges for each site using FermiDirac (for comparison)
        Ts = compute_site_tunelling(pTips, spos, beta=params['decay'], Amp=1.0)
        Qs = np.zeros(Es.shape)
        Is = np.zeros(Es.shape)
        for i in range(nsite):
            Qs[:,i] = occupancy_FermiDirac(Es[:,i], params['temperature'])
            Is[:,i] = Ts[:,i] * (1-Qs[:,i])
        
        Qtot = np.sum(Qs, axis=1)
        STM_simple = np.sum(Is, axis=1)
        
        # Calculate current using the Pauli solver
        Current = self.calculate_pauli_current(Es, params)
        
        # Scale Pauli current for visualization if needed
        max_current = np.max(np.abs(Current))
        if max_current > 0:
            STM = Current / max_current  # Normalize for comparison purposes
        else:
            STM = Current
        
        # Create new figure for 1D scan
        scan_fig = plt.figure(figsize=(10, 12))
        scan_fig.suptitle(f'1D Scan from ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})')
        
        # Subplot for distance vs Energies
        ax1 = scan_fig.add_subplot(311)
        ax1.set_title('Site Energies')
        ax1.set_xlabel('Distance [Å]')
        ax1.set_ylabel('Energy [eV]')
        
        for i in range(nsite):
            phi = params['phiRot'] + i * 2 * np.pi / nsite
            label = f'Site {i+1}'
            ax1.plot(distance, Es[:,i], label=label)
        
        # Add horizontal line for bias voltage
        ax1.axhline(y=params['VBias'], color='k', linestyle='--', label='Bias')
        
        # Add legend
        ax1.legend()
        ax1.grid(True)
        
        # Subplot for distance vs. Q
        ax2 = scan_fig.add_subplot(312)
        ax2.set_title('Site Occupancy')
        ax2.set_xlabel('Distance [Å]')
        ax2.set_ylabel('Charge [e]')
        
        for i in range(nsite):
            phi = params['phiRot'] + i * 2 * np.pi / nsite
            label = f'Site {i+1}'
            ax2.plot(distance, Qs[:,i], label=label)
        
        # Plot total charge
        ax2.plot(distance, Qtot, 'k-', label='Total')
        
        # Add legend
        ax2.legend()
        ax2.grid(True)
        
        # Subplot for distance vs. STM
        ax3 = scan_fig.add_subplot(313)
        ax3.set_title('STM Signal')
        ax3.set_xlabel('Distance [Å]')
        ax3.set_ylabel('Current [a.u.]')
        
        for i in range(nsite):
            phi = params['phiRot'] + i * 2 * np.pi / nsite
            label = f'Site {i+1}'
            ax3.plot(distance, Is[:,i], label=label)
        
        # Plot total STM
        ax3.plot(distance, STM, 'k-', label='Total')
        
        # Add legend
        ax3.legend()
        ax3.grid(True)
        
        # Adjust layout and show
        scan_fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_voltage_line_scan(self, start_point, end_point, pointPerAngstrom=5):
        """Plot simulated charge and experimental dI/dV along a line scan for different voltages"""
        if not hasattr(self, 'exp_dIdV') or self.exp_dIdV is None:
            print("No experimental data found")
            return
        
        params = self.get_param_values()
        L = params['L']
        nsite = params['nsite']
        
        # Create line coordinates
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate number of points based on distance
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        npoints = max(50, int(dist * pointPerAngstrom))
        
        x = np.linspace(x1, x2, npoints)
        y = np.linspace(y1, y2, npoints)
        distance = np.sqrt((x - x[0])**2 + (y - y[0])**2)
        
        # Get dimensions and extents of experimental data
        nx, ny = self.exp_dIdV[0].shape
        xmin, xmax = np.min(self.exp_X), np.max(self.exp_X)
        ymin, ymax = np.min(self.exp_Y), np.max(self.exp_Y)
        
        # Sample experimental data along the line for all voltage slices
        exp_profiles = []
        n_voltages = len(self.exp_biases)
        
        for v_idx in range(n_voltages):
            # Prepare interpolator for this voltage slice
            points = np.column_stack((self.exp_X.flatten(), self.exp_Y.flatten()))
            values = self.exp_dIdV[v_idx].flatten()
            # Use LinearNDInterpolator for better accuracy
            interp = LinearNDInterpolator(points, values)
            
            # Sample along the line using interpolator
            points_to_sample = np.column_stack((x, y))
            sampled_values = interp(points_to_sample)
            
            # Store the profile
            exp_profiles.append(sampled_values)
        
        # Create voltage-distance map for experimental dI/dV
        v_dist_map = np.zeros((n_voltages, npoints))
        for i, profile in enumerate(exp_profiles):
            v_dist_map[i, :] = profile
        
        # Create new figure for voltage scan
        v_scan_fig = plt.figure(figsize=(12, 9))
        v_scan_fig.suptitle(f'Voltage Line Scan from ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})')
        
        # Plot dI/dV vs. distance for each voltage
        ax1 = v_scan_fig.add_subplot(211)
        ax1.set_title('Experimental dI/dV vs. Distance')
        ax1.set_xlabel('Distance [Å]')
        ax1.set_ylabel('dI/dV [a.u.]')
        
        for i, v in enumerate(self.exp_biases):
            # Only plot every few voltages to avoid crowding
            if i % max(1, n_voltages // 10) == 0:  # Plot ~10 curves
                ax1.plot(distance, exp_profiles[i], label=f'{v:.2f} V')
        
        ax1.legend()
        ax1.grid(True)
        
        # Plot experimental dI/dV vs. voltage and distance as a 2D colormap
        ax2 = v_scan_fig.add_subplot(212)
        extent = [0, dist, np.min(self.exp_biases), np.max(self.exp_biases)]
        
        # Set color limits for better contrast
        vmax = np.max(np.abs(v_dist_map))
        im = ax2.imshow(v_dist_map, aspect='auto', origin='lower', extent=extent, 
                    cmap='seismic', vmin=-vmax, vmax=vmax)
        
        ax2.set_xlabel('Distance [Å]')
        ax2.set_ylabel('Bias Voltage [V]')
        ax2.set_title('Experimental dI/dV Map')
        
        # Add color bar
        cbar = v_scan_fig.colorbar(im, ax=ax2)
        cbar.set_label('dI/dV [a.u.]')
        
        # Adjust layout and show
        v_scan_fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
