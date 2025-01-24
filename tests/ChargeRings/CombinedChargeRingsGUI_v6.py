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

from GUITemplate import GUITemplate
from charge_rings_core import calculate_tip_potential, calculate_qdot_system
from charge_rings_plotting import plot_tip_potential, plot_qdot_system

class PlotType(Enum):
    """Enumeration of supported plot types"""
    IMAGE = auto()
    LINE = auto()

@dataclass
class PlotConfig:
    """Configuration for a plot, including its display properties and artists"""
    ax: plt.Axes
    title: str
    plot_type: PlotType
    xlabel: str = ''
    ylabel: str = ''
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    grid: bool = True
    animated: bool = True
    overlay_artists: List[Any] = field(default_factory=list)
    artist: Optional[Any] = None
    background: Optional[Any] = None

class PlotManager:
    """Manages plot configurations, initialization and updates with blitting support"""
    
    def __init__(self, fig: Figure):
        self.fig = fig
        self.plots: Dict[str, PlotConfig] = {}
        self.initialized = False
    
    def add_plot(self, name: str, config: PlotConfig) -> None:
        """Register a new plot configuration"""
        self.plots[name] = config
    
    def initialize_plots(self) -> None:
        """Initialize all registered plots with dummy data"""
        if self.initialized:
            return
            
        for name, cfg in self.plots.items():
            # Set common properties
            cfg.ax.set_title(cfg.title)
            cfg.ax.set_xlabel(cfg.xlabel)
            cfg.ax.set_ylabel(cfg.ylabel)
            if cfg.grid:
                cfg.ax.grid(True)
            
            # Create appropriate artist based on plot type
            if cfg.plot_type == PlotType.IMAGE:
                dummy_data = np.zeros((100, 100))
                cfg.artist = cfg.ax.imshow(
                    dummy_data,
                    animated=cfg.animated,
                    cmap=cfg.cmap
                )
                if cfg.clim:
                    cfg.artist.set_clim(*cfg.clim)
            elif cfg.plot_type == PlotType.LINE:
                cfg.artist, = cfg.ax.plot([], [], animated=cfg.animated)
        
        # Store backgrounds for blitting
        self.fig.canvas.draw()
        for cfg in self.plots.values():
            cfg.background = self.fig.canvas.copy_from_bbox(cfg.ax.bbox)
        
        self.initialized = True
    
    def restore_backgrounds(self) -> None:
        """Restore the background for all plots"""
        for cfg in self.plots.values():
            self.fig.canvas.restore_region(cfg.background)
    
    def update_plot(self, name: str, data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], extent: Optional[Tuple[float, float, float, float]] = None, clim: Optional[Tuple[float, float]] = None) -> None:
        """Update a plot with new data"""
        if not self.initialized:
            raise RuntimeError("Plots must be initialized before updating")
            
        cfg = self.plots[name]
        
        if cfg.plot_type == PlotType.IMAGE:
            cfg.artist.set_data(data)
            if extent is not None:
                cfg.artist.set_extent(extent)
            if clim is not None:
                cfg.artist.set_clim(*clim)
        elif cfg.plot_type == PlotType.LINE:
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValueError("Line plots require x and y data as tuple")
            x, y = data
            cfg.artist.set_data(x, y)
            cfg.ax.relim()
            cfg.ax.autoscale_view()
        
        # Draw main artist and any overlays
        cfg.ax.draw_artist(cfg.artist)
        for artist in cfg.overlay_artists:
            cfg.ax.draw_artist(artist)
    
    def blit(self) -> None:
        """Perform final blitting update"""
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

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
        
        # Configure plots
        self.plot_manager.add_plot('potential_1d',   PlotConfig( ax=self.ax1, title="1D Potential (z=0)",   plot_type=PlotType.LINE,  xlabel="x [Å]", ylabel="V [V]",                 grid=True ))
        self.plot_manager.add_plot('tip_potential',  PlotConfig( ax=self.ax2, title="Tip Potential",        plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="z [Å]", cmap='bwr',     grid=True ))
        self.plot_manager.add_plot('site_potential', PlotConfig( ax=self.ax3, title="Site Potential",       plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="z [Å]", cmap='seismic', grid=True ))
        self.plot_manager.add_plot('energies',       PlotConfig( ax=self.ax4, title="Site Energies",        plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="y [Å]", cmap='seismic', grid=True ))
        self.plot_manager.add_plot('total_charge',   PlotConfig( ax=self.ax5, title="Total Charge",         plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="y [Å]", cmap='seismic', grid=True ))
        self.plot_manager.add_plot('stm',            PlotConfig( ax=self.ax6, title="STM",                  plot_type=PlotType.IMAGE, xlabel="x [Å]", ylabel="y [Å]", cmap='inferno', grid=True ))
        self.plot_manager.add_plot('exp_didv',       PlotConfig( ax=self.ax7, title="Experimental dI/dV",   plot_type=PlotType.IMAGE, xlabel="X [Å]", ylabel="Y [Å]", cmap='seismic', grid=True ))
        self.plot_manager.add_plot('exp_current',    PlotConfig( ax=self.ax8, title="Experimental Current", plot_type=PlotType.IMAGE, xlabel="X [Å]", ylabel="Y [Å]", cmap='inferno', grid=True ))
        
        # Initialize all plots
        self.plot_manager.initialize_plots()
        
        # Load experimental data if available
        self.load_experimental_data()
        
        self.run()  # Call run() method
    
    def load_experimental_data(self):
        """Load experimental data from files"""
        try:
            self.exp_dIdV = np.load('data/dIdV.npy')
            self.exp_I = np.load('data/I.npy')
            self.exp_biases = np.load('data/biases.npy')
            self.exp_extent = [-20, 20, -20, 20]  # Hardcoded for now
            self.exp_idx = 0
        except FileNotFoundError:
            print("Warning: Experimental data files not found")
            self.exp_dIdV = None
            self.exp_I = None
            self.exp_biases = None
    
    def run(self):
        """Main calculation and plot update routine"""
        params = self.get_param_values()
        
        # Calculate tip potential
        tip_data = calculate_tip_potential(**params)
        Vtip = tip_data['Vtip']
        Esites = tip_data['Esites']
        V1d = tip_data['V1d']
        
        # Calculate quantum dot system
        qdot_data = calculate_qdot_system(**params)
        Es = qdot_data['Es'][:,:,0]  # Take first layer for 2D plot
        total_charge = qdot_data['total_charge'][:,:,0]  # Take first layer for 2D plot
        STM = qdot_data['STM'][:,:,0]  # Take first layer for 2D plot
        spos = qdot_data['spos']
        
        # Update plots with new data
        self.plot_manager.restore_backgrounds()
        
        # Update 1D potential plot
        x_coords = np.linspace(-params['L'], params['L'], len(V1d))
        self.plot_manager.update_plot('potential_1d', (x_coords, V1d))
        
        # Update image plots
        extent = [-params['L'], params['L'], -params['L'], params['L']]
        self.plot_manager.update_plot('tip_potential',  Vtip,         extent=extent, clim=(-params['VBias'], params['VBias']))
        self.plot_manager.update_plot('site_potential', Esites,       extent=extent)
        self.plot_manager.update_plot('energies',       Es,           extent=extent)
        self.plot_manager.update_plot('total_charge',   total_charge, extent=extent)
        self.plot_manager.update_plot('stm',            STM,          extent=extent)
        
        # Update experimental plots if data is available
        if hasattr(self, 'exp_dIdV') and self.exp_dIdV is not None:
            self.plot_manager.update_plot('exp_didv', self.exp_dIdV[self.exp_idx], extent=self.exp_extent)
            self.plot_manager.update_plot('exp_current', self.exp_I[self.exp_idx], extent=self.exp_extent)
        
        # Perform final blitting update
        self.plot_manager.blit()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
