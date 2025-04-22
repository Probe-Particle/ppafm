import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QFont
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple, Union
from enum import Enum, auto
import matplotlib.pyplot as plt

@dataclass
class PlotConfig:
    """Configuration for a plot, including its display properties and artists"""
    ax: plt.Axes
    title: str
    xlabel: str = ''
    ylabel: str = ''
    grid: bool = True
    animated: bool = True
    
    # Image plot properties
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    
    # Line plot properties
    styles: List[str] = field(default_factory=list)
    
    # Artists management
    image_artist: Optional[Any] = None      # Main imshow artist
    line_artists: List[Any] = field(default_factory=list)  # Line plot artists
    overlay_artists: List[Any] = field(default_factory=list)  # Additional overlay artists
    background: Optional[Any] = None
    
    # Additional plot-specific properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    update_layout: bool = False  # Whether to update layout for this plot
    force_redraw: bool = False   # Whether to force redraw for this plot

class PlotManager:
    """Manages plot configurations, initialization and updates with blitting support"""
    
    def __init__(self, fig: plt.Figure):
        self.fig = fig
        self.plots: Dict[str, PlotConfig] = {}
        self.initialized = False
        self.bRestoreBackground = False
        self.bUpdateLimits = True
        self.bBlitIndividual = False
    
    def add_plot(self, name: str, config: PlotConfig) -> None:
        """Register a new plot configuration"""
        self.plots[name] = config
    
    def initialize_plots(self) -> None:
        """Initialize all registered plots with dummy data"""
        if self.initialized:
            return
            
        for name, cfg in self.plots.items():
            # Set common properties with increased padding
            cfg.ax.set_title(cfg.title, pad=20)
            cfg.ax.set_xlabel(cfg.xlabel, labelpad=10)
            cfg.ax.set_ylabel(cfg.ylabel, labelpad=10)
            if cfg.grid:
                cfg.ax.grid(True)
            
            # Initialize image if cmap is specified
            if cfg.cmap:
                dummy_data = np.zeros((100, 100))
                cfg.image_artist = cfg.ax.imshow(dummy_data, animated=cfg.animated, cmap=cfg.cmap)
                if cfg.clim: 
                    cfg.image_artist.set_clim(*cfg.clim)
            
            # Initialize lines if styles are specified
            for style in cfg.styles:
                artist, = cfg.ax.plot([], [], style, animated=cfg.animated)
                cfg.line_artists.append(artist)
        
        # Initial draw and layout
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
        # Store backgrounds after layout is set
        for cfg in self.plots.values():
            cfg.background = self.fig.canvas.copy_from_bbox(cfg.ax.bbox)
        
        self.initialized = True
    
    def clear_overlays(self, name: str) -> None:
        """Clear all overlay artists for a plot"""
        cfg = self.plots[name]
        while cfg.overlay_artists:
            artist = cfg.overlay_artists.pop()
            artist.remove()
    
    def add_overlay(self, name: str, artist: Any) -> None:
        """Add a transient artist to be managed with blitting"""
        cfg = self.plots[name]
        cfg.overlay_artists.append(artist)
    
    def restore_backgrounds(self) -> None:
        """Restore the background for all plots"""
        if self.bRestoreBackground:
            print("restore_backgrounds")
            self.fig.tight_layout()
            self.fig.canvas.draw()
            for cfg in self.plots.values():
                cfg.background = self.fig.canvas.copy_from_bbox(cfg.ax.bbox)
            self.bRestoreBackground = False
        
        # Restore backgrounds
        for cfg in self.plots.values():
            self.fig.canvas.restore_region(cfg.background)
    
    def update_plot(self, name: str, 
                   data: Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]], 
                   extent: Optional[Tuple[float, float, float, float]] = None,
                   clim: Optional[Tuple[float, float]] = None) -> None:
        """Update a plot with new data. For image plots, data should be a 2D array.
        For line plots, data should be a list of (x, y) tuples."""
        if not self.initialized:
            raise RuntimeError("Plots must be initialized before updating")
        
        cfg = self.plots[name]
        
        # Update image if present
        if cfg.image_artist is not None and isinstance(data, np.ndarray):
            cfg.image_artist.set_data(data)
            
            if extent is not None:
                cfg.image_artist.set_extent(extent)
                if self.bUpdateLimits:
                    cfg.ax.set_xlim(extent[0], extent[1])
                    cfg.ax.set_ylim(extent[2], extent[3])
            
            if clim is not None:
                cfg.image_artist.set_clim(*clim)
        
        # Update lines if present
        elif cfg.line_artists and isinstance(data, list):
            for (x, y), artist in zip(data, cfg.line_artists):
                artist.set_data(x, y)
            
            if self.bUpdateLimits:
                # Auto-scale the axes to fit the line data
                cfg.ax.relim()
                cfg.ax.autoscale_view()
    
    def blit(self) -> None:
        """Update the figure with blitting"""
        if not self.initialized:
            return
            
        # Draw all artists for each plot
        for cfg in self.plots.values():
            if cfg.image_artist:
                cfg.ax.draw_artist(cfg.image_artist)
            for artist in cfg.line_artists:
                cfg.ax.draw_artist(artist)
            for artist in cfg.overlay_artists:
                cfg.ax.draw_artist(artist)
        
        # Blit everything to the screen
        if self.bBlitIndividual:
            for cfg in self.plots.values():
                self.fig.canvas.blit(cfg.ax.bbox)
        else:
            self.fig.canvas.blit(self.fig.bbox)

class GUITemplate(QtWidgets.QMainWindow):
    def __init__(self, title="Application GUI"):
        super().__init__()
        # set smaller default font
        app = QtWidgets.QApplication.instance()
        if app:
            app.setFont(QFont("Sans", 8))
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.main_widget = QtWidgets.QWidget(self)
        
        # Initialize parameter specifications
        self.param_specs = {}
        self.param_widgets = {}
        
        # Create GUI
        #self.create_gui()
        
    def create_gui(self):
        """Create the main GUI layout and widgets"""
        # Main layout (compact margins)
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        l00.setContentsMargins(2,2,2,2)
        l00.setSpacing(4)
        
        # Control panel layout (compact margins)
        l0 = QtWidgets.QVBoxLayout()
        l0.setContentsMargins(2,2,2,2)
        l0.setSpacing(4)
        l00.addLayout(l0)

        self.layout0 = l0
        
        # Create widgets for each parameter group
        current_group = None
        current_layout = None
        
        for param_name, spec in self.param_specs.items():
            # skip fidget-disabled parameters
            if not spec.get('fidget', True):
                continue
            
            # Create new group if needed
            if spec['group'] != current_group:
                current_group = spec['group']
                gb = QtWidgets.QGroupBox(current_group)
                l0.addWidget(gb)
                current_layout = QtWidgets.QVBoxLayout(gb)
                # group layout compact
                current_layout.setContentsMargins(2,2,2,2)
                current_layout.setSpacing(4)
            
            # Create widget layout
            hb = QtWidgets.QHBoxLayout()
            # row layout compact
            hb.setContentsMargins(0,0,0,0)
            hb.setSpacing(2)
            current_layout.addLayout(hb)
            hb.addWidget(QtWidgets.QLabel(f"{param_name}:"))
            
            # Create widget if fidget enabled, else label
            if spec.get('fidget', True):
                if spec['widget'] == 'double':
                    widget = QtWidgets.QDoubleSpinBox()
                    widget.setRange(*spec['range'])
                    widget.setValue(spec['value'])
                    widget.setSingleStep(spec['step'])
                    if 'decimals' in spec:
                        widget.setDecimals(spec['decimals'])
                elif spec['widget'] == 'int':
                    widget = QtWidgets.QSpinBox()
                    widget.setRange(*spec['range'])
                    widget.setValue(spec['value'])
                    if 'step' in spec:
                        widget.setSingleStep(spec['step'])
                widget.valueChanged.connect(self.on_parameter_change)
            else:
                widget = QtWidgets.QLabel(str(spec['value']))
            hb.addWidget(widget)
            self.param_widgets[param_name] = widget
        
        # Add common controls
        self.add_common_controls(l0)
        
        # Set the central widget
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
    
    def add_common_controls(self, layout):
        """Add common control buttons"""
        hb = QtWidgets.QHBoxLayout()
        hb.setContentsMargins(2,2,2,2)
        hb.setSpacing(4)
        self.common_controls_layout = hb
        layout.addLayout(hb)

        # Run Button
        btn = QtWidgets.QPushButton("Run")
        btn.clicked.connect(self.run)
        hb.addWidget(btn)
        
        # Auto-update checkbox
        self.cbAutoUpdate = QtWidgets.QCheckBox("Auto-update")
        self.cbAutoUpdate.setChecked(True)
        hb.addWidget(self.cbAutoUpdate)
        
        self.hbCommonControls = hb
        
        # Save/Load buttons
        hb = QtWidgets.QHBoxLayout()
        hb.setContentsMargins(2,2,2,2)
        hb.setSpacing(4)
        layout.addLayout(hb)
        btnSave = QtWidgets.QPushButton("Save Parameters")
        btnSave.clicked.connect(self.save_parameters)
        hb.addWidget(btnSave)
        btnLoad = QtWidgets.QPushButton("Load Parameters")
        btnLoad.clicked.connect(self.load_parameters)
        hb.addWidget(btnLoad)
        self.hbSaveLoad = hb
    
    def get_param_values(self):
        """Get current values of all parameters"""
        values = {}
        for name, spec in self.param_specs.items():
            if name in self.param_widgets:
                widget = self.param_widgets[name]
                if hasattr(widget, 'value'):
                    values[name] = widget.value()
                else:
                    try:
                        values[name] = float(widget.text())
                    except:
                        values[name] = widget.text()
            else:
                # use original spec value for skipped fidgets
                values[name] = spec.get('value')
        return values
    
    def set_param_values(self, values):
        """Set values for all parameters"""
        for name, value in values.items():
            if name in self.param_widgets:
                self.param_widgets[name].setValue(value)
    
    def save_parameters(self):
        """Save parameters to JSON file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON files (*.json)")
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            with open(filename, 'w') as f:
                json.dump(self.get_param_values(), f, indent=4)
    
    def load_parameters(self):
        """Load parameters from JSON file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                values = json.load(f)
                self.set_param_values(values)
                self.run()
    
    def on_parameter_change(self):
        """Handle parameter changes"""
        if self.cbAutoUpdate.isChecked():
            self.run()
    
    def run(self):
        """Main execution method to be implemented by child classes"""
        raise NotImplementedError("Child classes must implement run() method")

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = GUITemplate()
    aw.show()
    sys.exit(qApp.exec_())
