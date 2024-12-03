import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ScanWindow1D(QtWidgets.QDialog):
    def __init__(self, parent=None, start_point=None, end_point=None):
        super().__init__(parent)
        self.parent = parent
        self.start_point = start_point
        self.end_point = end_point
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('1D Scan Results')
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create figure
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0])  # For charge/current
        self.ax2 = self.fig.add_subplot(gs[1])  # For energy levels
        
        # Add a close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        self.setMinimumSize(800, 600)
        
    def update_plot(self, scan_points, charges, currents, energies, occupations):
        """Update the plot with new scan data
        
        Args:
            scan_points: Array of distances along scan line
            charges: Array of charges along scan line
            currents: Array of currents along scan line
            energies: 2D array of energy levels along scan line
            occupations: 2D array of occupations along scan line
        """
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot charge and current
        self.ax1.plot(scan_points, charges, 'b-', label='Charge')
        self.ax1.plot(scan_points, currents, 'r-', label='Current')
        self.ax1.set_xlabel('Distance [Å]')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot energy levels with occupation as alpha
        for i in range(energies.shape[1]):
            energy = energies[:, i]
            occ = occupations[:, i]
            self.ax2.scatter(scan_points, energy, c='b', alpha=occ, s=2)
            self.ax2.plot(scan_points, energy, 'b-', alpha=0.3)
        
        self.ax2.set_xlabel('Distance [Å]')
        self.ax2.set_ylabel('Energy [eV]')
        self.ax2.grid(True)
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
