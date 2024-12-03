import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class ScanWindow1D(QtWidgets.QWidget):
    def __init__(self, parent=None, start_point=None, end_point=None):
        super().__init__()
        self.parent = parent
        self.start_point = start_point
        self.end_point = end_point
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("1D Scan Results")
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0])  # Charge and current
        self.ax2 = self.fig.add_subplot(gs[1])  # Energy levels
        
        # Add labels
        self.ax1.set_xlabel('Distance (Å)')
        self.ax1.set_ylabel('Value')
        self.ax2.set_xlabel('Distance (Å)')
        self.ax2.set_ylabel('Energy (eV)')
        
        # Create twin axis for current
        self.ax1_twin = self.ax1.twinx()
        self.ax1_twin.set_ylabel('Current')
        
        self.canvas.draw()
        
        # Add a close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        self.setMinimumSize(800, 600)
    
    def update_plot(self, distances, charges, currents, energies, occupations, colors ):
        """Update plots with new data"""
        self.ax1.clear()
        self.ax1_twin.clear()
        self.ax2.clear()
        
        # Plot charge and current
        l1 = self.ax1.plot(distances, charges, 'b-', label='Charge')
        l2 = self.ax1_twin.plot(distances, currents, 'r-', label='Current')
        
        # Add legends
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        self.ax1.legend(lns, labs, loc='upper right')
        
        # Plot energy levels with occupation-based coloring
        for i in range(energies.shape[1]):
            #color = plt.cm.viridis(occupations[:,i])
            #self.ax2.scatter(distances, energies[:,i], c=color, s=10)
            self.ax2.plot(distances, energies[:,i], '-', c=colors[i] )
        
        # Add labels
        self.ax1.set_xlabel('Distance (Å)')
        self.ax1.set_ylabel('Charge')
        self.ax1_twin.set_ylabel('Current')
        self.ax2.set_xlabel('Distance (Å)')
        self.ax2.set_ylabel('Energy (eV)')
        
        # Update layout
        self.fig.tight_layout()
        self.canvas.draw()
