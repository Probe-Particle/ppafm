#!/usr/bin/python

import sys
import os
import json
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from TipMultipole import (
    makeCircle, makeRotMats, compute_site_energies,
    compute_site_tunelling, makePosXY, compute_V_mirror, occupancy_FermiDirac
)
from GUITemplate import GUITemplate

class ApplicationWindow(GUITemplate):
    def __init__(self):
        # First call parent constructor
        super().__init__("Combined Charge Rings GUI v2")
        
        # Then set parameter specifications
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
            'zV0':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 0.0),  'value': -2.5, 'step': 0.1},
            'zQd':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': 0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-1.0, 1.0), 'value': -0.1,'step': 0.01},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 100,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.3,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},
        }
        
        self.create_gui()
        # Add matplotlib canvas
        self.fig = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.fig)
        self.main_widget.layout().insertWidget(0, self.canvas)
        
        self.init_simulation()
        self.run()

    def init_simulation(self):
        """Initialize simulation with Python backend"""
        params = self.get_param_values()
        print( "params: ",  params)
        nsite = params['nsite']
        R = params['radius']
        
        # Setup sites on circle using Python implementation
        self.spos, phis = makeCircle(n=nsite, R=R)
        self.spos[:,2] = 0.0  # quantum dots are on the surface
        
        # Setup multipoles and site energies
        self.Esite = np.full(nsite, params['Esite'])
        self.rots  = makeRotMats(phis + params['phiRot'])
        
        # Initialize global parameters
        self.temperature = params['temperature']
        self.onSiteCoulomb = params['onSiteCoulomb']
    
    def calculateTipPotential(self, params):
        """Calculate tip potential data for X-Z projections"""
        # X-Z grid
        ps_xz, Xs, Zs = makePosXY(n=params['npix'], L=params['L'], axs=(0,2,1))
        
        # Tip position
        zT = params['z_tip'] + params['Rtip']
        tip_pos = np.array([0.0, 0.0, zT])
        
        # Calculate potentials
        self.tip_potential_data = {
            'Vtip':   compute_V_mirror(tip_pos, ps_xz,  VBias=params['VBias'],  Rtip=params['Rtip'], zV0=params['zV0']).reshape(params['npix'], params['npix']),
            'Esites': compute_site_energies( ps_xz, np.array([[0.0,0.0,params['zQd']]]), params['VBias'], params['Rtip'], zV0=params['zV0']).reshape(params['npix'], params['npix']),
            'ps_xz':  ps_xz,
            'extent': [-params['L'], params['L'], -params['L'], params['L']]
        }
        
        # Calculate 1D potential along x at z=0
        ps_1d = np.zeros((params['npix'], 3))
        ps_1d[:,0] = np.linspace(-params['L'], params['L'], params['npix'])
        ps_1d[:,2] = 0.0
        self.tip_potential_data['V1d'] = compute_V_mirror(tip_pos, ps_1d, VBias=params['VBias'], Rtip=params['Rtip'], zV0=params['zV0'])

    def calculateQdotSystem(self, params):
        zT = params['z_tip'] + params['Rtip']
        pTips, Xs, Ys = makePosXY(n=params['npix'], L=params['L'], p0=(0,0,zT))
        
        Es = compute_site_energies(
            pTips, 
            self.spos,
            VBias=params['VBias'],
            Rtip=params['Rtip'],
            zV0=params['zV0'],
            E0s=self.Esite
        )
        
        Ts = compute_site_tunelling(pTips, self.spos, beta=params['decay'], Amp=1.0)

        Qs  = np.zeros(Es.shape)
        Is  = np.zeros(Es.shape)
        for i in range(params['nsite']):
            Qs[:,i] = occupancy_FermiDirac( Es[:,i], self.temperature )
            Is[:,i] = Ts[:,i] * (1-Qs[:,i]) 
        
        self.qdot_system_data = { 
            'Es': Es.reshape(params['npix'], params['npix'], -1),
            'total_charge': np.sum(Qs, axis=1).reshape(params['npix'], params['npix'],-1),
            'STM': np.sum(Is, axis=1).reshape(params['npix'], params['npix'],-1),
            'pTips': pTips, 
            'extent': [-params['L'], params['L'], -params['L'], params['L']] 
        }

    def plotTipPotential(self):
        """Plot X-Z projections using precomputed data"""
        data = self.tip_potential_data
        params = self.get_param_values()
        
        # 1D Potential
        self.ax1.clear()
        x_coords = np.linspace(-data['extent'][1], data['extent'][1], params['npix'])
        self.ax1.plot(x_coords, data['V1d'], label='V_tip')
        self.ax1.plot(x_coords, data['V1d'] + params['Esite'], label='V_tip + E_site')
        self.ax1.plot(x_coords, x_coords*0.0 + params['VBias'], label='VBias')
        self.ax1.axhline(0.0, ls='--', c='k')
        self.ax1.set_title("1D Potential (z=0)")
        self.ax1.set_xlabel("x [Å]")
        self.ax1.set_ylabel("V [V]")
        self.ax1.grid()
        self.ax1.legend()
        
        # Tip Potential
        self.ax2.clear()
        zT = params['z_tip'] + params['Rtip']
        self.ax2.imshow(data['Vtip'], extent=data['extent'], cmap='bwr', origin='lower', vmin=-params['VBias'], vmax=params['VBias'])
        circ1, _ = makeCircle(16, R=params['Rtip'], axs=(0,2,1), p0=(0.0,0.0,zT))
        circ2, _ = makeCircle(16, R=params['Rtip'], axs=(0,2,1), p0=(0.0,0.0,2*params['zV0']-zT))
        self.ax2.plot(circ1[:,0], circ1[:,2], ':k')
        self.ax2.plot(circ2[:,0], circ2[:,2], ':k')
        self.ax2.axhline(params['zV0'], ls='--', c='k', label='mirror surface')
        self.ax2.axhline(params['zQd'], ls='--', c='g', label='Qdot height')
        self.ax2.axhline(params['z_tip'], ls='--', c='orange', label='Tip Height')
        self.ax2.set_title("Tip Potential")
        self.ax2.set_xlabel("x [Å]")
        self.ax2.set_ylabel("z [Å]")
        self.ax2.grid()
        self.ax2.legend()
        
        # Site Potential
        self.ax3.clear()
        self.ax3.imshow(data['Esites'], extent=data['extent'], cmap='bwr', origin='lower', vmin=-params['VBias'], vmax=params['VBias'])
        self.ax3.axhline(params['zV0'], ls='--', c='k', label='mirror surface')
        self.ax3.axhline(params['zQd'], ls='--', c='g', label='Qdot height')
        self.ax3.legend()
        self.ax3.set_title("Site Potential")
        self.ax3.set_xlabel("x [Å]")
        self.ax3.set_ylabel("z [Å]")
        self.ax3.grid()
    
    def plotQdotSystem(self):
        """Plot X-Y projections using precomputed data"""
        data = self.qdot_system_data
        params = self.get_param_values()
        
        # Energies
        self.ax4.clear()
        bUseMax = True
        if bUseMax:
            Eplot = np.max(data['Es'], axis=2)
            vmax = np.abs(Eplot).max()
            str_mode = '(max)'
        else:
            Eplot = np.sum(data['Es'], axis=2)
            vmax = np.abs(Eplot).max()
            str_mode = '(sum)'
        im = self.ax4.imshow(Eplot, origin="lower", extent=data['extent'], cmap='bwr', vmin=-vmax, vmax=vmax)
        self.ax4.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im, ax=self.ax4)
        self.ax4.set_title("Energies "+str_mode)
        self.ax4.set_xlabel("x [Å]")
        self.ax4.set_ylabel("y [Å]")
        
        # Site Charge
        self.ax5.clear()
        im = self.ax5.imshow(data['total_charge'], origin="lower", extent=data['extent'])
        self.ax5.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im, ax=self.ax5)
        self.ax5.set_title("Site Charge")
        self.ax5.set_xlabel("x [Å]")
        self.ax5.set_ylabel("y [Å]")
        
        # STM
        self.ax6.clear()
        im = self.ax6.imshow(data['STM'], origin="lower", extent=data['extent'])
        self.ax6.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im, ax=self.ax6)
        self.ax6.set_title("STM")
        self.ax6.set_xlabel("x [Å]")
        self.ax6.set_ylabel("y [Å]")
    
    def run(self):
        """Main execution method"""
        self.init_simulation()
        params = self.get_param_values()
        
        # Create 2x3 grid of plots
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3)
        
        # Top row: X-Z projections
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        
        # Bottom row: X-Y projections
        self.ax4 = self.fig.add_subplot(gs[1, 0])
        self.ax5 = self.fig.add_subplot(gs[1, 1])
        self.ax6 = self.fig.add_subplot(gs[1, 2])
        
        # Perform calculations
        self.calculateTipPotential(params)
        self.calculateQdotSystem(params)
        
        # Update all plots
        self.plotTipPotential()
        self.plotQdotSystem()
        
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())