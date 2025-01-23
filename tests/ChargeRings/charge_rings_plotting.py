#!/usr/bin/python

import numpy as np
from TipMultipole import makeCircle

def plot_tip_potential(ax1, ax2, ax3, data, params):
    """
    Plot X-Z projections of tip potential
    
    Args:
        ax1: Matplotlib axis for 1D potential plot
        ax2: Matplotlib axis for tip potential plot
        ax3: Matplotlib axis for site potential plot
        data: Dictionary containing calculated data from calculate_tip_potential
        params: Dictionary containing simulation parameters
    """
    # 1D Potential
    ax1.clear()
    x_coords = np.linspace(-data['extent'][1], data['extent'][1], params['npix'])
    ax1.plot(x_coords, data['V1d'], label='V_tip')
    ax1.plot(x_coords, data['V1d'] + params['Esite'], label='V_tip + E_site')
    ax1.plot(x_coords, x_coords*0.0 + params['VBias'], label='VBias')
    ax1.axhline(0.0, ls='--', c='k')
    ax1.set_title("1D Potential (z=0)")
    ax1.set_xlabel("x [Å]")
    ax1.set_ylabel("V [V]")
    ax1.grid()
    ax1.legend()
    
    # Tip Potential
    ax2.clear()
    zT = params['z_tip'] + params['Rtip']
    ax2.imshow(data['Vtip'], extent=data['extent'], cmap='bwr', 
               origin='lower', vmin=-params['VBias'], vmax=params['VBias'])
    circ1, _ = makeCircle(16, R=params['Rtip'], axs=(0,2,1), p0=(0.0,0.0,zT))
    circ2, _ = makeCircle(16, R=params['Rtip'], axs=(0,2,1), 
                         p0=(0.0,0.0,2*params['zV0']-zT))
    ax2.plot(circ1[:,0], circ1[:,2], ':k')
    ax2.plot(circ2[:,0], circ2[:,2], ':k')
    ax2.axhline(params['zV0'], ls='--', c='k', label='mirror surface')
    ax2.axhline(params['zQd'], ls='--', c='g', label='Qdot height')
    ax2.axhline(params['z_tip'], ls='--', c='orange', label='Tip Height')
    ax2.set_title("Tip Potential")
    ax2.set_xlabel("x [Å]")
    ax2.set_ylabel("z [Å]")
    ax2.grid()
    ax2.legend()
    
    # Site Potential
    ax3.clear()
    ax3.imshow(data['Esites'], extent=data['extent'], cmap='bwr', 
               origin='lower', vmin=-params['VBias'], vmax=params['VBias'])
    ax3.axhline(params['zV0'], ls='--', c='k', label='mirror surface')
    ax3.axhline(params['zQd'], ls='--', c='g', label='Qdot height')
    ax3.legend()
    ax3.set_title("Site Potential")
    ax3.set_xlabel("x [Å]")
    ax3.set_ylabel("z [Å]")
    ax3.grid()

def plot_qdot_system(ax4, ax5, ax6, data, params):
    """
    Plot X-Y projections of quantum dot system
    
    Args:
        ax4: Matplotlib axis for energies plot
        ax5: Matplotlib axis for total charge plot
        ax6: Matplotlib axis for STM plot
        data: Dictionary containing calculated data from calculate_qdot_system
        params: Dictionary containing simulation parameters
    """
    # Energies
    ax4.clear()
    Eplot = np.max(data['Es'], axis=2)  # Using max mode
    vmax = np.abs(Eplot).max()
    ax4.imshow(Eplot, extent=data['extent'], cmap='bwr', 
               origin='lower', vmin=-vmax, vmax=vmax)
    
    # Plot quantum dot positions
    for i in range(params['nsite']):
        ax4.plot(data['spos'][i,0], data['spos'][i,1], 'ko')
    ax4.set_title(f"Energies (max)")
    ax4.set_xlabel("x [Å]")
    ax4.set_ylabel("y [Å]")
    ax4.grid()
    
    # Total Charge
    ax5.clear()
    ax5.imshow(data['total_charge'].reshape(params['npix'],params['npix']), 
               extent=data['extent'], cmap='bwr', origin='lower')
    for i in range(params['nsite']):
        ax5.plot(data['spos'][i,0], data['spos'][i,1], 'ko')
    ax5.set_title("Total Charge")
    ax5.set_xlabel("x [Å]")
    ax5.set_ylabel("y [Å]")
    ax5.grid()
    
    # STM
    ax6.clear()
    ax6.imshow(data['STM'].reshape(params['npix'],params['npix']), 
               extent=data['extent'], cmap='gray', origin='lower')
    for i in range(params['nsite']):
        ax6.plot(data['spos'][i,0], data['spos'][i,1], 'ro')
    ax6.set_title("STM")
    ax6.set_xlabel("x [Å]")
    ax6.set_ylabel("y [Å]")
    ax6.grid()
