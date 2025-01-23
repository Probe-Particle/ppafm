#!/usr/bin/python

import numpy as np
from TipMultipole import makeCircle

def plot_tip_potential(ax1, ax2, ax3, *, Vtip, Esites, ps_xz, V1d, extent,
                      npix, L, z_tip, Rtip, VBias, zV0, zQd, Esite, **kwargs):
    """
    Plot X-Z projections of tip potential
    
    Args:
        ax1: Matplotlib axis for 1D potential plot
        ax2: Matplotlib axis for tip potential plot
        ax3: Matplotlib axis for site potential plot
        Vtip: Tip potential data
        Esites: Site energies data
        ps_xz: X-Z positions
        V1d: 1D potential data
        extent: Plot extent parameters
        npix: Number of pixels for grid
        L: Size of simulation box
        z_tip: Tip height
        Rtip: Tip radius
        VBias: Bias voltage
        zV0: Mirror surface position
        zQd: Quantum dot height
        Esite: Site energy
    """
    # 1D Potential
    ax1.clear()
    x_coords = np.linspace(-extent[1], extent[1], npix)
    ax1.plot(x_coords, V1d, label='V_tip')
    ax1.plot(x_coords, V1d + Esite, label='V_tip + E_site')
    ax1.plot(x_coords, x_coords*0.0 + VBias, label='VBias')
    ax1.axhline(0.0, ls='--', c='k')
    ax1.set_title("1D Potential (z=0)")
    ax1.set_xlabel("x [Å]")
    ax1.set_ylabel("V [V]")
    ax1.grid()
    ax1.legend()
    
    # Tip Potential
    ax2.clear()
    zT = z_tip + Rtip
    ax2.imshow(Vtip, extent=extent, cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias)
    circ1, _ = makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,zT))
    circ2, _ = makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0,0.0,2*zV0-zT))
    ax2.plot(circ1[:,0], circ1[:,2], ':k')
    ax2.plot(circ2[:,0], circ2[:,2], ':k')
    ax2.axhline(zV0, ls='--', c='k', label='mirror surface')
    ax2.axhline(zQd, ls='--', c='g', label='Qdot height')
    ax2.axhline(z_tip, ls='--', c='orange', label='Tip Height')
    ax2.set_title("Tip Potential")
    ax2.set_xlabel("x [Å]")
    ax2.set_ylabel("z [Å]")
    ax2.grid()
    ax2.legend()
    
    # Site Potential
    ax3.clear()
    ax3.imshow(Esites, extent=extent, cmap='bwr', origin='lower', vmin=-VBias, vmax=VBias)
    ax3.axhline(zV0, ls='--', c='k', label='mirror surface')
    ax3.axhline(zQd, ls='--', c='g', label='Qdot height')
    ax3.legend()
    ax3.set_title("Site Potential")
    ax3.set_xlabel("x [Å]")
    ax3.set_ylabel("z [Å]")
    ax3.grid()

def plot_qdot_system(ax4, ax5, ax6, *, Es, total_charge, STM, pTips, extent, spos,
                    nsite, npix, VBias, **kwargs):
    """
    Plot X-Y projections of quantum dot system
    
    Args:
        ax4: Matplotlib axis for energies plot
        ax5: Matplotlib axis for total charge plot
        ax6: Matplotlib axis for STM plot
        Es: Site energies data
        total_charge: Total charge distribution data
        STM: STM signal data
        pTips: Tip positions
        extent: Plot extent parameters
        spos: Site positions
        nsite: Number of sites
        npix: Number of pixels for grid
        VBias: Bias voltage
    """
    # Energies
    ax4.clear()
    Eplot = np.max(Es, axis=2)  # Using max mode
    vmax = np.abs(Eplot).max()
    ax4.imshow(Eplot, extent=extent, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
    
    # Plot quantum dot positions
    for i in range(nsite):
        ax4.plot(spos[i,0], spos[i,1], 'ko')
    ax4.set_title(f"Energies (max)")
    ax4.set_xlabel("x [Å]")
    ax4.set_ylabel("y [Å]")
    ax4.grid()
    
    # Total Charge
    ax5.clear()
    ax5.imshow(total_charge.reshape(npix,npix), extent=extent, cmap='bwr', origin='lower')
    for i in range(nsite):
        ax5.plot(spos[i,0], spos[i,1], 'ko')
    ax5.set_title("Total Charge")
    ax5.set_xlabel("x [Å]")
    ax5.set_ylabel("y [Å]")
    ax5.grid()
    
    # STM
    ax6.clear()
    ax6.imshow(STM.reshape(npix,npix), extent=extent, cmap='gray', origin='lower')
    for i in range(nsite):
        ax6.plot(spos[i,0], spos[i,1], 'ro')
    ax6.set_title("STM")
    ax6.set_xlabel("x [Å]")
    ax6.set_ylabel("y [Å]")
    ax6.grid()
