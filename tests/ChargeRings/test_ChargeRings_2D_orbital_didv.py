#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from pyProbeParticle import GridUtils as GU
from pyProbeParticle import photo
from orbital_utils import load_orbital, plotMinMax, photonMap2D_stamp, calculate_stm_maps, calculate_didv

# ========== Setup Parameters

# System parameters
V_Bias    = 1.0
Q_tip     = 0.13
cCouling  = 0.02 # 0.0
E_Fermi   = 0.0
z_tip     = 5.0
L         = 20.0
npix      = 400
decay     = 0.5
T         = 2.0
dQ        = 0.004  # Charge difference for dI/dV

# QD system setup
nsite  = 3
R      = 7.0
phiRot = 0.1 + np.pi/2

Q0  = 1.0
Qzz = 15.0 #* 0.0

# Canvas parameters
Lcanv = 60.0
dCanv = 0.2
decay_conv = 0.5

# Energy of states on the sites
Esite = [-0.2, -0.2, -0.2]

bDebug = True

def main():
    # Setup system geometry
    phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
    spos = np.zeros((nsite, 3))
    spos[:,0] = np.cos(phis)*R
    spos[:,1] = np.sin(phis)*R
    angles = phis + phiRot
    
    # Setup site multipoles
    mpols = np.zeros((nsite, 10))
    mpols[:,0] = Q0
    mpols[:,4] = Qzz
    
    # Initialize global parameters for ChargeRings
    rots = chr.makeRotMats(phis + phiRot, nsite)
    chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T, onSiteCoulomb=3.0)
    
    # Set up configuration basis
    confstrs = ["000","001","010","100","110","101","011","111"]
    confs = chr.confsFromStrings(confstrs)
    nconfs = chr.setSiteConfBasis(confs)
    
    # Load QD orbital and print information
    orbital_data, orbital_lvec = load_orbital("QD.cub")
    print("Orbital shape:", orbital_data.shape)
    print("Lattice vectors:")
    print(orbital_lvec)
    
    # Get physical dimensions from lattice vectors
    Lx = abs(orbital_lvec[1,0])
    Ly = abs(orbital_lvec[2,1])
    print(f"Molecular orbital dimensions: Lx={Lx:.3f} Å, Ly={Ly:.3f} Å")
    
    # Setup canvas
    ncanv = int(np.ceil(Lcanv/dCanv))
    canvas_shape = (ncanv, ncanv)
    canvas_dd = np.array([dCanv, dCanv])
    print(f"Canvas shape: {canvas_shape}")
    print(f"Canvas spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")
    
    # Create tip field
    tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)
    
    # Prepare orbital data - integrate over z
    orbital_data = np.transpose(orbital_data, (2, 1, 0))
    orbital_2D = np.sum(orbital_data[:, :, orbital_data.shape[2]//2:], axis=2)
    orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)
    
    # Define center region
    crop_center = (canvas_shape[0]//2, canvas_shape[1]//2)
    crop_size = (canvas_shape[0]//4, canvas_shape[1]//4)
    
    # Calculate physical dimensions of center region
    center_Lx = 2 * crop_size[0] * canvas_dd[0]
    center_Ly = 2 * crop_size[1] * canvas_dd[1]
    print(f"Center region physical size: {center_Lx:.3f} x {center_Ly:.3f} Å")
    
    # Create grid for the center region
    x = np.linspace(-center_Lx/2, center_Lx/2, 2*crop_size[0])
    y = np.linspace(-center_Ly/2, center_Ly/2, 2*crop_size[1])
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Create tip positions array for the center region
    ps = np.zeros((len(x) * len(y), 3))
    ps[:,0] = X.flatten()
    ps[:,1] = Y.flatten()
    ps[:,2] = z_tip
    
    # Calculate dI/dV
    dIdQ, I_1, I_2 = calculate_didv(orbital_2D, orbital_lvec, spos, angles, canvas_dd, canvas_shape, tipWf, ps, Q_tip, dQ, E_Fermi, V_Bias, decay, T, crop_center, crop_size)
    
    # Plot results
    extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]
    
    # Plot current at Q_tip
    plt.figure(figsize=(8,8))
    plt.imshow(I_1, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(label='Current')
    plt.title(f'STM Current (Q_tip = {Q_tip:.3f})')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    plt.savefig("test_ChargeRings_2D_orbital_didv_I1.png", bbox_inches='tight')
    plt.close()
    
    # Plot current at Q_tip + dQ
    plt.figure(figsize=(8,8))
    plt.imshow(I_2, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(label='Current')
    plt.title(f'STM Current (Q_tip = {Q_tip + dQ:.3f})')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    plt.savefig("test_ChargeRings_2D_orbital_didv_I2.png", bbox_inches='tight')
    plt.close()
    
    # Plot dI/dV
    plt.figure(figsize=(8,8))
    plt.imshow(dIdQ, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(label='dI/dV')
    plt.title('dI/dV Map')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    plt.savefig("test_ChargeRings_2D_orbital_didv_map.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
