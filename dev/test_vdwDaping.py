#!/usr/bin/python -u

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm.common as PPU
import ppafm.core   as PPC

# ======== Setup
iPP   = 8
iZs   = [1,6,7,8] 
iAtom =  0

# ======== Main

FFparams            = PPU.loadSpecies( )
elem_dict           = PPU.getFFdict(FFparams); # print elem_dict

REs  = PPU.getAtomsRE( iPP, iZs, FFparams )
cLJs = PPU.getAtomsLJ( iPP, iZs, FFparams )

print( "REs  ", REs )
print( "cLJs ", cLJs )

xs = np.arange(0.0,6.0,0.1)

print(xs)

Es_LJ,Fs_LJ = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=-2 )
Es,Fs       = PPC.evalRadialFF( xs, REs [iAtom,:], kind=1 )

vmax=REs[iAtom,1]

plt.subplot(2,1,1); plt.plot(xs,Es); plt.plot(xs,Es_LJ,'--k'); plt.axvline(REs[iAtom,0], ls=':', c='k'); plt.ylim(vmax*-2,vmax) ;plt.grid()
plt.subplot(2,1,2); plt.plot(xs,Fs); plt.plot(xs,Fs_LJ,'--k'); plt.axvline(REs[iAtom,0], ls=':', c='k'); plt.ylim(vmax*-2,vmax) ;plt.grid()

plt.show()