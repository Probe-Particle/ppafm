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
#iZs   = [1] 
iAtom =  3

iVdWModel=3
if len(sys.argv)>1: iVdWModel=int(sys.argv[1])

ADamps = [ 180.0, 0.5, 0.5, 0.03, 0.01 ]

model_names = [
#   "LJ_RE",     # -3
#   "LJ   ",     # -2
#   "invR6",     # -1
    "VdW_C6",    # 0
    "VdW_R2",    # 1
    "VdW_R4",    # 2
    "VdW_invR4", # 3
    "VdW_invR8", # 4
]


# ======== Functions

def plotDecor( x0, vmax ):
    plt.axvline( x0, ls=':', c='k'); 
    plt.axhline( 0, ls='-' , c='k'); 
    plt.ylim(vmax*-2,vmax) 
    plt.grid()

def deriv( xs, Es, d=None ):
    if d is None : d=xs[1]-xs[0]  #;print( "d ", d )
    Fs = (Es[2:]-Es[:-2])/(2.*d)
    xs = xs[1:-1]
    return Fs, xs

# ======== Main

FFparams            = PPU.loadSpecies( )
elem_dict           = PPU.getFFdict(FFparams); # print elem_dict

REs  = PPU.getAtomsRE( iPP, iZs, FFparams )   ;print( "REs  ", REs )
cLJs = PPU.getAtomsLJ( iPP, iZs, FFparams )   ;print( "cLJs ", cLJs )


xs = np.arange(0.0,6.0,0.1)                   #;print(xs)

'''
Es_LJ,Fs_LJ = PPC.evalRadialFF( xs, REs[iAtom,:], kind=-3 ) #;print( "Es_LJ", Es_LJ )  # LJ  -C6/r^6 + C12/r^12
#Es_LJ,Fs_LJ = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=-2 ) ;print( "Es_LJ", Es_LJ )  # LJ  -C6/r^6 + C12/r^12
Es_r6,Fs_r6 = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=0 ) #;print( "Es_r6", Es_r6 )   # vdW -C6/r^6

Es,Fs       = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=0  )   
#Es,Fs       = PPC.evalRadialFF( xs, REs [iAtom,:], kind=0  )   

plt.figure(figsize=(8,16))
plt.subplot(2,1,1);    plt.plot(xs,Es_LJ   ,':k'); plt.plot(xs,Es_r6   ,'--k'); plt.plot(xs,Es   );       plotDecor( R0, E0*2 )       
plt.subplot(2,1,2);    plt.plot(xs,Fs_LJ*-1,':k'); plt.plot(xs,Fs_r6*-1,'--k'); plt.plot(xs,Fs*-1);       plotDecor( R0, E0*5 ) 
'''

coefs = REs
if iVdWModel in {-2,-1,0}: coefs = cLJs


name=model_names[iVdWModel]
nz = len(iZs)
plt.figure(figsize=(5*nz,5*2))
for i in range(nz):
    iAtom = i

    R0=REs[iAtom,0]
    E0=REs[iAtom,1]
    Es_LJ,Fs_LJ = PPC.evalRadialFF( xs, REs[iAtom,:], kind=-3 ) #;print( "Es_LJ", Es_LJ )  # LJ  -C6/r^6 + C12/r^12
    #Es_LJ,Fs_LJ = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=-2 ) ;print( "Es_LJ", Es_LJ )  # LJ  -C6/r^6 + C12/r^12
    Es_r6,Fs_r6 = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=-1 ) #;print( "Es_r6", Es_r6 )   # vdW -C6/r^6
    ADamp = ADamps[iVdWModel]
    #Es,Fs      = PPC.evalRadialFF( xs, cLJs[iAtom,:], kind=iVdWModel  )   
    Es,Fs       = PPC.evalRadialFF( xs, coefs[iAtom,:], kind=iVdWModel, ADamp=ADamp )   

    Fs_num,xs_  = deriv( xs, Es )

    plt.subplot(2,nz,i   +1);    plt.plot(xs,Es_LJ   ,':k'); plt.plot(xs,Es_r6   ,'--k'); plt.plot(xs,Es   );                                   plotDecor( R0, E0*5 );  plt.title("%s(Adamp=%5.2f) %s-%s" %(name,ADamp, (FFparams[iZs[i]-1][4]).decode(), (FFparams[iPP-1][4]).decode() ) ) #; plt.title("Interaction %i-%i" %(iZs[i], iPP) )
    plt.subplot(2,nz,i+nz+1);    plt.plot(xs,Fs_LJ*-1,':k'); plt.plot(xs,Fs_r6*-1,'--k'); plt.plot(xs,Fs*-1);  plt.plot(xs_,Fs_num*-1, ':r');   plotDecor( R0, E0*7 ); 

plt.subplot(2,nz,  +1); plt.ylabel("Energy [eV]");
plt.subplot(2,nz,nz+1); plt.ylabel("Force  [eV/A]");
plt.savefig(name+".png",bbox_inches='tight')

plt.show()