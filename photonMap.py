#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
import __main__ as main
import numpy as np
#import GridUtils as GU
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT

# ======== Main

def getMGrid2D(nDim, dd):
    'returns coordinate arrays X, Y, Z'
    (dx, dy) = dd
    XY = np.mgrid[0:nDim[0],0:nDim[1]].astype(float)
    yshift = nDim[1]/2;  yshift_ = yshift;
    xshift = nDim[0]/2;  xshift_ = xshift;
    if( nDim[1]%2 != 0 ):  yshift_ += 1.0
    if( nDim[0]%2 != 0 ):  xshift_ += 1.0
    X = XY[0] - xshift_;
    Y = XY[1] - yshift_;
    Y = dy*np.roll( Y, yshift, axis=1)
    X = dx*np.roll( X, xshift, axis=0)
    return X, Y, (xshift, yshift)

def makeTipField2d( sh, dd, z=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    Vtip = np.zeros( (sh[:2]) )
    X,Y,shifts  = getMGrid2D( sh, dd )
    X *= dd[0]; Y *= dd[1];
    print "Z = ", z
    radial = 1/np.sqrt( X**2 + Y**2 + sigma**2  + z**2 ) 
    if multipole_dict is not None:	# multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.iteritems():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    #Vtip = X
    Vtip = radial
    return Vtip, shifts

def convFFT(F1,F2):
    return np.fft.ifftn( np.fft.fftn(F1) * np.fft.fftn(F2) )

def photonMap2D( rhoTrans, tipDict, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    sh = rhoTrans.shape
    print "shape: ", sh
    #rho  = np.zeros( (sh[:2]) )
    rho  = np.sum(  rhoTrans, axis=2) 
    #Vtip = np.zeros( (sh[:2]) )
    print lvec
    dx   = lvec[3][2]/sh[0]; print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; print lvec[2][1],sh[1]
    dd = (dx,dy)
    print dd
    Vtip, shifts = makeTipField2d( sh[:2], dd, z=z, sigma=sigma, multipole_dict=multipole_dict )
    phmap  = convFFT(Vtip,rho).real
    print "rho  ", rho.shape  
    print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=0)
    return phmap, Vtip, rho 



def photonMap2D_stamp( rhoTrans, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[1.0], ncanv=(300,300) ):
    sh = rhoTrans.shape
    print "shape: ", sh
    #rho  = np.zeros( (sh[:2]) )
    rho    = np.sum(  rhoTrans, axis=2) 
    canvas = np.zeros(ncanv)

    for i in range(len(rots)):
        GU.stampToGrid2D( canvas, rho, poss[i], rots[i], dd=[1.0,1.0], coef=coefs[i] )

    #Vtip = np.zeros( (sh[:2]) )
    print lvec
    dx   = lvec[3][2]/sh[0]; print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; print lvec[2][1],sh[1]
    dd = (dx,dy)
    print dd

    #canvas = np.zeros((300,300))

    Vtip, shifts = makeTipField2d( ncanv[:2], dd, z=z, sigma=sigma, multipole_dict=multipole_dict )
    phmap  = convFFT(Vtip,canvas).real
    print "rho  ", rho.shape  
    print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=0)
    return phmap, Vtip, canvas


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option( "-H", "--homo",   action="store", type="string", default="homo.xsf", help="orbital of electron hole;    3D data-file (.xsf,.cube)")
    parser.add_option( "-L", "--lumo",   action="store", type="string", default="lumo.xsf", help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    parser.add_option( "-R", "--radius", action="store", type="float",  default="1.0", help="tip radius")
    parser.add_option( "-z", "--ztip", action="store", type="float",  default="5.0", help="tip above substrate")
    parser.add_option( "-t", "--tip",    action="store", type="string", default="s",   help="tip compositon s,px,py,pz,d...")

    #parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
    (options, args) = parser.parse_args()

    #rho1, lvec1, nDim1, head1 = GU.loadXSF("./pyridine/CHGCAR.xsf")
    #rho2, lvec2, nDim2, head2 = GU.loadXSF("./CO_/CHGCAR.xsf")

    print( ">>> Loading HOMO from ", options.homo, " ... " )
    homo, lvecH, nDimH, headH = GU.loadCUBE( options.homo )
    print( ">>> Loading LUMO from ", options.lumo, " ... " )
    lumo, lvecL, nDimL, headL = GU.loadCUBE( options.lumo )
    rhoTrans = homo*lumo

    '''
    rho  = np.sum(  rhoTrans, axis=2)
    canvas = np.zeros((300,300))
    #dx   = lvec[3][2]/sh[0]; print lvec[3][2],sh[0]
    #dy   = lvec[2][1]/sh[1]; print lvec[2][1],sh[1]
    #dd = (dx,dy)
    GU.stampToGrid2D( canvas, rho, [2.0,60.0], np.pi/4.0, dd=[1.0,1.0] )
    plt.imshow(canvas)
    plt.show()
    '''

    
    #tip =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
    tipDict =  { 's': 1.0  }

    #phmap, Vtip, rho =  photonMap2D( rhoTrans, tipDict, lvecH, z=0.5, sigma=0.0, multipole_dict=tipDict )

    rots =[0.0]
    poss =[ [200.0,200.0] ]
    coefs=[1.0]
    phmap, Vtip, rho =  photonMap2D_stamp( rhoTrans, lvecH, z=0.5, sigma=1.0, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow( rho      ); plt.colorbar(); plt.title('Transient Density')
    plt.subplot(1,3,2); plt.imshow( Vtip     ); plt.colorbar(); plt.title('Tip Field')
    plt.subplot(1,3,3); plt.imshow( phmap**2 ); plt.colorbar(); plt.title('Photon Map')

    rots =[0.0,0.0]
    poss =[ [200.0,50.0] ,  [200.0,200.0] ]
    coefs=[1.0,1.0]
    phmap, Vtip, rho =  photonMap2D_stamp( rhoTrans, lvecH, z=0.5, sigma=1.0, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow( rho      ); plt.colorbar(); plt.title('Transient Density')
    plt.subplot(1,3,2); plt.imshow( Vtip     ); plt.colorbar(); plt.title('Tip Field')
    plt.subplot(1,3,3); plt.imshow( phmap**2 ); plt.colorbar(); plt.title('Photon Map')
    plt.show()
    

    '''
    
    #print( ">>> Evaluating convolution E(R) = A*Integral_r ( rho_tip^B(r-R) * rho_sample^B(r) ) using FFT ... " )
    #Fx,Fy,Fz,E = fFFT.potential2forces_mem( rhoTrans, lvecH, nDimH, rho=tip, doForce=True, doPot=True, deleteV=False )

    Vtip =  fFFT.getMultiploleGrid( lvecH, nDimH, sigma=1.0, multipole_dict=tip, tilt=0.0 )
    #getMultiplole( sampleSize, X, Y, Z, dd, sigma=0.7, multipole_dict=None, tilt=0.0 ):

    couplings = fFFT.convolveFFT( rhoTrans, Vtip, lvecH, nDimH )

    GU.saveXSF( "couplMap.xsf", couplings, lvecH, head=headH )
    GU.saveXSF( "Vtip.xsf", Vtip, lvecH, head=headH )

    GU.saveXSF( "HOMO.xsf", homo, lvecH, head=headH )
    GU.saveXSF( "LUMO.xsf", lumo, lvecL, head=headL )
    GU.saveXSF( "rhoTrans_HOMO_LUMO.xsf", rhoTrans, lvecH, head=headH )
    '''

