#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

'''
TODO:
 * Check units of tip size 
 * try 3D effects (not flatened z-axis)
 * complex coeficients for molecular orbital phase
 * draw position and orientaion of molecules
 * move molecules by mouse

'''





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

def makeBox( pos, rot, a=10.0,b=20.0, byCenter=True ):
    c= np.cos(rot)
    s=-np.sin(rot)
    x=pos[0]; y=pos[1]
    if byCenter:
        ca=c*a*0.5;sa=s*a*0.5; 
        cb=c*b*0.5;sb=s*b*0.5; 
        xs=[x-ca+sb,x+ca+sb,x+ca-sb,x-ca-sb,x-ca+sb]
        ys=[y-sa-cb,y+sa-cb,y+sa+cb,y-sa+cb,y-sa-cb]
    else:
        xs=[x,x+c*a,x+c*a-s*b,x-s*b,x]
        ys=[y,y+s*a,y+s*a+c*b,y+c*b,y]
    return xs,ys

def plotBoxes( poss, rots, lvec, ax=None ):
    if ax is None:
        ax = plt.gca()
    #print "lvec ", lvecH
    for i in range(len(poss)):
        #xs,ys = makeBox( poss[i], rots[i], a=lvec[2][1],b=lvec[3][2] )
        xs,ys = makeBox( poss[i], rots[i], a=lvec[3][2],b=lvec[2][1] )
        ax.plot(xs,ys)
        #plt.plot(xs[0],ys[0],'o')

def shiftHalfAxis( X, d, n, ax=1 ):
    shift = n/2; 
    if( n%2 != 0 ):  shift += 1
    X -= shift
    return np.roll( X, shift, axis=ax), shift

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
    Y = dy * np.roll( Y, yshift, axis=1)
    X = dx * np.roll( X, xshift, axis=0)
    #print X[:,0]
    return X, Y, (xshift, yshift)

def getMGrid3D( nDim, dd ):
    'returns coordinate arrays X, Y, Z'
    (dx,dy,dz) = dd
    (nx,ny,nz) = nDim[:3]
    #print dd,nDim
    XYZ = np.mgrid[0:nx,0:ny,0:nz].astype(float)
    X,xshift = shiftHalfAxis( XYZ[0], dx, nx, ax=0 )
    Y,yshift = shiftHalfAxis( XYZ[1], dy, ny, ax=1 )
    #Z,zshift = shiftHalfAxis( XYZ[2], dz, nz, ax=2 )
    Z = XYZ[2]
    return X*dx, Y*dy, Z*dz, (xshift, yshift, 0)

def makeTipField2D( sh, dd, z=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    Vtip = np.zeros( sh[:2] )
    #X,Y,shifts  = getMGrid2D( sh, dd )
    Y,X,shifts  = getMGrid2D( sh, dd )
    #X *= dd[0]; Y *= dd[1];  # this is already done in getMGrid
    #print "Z = ", z
    #print "(xmax,ymax) = ", X[-1,-1],Y[-1,-1],  X[0,0],Y[0,0]
    radial = 1/np.sqrt( X**2 + Y**2  + z**2 + sigma**2  ) 
    #print "radial ", radial[:,radial.shape[1]/2]
    if multipole_dict is not None:	# multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.iteritems():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    #Vtip = X
    #Vtip = radial
    #print "Vtip ", Vtip[:,Vtip.shape[1]/2]
    return Vtip, shifts

def makeTipField3D( sh, dd, z0=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    Vtip = np.zeros( sh )
    X,Y,Z,shifts  = getMGrid3D( sh, dd )
    Z += z0
    #X *= dd[0]; Y *= dd[1];  # this is already done in getMGrid
    #print "Z = ", z
    #print "(xmax,ymax) = ", X[-1,-1],Y[-1,-1],  X[0,0],Y[0,0]
    radial = 1/np.sqrt( X**2 + Y**2  + Z**2 + sigma**2 ) 
    #print "radial ", radial[:,radial.shape[1]/2]
    if multipole_dict is not None:	# multipole_dict should be dictionary like { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
        Vtip = np.zeros( np.shape(radial) )
        for kind, coef in multipole_dict.iteritems():
            Vtip += radial * coef * fFFT.getSphericalHarmonic( X, Y, Z, kind=kind, tilt=0 )
    else:
        Vtip = radial
    #print "Vtip.shape     ", Vtip.shape
    Vtip = Vtip.transpose((2,1,0)).copy()
    #print " -> Vtip.shape ", Vtip.shape
    #print "shifts ", shifts
    return Vtip, shifts

'''
def coordConv(poss,rots,center,nn):
    #poss - center positions of rhoTrans in canvas with respect to canvas center in points
    #rots - rotations in RAD
    #center - indexes of rhoTrans center (point of rotation)
    #nn - array defining the canvas size, in points
    a = center[0]
    b = center[1]
    for i in range(len(rots)):
        nposs = poss
        ph = rots[i]
        x  = poss[i][0]
        y  = poss[i][1]
        ca = np.cos(ph)
        sa = np.sin(ph)
        nposs[i][0] = x - ca*a + sa*b + nn[0]/2.
        nposs[i][1] = y + sa*a + ca*b + nn[1]/2.
    return nposs
'''

def convFFT(F1,F2):
    return np.fft.ifftn( np.fft.fftn(F1) * np.fft.fftn(F2) )

def photonMap2D( rhoTrans, tipDict, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0} ):
    sh = rhoTrans.shape
    #print "shape: ", sh
    #rho  = np.zeros( (sh[:2]) )
    rho  = np.sum(  rhoTrans, axis=2) 
    #Vtip = np.zeros( (sh[:2]) )
    #print lvec
    dx   = lvec[3][2]/sh[0]; #print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; #print lvec[2][1],sh[1]
    dd = (dx,dy)
    #print " (dx,dy) ", dd
    Vtip, shifts = makeTipField2D( sh[:2], dd, z=z, sigma=sigma, multipole_dict=multipole_dict )
    renorm = 1./( (Vtip**2).sum() * (rho**2).sum() )
    phmap  = convFFT(Vtip,rho).real * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=0)
    return phmap, Vtip, rho , dd



def photonMap2D_stamp( rhoTrans, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], ncanv=(300,300) ):
    sh = rhoTrans.shape
    #print "shape: ", sh
    #print lvec
    dx   = lvec[3][2]/sh[0]; #print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; #print lvec[2][1],sh[1]
    dd = (dx,dy)
    #print " (dx,dy) ", dd

    #rho   = np.zeros( (sh[:2]) )
    rho    = np.sum  (  rhoTrans, axis=2 ) 
    rho    = rho.astype( np.complex128 ) 
    canvas = np.zeros( ncanv, dtype=np.complex128 ) 
    
    #rho    = np.ascontiguousarray( rho,    dtype=np.complex128 )
    #canvas = np.ascontiguousarray( canvas, dtype=np.complex128 )

    for i in range(len(poss)):
        #GU.stampToGrid2D( canvas, rho, poss[i], rots[i], dd=[1.0,1.0], coef=coefs[i] )
        coef = complex( coefs[i][0], coefs[i][1] )
        #print  i,poss[i], rots[i],  coef
        pos = [ poss[i][0]+ncanv[0]*0.5*dx,   poss[i][1]+ncanv[1]*0.5*dy  ]
        GU.stampToGrid2D_complex( canvas, rho, pos, rots[i], dd=dd, coef=coef )

    #canvas = np.zeros((300,300))

    Vtip, shifts = makeTipField2D( ncanv[:2], dd, z=z, sigma=sigma, multipole_dict=multipole_dict )
    #renorm = 1./( (Vtip**2).sum() * (rho.real**2+rho.imag**2).sum() )
    renorm = 1./( (Vtip**2).sum() )
    phmap  = convFFT(Vtip,canvas) * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=0)
    return phmap, Vtip, canvas, dd


def photonMap3D_stamp( rhoTrans, lvec, z=10.0, sigma=1.0, multipole_dict={'s':1.0}, rots=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], ncanv=(300,300),  ):
    sh  = rhoTrans.shape
    if len(ncanv)<3:
        ncanv = ( (ncanv[0],ncanv[1],sh[2]) )
    canvas = np.zeros( ncanv[::-1], dtype=np.complex128 )
    rho    = rhoTrans.transpose((2,1,0)).astype( np.complex128 ).copy()
    #print "shape: stamp ", rho.shape, " canvas ", canvas.shape
    #print lvec
    dx   = lvec[3][2]/sh[0]; #print lvec[3][2],sh[0]
    dy   = lvec[2][1]/sh[1]; #print lvec[2][1],sh[1]
    dz   = lvec[1][0]/sh[2]; #print lvec[1][0],sh[2]
    dd = (dx,dy,dz);    #print " dd ", dd
 
    for i in range(len(poss)):
        coef = complex( coefs[i][0], coefs[i][1] )
        pos_ = poss[i]
        if len(pos_)<3:
            pos_.append(0.0)
        pos = [ pos_[0]+ncanv[0]*0.5*dx,   pos_[1]+ncanv[1]*0.5*dy, pos_[2] ]
        GU.stampToGrid3D_complex( canvas, rho, pos, rots[i], dd=dd, coef=coef )

    Vtip, shifts = makeTipField3D( ncanv, dd, z0=z, sigma=sigma, multipole_dict=multipole_dict )
    #renorm = 1./( (Vtip**2).sum() * (rho.real**2+rho.imag**2).sum() )
    renorm = 1./( (Vtip**2).sum() )
    phmap  = convFFT(Vtip,canvas) * renorm
    #print "rho  ", rho.shape  
    #print "Vtip ", Vtip.shape  
    #Vtip = np.roll( Vtip, -shifts[2], axis=2)
    Vtip = np.roll( Vtip, -shifts[1], axis=1)
    Vtip = np.roll( Vtip, -shifts[0], axis=2)
    return phmap, Vtip, canvas, dd

def solveExcitonSystem( e, rhoTrans, lvec, poss, rots, ndim=None ):
    '''
    Solve coupled excitonic system according to :
    https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Book%3A_Time_Dependent_Quantum_Mechanics_and_Spectroscopy_(Tokmakoff)/15%3A_Energy_and_Charge_Transfer/15.03%3A_Excitons_in_Molecular_Aggregates
    '''
    n = range(len(poss))
    H = np.eye(n)
    if ndim is not None: # down-sample ?
        rhoTrans = GU.downSample3D( rhoTrans, ndim=ndim )
        GU.saveXSF("rhoTrans_donw.xsf", rhoTrans,  )
    poss = np.array(poss)
    lvec=np.array(lvec[1:])
    print "lvec ", lvec
    for i in range(n):
        ci = np.cos(rot[i])
        si = np.sin(rot[i])
        rot1 = np.array([[ci,-si,0.],[si,ci,0.],[0.,0.,1.]])
        rot1 = np.dot( rot1, lvec )    # ToDo : check rotation is correct
        for j in range(i):
            dpos = poss[i] - poss[j]
            cj = np.cos(rot[j])
            sj = np.sin(rot[j])
            rot2 = np.array([[cj,-sj,0.],[sj,cj,0.],[0.,0.,1.]])
            rot2 = np.dot( rot2, lvec )
            eij = GU.coulombGrid_brute( rhoTrans, rhoTrans, dpos=dpos, rot1=rot1, rot2=rot2 )
            H[i,j]=eij
            H[j,i]=eij
    es,vs = np.linalg.eig(H)
    return es,vs,H

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

    
    #tipDict =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
    #tipDict =  { 's': 1.0, 'py':1.0  }
    tipDict =  { 's': 1.0, 'dy2':1.0  }
    #tipDict =  { 's': 1.0 }
    #tipDict =  { 'px': 1.0  }
    #tipDict =  { 'py': 1.0  }

    #phmap, Vtip, rho =  photonMap2D( rhoTrans, tipDict, lvecH, z=0.5, sigma=0.0, multipole_dict=tipDict )

    '''
    rots =[0.0]
    poss =[ [200.0,200.0] ]
    coefs=[1.0]
    phmap, Vtip, rho =  photonMap2D_stamp( rhoTrans, lvecH, z=0.5, sigma=1.0, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow( rho      ); plt.colorbar(); plt.title('Transient Density')
    plt.subplot(1,3,2); plt.imshow( Vtip     ); plt.colorbar(); plt.title('Tip Field')
    plt.subplot(1,3,3); plt.imshow( phmap**2 ); plt.colorbar(); plt.title('Photon Map')
    '''

    fromDeg = np.pi/180.

    rots  =[-30.0*fromDeg,45.0*fromDeg]
    #poss =[ [10.0,5.0] ,  [10.0,10.0] ]
    poss  =[ [-5.0,10.0] ,  [5.0,-5.0] ]
    #poss =[ [0.0,10.0]  ]
    #poss =[ [200.0,50.0] ,  [50.0,50.0] ]
    #coefs=[ [1.0,0.0],      [0.0,1.0]     ]
    coefs=[ [1.0,0.0],      [-1.0,0.0]     ]

    #rots =[0.0]
    #poss =[ [300.0,50.0]]
    #coefs=[ [1.0,0.0]   ]

    b2D = False
    #b2D = True

    if b2D:
        phmap, Vtip, rho, dd =  photonMap2D_stamp( rhoTrans, lvecH, z=5.0, sigma=1.0, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )
        (dx,dy)=dd
    else:
        phmap_, Vtip_, rho_, dd =  photonMap3D_stamp( rhoTrans, lvecH, z=5.0, sigma=1.0, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=(500,500) )
        phmap = np.sum(phmap_,axis=0)
        Vtip  = np.sum(Vtip_ ,axis=0)
        rho   = np.sum(rho_  ,axis=0)
        (dx,dy,dz)=dd

    print "dd ",  dd
    sh=phmap.shape
    extent=( -sh[0]*dd[0]*0.5,sh[0]*dd[0]*0.5,   -sh[1]*dd[1]*0.5, sh[1]*dd[1]*0.5   )
    
    #print "xs ", xs
    #print "ys ", ys
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,2); plt.imshow( Vtip.real, extent=extent , origin='image'                    ); plt.xlabel('X[A]'); plt.ylabel('Y[A]'); plt.colorbar(); plt.title('Tip Field')
    #plt.subplot(1,3,1); plt.imshow( rho.real  **2 + rho.imag  **2, origin='image' ); plt.colorbar(); plt.title('Transient Density')
    plt.subplot(1,3,1); plt.imshow( rho.real   ,extent=extent, origin='image'                    ); plt.xlabel('X[A]'); plt.ylabel('Y[A]'); plt.colorbar(); plt.title('Transient Density')
    #print "lvec ", lvecH
    #for i in range(len(poss)):
    #    xs,ys = makeBox( poss[i], rots[i], a=lvecH[2][1],b=lvecH[3][2] )
    #    plt.plot(xs,ys)
    #    #plt.plot(xs[0],ys[0],'o')
    plotBoxes( poss, rots, lvecH )
    plt.subplot(1,3,3); plt.imshow( phmap.real**2 + phmap.imag**2, extent=extent, origin='image' ); plt.xlabel('X[A]'); plt.ylabel('Y[A]'); plt.colorbar(); plt.title('Photon Map')
    
    
    plt.figure()
    plt.plot( np.arange(Vtip.shape[0])*dx, Vtip[:,Vtip.shape[1]/2] ); plt.grid(); # plt.ylim(0.,1.);
    #print "Vtip outside: ", Vtip[:,Vtip.shape[1]/2]
    
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

