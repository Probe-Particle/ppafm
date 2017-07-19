#!/usr/bin/python

import os
import sys
import numpy     as np
import GridUtils as GU
import fieldFFT  as fFFT
import common    as PPU

import core
import cpp_utils

# ===== constants 
Fmax_DEFAULT = 100.0

# overall procedure for importing the sample geometry:

def perpareArrays( FF, Vpot ):
	if ( FF is None ):
		gridN = PPU.params['gridN']
		FF = np.zeros( (gridN[2],gridN[1],gridN[0],3)    )
	else:
		PPU.params['gridN'] = np.shape( FF )	
	if ( Vpot ):
		V = np.zeros( (gridN[2],gridN[1],gridN[0])    )
	else:
		V=None
	core.setFF( gridF=FF, gridE=V )
	return FF, V 

def computeLJ( Rs, iZs, FFLJ=None, FFparams=None, Vpot=False ):
	if FFparams is None:
		raise ValueError("You should provide a list of LJ parameters!")
	FFLJ,VLJ = perpareArrays( FFLJ, Vpot )
	#print "FFLJ.shape, VLJ.shape :", FFLJ.shape, VLJ.shape
	cLJs     = PPU.getAtomsLJ( PPU.params['probeType'], iZs, FFparams )
	#core.setFF( gridF=FFLJ, gridE=VLJ )
	core.getLenardJonesFF( Rs, cLJs )
	return FFLJ, VLJ

def relaxedScan3D( xTips, yTips, zTips ):
	ntips = len(zTips); 
	print " zTips : ",zTips
	rTips = np.zeros((ntips,3))
	rs    = np.zeros((ntips,3))
	fs    = np.zeros((ntips,3))
	rTips[:,0] = 1.0
	rTips[:,1] = 1.0
	rTips[:,2] = zTips[::-1]  
	nx = len(zTips); ny = len(yTips ); nz = len(xTips);
	fzs    = np.zeros( ( nx,ny,nz ) );
	PPpos  = np.zeros( ( nx,ny,nz,3 ) );
	for ix,x in enumerate( xTips  ):
		sys.stdout.write('\033[K')
		sys.stdout.flush()
		sys.stdout.write("\rrelax ix: {}".format(ix))
		sys.stdout.flush()
		rTips[:,0] = x
		for iy,y in enumerate( yTips  ):
			rTips[:,1] = y
			itrav = core.relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
			fzs[:,iy,ix] = (fs[:,2].copy()) [::-1]
			PPpos[:,iy,ix,0] = rs[::-1,0] # - rTips[:,0]
			PPpos[:,iy,ix,1] = rs[::-1,1] # - rTips[:,1]
			PPpos[:,iy,ix,2] = rs[::-1,2] # - rTips[:,2]
	return fzs,PPpos

def Gauss(Evib, E0, w):
    return np.exp( -0.5*((Evib - E0)/w)**2);

def symGauss( Evib, E0, w):
    return Gauss(Evib, E0, w) - Gauss(Evib, -E0, w);

def computeLJFF(iZs, Rs, FFparams, Fmax=Fmax_DEFAULT, computeVpot=False, Vmax=None):
    print "--- Compute Lennard-Jones Force-filed ---"
    FFLJ, VLJ=computeLJ( Rs, iZs, FFLJ=None, FFparams=FFparams, Vpot=computeVpot)     # This function computes the LJ forces experienced  by the ProbeParticle                                              
    if Fmax is not  None:
        print "Limit vector field"
        GU.limit_vec_field( FFLJ, Fmax=Fmax )
        # remove too large values; keeps the same
        # direction; good for the visualization 
    if  Vmax != None and VLJ != None:
        VLJ[ VLJ > Vmax ] =  Vmax # remove too large values
    return FFLJ,VLJ

def computeELFF_pch(iZs,Rs,Qs,computeVpot, tip='s', Fmax=Fmax_DEFAULT ):
    tipKinds = {'s':0,'pz':1,'dz2':2}
    tipKind  = tipKinds[tip]
    print " ========= get electrostatic forcefiled from the point charges tip=%s %i " %(tip,tipKind)
    FFel,V = perpareArrays( None, computeVpot )
    core.getCoulombFF( Rs, Qs*PPU.CoulombConst, kind=tipKind )
    if Fmax is not  None:
        print "Limit vector field"
        GU.limit_vec_field( FFel, Fmax=Fmax )
    if computeVpot :
        Vmax = 10.0; V[ V>Vmax ] = Vmax
    return FFel,V

def computeElFF(V,lvec,nDim,tip,Fmax=None,computeVpot=False,Vmax=None, tilt=0.0 ):
    print " ========= get electrostatic forcefiled from hartree "
    rho = None
    multipole = None
    if tip in {'s','px','py','pz','dx2','dy2','dz2','dxy','dxz','dyz'}:
        rho = None
        multipole={tip:1.0}
    elif tip.endswith(".xsf"):
        rho, lvec_tip, nDim_tip, tiphead = GU.loadXSF(tip)
        if any(nDim_tip != nDim):
            sys.exit("Error: Input file for tip charge density has been specified, but the dimensions are incompatible with the Hartree potential file!")    
    print " computing convolution with tip by FFT "
    #Fel_x,Fel_y,Fel_z      = fFFT.potential2forces(V, lvec, nDim, rho=rho, sigma=PPU.params['sigma'], multipole = multipole)
    Fel_x,Fel_y,Fel_z, Vout = fFFT.potential2forces_mem( V, lvec, nDim, rho=rho, sigma=PPU.params['sigma'], multipole = multipole, tilt=tilt )
    FFel = GU.packVecGrid(Fel_x,Fel_y,Fel_z)
    del Fel_x,Fel_y,Fel_z
    return FFel

def meshgrid3d(xs,ys,zs):
    Xs,Ys,Zs = np.zeros()
    Xs,Ys = np.meshgrid(xs,ys)

def perform_relaxation (lvec,FFLJ,FFel=None,FFboltz=None,tipspline=None,bPPdisp=False):
    if tipspline is not None :
        try:
            print " loading tip spline from "+tipspline
            S = np.genfromtxt(tipspline )
            xs   = S[:,0].copy();  print "xs: ",   xs
            ydys = S[:,1:].copy(); print "ydys: ", ydys
            core.setTipSpline( xs, ydys )
            #Ks   = [0.0]
        except:
            print "cannot load tip spline from "+tipspline
            sys.exit()
    core.setFF( FFLJ )
    FF=None
    xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
    FF = FFLJ.copy()
    if ( FFel is not None):
        FF += FFel * PPU.params['charge']
        print "adding charge:", PPU.params['charge']
    if FFboltz != None :
        FF += FFboltz
#    GU.save_vec_field( 'FF', FF, lvec)
    core.setFF_Fpointer( FF )
    print "stiffness:", PPU.params['klat']
    core.setTip( kSpring = np.array((PPU.params['klat'],PPU.params['klat'],0.0))/-PPU.eVA_Nm )
    fzs,PPpos = relaxedScan3D( xTips, yTips, zTips )
    if bPPdisp:
        PPdisp=PPpos.copy()
        init_pos=np.array(np.meshgrid(xTips,yTips,zTips)).transpose(3,1,2,0)+np.array([PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],-PPU.params['r0Probe'][2]])
        PPdisp-=init_pos
    else:
        PPdisp = None
    return fzs,PPpos,PPdisp,lvecScan

