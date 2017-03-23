#!/usr/bin/python

import os
import sys
import numpy as np
import GridUtils as GU
#import fieldFFT
import fieldFFT       as fFFT
import common as PPU

import core
import cpp_utils

# ===== constants 
Fmax_DEFAULT = 100.0



def parseAtoms( atoms, autogeom = False, PBC = True, FFparams=None ):
	if FFparams is None:
		raise ValueError("You should provide a list of LJ parameters!")
	Rs = np.array([atoms[1],atoms[2],atoms[3]]); 
        Natoms=[]
        elem_dict={}
        for i,ff in enumerate(FFparams):
                elem_dict[ff[3]] = i+1
        for atm in atoms[0]:
                try:
                        Natoms.append(int(atm))
                except:
                        try:
                                Natoms.append(elem_dict[atm])
                        except:
                                raise ValueError("Did not find atomkind: "
                                "{}".format(atm))
	iZs=np.array( Natoms )
	if autogeom:
		print " autoGeom "
		PPU.autoGeom( Rs, shiftXY=True,  fitCell=True,  border=3.0 )
	Rs = np.transpose( Rs, (1,0) ).copy()
	Qs = np.array( atoms[4] )
	if PBC:
		iZs,Rs,Qs = PPU.PBCAtoms( iZs, Rs, Qs, avec=PPU.params['gridA'], bvec=PPU.params['gridB'] )
	return iZs,Rs,Qs


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
	C6,C12   = PPU.getAtomsLJ( PPU.params['probeType'], iZs, FFparams )
	#core.setFF( gridF=FFLJ, gridE=VLJ )
	core.getLenardJonesFF( Rs, C6, C12 )
	return FFLJ, VLJ

def computeCoulomb( Rs, Qs, FFel=None , Vpot=False ):
	FFel,Vel = perpareArrays( FFel, Vpot )
	#core.setFF( gridF=FFel, gridE=Vel )
	core.getCoulombFF ( Rs, Qs * PPU.CoulombConst )
	return FFel, Vel

"""
def prepareForceFields( store = True, storeXsf = False, autogeom = False, FFparams=None ):
	newEl = False
	newLJ = False
	head = None
	# --- try to load FFel or compute it from LOCPOT.xsf
	if ( os.path.isfile('FFel_x.xsf') ):
		print " FFel_x.xsf found "
		FFel, lvecEl, nDim, head = GU.loadVecField('FFel', FFel)
		PPU.lvec2params( lvecEl )
	else:
		print "F Fel_x.xsf not found "
		if ( xsfLJ  and os.path.isfile('LOCPOT.xsf') ):
			print " LOCPOT.xsf found "
			V, lvecEl, nDim, head = GU.loadXSF('LOCPOT.xsf')
			PPU.lvec2params( lvecEl )
			FFel_x,FFel_y,FFel_z = fieldFFT.potential2forces( V, lvecEl, nDim, sigma = 1.0 )
			FFel = GU.packVecGrid( FFel_x,FFel_y,FFel_z )
			del FFel_x,FFel_y,FFel_z
			GU.saveVecFieldXsf( 'FFel', FF, lvecEl, head = head )
		else:
			print " LOCPOT.xsf not found "
			newEl = True
	# --- try to load FFLJ 
	if ( os.path.isfile('FFLJ_x.xsf') ):
		print " FFLJ_x.xsf found "
		FFLJ, lvecLJ, nDim, head = GU.loadVecFieldXsf( 'FFLJ' )
		PPU.lvec2params( lvecLJ )
	else: 
		newLJ = True
	# --- compute Forcefield by atom-wise interactions 
	if ( newEl or newEl ):
                atoms     = basUtils.loadAtoms('geom.bas')
		iZs,Rs,Qs = parseAtoms( atoms, autogeom = autogeom, PBC =
                PPU.params['PBC'], FFparams = FFparams )
		lvec = PPU.params2lvec( )
		if head is None:
			head = GU.XSF_HEAD_DEFAULT
		if newLJ:
			FFLJ = computeLJ     ( Rs, iZs, FFparams=FFparams )
			GU.saveVecFieldXsf( 'FFLJ', FF, lvecEl, head = head )
		if newEl:
			FFel = computeCoulomb( Rs, Qs, FFel )
			GU.saveVecFieldXsf( 'FFel', FF, lvecEl, head = head )
	return FFLJ, FFel
"""		

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
        print ""
	return fzs,PPpos

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

def computeELFF_pch(iZs,Rs,Qs,computeVpot, Fmax=Fmax_DEFAULT ):
    print " ========= get electrostatic forcefiled from the point charges "
    FFel, V = computeCoulomb( Rs, Qs, FFel=None, Vpot=computeVpot  )
    if Fmax is not  None:
        print "Limit vector field"
        GU.limit_vec_field( FFel, Fmax=Fmax )
    if computeVpot :
        Vmax = 10.0; V[ V>Vmax ] = Vmax
    return FFel,V

def computeElFF(V,lvec,nDim,tip,Fmax=None,computeVpot=False,Vmax=None):
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
    Fel_x,Fel_y,Fel_z, Vout = fFFT.potential2forces_mem( V, lvec, nDim, rho=rho, sigma=PPU.params['sigma'], multipole = multipole )
    FFel = GU.packVecGrid(Fel_x,Fel_y,Fel_z)
    del Fel_x,Fel_y,Fel_z
    return FFel

def perform_relaxation (lvec,FFLJ,FFel=None,FFboltz=None,tipspline=None):
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
    PPdisp=PPpos.copy()
    init_pos=np.array(np.meshgrid(xTips,yTips,zTips)).transpose(3,1,2,0)+np.array([PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],-PPU.params['r0Probe'][2]])
    PPdisp-=init_pos
    return fzs,PPpos,PPdisp,lvecScan


