#!/usr/bin/python

import os
import sys
import numpy as np
import GridUtils as GU
#import fieldFFT
import fieldFFT       as fFFT
import common as PPU
#from scipy.interpolate import RegularGridInterpolator

import core
import cpp_utils


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


def prepareArrays( FFC, FFO, Vpot ):
	if ( FFC is None ):
		gridN = PPU.params['gridN']
		FFC = np.zeros( (gridN[2],gridN[1],gridN[0],3)    )
	else:
		PPU.params['gridN'] = np.shape( FFC )	
	if ( FFO is None ):
		gridN = PPU.params['gridN']
		FFO = np.zeros( (gridN[2],gridN[1],gridN[0],3)    )
	else:
		PPU.params['gridN'] = np.shape( FFO )	
	if ( Vpot ):
		VC = np.zeros( (gridN[2],gridN[1],gridN[0])    )
		VO = np.zeros( (gridN[2],gridN[1],gridN[0])    )
	else:
		VC=None
		VO=None
	core.setFFC( gridF=FFC, gridE=VC )
	core.setFFO( gridF=FFO, gridE=VO )
	return FFC,VC,FFO,VO 

def computeLJ( Rs, iZs, FFLJC=None,FFLJO=None,FFparams=None, Vpot=False ):
	if FFparams is None:
		raise ValueError("You should provide a list of LJ parameters!")
	FFLJC,VLJC,FFLJO,VLJO = prepareArrays( FFLJC,FFLJO, Vpot )
	C6,C12   = PPU.getAtomsLJ( PPU.params['Catom'], iZs, FFparams )
	#print "C6", C6
	#print "Catom", PPU.params['Catom']
	core.getCLenardJonesFF( Rs, C6, C12 )
	C6,C12   = PPU.getAtomsLJ( PPU.params['Oatom'], iZs, FFparams )
	#print "C6", C6
	#print "Oatom", PPU.params['Oatom']
	#sys.exit()
	core.getOLenardJonesFF( Rs, C6, C12 )
	return FFLJC,VLJC,FFLJO,VLJO

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
	rCs    = np.zeros((ntips,3))
	rOs    = np.zeros((ntips,3))
	fCs    = np.zeros((ntips,3))
	fOs    = np.zeros((ntips,3))
	rTips[:,0] = 1.0
	rTips[:,1] = 1.0
	rTips[:,2] = zTips[::-1]  
	nx = len(zTips); ny = len(yTips ); nz = len(xTips);
	fzs    = np.zeros( ( nx,ny,nz ) );
	PPCpos  = np.zeros( ( nx,ny,nz,3 ) );
	PPOpos  = np.zeros( ( nx,ny,nz,3 ) );
	for ix,x in enumerate( xTips  ):
		sys.stdout.write('\033[K')
		sys.stdout.flush()
		sys.stdout.write("\rrelax ix: {}".format(ix))
		sys.stdout.flush()
		rTips[:,0] = x
		for iy,y in enumerate( yTips  ):
			rTips[:,1] = y
			itrav = core.relaxTipStroke( rTips, rCs, rOs, fCs, fOs) / float( len(zTips) )
			fzs[:,iy,ix] = ((fCs+fOs)[:,2].copy()) [::-1]
			PPCpos[:,iy,ix,0] = rCs[::-1,0] # - rTips[:,0]
			PPCpos[:,iy,ix,1] = rCs[::-1,1] # - rTips[:,1]
			PPCpos[:,iy,ix,2] = rCs[::-1,2] # - rTips[:,2]
			PPOpos[:,iy,ix,0] = rOs[::-1,0] # - rTips[:,0]
			PPOpos[:,iy,ix,1] = rOs[::-1,1] # - rTips[:,1]
			PPOpos[:,iy,ix,2] = rOs[::-1,2] # - rTips[:,2]
        print ""
	return fzs,PPCpos,PPOpos

def computeLJFF(iZs, Rs, FFparams, Fmax=None, computeVpot=False, Vmax=None):
    print "--- Compute Lennard-Jones Force-filed ---"
    FFLJC,VLJC,FFLJO,VLJO=computeLJ( Rs, iZs, FFLJC=None, FFLJO=None, FFparams=FFparams,   # This function computes the LJ forces experienced 
                            Vpot=computeVpot)                        # by the ProbeParticle
    if Fmax is not  None:
        print "Limit vector field"
        GU.limit_vec_field( FFLJ, Fmax=Fmax )
        # remove too large values; keeps the same
        # direction; good for the visualization 
    if  Vmax != None and VLJ != None:
    	VLJ[ VLJ > Vmax ] =  Vmax # remove too large values
    return FFLJC,VLJC,FFLJO,VLJO


def computeElFF(V,lvec,nDim,tip,sigma,Fmax=None,computeVpot=False,Vmax=None):
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
    Fel_x,Fel_y,Fel_z = fFFT.potential2forces(V, lvec, nDim, rho=rho, 
    sigma=sigma, multipole = multipole)
    FFel = GU.packVecGrid(Fel_x,Fel_y,Fel_z)
    del Fel_x,Fel_y,Fel_z
    return FFel

def computeCoulomb( Rs, Qs, FFel=None , Vpot=False ):
	tmp1, tmp2, FFel,Vel = prepareArrays( FFel, FFel, Vpot ); del tmp1, tmp2;
	#core.setFF( gridF=FFel, gridE=Vel )
	core.getOCoulombFF ( Rs, Qs * PPU.CoulombConst )
	return FFel, Vel

def computeELFF_pch(iZs,Rs,Qs,computeVpot, Fmax=10.0 ):
    print " ========= get electrostatic forcefiled from the point charges "
    FFel, V = computeCoulomb( Rs, Qs, FFel=None, Vpot=computeVpot  )
    if Fmax is not  None:
        print "Limit vector field"
        GU.limit_vec_field( FFel, Fmax=Fmax )
    if computeVpot :
        Vmax = 10.0; V[ V>Vmax ] = Vmax
    return FFel,V


def getZTipForce(xTips, yTips, zTips, FFtipZ,shiftTz=0.0):
    zTips+=shiftTz
#    print zTip,s
    scangrid=np.array(np.meshgrid(xTips, yTips,
                      zTips)).transpose([3,1,2,0]).copy()
#    print scangrid
    cell=np.array([PPU.params['gridA'],PPU.params['gridB'],PPU.params['gridC']]).copy()
    FtipRes=GU.interpolate_cartesian(FFtipZ,scangrid,cell)

    return FtipRes



def perform_relaxation(lvec,FFLJC,FFLJO=None,FFel=None,FFTip=None,
    FFboltz=None,tipspline=None):
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
    core.setFFC( FFLJC )
    core.setFFO( FFLJO )
    FFC=None
    FFO=None
    xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
    FFC = FFLJC.copy()
    FFO = FFLJO.copy()
    if ( FFel is not None):
        FFC += FFel * PPU.params['Ccharge']
        print "adding C charge:", PPU.params['Ccharge']
        FFO += FFel * PPU.params['Ocharge']
        print "adding O charge:", PPU.params['Ocharge']
    core.setFFC_Fpointer( FFC )
    core.setFFO_Fpointer( FFO )
    print "C stiffness:", PPU.params['Cklat']
    print "O stiffness:", PPU.params['Oklat']
    core.setTip( CkSpring =
    np.array((PPU.params['Cklat'],PPU.params['Cklat'],0.0))*-1.0,
    OkSpring =
    np.array((PPU.params['Oklat'],PPU.params['Oklat'],0.0))*-1.0 )


    fzs,PPCpos,PPOpos = relaxedScan3D( xTips, yTips, zTips )
    if FFTip is not None and ((PPU.params['tip'] != None)and(PPU.params['tip'] != 'None')and(PPU.params['tip'] != "None")):
        print "Adding the metallic tip vertical force according to tipcharge "
        FFTip*=PPU.params['tipcharge']
        fzs+= getZTipForce(xTips, yTips, zTips, FFTip,shiftTz=PPU.params['tipZdisp'])
        print "Finished with adding the metallic tip vertical force"
    elif FFTip is not None:
        print "Adding the metallic tip vertical force from Cu charges"
        fzs += getZTipForce(xTips, yTips, zTips, FFTip*PPU.params['ChargeCuDown'])
        fzs += getZTipForce(xTips, yTips, zTips, FFTip*PPU.params['ChargeCuUp'],shiftTz=PPU.params['CuUpshift'] )
        print "Finished with adding the metallic tip vertical force"
#    PPdisp=PPpos.copy()
#    init_pos=np.array(np.meshgrid(xTips,yTips,zTips)).transpose(3,1,2,0)+np.array([PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],-PPU.params['r0Probe'][2]])
#    PPdisp-=init_pos
    return fzs,PPCpos,PPOpos,lvecScan


