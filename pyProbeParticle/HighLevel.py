#!/usr/bin/python

import numpy as np
import os
import GridUtils as GU
import fieldFFT
import common as PPU

import core
import cpp_utils


def parseAtoms( atoms, autogeom = False, PBC = True ):
	Rs       = np.array([atoms[1],atoms[2],atoms[3]]);  
	iZs      = np.array( atoms[0] )
	if autogeom:
		print " autoGeom "
		PPU.autoGeom( Rs, shiftXY=True,  fitCell=True,  border=3.0 )
	Rs = np.transpose( Rs, (1,0) ).copy() 
	Qs = np.array( atoms[4] )
	if PBC:
		iZs,Rs,Qs = PPU.PBCAtoms( iZs, Rs, Qs, avec=PPU.params['gridA'], bvec=PPU.params['gridB'] )
	return iZs,Rs,Qs

def computeLJ( Rs, iZs, FFLJ=None, FFparams=None ):
	if ( FFLJ is None ):
                print "Here"
		gridN = PPU.params['gridN']
		FFLJ = np.zeros( (gridN[2],gridN[1],gridN[0],3)    )
	else:
		PPU.params['gridN'] = np.shape( FFLJ )	
#	iZs,Rs,Qs = parseAtoms( )
	if FFparams is None:
		FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
	C6,C12   = PPU.getAtomsLJ( PPU.params['probeType'], iZs, FFparams )
	core.setFF( FFLJ )
	core.getLenardJonesFF( Rs, C6, C12 )
	return FFLJ

def computeCoulomb( Rs, Qs, FFel=None ):
	if ( FFel is None ):
		gridN = PPU.params['gridN']
		FFel = np.zeros( (gridN[0],gridN[1],gridN[2],3)    )
	else:
		PPU.params['gridN'] = np.shape( FFel )	
	core.setFF( FFel )
	FFel  = core.getCoulombFF ( Rs, Qs * CoulombConst )
	return FFel

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
		atoms     = basUtils.loadAtoms('geom.bas', elements.ELEMENT_DICT )
		iZs,Rs,Qs = parseAtoms( atoms, autogeom = autogeom, PBC = PPU.params['PBC'] )
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
		print "relax ix:", ix
		rTips[:,0] = x
		for iy,y in enumerate( yTips  ):
			rTips[:,1] = y
			itrav = core.relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
			fzs[:,iy,ix] = (fs[:,2].copy()) [::-1]
			PPpos[:,iy,ix,0] = rs[::-1,0] # - rTips[:,0]
			PPpos[:,iy,ix,1] = rs[::-1,1] # - rTips[:,1]
			PPpos[:,iy,ix,2] = rs[::-1,2] # - rTips[:,2]
	return fzs,PPpos

