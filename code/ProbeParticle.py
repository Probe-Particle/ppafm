#!/usr/bin/python

import numpy as np
from   ctypes import c_int, c_double, c_char_p
import ctypes
import os
import GridUtils as GU
import libFFTfin
import ProbeParticleUtils as PPU

# ====================== constants

# ==============================
# ============================== Pure python functions
# ==============================

LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
print " ProbeParticle Library DIR = ", LIB_PATH

def autoGeom( Rs, shiftXY=False, fitCell=False, border=3.0 ):
	'''
	set Force-Filed and Scanning supercell to fit optimally given geometry
	then shifts the geometry in the center of the supercell
	'''
	zmax=max(Rs[2]); 	Rs[2] -= zmax
	print " autoGeom substracted zmax = ",zmax
	xmin=min(Rs[0]); xmax=max(Rs[0])
	ymin=min(Rs[1]); ymax=max(Rs[1])
	if fitCell:
		PPU.params[ 'gridA' ][0] = (xmax-xmin) + 2*border
		PPU.params[ 'gridA' ][1] = 0
		PPU.params[ 'gridB' ][0] = 0
		PPU.params[ 'gridB' ][1] = (ymax-ymin) + 2*border
		PPU.params[ 'scanMin' ][0] = 0
		PPU.params[ 'scanMin' ][1] = 0
		PPU.params[ 'scanMax' ][0] = PPU.params[ 'gridA' ][0]
		PPU.params[ 'scanMax' ][1] = PPU.params[ 'gridB' ][1]
		print " autoGeom changed cell to = ", PPU.params[ 'scanMax' ]
	if shiftXY:
		dx = -0.5*(xmin+xmax) + 0.5*( PPU.params[ 'gridA' ][0] + PPU.params[ 'gridB' ][0] ); Rs[0] += dx
		dy = -0.5*(ymin+ymax) + 0.5*( PPU.params[ 'gridA' ][1] + PPU.params[ 'gridB' ][1] ); Rs[1] += dy;
		print " autoGeom moved geometry by ",dx,dy

def PBCAtoms( Zs, Rs, Qs, avec, bvec, na=None, nb=None ):
	'''
	multiply atoms of sample along supercell vectors
	the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
	'''
	Zs_ = []
	Rs_ = []
	Qs_ = []
	if na is None:
		na=PPU.params['nPBC'][0]
	if nb is None:
		nb=PPU.params['nPBC'][1]
	for i in range(-na,na+1):
		for j in range(-nb,nb+1):
			for iatom in range(len(Zs)):
				x = Rs[iatom][0] + i*avec[0] + j*bvec[0]
				y = Rs[iatom][1] + i*avec[1] + j*bvec[1]
				#if (x>xmin) and (x<xmax) and (y>ymin) and (y<ymax):
				Zs_.append( Zs[iatom]          )
				Rs_.append( (x,y,Rs[iatom][2]) )
				Qs_.append( Qs[iatom]          )
	return np.array(Zs_).copy(), np.array(Rs_).copy(), np.array(Qs_).copy()	

def get_C612( i, j, FFparams ):
	'''
	compute Lenard-Jones coefitioens C6 and C12 pair of atoms i,j
	'''
	#print i, j, FFparams[i], FFparams[j]
	Rij = FFparams[i][0] + FFparams[j][0]
	Eij = np.sqrt( FFparams[i][1] * FFparams[j][1] )
	return 2*Eij*(Rij**6), Eij*(Rij**12)

def getAtomsLJ(  iZprobe, iZs,  FFparams ):
	'''
	compute Lenard-Jones coefitioens C6 and C12 for interaction between atoms in list "iZs" and probe-particle "iZprobe"
	'''
	n   = len(iZs)
	C6  = np.zeros(n)
	C12 = np.zeros(n)
	for i in range(n):
		C6[i],C12[i] = get_C612( iZprobe-1, iZs[i]-1, FFparams )
	return C6,C12

# ==============================
# ============================== interface to C++ core 
# ==============================

name='ProbeParticle'
ext='_lib.so'

# recompilation of C++ dynamic librady ProbeParticle_lib.so from ProbeParticle.cpp
def recompile():
	CWD=os.getcwd()
	os.chdir(LIB_PATH)
	os.system("make PP")
	os.chdir(CWD)

# if binary of ProbeParticle_lib.so is deleted => recompile it

#makeclean()

if not os.path.exists(LIB_PATH+"/"+name+ext):
	recompile()

lib    = ctypes.CDLL(LIB_PATH+"/"+name+ext )    # load dynamic librady object using ctypes 

# define used numpy array types for interfacing with C++

array1i = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags='CONTIGUOUS')

# ========
# ======== Python warper function for C++ functions
# ========

# void setFF( int * n, double * grid, double * step,  )
lib.setFF.argtypes = [array1i,array4d,array2d]
lib.setFF.restype  = None
def setFF( grid, cell = None ):
	n_    = np.shape(grid)
	n     = np.array( (n_[2],n_[1],n_[0]) ).astype(np.int32)
	if cell is None:
		cell = np.array([
		PPU.params['gridA'],
		PPU.params['gridB'],
		PPU.params['gridC'],
		]).copy() 
	lib.setFF( n, grid, cell )

# void setFF( int * n, double * grid, double * step,  )
lib.setFF_Pointer.argtypes = [array4d]
lib.setFF_Pointer.restype  = None
def setFF_Pointer( grid ):
	lib.setFF_Pointer( grid )

#void setRelax( int maxIters, double convF2, double dt, double damping )
lib.setRelax.argtypes = [ c_int, c_double, c_double, c_double ]
lib.setRelax.restype  = None
def setRelax( maxIters  = 1000, convF2 = 1.0e-4, dt = 0.1, damping = 0.1 ):
	lib.setRelax( maxIters, convF*convF, dt, damping )

#void setFIRE( double finc, double fdec, double falpha )
lib.setFIRE.argtypes = [ c_double, c_double, c_double ]
lib.setFIRE.restype  = None
def setFIRE( finc = 1.1, fdec = 0.5, falpha  = 0.99 ):
	lib.setFIRE( finc, fdec, falpha )


#void setTip( double lRad, double kRad, double * rPP0, double * kSpring )
lib.setTip.argtypes = [ c_double, c_double, array1d, array1d ]
lib.setTip.restype  = None
def setTip( lRadial=None, kRadial=None, rPP0=None, kSpring=None	):
	if lRadial is None:
		lRadial=PPU.params['r0Probe'][2]
	if kRadial is  None:
		kRadial=PPU.params['stiffness'][2]/-PPU.eVA_Nm
	if rPP0 is  None:
		rPP0=np.array((PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],0.0))
	if kSpring is  None: 
		kSpring=np.array((PPU.params['stiffness'][0],PPU.params['stiffness'][1],0.0))/-PPU.eVA_Nm 
	print " IN setTip !!!!!!!!!!!!!! "
	print " lRadial ", lRadial
	print " kRadial ", kRadial
	print " rPP0 ", rPP0
	print " kSpring ", kSpring
	lib.setTip( lRadial, kRadial, rPP0, kSpring )


# void getClassicalFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getLenardJonesFF.argtypes  = [ c_int,       array2d,      array1d,     array1d     ]
lib.getLenardJonesFF.restype   = None
def getLenardJonesFF( Rs, C6, C12 ):
	natom = len(Rs) 
	lib.getLenardJonesFF( natom, Rs, C6, C12 )

# void getCoulombFF       (    int natom,   double * Rs_, double * C6, double * C12 )
lib.getCoulombFF.argtypes  = [ c_int,       array2d,      array1d   ]
lib.getCoulombFF.restype   = None
def getCoulombFF( Rs, kQQs ):
	natom = len(Rs) 
	lib.getCoulombFF( natom, Rs, kQQs )

# int relaxTipStroke ( int probeStart, int nstep, double * rTips_, double * rs_, double * fs_ )
lib.relaxTipStroke.argtypes  = [ c_int, c_int, c_int,  array2d, array2d, array2d ]
lib.relaxTipStroke.restype   = c_int
def relaxTipStroke( rTips, rs, fs, probeStart=1, relaxAlg=1 ):
	n = len(rTips) 
	return lib.relaxTipStroke( probeStart, relaxAlg, n, rTips, rs, fs )


# ============= Hi-Level Macros

def parseAtoms( atoms, autogeom = False, PBC = True ):
	Rs       = np.array([atoms[1],atoms[2],atoms[3]]);  
	iZs      = np.array( atoms[0] )
	if autogeom:
		print " autoGeom "
		autoGeom( Rs, shiftXY=True,  fitCell=True,  border=3.0 )
	Rs = np.transpose( Rs, (1,0) ).copy() 
	Qs = np.array( atoms[4] )
	if PBC:
		iZs,Rs,Qs = PBCAtoms( iZs, Rs, Qs, avec=PPU.params['gridA'], bvec=PPU.params['gridB'] )
	return iZs,Rs,Qs

def computeLJ( Rs, iZs, FFLJ=None, FFparams=None ):
	if ( FFLJ is None ):
		gridN = PPU.params['gridN']
		FFLJ = np.zeros( (gridN[0],gridN[1],gridN[2],3)    )
	else:
		PPU.params['gridN'] = np.shape( FFLJ )	
#	iZs,Rs,Qs = parseAtoms( )
	if FFparams is None:
		FFparams = PPU.loadSpecies( LIB_PATH+'/defaults/atomtypes.ini' )
	C6,C12   = getAtomsLJ( PPU.params['probeType'], iZs, FFparams )
	setFF( FFLJ )
	getLenardJonesFF( Rs, C6, C12 )
	return FFLJ

def computeCoulomb( Rs, Qs, FFel=None ):
	if ( FFel is None ):
		gridN = PPU.params['gridN']
		FFel = np.zeros( (gridN[0],gridN[1],gridN[2],3)    )
	else:
		PPU.params['gridN'] = np.shape( FFel )	
	setFF( FFel )
	FFel  = getCoulombFF ( Rs, Qs * CoulombConst )
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
			FFel_x,FFel_y,FFel_z = libFFTfin.potential2forces( V, lvecEl, nDim, sigma = 1.0 )
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
			itrav = relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
			fzs[:,iy,ix] = (fs[:,2].copy())[::-1]
			PPpos[:,iy,ix,0] = rs[::-1,0] # - rTips[:,0]
			PPpos[:,iy,ix,1] = rs[::-1,1] # - rTips[:,1]
			PPpos[:,iy,ix,2] = rs[::-1,2] # - rTips[:,2]
	return fzs,PPpos

