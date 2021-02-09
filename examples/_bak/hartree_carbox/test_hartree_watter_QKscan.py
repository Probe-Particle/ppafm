#!/usr/bin/python

#import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import os
import numpy as np
import matplotlib.pyplot as plt
import elements
#import XSFutils
import basUtils


from memory_profiler import profile




print(" # ========== make & load  ProbeParticle C++ library ") 

def makeclean( ):
	import os
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]

makeclean( )  # force to recompile 

import ProbeParticle as PP

print(" >> WARNING!!! OVEWRITING SETTINGS by params.ini  ")

PP.loadParams( 'params_watter.ini' )

#Fx,lvec,nDim,head=XSFutils.loadXSF('Fx.xsf')
#Fy,lvec,nDim,head=XSFutils.loadXSF('Fy.xsf')
#Fz,lvec,nDim,head=XSFutils.loadXSF('Fz.xsf')

Fx,lvec,nDim,head=PP.loadXSF('Fx.xsf')
Fy,lvec,nDim,head=PP.loadXSF('Fy.xsf')
Fz,lvec,nDim,head=PP.loadXSF('Fz.xsf')

PP.params['gridA'] = lvec[ 1,:  ].copy()
PP.params['gridB'] = lvec[ 2,:  ].copy()
PP.params['gridC'] = lvec[ 3,:  ].copy()
PP.params['gridN'] = nDim.copy()

FFLJ  = np.zeros( (nDim[0],nDim[1],nDim[2],3) )
FFel  = np.zeros( np.shape( FFLJ ) )

FFel[:,:,:,0]=Fx
FFel[:,:,:,1]=Fy
FFel[:,:,:,2]=Fz

del Fx; del Fy; del Fz


cell =np.array([
PP.params['gridA'],
PP.params['gridB'],
PP.params['gridC'],
]).copy() 
gridN = PP.params['gridN']

print("cell", cell)

PP.setFF( FFLJ, cell  )

print(" # ============ define atoms ")

atoms    = basUtils.loadAtoms('watter4NaCl-2.xyz')
Rs       = np.array([atoms[1],atoms[2],atoms[3]]);  
iZs      = np.array( atoms[0])

if not PP.params['PBC' ]:
	print(" NO PBC => autoGeom ")
	PP.autoGeom( Rs, shiftXY=True,  fitCell=True,  border=3.0 )
	print(" NO PBC => params[ 'gridA'   ] ", PP.params[ 'gridA' ]) 
	print(" NO PBC => params[ 'gridB'   ] ", PP.params[ 'gridB'   ])
	print(" NO PBC => params[ 'gridC'   ] ", PP.params[ 'gridC'   ])
	print(" NO PBC => params[ 'scanMin' ] ", PP.params[ 'scanMin' ])
	print(" NO PBC => params[ 'scanMax' ] ", PP.params[ 'scanMax' ])

Rs = np.transpose( Rs, (1,0) ).copy() 
Qs = np.array( atoms[4] )

if PP.params['PBC' ]:
	iZs,Rs,Qs = PP.PBCAtoms( iZs, Rs, Qs, avec=PP.params['gridA'], bvec=PP.params['gridB'] )

print("shape( Rs )", np.shape( Rs )); 

print(" # ============ define Scan and allocate arrays   - do this before simulation, in case it will crash ")

dz    = PP.params['scanStep'][2]
zTips = np.arange( PP.params['scanMin'][2], PP.params['scanMax'][2]+0.00001, dz )[::-1];


PP.setTip()

xTips  = np.arange( PP.params['scanMin'][0], PP.params['scanMax'][0]+0.00001, 0.1 )
yTips  = np.arange( PP.params['scanMin'][1], PP.params['scanMax'][1]+0.00001, 0.1 )
extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )





lvecScan =np.array([
PP.params['scanMin'],
[        PP.params['scanMax'][0],0.0,0.0],
[0.0,    PP.params['scanMax'][1],0.0    ],
[0.0,0.0,PP.params['scanMax'][2]        ]
]).copy() 

headScan='''
ATOMS
 1   0.0   0.0   0.0

BEGIN_BLOCK_DATAGRID_3D                        
   some_datagrid      
   BEGIN_DATAGRID_3D 
'''


nslice = 10;

#quit()

# ==============================================
#   The costly part of simulation starts here
# ==============================================

print(" # =========== Sample LenardJones ")

#xsfLJ       = True
xsfLJ       = False
recomputeLJ = True

if (     xsfLJ  and os.path.isfile('FFLJ_x.xsf')):
	recomputeLJ = False
if ((not xsfLJ) and os.path.isfile('FFLJ_y.npy')):
	recomputeLJ = False

if recomputeLJ:
	FFparams = PP.loadSpecies        ( 'atomtypes.ini'  )
	C6,C12   = PP.getAtomsLJ( PP.params['probeType'], iZs, FFparams )
	PP.setFF( FFLJ, cell  )
	PP.setFF_Pointer( FFLJ )
	PP.getLenardJonesFF( Rs, C6, C12 )
	if xsfLJ:
		PP.saveXSF('FFLJ_x.xsf', head, lvec, FFLJ[:,:,:,0] )
		PP.saveXSF('FFLJ_y.xsf', head, lvec, FFLJ[:,:,:,1] )
		PP.saveXSF('FFLJ_z.xsf', head, lvec, FFLJ[:,:,:,2] )
	else:
		np.save('FFLJ_x.npy', FFLJ[:,:,:,0] )
		np.save('FFLJ_y.npy', FFLJ[:,:,:,1] )
		np.save('FFLJ_z.npy', FFLJ[:,:,:,2] )
else:
	if xsfLJ:
		FFLJ[:,:,:,0],lvec,nDim,head=PP.loadXSF('FFLJ_x.xsf')
		FFLJ[:,:,:,1],lvec,nDim,head=PP.loadXSF('FFLJ_y.xsf')
		FFLJ[:,:,:,2],lvec,nDim,head=PP.loadXSF('FFLJ_z.xsf')
	else:
		FFLJ[:,:,:,0] = np.load('FFLJ_x.npy' )
		FFLJ[:,:,:,1] = np.load('FFLJ_y.npy' )
		FFLJ[:,:,:,2] = np.load('FFLJ_z.npy' )

# ======= plot 
#plt.figure(figsize=( 5*nslice,5 )); plt.title( ' FF LJ ' )
#for i in range(nslice):
#	plt.subplot( 1, nslice, i+1 )
#	plt.imshow( FF[i,:,:,2], origin='image', interpolation='nearest' )

#@profile
def relaxedScan3D( xTips, yTips, zTips ):
	ntips = len(zTips); 
	print(" zTips : ",zTips)
	rTips = np.zeros((ntips,3))
	rs    = np.zeros((ntips,3))
	fs    = np.zeros((ntips,3))
	rTips[:,0] = 1.0
	rTips[:,1] = 1.0
	rTips[:,2] = zTips 
	fzs    = np.zeros(( len(zTips), len(yTips ), len(xTips ) ));
	for ix,x in enumerate( xTips  ):
		print("relax ix:", ix)
		rTips[:,0] = x
		for iy,y in enumerate( yTips  ):
			rTips[:,1] = y
			itrav = PP.relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
			fzs[:,iy,ix] = fs[:,2].copy()
	return fzs

#@profile
def plotImages( prefix, F, slices ):
	for ii,i in enumerate(slices):
		print(" plotting ", i)
		plt.figure( figsize=( 10,10 ) )
		plt.imshow( F[i], origin='image', interpolation=PP.params['imageInterpolation'], cmap=PP.params['colorscale'], extent=extent )
		z = zTips[i] - PP.params['moleculeShift' ][2]
		plt.colorbar();
		plt.xlabel(r' Tip_x $\AA$')
		plt.ylabel(r' Tip_y $\AA$')
		plt.title( r"Tip_z = %2.2f $\AA$" %z  )
		plt.savefig( prefix+'_%3.3i.png' %i, bbox_inches='tight' )
		plt.close()

#Ks   = [ 0.5  ]
#Qs   = [ -0.2 ]
#Amps = [ 0.5, 1.0, 2.0, 4.0 ]


Ks   = [ 0.25, 0.5, 1.0 ]
Qs   = [ -0.2, 0.0, +0.2 ]
Amps = [ 2.0 ]

#@profile
def main():
	for iq,Q in enumerate( Qs ):
		FF = FFLJ + FFel * Q
		PP.setFF_Pointer( FF )
		for ik,K in enumerate( Ks ):
			dirname = "Q%1.2fK%1.2f" %(Q,K)
			os.makedirs( dirname )
			PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )
			fzs = relaxedScan3D( xTips, yTips, zTips )
			PP.saveXSF( dirname+'/OutFz.xsf', headScan, lvecScan, fzs )
			for iA,Amp in enumerate( Amps ):
				AmpStr = "/Amp%2.2f" %Amp
				print("Amp= ",AmpStr)
				os.makedirs( dirname+AmpStr )
				dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=Amp/dz )
				plotImages( dirname+AmpStr+"/df", dfs, slices = list(range( 0, len(dfs))) )

main()

print(" ***** ALL DONE ***** ")

#plt.show()




