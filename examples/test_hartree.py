#!/usr/bin/python

#import matplotlib
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import elements
import GridUtils as GU
#import XSFutils
import basUtils
import ProbeParticle as PP

from optparse import OptionParser


parser = OptionParser()
parser.add_option(      "--dfrange", action="store", type="float", help="Range of plotted frequency shift (df)", nargs=2)
(options, args) = parser.parse_args()


print " >> WARNING!!! OVEWRITING SETTINGS by params.ini  "

PP.loadParams( 'params.ini' )



Fx,lvec,nDim,head=GU.loadXSF('FFel_x.xsf')
Fy,lvec,nDim,head=GU.loadXSF('FFel_y.xsf')
Fz,lvec,nDim,head=GU.loadXSF('FFel_z.xsf')

PP.params['gridA'] = lvec[ 1,:  ].copy()
PP.params['gridB'] = lvec[ 2,:  ].copy()
PP.params['gridC'] = lvec[ 3,:  ].copy()
PP.params['gridN'] = nDim.copy()

FF   = np.zeros( (nDim[0],nDim[1],nDim[2],3) )
FFLJ = np.zeros( np.shape( FF ) )
FFel = np.zeros( np.shape( FF ) )

FFel[:,:,:,0]=Fx
FFel[:,:,:,1]=Fy
FFel[:,:,:,2]=Fz


cell =np.array([
PP.params['gridA'],
PP.params['gridB'],
PP.params['gridC'],
]).copy() 
gridN = PP.params['gridN']


print "cell", cell



PP.setFF( FF, cell  )


print " # ============ define atoms "


atoms    = basUtils.loadAtoms('input.xyz', elements.ELEMENT_DICT )
Rs       = np.array([atoms[1],atoms[2],atoms[3]]);  
iZs      = np.array( atoms[0])

if not PP.params['PBC' ]:
	print " NO PBC => autoGeom "
	PP.autoGeom( Rs, shiftXY=True,  fitCell=True,  border=3.0 )
	print " NO PBC => params[ 'gridA'   ] ", PP.params[ 'gridA' ] 
	print " NO PBC => params[ 'gridB'   ] ", PP.params[ 'gridB'   ]
	print " NO PBC => params[ 'gridC'   ] ", PP.params[ 'gridC'   ]
	print " NO PBC => params[ 'scanMin' ] ", PP.params[ 'scanMin' ]
	print " NO PBC => params[ 'scanMax' ] ", PP.params[ 'scanMax' ]


'''
Rs[0] += PP.params['moleculeShift' ][0]          # shift molecule so that we sample reasonable part of potential 
Rs[1] += PP.params['moleculeShift' ][1]          
Rs[2] += PP.params['moleculeShift' ][2]          
'''

Rs     = np.transpose( Rs, (1,0) ).copy() 

Qs = np.array( atoms[4] )

if PP.params['PBC' ]:
	iZs,Rs,Qs = PP.PBCAtoms( iZs, Rs, Qs, avec=PP.params['gridA'], bvec=PP.params['gridB'] )

print "shape( Rs )", np.shape( Rs ); 
#print "Rs : ",Rs








print " # ============ define Scan and allocate arrays   - do this before simulation, in case it will crash "

dz    = PP.params['scanStep'][2]
zTips = np.arange( PP.params['scanMin'][2], PP.params['scanMax'][2]+0.00001, dz )[::-1];
ntips = len(zTips); 
print " zTips : ",zTips
rTips = np.zeros((ntips,3))
rs    = np.zeros((ntips,3))
fs    = np.zeros((ntips,3))

rTips[:,0] = 1.0
rTips[:,1] = 1.0
rTips[:,2] = zTips 

PP.setTip()

xTips  = np.arange( PP.params['scanMin'][0], PP.params['scanMax'][0]+0.00001, 0.1 )
yTips  = np.arange( PP.params['scanMin'][1], PP.params['scanMax'][1]+0.00001, 0.1 )
extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
fzs    = np.zeros(( len(zTips), len(yTips ), len(xTips ) ));

nslice = 10;


fname=os.path.dirname(sys.argv[0])
dname=os.path.dirname(fname)
FFparams = PP.loadSpecies( dname+'/code/defaults/atomtypes.ini'  )
#FFparams = PP.loadSpecies        ( 'atomtypes.ini'  )
C6,C12   = PP.getAtomsLJ( PP.params['probeType'], iZs, FFparams )

print " # ============ define Grid "

cell =np.array([
PP.params['gridA'],
PP.params['gridB'],
PP.params['gridC'],
]).copy() 

gridN = PP.params['gridN']







#quit()

# ==============================================
#   The costly part of simulation starts here
# ==============================================

print " # =========== Sample LenardJones "

PP.setFF( FF, cell  )
PP.setFF_Pointer( FF )
PP.getLenardJonesFF( Rs, C6, C12 )

plt.figure(figsize=( 5*nslice,5 )); plt.title( ' FF LJ ' )
for i in range(nslice):
	plt.subplot( 1, nslice, i+1 )
	plt.imshow( FF[i,:,:,2], origin='image', interpolation='nearest' )


withElectrostatics = ( abs( PP.params['charge'] )>0.001 )
if withElectrostatics: 
	print " # =========== Sample Coulomb "
	FF += FFel*PP.params['charge']
	PP.setFF_Pointer( FF )


del FFel


plt.figure(figsize=( 5*nslice,5 )); plt.title( ' FF total ' )
for i in range(nslice):
	plt.subplot( 1, nslice, i+1 )
	plt.imshow( FF[i,:,:,2], origin='image', interpolation='nearest' )



print " # ============  Relaxed Scan 3D "

for ix,x in enumerate( xTips  ):
	print "relax ix:", ix
	rTips[:,0] = x
	for iy,y in enumerate( yTips  ):
		rTips[:,1] = y
		itrav = PP.relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
		fzs[:,iy,ix] = fs[:,2].copy()
		#print itrav
		#if itrav > 100:
		#	print " bad convergence > %i iterations per pixel " % itrav
		#	print " exiting "
		#	break
		

print " # ============  convert Fz -> df "

dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=int(PP.params['Amplitude']/dz) )

print " # ============  Plot Relaxed Scan 3D "

#slices = range( PP.params['plotSliceFrom'], PP.params['plotSliceTo'], PP.params['plotSliceBy'] )
#print "plotSliceFrom, plotSliceTo, plotSliceBy : ", PP.params['plotSliceFrom'], PP.params['plotSliceTo'], PP.params['plotSliceBy']
#print slices 
#nslice = len( slices )

slices = range( 0, len(dfs) )

for ii,i in enumerate(slices):
	print " plotting ", i
	plt.figure( figsize=( 10,10 ) )
	if(options.dfrange != None):
		fmin = options.dfrange[0]
		fmax = options.dfrange[1]
		plt.imshow( dfs[i], origin='image', interpolation=PP.params['imageInterpolation'], vmin=fmin, vmax=fmax, cmap=PP.params['colorscale'], extent=extent )
	else:
		plt.imshow( dfs[i], origin='image', interpolation=PP.params['imageInterpolation'], cmap=PP.params['colorscale'], extent=extent )
	z = zTips[i] - PP.params['moleculeShift' ][2]
	plt.colorbar();
	plt.xlabel(r' Tip_x $\AA$')
	plt.ylabel(r' Tip_y $\AA$')
	plt.title( r"df Tip_z = %2.2f $\AA$" %z  )
	plt.savefig( 'df_%04i.png' %i, bbox_inches='tight' )


print " ***** ALL DONE ***** "
