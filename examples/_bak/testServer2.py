#!/usr/bin/python

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import elements

#print dir( elements ) 


import basUtils

print(" # ========== make & load  ProbeParticle C++ library ") 

def makeclean( ):
    import os
    [ os.remove(f) for f in os.listdir(".") if f.endswith(".so") ]
    [ os.remove(f) for f in os.listdir(".") if f.endswith(".o") ]
    [ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]

#makeclean( )  # force to recompile 

import  ProbeParticle as PP

print(" # ==========  server interface file I/O ")

PP.loadParams( 'params.ini' )

print(" # ============ define atoms ")

#bas      = basUtils.loadBas('surf.bas')[0]
#bas      = basUtils.loadBas('PTCDA_Ruslan_1x1.bas')[0]
#bas      = basUtils.loadBas('GrN6x6.bas')[0]

atoms    = basUtils.loadAtoms('input.xyz')
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


#Rs[0] += PP.params['moleculeShift' ][0]          # shift molecule so that we sample reasonable part of potential 
#Rs[1] += PP.params['moleculeShift' ][1]          
#Rs[2] += PP.params['moleculeShift' ][2]          
Rs     = np.transpose( Rs, (1,0) ).copy() 

Qs = np.array( atoms[4] )

if PP.params['PBC' ]:
    iZs,Rs,Qs = PP.PBCAtoms( iZs, Rs, Qs, avec=PP.params['gridA'], bvec=PP.params['gridB'] )

print("shape( Rs )", np.shape( Rs )); 
#print "Rs : ",Rs


print(" # ============ define Scan and allocate arrays   - do this before simulation, in case it will crash ")

dz    = PP.params['scanStep'][2]
zTips = np.arange( PP.params['scanMin'][2], PP.params['scanMax'][2]+0.00001, dz )[::-1];
ntips = len(zTips); 
print(" zTips : ",zTips)
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

atomTypesFile = os.path.dirname(sys.argv[0]) + '/../code/defaults/atomtypes.ini'
FFparams = PP.loadSpecies( atomTypesFile  )
C6,C12   = PP.getAtomsLJ( PP.params['probeType'], iZs, FFparams )

print(" # ============ define Grid ")

cell =np.array([
PP.params['gridA'],
PP.params['gridB'],
PP.params['gridC'],
]).copy() 

gridN = PP.params['gridN']

FF   = np.zeros( (gridN[2],gridN[1],gridN[0],3) )

#quit()

# ==============================================
#   The costly part of simulation starts here
# ==============================================

print(" # =========== Sample LenardJones ")

PP.setFF( FF, cell  )
PP.setFF_Pointer( FF )
PP.getLenardJonesFF( Rs, C6, C12 )

plt.figure(figsize=( 5*nslice,5 )); plt.title( ' FF LJ ' )
'''
for i in range(nslice):
    plt.subplot( 1, nslice, i+1 )
    plt.imshow( FF[i,:,:,2], origin='image', interpolation='nearest' )
'''

withElectrostatics = ( abs( PP.params['charge'] )>0.001 )
if withElectrostatics: 
    print(" # =========== Sample Coulomb ")
    FFel = np.zeros( np.shape( FF ) )
    CoulombConst = -14.3996448915;  # [ e^2 eV/A ]
    Qs *= CoulombConst
    #print Qs
    PP.setFF_Pointer( FFel )
    PP.getCoulombFF ( Rs, Qs )
    plt.figure(figsize=( 5*nslice,5 )); plt.title( ' FFel ' )
    '''
    for i in range(nslice):
        plt.subplot( 1, nslice, i+1 )
        plt.imshow( FFel[i,:,:,2], origin='image', interpolation='nearest' )
    '''
    FF += FFel*PP.params['charge']
    PP.setFF_Pointer( FF )
    del FFel
'''
plt.figure(figsize=( 5*nslice,5 )); plt.title( ' FF total ' )
for i in range(nslice):
    plt.subplot( 1, nslice, i+1 )
    plt.imshow( FF[i,:,:,2], origin='image', interpolation='nearest' )
'''

print(" # ============  Relaxed Scan 3D ")

for ix,x in enumerate( xTips  ):
    print("relax ix:", ix)
    rTips[:,0] = x
    for iy,y in enumerate( yTips  ):
        rTips[:,1] = y
        itrav = PP.relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
        fzs[:,iy,ix] = fs[:,2].copy()
        #print itrav
        #if itrav > 100:
        #    print " bad convergence > %i iterations per pixel " % itrav
        #    print " exiting "
        #    break
        

print(" # ============  convert Fz -> df ")

dfs = PP.Fz2df( fzs, dz = dz, k0 = PP.params['kCantilever'], f0=PP.params['f0Cantilever'], n=int(PP.params['Amplitude']/dz) )

print(" # ============  Plot Relaxed Scan 3D ")

#slices = range( PP.params['plotSliceFrom'], PP.params['plotSliceTo'], PP.params['plotSliceBy'] )
#print "plotSliceFrom, plotSliceTo, plotSliceBy : ", PP.params['plotSliceFrom'], PP.params['plotSliceTo'], PP.params['plotSliceBy']
#print slices 
#nslice = len( slices )

slices = list(range( 0, len(dfs)))

for ii,i in enumerate(slices):
    print(" plotting ", i)
    plt.figure( figsize=( 10,10 ) )
    plt.imshow( dfs[i], origin='image', interpolation=PP.params['imageInterpolation'], cmap=PP.params['colorscale'], extent=extent )
#    z = zTips[i] - PP.params['moleculeShift' ][2]
    z = zTips[i] 
    plt.colorbar();
    plt.xlabel(r' Tip_x $\AA$')
    plt.ylabel(r' Tip_y $\AA$')
    plt.title( r"df Tip_z = %2.2f $\AA$" %z  )
    plt.savefig( 'df_%04i.png' %i, bbox_inches='tight' )


print(" ***** ALL DONE ***** ")

#plt.show()

