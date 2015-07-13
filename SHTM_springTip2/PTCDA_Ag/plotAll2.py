#!/usr/bin/python

from STHM_Utils import *
from xsfutil import *
from basUtils import *
from pylab import *
import sys,os






# ========== Settings

ilist =  range(27,42)

bd    =  1

#startx = -2
#starty = -2
#sizex  = 25
#sizey  = 25

startx = 0
starty = 0
sizex  = 12.57348
sizey  = 18.912

ix0  = 20
iy0  = 40
iszx = 200
iszy = 200

def cut( F ):
	#return F[:,ix0:ix0+iszx,iy0:iy0+iszy]
	#return transpose( F , (0,2,1) )
	return F

dfmin = -10; dfmax = 50


cdict = {
'red':   ( (0.0, 0.0, 0.0),  (0.3333, 0.0, 0.0), (0.6666, 1.0, 1.0), (1.0, 1.0, 1.0)   ),
'green': ( (0.0, 0.0, 0.0),  (0.3333, 1.0, 1.0), (0.6666, 1.0, 1.0), (1.0, 1.0, 1.0)   ),
'blue':  ( (0.0, 0.0, 0.0),  (0.3333, 0.0, 0.0), (0.6666, 0.0, 0.0), (1.0, 1.0, 1.0)   )       
 }
cdict = {
'red':   ( (0.0, 0.0, 0.0),  (0.25, 0.0, 0.0), (0.50, 1.0, 1.0), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)   ),
'green': ( (0.0, 0.0, 0.0),  (0.25, 1.0, 1.0), (0.50, 1.0, 1.0), (0.75, 0.0, 0.0), (1.0, 1.0, 1.0)   ),
'blue':  ( (0.0, 0.0, 0.0),  (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 1.0, 1.0)   )       
 }
cdict = {
'red':   ( (0.0, 0.0, 0.0),  (0.2, 0.2, 0.2),  (1.0, 1.0, 1.0)   ),
'green': ( (0.0, 0.0, 0.0),  (0.2, 0.2, 0.2),  (1.0, 1.0, 1.0)   ),
'blue':  ( (0.0, 0.0, 0.0),  (0.2, 1.0, 1.0),  (1.0, 1.0, 1.0)   )       
 }
cmap= LinearSegmentedColormap('BlueRed1', cdict)

# =========== prepare everything

#extent=( startx, startx+sizex, starty, starty + sizey )
extent=( starty, starty + sizey, startx, startx+sizex )

#atoms = loadBas('../surf.bas')[0]
atoms = loadBas('../surf_plot.bas')[0]
#atoms = None
bonds = findBondsSimple(atoms, 1.7 ) 
withcolorbar = False

# =========== Plot individaul datafiles

if os.path.isfile('df.xsf'):
	df_data,lvec, nDim, head = loadXSF('df.xsf')
else:
	data_Fz,lvec, nDim, head = loadXSF('OutFz.xsf')
	df_data = F2df( data_Fz, dz = 0.1, k0 = 1800.0, f0=30300.0 )
	saveXSF('df.xsf', head, lvec, df_data )

df_data = cut( df_data )

plotWithAtoms( df_data[:,bd:-bd,bd:-bd], ilist, extent, dz = 0.1,  atoms=atoms, cmap=cmap );                           savefig('df.png',          bbox_inches='tight', pad_inches=0 )
plotWithAtoms( df_data[:,bd:-bd,bd:-bd], ilist, extent, dz = 0.1,  atoms=None,  cmap=cmap );                           savefig('df_noAtoms.png',  bbox_inches='tight', pad_inches=0 )
plotWithAtoms( df_data[:,bd:-bd,bd:-bd], ilist, extent, dz = 0.1,  atoms=None,  cmap=cmap, vmin=dfmin, vmax=dfmax );   savefig('df_absolute.png', bbox_inches='tight', pad_inches=0 )

Xs,lvec, nDim, head = loadXSF('OutX.xsf'); Xs = cut( Xs )
Ys,lvec, nDim, head = loadXSF('OutY.xsf'); Xs = cut( Xs )
plotDeviations_RG    ( Xs[:,bd:-bd,bd:-bd],Ys[:,bd:-bd,bd:-bd], ilist, extent,  atoms=atoms );                savefig('Deviations_RG.png',     bbox_inches='tight', pad_inches=0 )
#plotDeviations_Points( Xs[:,bd:-bd,bd:-bd],Ys[:,bd:-bd,bd:-bd], ilist, extent,  atoms=atoms, bonds=bonds );   savefig('Deviations_Points_1.png', bbox_inches='tight', pad_inches=0 )
#plotDeviations_Points( Xs[:,bd:-bd,bd:-bd],Ys[:,bd:-bd,bd:-bd], ilist, extent,  atoms=atoms, bonds=bonds, s=2, step=4, alpha=1 );   savefig('Deviations_2_Points.png', bbox_inches='tight', pad_inches=0 )
plotDeviations_Points( Xs[:,bd:-bd,bd:-bd],Ys[:,bd:-bd,bd:-bd], ilist, extent,  atoms=atoms, bonds=bonds, s=1, step=2, alpha=1 );   savefig('Deviations_3_Points.png', bbox_inches='tight', pad_inches=0 )

'''
#T2,lvec, nDim, head = loadXSF('OutT2.xsf')
#plotWithAtoms( T2[:,bd:-bd,bd:-bd], ilist, extent, dz = 0.1,  atoms=atoms, cmap=cmap );    savefig('T2.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( T2[:,bd:-bd,bd:-bd], ilist, extent, dz = 0.1,  atoms=None, cmap=cmap );     savefig('T2_noAtoms.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( log(T2[:,bd:-bd,bd:-bd]), ilist, extent, dz = 0.1,  atoms=atoms, cmap='gray' );    savefig('T2_log.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( log(T2[:,bd:-bd,bd:-bd]), ilist, extent, dz = 0.1,  atoms=None, cmap='gray' );     savefig('T2_log_noAtoms.png',     bbox_inches='tight', pad_inches=0 )


Eig1,lvec, nDim, head   = loadXSF('OutEig1.xsf')
Eig2,lvec, nDim, head   = loadXSF('OutEig2.xsf')

plotWithAtoms( Eig1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar);      savefig('Eig1.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( Eig2, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar);       savefig('Eig2.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( Eig1, ilist, extent, dz = 0.1,   atoms=None,   withcolorbar=withcolorbar);      savefig('Eig1_noAtoms.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( Eig2, ilist, extent, dz = 0.1,   atoms=None,   withcolorbar=withcolorbar);       savefig('Eig2_noAtoms.png',     bbox_inches='tight', pad_inches=0 )

T1,lvec, nDim, head   = loadXSF('OutT1.xsf')
T2,lvec, nDim, head   = loadXSF('OutT2.xsf')
TSS1,lvec, nDim, head = loadXSF('OutTSS1.xsf')
TSS2,lvec, nDim, head = loadXSF('OutTSS2.xsf')
#R1,lvec, nDim, head   = loadXSF('OutR1.xsf')


#plotWithAtoms( T1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('T1.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( T2, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar);       savefig('T2.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( T1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('T1.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( T2, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar);       savefig('T2.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( TSS1, ilist, extent, dz = 0.1, atoms=atoms,  withcolorbar=withcolorbar );      savefig('TSS1.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( TSS2, ilist, extent, dz = 0.1, atoms=atoms,  withcolorbar=withcolorbar);       savefig('TSS2.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( R1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('R1.png',     bbox_inches='tight', pad_inches=0 )

atoms = None

plotWithAtoms( T1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('T1_noAtoms.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( T2, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar);       savefig('T2_noAtoms.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( TSS1, ilist, extent, dz = 0.1, atoms=atoms,  withcolorbar=withcolorbar );      savefig('TSS1_noAtoms.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( TSS2, ilist, extent, dz = 0.1, atoms=atoms,  withcolorbar=withcolorbar);       savefig('TSS2_noAtoms.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( R1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('R1_noAtoms.png',     bbox_inches='tight', pad_inches=0 )

withcolorbar = True

plotWithAtoms( T1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('T1_cbar.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( T2, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar);       savefig('T2_cbar.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( TSS1, ilist, extent, dz = 0.1, atoms=atoms,  withcolorbar=withcolorbar );      savefig('TSS1_cbar.png',     bbox_inches='tight', pad_inches=0 )
plotWithAtoms( TSS2, ilist, extent, dz = 0.1, atoms=atoms,  withcolorbar=withcolorbar);       savefig('TSS2_cbar.png',     bbox_inches='tight', pad_inches=0 )
#plotWithAtoms( R1, ilist, extent, dz = 0.1,   atoms=atoms,  withcolorbar=withcolorbar );      savefig('R1_cbar.png',     bbox_inches='tight', pad_inches=0 )

'''

#show()
