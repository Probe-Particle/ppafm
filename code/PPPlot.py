#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt

params = None


# =========== Utils

def plotBonds( xyz, bonds ):
	for b in bonds:
		i=b[0]; j=b[1]
		plt.arrow(xyz[1][i], xyz[2][i], xyz[1][j]-xyz[1][i], xyz[2][j]-xyz[2][i], head_width=0.0, head_length=0.0,  fc='k', ec='k', lw= 1.0,ls='solid' )


def plotAtoms( xyz, atomSize=0.3, edge=True, ec='k', color='w', colors=None ):
	plt.fig = plt.gcf()
	if colors is None:
		colors=[ color ]*100
	for i in range(len(xyz[1])):
		fc = colors[xyz[0][i]]
		if not edge:
			ec=fc
		circle=plt.Circle( (xyz[1][i],xyz[2][i]), atomSize, fc=fc, ec=ec  )
		plt.fig.gca().add_artist(circle)

def plotGeom( atoms=None, bonds=None, atomSize=0.3 ):
	if (bonds is not None) and (atoms is not None):
		plotBonds( atoms, bonds )	
	if atoms is not None:
		plotAtoms( atoms, atomSize=atomSize )

def colorize_XY2RG( Xs, Ys ):
	r = np.sqrt(Xs**2 + Ys**2)
	vmax = r[5:-5,5:-5].max()
	Red   = 0.5*Xs/vmax + 0.5
	Green = 0.5*Ys/vmax + 0.5
	c = np.array( (Red, Green, 0.5*np.ones(np.shape(Red)) )  )  # -->  array of (3,n,m) shape, but need (n,m,3)
	c = c.swapaxes(0,2) 
	c = c.swapaxes(0,1) 
	return c, vmax

# =========== plotting functions

def plotImages( prefix, F, slices, extent=None, zs = None, figsize=(10,10) ):
	for ii,i in enumerate(slices):
		print " plotting ", i
		plt.figure( figsize=figsize )
		plt.imshow( F[i], origin='image', interpolation=params['imageInterpolation'], cmap=params['colorscale'], extent=extent )
		plt.colorbar();
		plt.xlabel(r' Tip_x $\AA$')
		plt.ylabel(r' Tip_y $\AA$')
		z = i*params['scanStep'][2]
		if zs is not None:
			z = zs[i]
		plt.title( r"Tip_z = %2.2f $\AA$" %z  )
		plt.savefig( prefix+'_%3.3i.png' %i, bbox_inches='tight' )
		plt.close()

def plotVecFieldRG( prefix, dXs, dYs, slices, extent=None, zs = None, figsize=(10,10) ):
	for ii,i in enumerate(slices):
		print " plotting ", i
		plt.figure( figsize=( 10,10 ) )
		HSBs,vmax = colorize_XY2RG(dXs[i],dYs[i])
		plt.imshow( HSBs, extent=extent, origin='image', interpolation=PP.params['imageInterpolation'] ) 
		plt.xlabel(r' Tip_x $\AA$')
		plt.ylabel(r' Tip_y $\AA$')
		z = i*params['scanStep'][2]
		if zs is not None:
			z = zs[i]
		plt.title( r"Tip_z = %2.2f $\AA$" %z  )
		plt.savefig( prefix+'_%3.3i.png' %i, bbox_inches='tight' )
		plt.close()

def plotDistortions( prefix, Rs, slices, extent=None, zs = None, by=2, figsize=(10,10) ):
	for ii,i in enumerate(slices):
		print " plotting ", i
		plt.figure( figsize=figsize )
		plt.plot( Rs[i,::by,::by,0].flat, Rs[i,::by,::by,1].flat, 'r.', markersize=0.5 )
		plt.imshow( Rs[i,:,:,2], origin='image', interpolation=params['imageInterpolation'], cmap=params['colorscale'], extent=extent )
		plt.colorbar();
		plt.xlabel(r' Tip_x $\AA$')
		plt.ylabel(r' Tip_y $\AA$')
		z = i*params['scanStep'][2]
		if zs is not None:
			z = zs[i]
		plt.title( r"Tip_z = %2.2f $\AA$" %z  )
		plt.savefig( prefix+'_%3.3i.png' %i, bbox_inches='tight' )
		plt.close()
