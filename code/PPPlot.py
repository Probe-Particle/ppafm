#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt

params = None

def plotImages( prefix, F, slices, extent=None, zs = None ):
	for ii,i in enumerate(slices):
		print " plotting ", i
		plt.figure( figsize=( 10,10 ) )
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
