#!/usr/bin/python

## read Nanonis sxm files with data on discharge rings in TBTAP trimers
## plot a grid of all dI/dV or current plots
## Vladislav Pokorny; pokornyv@fzu.cz; 2024

import numpy as np
import scipy as sp

from sys import argv,exit
from os.path import split
from time import ctime

import matplotlib.pyplot as plt
from matplotlib import colormaps

import nanonispy2 as nap

## plot current maps (1) or dI/dV maps (0)
PlotCurrent = 0

## read positions of points where we want to extract dIdV or current
try:
	x1,y1 = float(argv[1]),float(argv[2])
except IndexError:
	x1,y1 = -62.97,82.39 ## approx. center of the grid

try:
	#-63.06 82.11 is an interesting point as it crosses both a ring and a negative dip in dI/dV
	#-63.0832  82.1188 is a global minimum of dI/dV for V=0.75 V
	x2,y2 = float(argv[3]),float(argv[4]) 
	n2ndpoint = 1
except IndexError:
	n2ndpoint = 0


path = "/home/prokop/Desktop/Charing_Rings/prokop_copy/data/"


## number of data sets
NPoint = 14
bias_out_A = np.zeros(NPoint)
dIdV_out_A = np.zeros(NPoint)
I_out_A    = np.zeros(NPoint)
if n2ndpoint:
	dIdV_out2_A = np.zeros(NPoint)
	I_out2_A    = np.zeros(NPoint)

## plot the data
fig, ax = plt.subplots(ncols=4,nrows=4,figsize=(15,13))

if PlotCurrent:
	fname_out = 'point_x'+str(x1)+'_y'+str(y1)+'_current'
else:
	fname_out = 'point_x'+str(x1)+'_y'+str(y1)+'_dIdV'

with open(fname_out+'.dat','w') as fout:
	fout.write('bias [V]\tx [nm]\ty [nm]\tpixel x\t pixel y\tdIdV[x,y]\n')
	
	k = 0
	for i in [42,41,40,39,38,43,44,45,46,47,48,49,50,51]: ## ordered by increasing voltage
		a,b = int(k/4),k%4
		fname_in = path+'/TBTPA on Pb11100'+str(i)+'.sxm'
		print(60*'#')
		print('Reading file '+fname_in)
		scan = nap.read.Scan(fname_in)
		print('Data basename: '+scan.basename)
	
		bias = -scan.header['bias'] ## the file contains tip voltage
		print('Bias voltage: {0: .5f}'.format(bias))
		print('Resolution: '+str(scan.header['scan_pixels']))
	
		X_A     = 1e9* scan.signals['X']['forward']
		Y_A     = 1e9* scan.signals['Y']['forward']
		## these data are stored by columns!
		dIdV_A =  1e12*scan.signals['LI_Demod_1_X']['forward']
		I_A    = -1e12*scan.signals['Current']['forward']

		xmin,xmax = np.amin(X_A),np.amax(X_A)
		ymin,ymax = np.amin(Y_A),np.amax(Y_A)
		if PlotCurrent: maxval1 = np.amax(np.fabs(I_A))
		else:           maxval1 = np.amax(np.fabs(dIdV_A))

		minval = np.amin(dIdV_A)
		minpos = np.argmin(dIdV_A)
		minx = int(minpos / X_A.shape[0])
		miny = minpos % X_A.shape[0]
		xx = X_A[minx,miny]
		yy = Y_A[minx,miny]
		print('Minimum of d/dV: {0: .6f} for pixel {1: 3d} {2: 3d} = {3: .4f} {4: .4f}'.format(minval,minx,miny,xx,yy))

		## find the position of the point from input
		idxx = int(np.fabs((x1-xmin))/(np.fabs(xmax-xmin))*X_A.shape[0])
		idxy = Y_A.shape[1]-1-int(np.fabs((y1-ymin))/(np.fabs(ymax-ymin))*Y_A.shape[1])

		if PlotCurrent:
			p1 = ax[a,b].imshow(I_A,aspect='equal',cmap='inferno',vmin=0.0, vmax=600.0,  extent =[xmin,xmax,ymin,ymax])
		else: ## dI/dV
			p1 = ax[a,b].imshow(dIdV_A,aspect='equal',cmap='seismic',vmin=-maxval1, vmax=maxval1, extent =[xmin,xmax,ymin,ymax])
		ax[a,b].set_title(r'voltage {0: .4f} V'.format(bias), fontsize=8)
		ax[a,b].scatter(x=[x1], y=[y1], c='#008800', s=20, marker = 'x')
		if n2ndpoint:
			## find the position of the second point from input
			idxx2=int(np.fabs((x2-xmin))/(np.fabs(xmax-xmin))*X_A.shape[0])
			idxy2=Y_A.shape[1]-1-int(np.fabs((y2-ymin))/(np.fabs(ymax-ymin))*Y_A.shape[1])
			ax[a,b].scatter(x=[x2], y=[y2], c='royalblue', s=20, marker = 'x')
		plt.colorbar(p1,fraction=0.046, pad=0.05, ax=ax[a,b])

		bias_out_A[k] = bias 
		dIdV_out_A[k] = dIdV_A[idxy,idxx]
		I_out_A[k] = I_A[idxy,idxx]
		
		if n2ndpoint:
			dIdV_out2_A[k] = dIdV_A[idxy2,idxx2]
			I_out2_A[k] = I_A[idxy2,idxx2]
		fout.write('{0: .5f}\t{1: .2f}\t{2: .2f}\t{3: 3d}\t{4: 3d}\t{5: 2e}\n'.format(bias,x1,y1,idxx,idxy,dIdV_A[idxx,idxy]))
		k += 1

## plot the current / dIdV at given points as last two panels
if PlotCurrent:
	ax[a,b+1].plot(bias_out_A, I_out_A, color='#008800', linewidth=2.0,linestyle='solid', marker='o', markersize=4)
else: ## dI/dV
	ax[a,b+1].plot(bias_out_A, dIdV_out_A, color='#008800', linewidth=2.0,linestyle='solid', marker='o', markersize=4)
if not PlotCurrent: ax[a,b+1].axhline(y = 0, color='black', linewidth=1.0)
ax[a,b+1].set_title(r'voltage dependence at [{0: .3f} {1: .3f}]'.format(x1,y1), fontsize=8)
ax[a,b+1].grid(True,color='#eeeeee')
ax[a,b+1].set_xlabel(r'$V$ [V]', fontsize=10)
if PlotCurrent:
	ax[a,b+1].set_ylabel(r'$I$ [pA]', fontsize=10)
else:
	ax[a,b+1].set_ylabel(r'$dI/dV$ [pS]', fontsize=10)

if n2ndpoint:
	if PlotCurrent:
		ax[a,b+2].plot(bias_out_A, I_out2_A, color='royalblue', linewidth=2.0,linestyle='solid', marker='o', markersize=4)
	else: ## dI/dV
		ax[a,b+2].plot(bias_out_A, dIdV_out2_A, color='royalblue', linewidth=2.0,linestyle='solid', marker='o', markersize=4)
	if not PlotCurrent: ax[a,b+2].axhline(y = 0, color='black', linewidth=1.0)
	ax[a,b+2].set_title(r'voltage dependence at [{0: .3f} {1: .3f}]'.format(x2,y2), fontsize=8)
	ax[a,b+2].grid(True,color='#eeeeee')
	ax[a,b+2].set_xlabel(r'$V$ [V]', fontsize=10)
	if PlotCurrent:
		ax[a,b+2].set_ylabel(r'$I$ [pA]', fontsize=10)
	else:
		ax[a,b+2].set_ylabel(r'$dI/dV$ [pS]', fontsize=10)

plt.tight_layout()
plt.savefig(fname_out+'.pdf', format='pdf', bbox_inches='tight')

## rings_grid.py END ##

