#!/usr/bin/python

from xsfutil import *
from pylab import *
import sys,os

Fpauli,lvec, nDim, head = loadXSF('FFpauli_3.xsf')
#Fvdw,lvec, nDim, head   = loadXSF('FFvdw_3.xsf')
#F,lvec, nDim, head = loadXSF('FFLJ_3.xsf')

islice = 40


poss   = [ [105, 75], [117,77], [112,64] ]
colors = [ 'k', 'r', 'b' ]
labels = [ 'hole', 'Cside', 'Caxis' ]

zs = array(range(len(F))) * 0.1


'''
figure(figsize=(5,5))
#fmin = 100000
for i,pos in enumerate(poss):
	fline  = F[:,pos[1],pos[0]].copy()
	#F[:,pos[1],pos[0]] = 0
	#flinmin = fline.min()
	#if flinmin < fmin:
	#	flinmin = fmin
	plot( zs, fline, color=colors[i], label=labels[i] )
	plot( zs, Fvdw[:,pos[1],pos[0]], color=colors[i], ls='--' )

grid()
legend(loc=4)
#ylim( fmin*1.2, -fmin*1.2 )
ylim( -0.1, 0.0 )
xlim( 3.0, 10.0 )

figure()
imshow( F[islice], cmap='gray' )
for i,pos in enumerate(poss):
	plot( pos[0],pos[1], 'o', color=colors[i] )

figure()
imshow( Fvdw[islice], cmap='gray' )
for i,pos in enumerate(poss):
	plot( pos[0],pos[1], 'o', color=colors[i] )
'''

figure()
imshow( Fvdw[islice], cmap='gray' )
for i,pos in enumerate(poss):
	plot( pos[0],pos[1], 'o', color=colors[i] )

show()



