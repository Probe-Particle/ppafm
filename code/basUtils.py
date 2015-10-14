#!/usr/bin/python

import Element
import math

def loadBas(name):
	xyzs = []
	f = open(name,"r")
	while True:
		n=0;
		l = f.readline()
		#print "--",l,"--"
		try:
			n=int(l)
		except ValueError:
			break
		if (n>0):
			n=int(l)
			#f.readline()
			e=[];x=[];y=[]; z=[];
			for i in xrange(n):
				l=f.readline().split()
				#print l
				e.append( int(l[0]) )
	   			x.append( float(l[1]) )
	   			y.append( float(l[2]) )
	   			z.append( float(l[3]) )
			xyzs.append( [e,x,y,z] )
		else:
			break
		f.readline()
	f.close()
	return xyzs

def loadAtoms( name, ELEMENT_DICT ):
	xyzs = []
	f = open(name,"r")
	n=0;
	l = f.readline()
	#print "--",l,"--"
	try:
		n=int(l)
	except ValueError:
		return
	if (n>0):
		n=int(l)
		e=[];x=[];y=[]; z=[]; q=[]
		i = 0;
		while( i<n ):
			line = f.readline() 
			words=line.split()
			nw = len( words)
			ie = None
			if( nw >=4 ):
				try :
					ie = int( words[0] )
				except:
					if words[0] in ELEMENT_DICT:
						ie = ELEMENT_DICT[ words[0] ][0]
			if ie is not None:
				e.append( ie )
	   			x.append( float(words[1]) )
	   			y.append( float(words[2]) )
	   			z.append( float(words[3]) )
				if ( nw >=5 ):
					q.append( float(words[4]) )
				else:
					q.append( 0.0 )				
				i+=1
			else:
				print " skiped line : ", line
	f.close()
	return [ e,x,y,z,q ]

def findBonds( xyz, sc ):
	bonds = []
	n = len(xyz[0])
	for i in xrange(n):
		for j in xrange(i):
			dx=xyz[1][j]-xyz[1][i]
			dy=xyz[2][j]-xyz[2][i]
			dz=xyz[3][j]-xyz[3][i]
			r=math.sqrt(dx*dx+dy*dy+dz*dz)
			bondlength=Element.rad[xyz[0][i]-1]+ Element.rad[xyz[0][j]-1]
			if (r<( sc * bondlength)) :
				bonds.append([i,j])
	return bonds

def findBondsSimple( xyz, rmax ):
	bonds = []
	n = len(xyz[0])
	for i in xrange(n):
		for j in xrange(i):
			dx=xyz[1][j]-xyz[1][i]
			dy=xyz[2][j]-xyz[2][i]
			dz=xyz[3][j]-xyz[3][i]
			r=math.sqrt(dx*dx+dy*dy+dz*dz)
			if (r<rmax) :
				bonds.append([i,j])
	return bonds


def multCell( xyz, cel, m=(2,2,1) ):
	n = len(xyz[0])
	mtot = m[0]*m[1]*m[2]*n
	es = [None] * mtot
	xs = [None] * mtot
	ys = [None] * mtot
	zs = [None] * mtot
	j = 0
	for ia in range(m[0]):
		for ib in range(m[1]):
			for ic in range(m[2]):
				dx = ia*cel[0][0] + ib*cel[1][0] + ic*cel[2][0]
				dy = ia*cel[0][1] + ib*cel[1][1] + ic*cel[2][1]
				dz = ia*cel[0][2] + ib*cel[1][2] + ic*cel[2][2]
				for i in xrange(n):
					es[j]=xyz[0][i]
					xs[j]=xyz[1][i] + dx
					ys[j]=xyz[2][i] + dy
					zs[j]=xyz[3][i] + dz
					j+=1
	return [es,xs,ys,zs]




