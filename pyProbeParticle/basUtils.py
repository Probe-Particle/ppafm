#!/usr/bin/python

import elements
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

def loadAtoms( name, ELEMENT_DICT = elements.ELEMENT_DICT ):
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
				print " skipped line : ", line
	f.close()
	return [ e,x,y,z,q ]



def loadAtomsCUBE( fname, ELEMENT_DICT = elements.ELEMENT_DICT ):
	bohrRadius2angstroem = 0.5291772109217 # find a good place for this
	e=[];x=[];y=[]; z=[]; q=[]
	f = open(fname )
	#First two lines of the header are comments
	header1=f.readline()
	header2=f.readline()
	#The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
	sth0 = f.readline().split()
	#The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
	sth1 = f.readline().split()
	sth2 = f.readline().split()
	sth3 = f.readline().split()

	shift = [float(sth0[1]), float(sth0[2]), float(sth0[3])]
	nlines = int(sth0[0])
	for i in range(nlines):
		l=f.readline().split()
		r=[float(l[2]),float(l[3]),float(l[4])]
#		print l
		x.append( (r[0] - shift[0]) * bohrRadius2angstroem )
		y.append( (r[1] - shift[1]) * bohrRadius2angstroem )
		z.append( (r[2] - shift[2]) * bohrRadius2angstroem )
#		print float(l[2])*bohrRadius2angstroem, float(l[3])*bohrRadius2angstroem, float(l[4])*bohrRadius2angstroem
		e.append( int(  l[0]) )
		q.append(0.0)
	f.close()
	return [ e,x,y,z,q ]


def loadCellCUBE( fname ):
	bohrRadius2angstroem = 0.5291772109217 # find a good place for this
	f = open(fname )
	#First two lines of the header are comments
	header1=f.readline()
	header2=f.readline()
	#The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
	line = f.readline().split()
	n0 = int(line[0])
	c0 =[ float(s) for s in line[1:4] ]

	#The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
	line = f.readline().split()
	n1 = int(line[0])
	c1 =[ float(s) for s in line[1:4] ]

	line = f.readline().split()
	n2 = int(line[0])
	c2 =[ float(s) for s in line[1:4] ]

	line = f.readline().split()
	n3 = int(line[0])
	c3 =[ float(s) for s in line[1:4] ]

#	cell0 = [c0[0]*   bohrRadius2angstroem, c0[1]   *bohrRadius2angstroem, c0[2]   *bohrRadius2angstroem]
	cell0 = [0.0, 0.0, 0.0]
	cell1 = [c1[0]*n1*bohrRadius2angstroem, c1[1]*n1*bohrRadius2angstroem, c1[2]*n1*bohrRadius2angstroem]
	cell2 = [c2[0]*n2*bohrRadius2angstroem, c2[1]*n2*bohrRadius2angstroem, c2[2]*n2*bohrRadius2angstroem]
	cell3 = [c3[0]*n3*bohrRadius2angstroem, c3[1]*n3*bohrRadius2angstroem, c3[2]*n3*bohrRadius2angstroem]
	f.close()
	return [ cell0, cell1, cell2, cell3 ]

def loadNCUBE( fname ):
	bohrRadius2angstroem = 0.5291772109217 # find a good place for this
	f = open(fname )
	#First two lines of the header are comments
	header1=f.readline()
	header2=f.readline()
	#The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
	sth0 = f.readline().split()
	#The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
	sth1 = f.readline().split()
	sth2 = f.readline().split()
	sth3 = f.readline().split()
	f.close()
	return [ int(sth1[0]), int(sth2[0]), int(sth3[0]) ]


def findBonds( atoms, sc, ELEMENTS = elements.ELEMENTS ):
	bonds = []
	es = atoms[0]
	xs = atoms[1]
	ys = atoms[2]
	zs = atoms[3]
	n = len( xs )
	for i in xrange(n):
		for j in xrange(i):
			dx=xs[j]-xs[i]
			dy=ys[j]-ys[i]
			dz=zs[j]-zs[i]
			r=math.sqrt( dx*dx + dy*dy + dz*dz )
			ii = es[i]-1
			jj = es[j]-1	
			bondlength=ELEMENTS[ ii ][6]+ ELEMENTS[ jj ][6]
			if (r<( sc * bondlength)) :
				bonds.append( (i,j) )
	return bonds

def findBondsSimple( xyz, rmax ):
	bonds = []
	xs = atoms[1]
	ys = atoms[2]
	zs = atoms[3]
	n = len( xs )
	for i in xrange(n):
		for j in xrange(i):
			dx=xs[j]-xs[i]
			dy=ys[j]-ys[i]
			dz=zs[j]-zs[i]
			r=math.sqrt(dx*dx+dy*dy+dz*dz)
			if (r<rmax) :
				bonds.append( (i,j) )
	return bonds

def getAtomColors( atoms, ELEMENTS = elements.ELEMENTS ):
	colors=[]
	es = atoms[0]
	for i,e in enumerate( es ): 
		colors.append( ELEMENTS[ e - 1 ][7] )
	return colors

def multCell( xyz, cel, m=(2,2,1) ):
	n = len(xyz[0])
	mtot = m[0]*m[1]*m[2]*n
	es = [None] * mtot
	xs = [None] * mtot
	ys = [None] * mtot
	zs = [None] * mtot
	j  = 0
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




