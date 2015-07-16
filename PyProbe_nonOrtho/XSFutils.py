#!/usr/bin/python

import pylab
import ctypes
from   ctypes import *
import os
import numpy as np




name='ProbeParticle_lib'
ext='.so'
'''
def recompile( 
                LFLAGS="",
                #FFLAGS="-Og -g -Wall"
                FFLAGS="-std=c99 -O3 -ffast-math -ftree-vectorize"
        ):
        import os
        print " ===== COMPILATION OF : "+name+".cpp"
        print  os.getcwd()
        os.system("g++ "+FFLAGS+" -c -fPIC "+name+".cpp -o "+name+".o"+LFLAGS)
        os.system("g++ "+FFLAGS+" -shared -Wl,-soname,"+name+ext+" -o "+name+ext+" "+name+".o"+LFLAGS)
'''







def readUpTo( filein, keyword ):
	i = 0
	linelist = []
	while True :
		line = filein.readline()
		linelist.append(line)
		i=i+1
		if 	((not line) or (keyword in line)): break;
	return i,linelist

def readNums(filein):
	out = []
	while True :
		line = filein.readline()
		if (not line): break;
		words = line.split()
		try: float(words[0])
		except ValueError: break;
		out = out + [ float(iii) for iii in words  ]
	return pylab.array(out)

def readNumsUpTo(filename, dimensions):
#        recompile()
        lib    = ctypes.CDLL("./"+name+ext )
        array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array1i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
        lib.ReadNumsUpTo_C.argtypes  = [c_char_p, array1d, array1i]
        lib.ReadNumsUpTo_C.restype   = c_int
        N_arry=np.zeros((dimensions[0]*dimensions[1]*dimensions[2]), dtype = np.double)
        lib.ReadNumsUpTo_C(filename, N_arry, dimensions)
	return pylab.array(N_arry)

def readmat(filein, n):
	temp = []
	for i in range(n):
		temp.append( [ float(iii) for iii in filein.readline().split() ] )
	return pylab.array(temp)

def seekWord(filein, word):
	line = filein.readline()
	while line and not (word in line):
		line = filein.readline()

def loadXSF_old(fname):
	filein = open(fname )
	startline, head = readUpTo(filein, "BEGIN_DATAGRID_3D_")
	nDim = [ int(iii) for iii in filein.readline().split() ]
	nDim.reverse()
	nDim = pylab.array( nDim )
	lvec = readmat(filein, 4)
	#F = readNums(filein)
	F = readNumsUpTo(filein,"END_DATAGRID_3D")
	print "pylab.shape(F)",pylab.shape(F)
	print "nDim",nDim
	filein.close()
	#F = pylab.genfromtxt(fname, skip_header=startline+1, skip_footer=2, usecols = (1,2,3,4,5) )
	FF = pylab.reshape (F.flat, nDim )
	return FF,lvec, nDim, head
	#return 1.0,lvec, nDim, head

def iter_loadtxt(filename, delimiter=',', skip_header=0, dtype=float):
	print "iter_loadtxt "
	def iter_func():
		with open(filename, 'r') as infile:
			for _ in range(skip_header):
				next(infile)
			for line in infile:
				line = line.rstrip().split(delimiter)
				try:
					float(line[0])
					for item in line:
						yield dtype(item)
				except ValueError:
					print "Not a float"
					break
		iter_loadtxt.rowlength = len(line)
	data = pylab.fromiter(iter_func(), dtype=dtype)
	data = data.reshape((-1, iter_loadtxt.rowlength))
	return data

def loadXSF(fname):
	filein = open(fname )
	#startline, head = readUpTo(filein, "BEGIN_DATAGRID_3D_")
	startline, head = readUpTo(filein, "DATAGRID_3D_")
	nDim = [ int(iii) for iii in filein.readline().split() ]
	nDim.reverse()
#	nDim = pylab.array( nDim, dtype=np.int32 )
	nDim = pylab.array( nDim)
	lvec = readmat(filein, 4)
	line = filein.readline()
	perline = len(line.split())
	rewind = len(line)
	if(perline==0):
		line = filein.readline()
		rewind += len(line)
		perline = len(line.split())
	ntot = nDim[0]*nDim[1]*nDim[2]
	nrest = ntot%perline
	print ntot,ntot/perline,perline,nrest
#	if (( perline > 1 )&(nrest!=0)):    
	print "load "+fname+" using readNumsUpTo (very fast)"    
	filein.seek(-rewind,1)
	filein.close()
	F = readNumsUpTo(fname,nDim.astype(np.int32).copy())

        print "Done"
        print nDim
	FF = pylab.reshape (F, nDim )
	return FF,lvec, nDim, head

def writeArr(f, arr):
    f.write(" ".join(str(x) for x in arr) + "\n")

def writeArr2D(f, arr):
	for vec in arr:
		writeArr(f,vec)

def saveXSF(fname, head, lvec, data ):
	fileout = open(fname, 'w')
	for line in head:
		fileout.write(line)
	nDim = pylab.shape(data)
	writeArr (fileout, (nDim[2],nDim[1],nDim[0]) )
	writeArr2D(fileout,lvec)
	for r in data.flat:
		fileout.write( "%10.5e\n" % r )
	fileout.write ("   END_DATAGRID_3D\n")
	fileout.write ("END_BLOCK_DATAGRID_3D\n")



