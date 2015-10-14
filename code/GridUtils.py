#!/usr/bin/python

import numpy as np
from   ctypes import c_int, c_double, c_char_p
import ctypes
import os

# ==============================
# ============================== interface to C++ core 
# ==============================

LIB_PATH = os.path.dirname( os.path.realpath(__file__) )
print " ProbeParticle Library DIR = ", LIB_PATH

name='GridUtils'
ext='_lib.so'


# recompilation of C++ dynamic librady ProbeParticle_lib.so from ProbeParticle.cpp
def recompile( 
		LFLAGS="",
		#FFLAGS="-Og -g -Wall"
		#FFLAGS="-std=c99 -O3 -ffast-math -ftree-vectorize"
		FFLAGS="-std=c++11 -O3 -ffast-math -ftree-vectorize"
	):
	import os
	print " ===== COMPILATION OF : "+name+".cpp"+"   "+name+ext
	CWD = os.getcwd()
	os.chdir( LIB_PATH );   print " >> WORKDIR: ", os.getcwd()
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".so")  ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".o")   ]
	[ os.remove(f) for f in os.listdir(".") if f.endswith(".pyc") ]
	os.system("g++ "+FFLAGS+" -c -fPIC "+name+".cpp -o "+name+".o"+LFLAGS)
	os.system("g++ "+FFLAGS+" -shared -Wl,-soname,"+name+ext+" -o "+name+ext+" "+name+".o"+LFLAGS)
	os.chdir(CWD);          print " >> WORKDIR: ", os.getcwd()

# if binary of ProbeParticle_lib.so is deleted => recompile it
if not os.path.exists(LIB_PATH+"/"+name+ext ):
	recompile()

lib    = ctypes.CDLL(LIB_PATH+"/"+name+ext )    # load dynamic librady object using ctypes 

# define used numpy array types for interfacing with C++

array1i = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags='CONTIGUOUS')

# ==============  Xsf

XSF_HEAD_DEFAULT = headScan='''
ATOMS
 1   0.0   0.0   0.0

BEGIN_BLOCK_DATAGRID_3D                        
   some_datagrid      
   BEGIN_DATAGRID_3D_whatever 
'''

lib.ReadNumsUpTo_C.argtypes  = [c_char_p, array1d, array1i]
lib.ReadNumsUpTo_C.restype   = c_int
def readNumsUpTo(filename, dimensions):
	N_arry=np.zeros( (dimensions[0]*dimensions[1]*dimensions[2]), dtype = np.double )
	lib.ReadNumsUpTo_C( filename, N_arry, dimensions )
	return N_arry

def readUpTo( filein, keyword ):
	i = 0
	linelist = []
	while True :
		line = filein.readline()
		linelist.append(line)
		i=i+1
		if 	((not line) or (keyword in line)): break;
	return i,linelist

def readmat(filein, n):
	temp = []
	for i in range(n):
		temp.append( [ float(iii) for iii in filein.readline().split() ] )
	return np.array(temp)

def writeArr(f, arr):
    f.write(" ".join(str(x) for x in arr) + "\n")

def writeArr2D(f, arr):
	for vec in arr:
		writeArr(f,vec)

def saveXSF(fname, data, lvec, head=XSF_HEAD_DEFAULT ):
	fileout = open(fname, 'w')
	for line in head:
		fileout.write(line)
	nDim = np.shape(data)
	writeArr (fileout, (nDim[2],nDim[1],nDim[0]) )
	writeArr2D(fileout,lvec)
	for r in data.flat:
		fileout.write( "%10.5e\n" % r )
	fileout.write ("   END_DATAGRID_3D\n")
	fileout.write ("END_BLOCK_DATAGRID_3D\n")

def loadXSF(fname):
	filein = open(fname )
	#startline, head = readUpTo(filein, "BEGIN_DATAGRID_3D_")
	startline, head = readUpTo(filein, "DATAGRID_3D_")
	nDim = [ int(iii) for iii in filein.readline().split() ]
	nDim.reverse()
	nDim = np.array( nDim)
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
	print "load "+fname+" using readNumsUpTo (very fast)"    
	filein.seek(-rewind,1)
	filein.close()
	F = readNumsUpTo(fname,nDim.astype(np.int32).copy())
        print "Done"
        print nDim
	FF = np.reshape (F, nDim )
	return FF,lvec, nDim, head



# =============== Vector Field

def packVecGrid( Fx, Fy, Fz, FF = None ):
	if FF is None:
		nDim = np.shape( Fx )
		FF = np.zeros( (nDim[0],nDim[1],nDim[2],3) )
	FF[:,:,:,0]=Fx; 	FF[:,:,:,1]=Fy;	FF[:,:,:,2]=Fz
	return FF

def unpackVecGrid( FF ):
	return FF[:,:,:,0].copy(), FF[:,:,:,1].copy(), FF[:,:,:,2].copy()

def loadVecFieldXsf( fname, FF = None ):
	Fx,lvec,nDim,head=loadXSF(fname+'_x.xsf')
	Fy,lvec,nDim,head=loadXSF(fname+'_y.xsf')
	Fz,lvec,nDim,head=loadXSF(fname+'_z.xsf')
	FF = packVecGrid( Fx, Fy, Fz, FF )
	del Fx,Fy,Fz
	return FF, lvec, nDim, head

def loadVecFieldNpy( fname, FF = None ):
	Fx = np.load(fname+'_x.npy' )
	Fy = np.load(fname+'_y.npy' )
	Fz = np.load(fname+'_z.npy' )
	FF = packVecGrid( Fx, Fy, Fz, FF )
	del Fx,Fy,Fz
	return FF

def saveVecFieldXsf( fname, FF, lvec, head = XSF_HEAD_DEFAULT ):
	saveXSF(fname+'_x.xsf', FF[:,:,:,0], lvec, head )
	saveXSF(fname+'_y.xsf', FF[:,:,:,1], lvec, head )
	saveXSF(fname+'_z.xsf', FF[:,:,:,2], lvec, head )

def saveVecFieldNpy( fname, FF ):
	np.save(fname+'_x.npy', FF[:,:,:,0] )
	np.save(fname+'_y.npy', FF[:,:,:,1] )
	np.save(fname+'_z.npy', FF[:,:,:,2] )























