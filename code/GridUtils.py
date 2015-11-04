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

def recompile():
        current_directory=os.getcwd()
        os.chdir(os.path.dirname( os.path.realpath(__file__) ) )
        os.system("make GU")
        os.chdir(current_directory)

if not os.path.exists(LIB_PATH+"/"+name+ext):  # check if lib exist
	recompile()

lib    = ctypes.CDLL(LIB_PATH+"/"+name+ext )    # load dynamic librady object using ctypes 

# define used numpy array types for interfacing with C++

array1i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# ==============  Xsf

XSF_HEAD_DEFAULT = headScan='''
ATOMS
 1   0.0   0.0   0.0

BEGIN_BLOCK_DATAGRID_3D                        
   some_datagrid      
   BEGIN_DATAGRID_3D_whatever 
'''

lib.ReadNumsUpTo_C.argtypes  = [c_char_p, array1d, array1i, c_int]
lib.ReadNumsUpTo_C.restype   = c_int
def readNumsUpTo(filename, dimensions, noline):
        N_arry=np.zeros( (dimensions[0]*dimensions[1]*dimensions[2]), dtype = np.double )
        lib.ReadNumsUpTo_C( filename, N_arry, dimensions, noline )
        return N_arry

def readUpTo( filein, keyword ):
        i = 0
        linelist = []
        while True :
                line = filein.readline()
                linelist.append(line)
                i=i+1
                if      ((not line) or (keyword in line)): break;
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
        startline, head = readUpTo(filein, "DATAGRID_3D_")              # startline - number of the line with DATAGRID_3D_. Dinensions are located in the next line
	nDim = [ int(iii) for iii in filein.readline().split() ]        # reading 1 line with dimensions
	nDim.reverse()
	nDim = np.array( nDim)
	lvec = readmat(filein, 4)                                       # reading 4 lines where 1st line is origin of datagrid and 3 next lines are the cell vectors
	filein.close()
        print nDim
	print "GridUtils| Load "+fname+" using readNumsUpTo "    
	F = readNumsUpTo(fname,nDim.astype(np.int32).copy(), startline+5)

        print "GridUtils| Done"
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

def loadCUBE(fname):
        filein = open(fname )

#First two lines of the header are comments
	header1=filein.readline()
	header2=filein.readline()
	
#The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
        sth0 = filein.readline().split()
#The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
        sth1 = filein.readline().split()
        sth2 = filein.readline().split()
        sth3 = filein.readline().split()
        filein.close()
        nDim = np.array( [int(sth1[0]),int(sth2[0]),int(sth3[0])] )
        lvec = np.zeros((4, 3))
        for jj in range(3):
            lvec[0,jj]=float(sth0[jj+1])
            lvec[1,jj]=float(sth1[jj+1])*int(sth1[0])*0.529177249
            lvec[2,jj]=float(sth2[jj+1])*int(sth2[0])*0.529177249
            lvec[3,jj]=float(sth3[jj+1])*int(sth3[0])*0.529177249
        print "GridUtils| Load "+fname+" using readNumsUpTo"  
	noline = 6+int(sth0[0])
        F = readNumsUpTo(fname,nDim.astype(np.int32).copy(),noline)
        print "GridUtils| np.shape(F): ",np.shape(F)
        print "GridUtils| nDim: ",nDim
        print nDim
        FF = np.reshape(F, nDim ).transpose((2,1,0)).copy()  # Transposition of the array to have the same order of data as in XSF file
	nDim=[nDim[2],nDim[1],nDim[0]]                          # Setting up the corresponding dimensions. 
        head = []
        head.append("BEGIN_BLOCK_DATAGRID_3D \n")
        head.append("g98_3D_unknown \n")
        head.append("DATAGRID_3D_g98Cube \n")
        return FF,lvec, nDim, head
