#!/usr/bin/python

import pylab
import ctypes
from   ctypes import *
import os
import numpy as np




name=os.path.dirname(__file__)+'/GridUtils_lib'
ext='.so'

def recompile():
        current_directory=os.getcwd()
        os.chdir(os.path.dirname(__file__))
        os.system("make GU")
        os.chdir(current_directory)

def readUpTo( filein, keyword ):
        i = 0
        linelist = []
        while True :
                line = filein.readline()
                linelist.append(line)
                i=i+1
                if      ((not line) or (keyword in line)): break;
        return i,linelist

def readNumsUpTo(filename, dimensions, noline):
        recompile()
        lib    = ctypes.CDLL(name+ext )
        array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array1i = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
        lib.ReadNumsUpTo_C.argtypes  = [c_char_p, array1d, array1i, c_int]
        lib.ReadNumsUpTo_C.restype   = c_int
        N_arry=np.zeros((dimensions[0]*dimensions[1]*dimensions[2]), dtype = np.double)
        lib.ReadNumsUpTo_C(filename, N_arry, dimensions, noline)
        return pylab.array(N_arry)

def readmat(filein, n):
        temp = []
        for i in range(n):
                temp.append( [ float(iii) for iii in filein.readline().split() ] )
        return pylab.array(temp)

def loadXSF(fname):
	filein = open(fname )
	startline, head = readUpTo(filein, "DATAGRID_3D_")              # startline - number of the line with DATAGRID_3D_. Dinensions are located in the next line
        print startline
	nDim = [ int(iii) for iii in filein.readline().split() ]        # reading 1 line with dimensions
	nDim.reverse()
	nDim = pylab.array( nDim)
	lvec = readmat(filein, 4)                                       # reading 4 lines where 1st line is origin of datagrid and 3 next lines are the cell vectors
	filein.close()
        print nDim
	print "GridUtils| Load "+fname+" using readNumsUpTo "    
	F = readNumsUpTo(fname,nDim.astype(np.int32).copy(), startline+5)

        print "GridUtils| Done"
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
        nDim = pylab.array( [int(sth1[0]),int(sth2[0]),int(sth3[0])] )
        lvec = np.zeros((4, 3))
        for jj in range(3):
            lvec[0,jj]=float(sth0[jj+1])
            lvec[1,jj]=float(sth1[jj+1])*int(sth1[0])*0.529177249
            lvec[2,jj]=float(sth2[jj+1])*int(sth2[0])*0.529177249
            lvec[3,jj]=float(sth3[jj+1])*int(sth3[0])*0.529177249
        print "GridUtils| Load "+fname+" using readNumsUpTo"  
	noline = 6+int(sth0[0])
        F = readNumsUpTo(fname,nDim.astype(np.int32).copy(),noline)
        print "GridUtils| pylab.shape(F)",pylab.shape(F)
        print "GridUtils| nDim",nDim
        print nDim
        FF = pylab.reshape(F, nDim ).transpose((2,1,0)).copy()  # Transposition of the array to have the same order of data as in XSF file
	nDim=[nDim[2],nDim[1],nDim[0]]                          # Setting up the corresponding dimensions. 
        head = []
        head.append("BEGIN_BLOCK_DATAGRID_3D \n")
        head.append("g98_3D_unknown \n")
        head.append("DATAGRID_3D_g98Cube \n")
        return FF,lvec, nDim, head
