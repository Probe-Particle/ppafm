#!/usr/bin/python

import ctypes
from ctypes import c_char_p, c_double, c_int

import numpy as np

from . import cpp_utils

# ==============================

bohrRadius2angstroem = 0.5291772109217
Hartree2eV           = 27.211396132

# ============================== interface to C++ core

cpp_name='GridUtils'
cpp_utils.make("GU")
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes

# define used numpy array types for interfacing with C++
array1i = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array1d = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')
array4d = np.ctypeslib.ndpointer(dtype=np.double, ndim=4, flags='CONTIGUOUS')


# ============== Filters

def renorSlice( F ):
	vranges = []
	for i in range( len(F) ):
		Fi = F[i]
		vmin = np.nanmin( Fi )
		vmax = np.nanmax( Fi )
		F[i] -= vmin;
		F[i] /= ( vmax - vmin )
		vranges.append( (vmin,vmax) )
	return vranges

# ==============  Cutting, Sampling, Interpolation ...

#	void interpolate_gridCoord( int n, Vec3d * pos_list, double * data )
lib.interpolate_gridCoord.argtypes = [ c_int, array2d, array3d, array1d ]
lib.interpolate_gridCoord.restype   = None
interpolate_gridCoord               = lib.interpolate_gridCoord

#	void interpolateLine_gridCoord( int n, Vec3d * p1, Vec3d * p2, double * data, double * out )
lib.interpolateLine_gridCoord.argtypes = [ c_int, array1d, array1d, array3d, array1d ]
lib.interpolateLine_gridCoord.restype  = None
#interpolateLine_gridCoord                   = lib.interpolateLine_gridCoord

#	void interpolateLine_gridCoord( int n, Vec3d * p1, Vec3d * p2, double * data, double * out )
lib.interpolateLine_cartes.argtypes = [ c_int, array1d, array1d, array3d, array1d ]
lib.interpolateLine_cartes.restype  = None
#interpolateLine_gridCoord                   = lib.interpolateLine_gridCoord

#	void interpolateQuad_gridCoord( int * nij, Vec3d * p00, Vec3d * p01, Vec3d * p10, Vec3d * p11, double * data, double * out )
lib.interpolateQuad_gridCoord.argtypes = [ array1i, array1d, array1d, array1d, array1d, array3d, array2d ]
lib.interpolateQuad_gridCoord.restype  = None
#interpolateQuad_gridCoord              = lib.interpolateQuad_gridCoord

#	void interpolate_cartesian( int n, Vec3d * pos_list, double * data, double * out )
lib.interpolate_cartesian.argtypes  = [ c_int, array4d, array3d, array3d  ]
lib.interpolate_cartesian.restype   = None
#interpolate_cartesian               = lib.interpolate_cartesian

#	void setGridCell( double * cell )
lib.setGridCell.argtypes  = [array2d]
lib.setGridCell.restype   = None
setGridCell = lib.setGridCell

#	void setGridN( int * n )
lib.setGridN.argtypes  = [array1i]
lib.setGridN.restype   = None
setGridN = lib.setGridN

def interpolateLine( F, p1, p2, sz=500, cartesian=False ):
	result = np.zeros( sz )
	p00 = np.array ( p1, dtype='float64' )
	p01 = np.array ( p2, dtype='float64' )
	if( cartesian ):
		lib.interpolateLine_cartes   ( sz, p00, p01, F, result )
	else:
		lib.interpolateLine_gridCoord( sz, p00, p01, F, result )
	return result

def interpolateQuad( F, p00, p01, p10, p11, sz=(500,500) ):
    result = np.zeros( sz )
    npxy   = np.array( sz, dtype='int32' )
    p00 = np.array ( p00, dtype='float64' )
    p01 = np.array ( p01, dtype='float64' )
    p10 = np.array ( p10, dtype='float64' )
    p11 = np.array ( p11, dtype='float64' )
    lib.interpolateQuad_gridCoord( npxy, p00, p01, p10, p11, F, result )
    return result

def interpolate_cartesian( F, pos, cell=None, result=None ):
	if cell is not None:
		setGridCell( cell )
	nDim = np.array(pos.shape)
	print(nDim)
	if result is None:
		result = np.zeros( (nDim[0],nDim[1],nDim[2]) )
	n  = nDim[0]*nDim[1]*nDim[2]
	lib.interpolate_cartesian( n, pos, F, result )
	return result

def verticalCut( F, p1, p2, sz=(500,500) ):
	result = np.zeros( sz )
	npxy   = npxy = np.array( sz, dtype='int32' )
	p00 = np.array ( ( p1[0],p1[1],p1[2] ), dtype='float64' )
	p01 = np.array ( ( p2[0],p2[1],p1[2] ), dtype='float64' )
	p10 = np.array ( ( p1[0],p1[1],p2[2] ), dtype='float64' )
	p11 = np.array ( ( p2[0],p2[1],p2[2] ), dtype='float64' )
	lib.interpolateQuad_gridCoord( npxy, p00, p01, p10, p11, F, result )
	return result

def dens2Q_CHGCARxsf(data, lvec):
    nDim = data.shape
    Ntot = nDim[0]*nDim[1]*nDim[2]
    Vtot = np.linalg.det( lvec[1:] )
    print("dens2Q Volume    : ", Vtot)
    print("dens2Q Ntot      : ", Ntot)
    print("dens2Q Vtot/Ntot : ", Vtot/Ntot)
    #Qsum = rho1.sum()
    return Vtot/Ntot

#double cog( double * data_, double* center ){
lib.cog.argtypes = [ array3d, array1d ]
lib.cog.restype  = c_double
def cog( data ):
    center = np.zeros(3)
    Hsum = lib.cog( data, center );
    return center, Hsum

#sphericalHist( double * data_, double* center, double dr, int n, double* Hs, double* Ws ){
lib.sphericalHist.argtypes = [ array3d, array1d, c_double, c_int, array1d, array1d ]
lib.sphericalHist.restype  = None
def sphericalHist( data, center, dr, n ):
    Hs = np.zeros(n); Ws = np.zeros(n);
    rs = np.arange( 0, n ) * dr
    lib.sphericalHist( data, center, dr, n, Hs, Ws );
    return rs, Hs, Ws

# ==============  String / File IO utils

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

#    int ReadNumsUpTo_C (char *fname, double *numbers, int * dims, int noline)
lib.ReadNumsUpTo_C.argtypes  = [c_char_p, array1d, array1i, c_int]
lib.ReadNumsUpTo_C.restype   = c_int
def readNumsUpTo(filename, dimensions, noline):
	N_arry=np.zeros( (dimensions[0]*dimensions[1]*dimensions[2]), dtype = np.double )
	lib.ReadNumsUpTo_C( filename.encode(), N_arry, dimensions, noline )
	return N_arry

# =================== binary dbl (double)

def parseNameString( name ):
	words = name.split("_")
	#print words
	prefix = words[0]
	shape = [ int(word) for word in words[1:] ]
	return prefix,shape

def loadFromDbl( name ):
	prefix,ndim = parseNameString( name )
	F = np.fromfile ( name+'.dbl' )
	F = np.reshape  ( F, (ndim[2],ndim[1],ndim[0]) )
	F = np.transpose( F, (2,1,0) )
	return np.ascontiguousarray( F )

# =================== XSF

XSF_HEAD_DEFAULT = headScan='''
ATOMS
 1   0.0   0.0   0.0

BEGIN_BLOCK_DATAGRID_3D
   some_datagrid
   BEGIN_DATAGRID_3D_whatever
'''

def orthoLvec( sh, dd ):
    return [
        [0,0,0],
        [sh[2]*dd[0],0,0],
        [0,sh[1]*dd[1],0],
        [0,0,sh[0]*dd[2]]
    ]


def saveXSF(fname, data, lvec=None, dd=None, head=XSF_HEAD_DEFAULT, verbose=1 ):
    if verbose > 0: print("Saving xsf", fname)
    fileout = open(fname, 'w')
    if lvec is None:
        if dd is None:
            dd=[1.0,1.0,1.0]
        lvec = orthoLvec( data.shape, dd )
    for line in head:
        fileout.write(line)
    nDim = np.shape(data)
    writeArr (fileout, (nDim[2]+1,nDim[1]+1,nDim[0]+1) )
    writeArr2D(fileout,lvec)
    data2 = np.zeros(np.array(nDim)+1);   # These crazy 3 lines are here since the first and the last cube
    data2[:-1,:-1,:-1] = data;  # in XSF in every direction is the same
    data2[-1,:,:]=data2[0,:,:];data2[:,-1,:]=data2[:,0,:];data2[:,:,-1]=data2[:,:,0];
    for r in data2.flat:
        fileout.write( "%10.5e\n" % r )
    fileout.write ("   END_DATAGRID_3D\n")
    fileout.write ("END_BLOCK_DATAGRID_3D\n")

def loadXSF(fname, xyz_order=False, verbose=True):
	filein = open( fname )
	startline, head = readUpTo(filein, "BEGIN_DATAGRID_3D")              # startline - number of the line with DATAGRID_3D_. Dinensions are located in the next line
	nDim = [ int(iii) for iii in filein.readline().split() ]        # reading 1 line with dimensions
	nDim.reverse()
	nDim = np.array( nDim)
	lvec = readmat(filein, 4)                                       # reading 4 lines where 1st line is origin of datagrid and 3 next lines are the cell vectors
	filein.close()
	if verbose: print("nDim xsf (= nDim + [1,1,1] ):", nDim)
	if verbose: print("GridUtils| Load "+fname+" using readNumsUpTo ")
	F = readNumsUpTo(fname,nDim.astype(np.int32).copy(), startline+5)
	if verbose: print("GridUtils| Done")
	FF = np.reshape(F, nDim)[:-1,:-1,:-1]
	if xyz_order:
		FF = FF.transpose((2, 1, 0))
    # FF is not C_CONTIGUOUS without copy
	FF = FF.copy()
	return FF, lvec, nDim-1, head

def getFromHead_PRIMCOORD( head ):
	Zs = None; Rs = None;
	for i,line in enumerate( head ):
		if "PRIMCOORD" in line:
			natoms = int( head[i+1].split()[0] )
			Zs = np.zeros( natoms, dtype='int32' ); Rs = np.zeros( (natoms,3) )
			for j in range(natoms):
				words = head[i+j+2].split()
				Zs[j  ]    = int  ( words[ 0 ] )
				Rs[j,0] = float( words[ 1 ] )
				Rs[j,1] = float( words[ 2 ] )
				Rs[j,2] = float( words[ 3 ] )
	return Zs, Rs


# =================== Cube

def loadCUBE(fname, xyz_order=False, verbose=True):
	filein = open(fname )
	#First two lines of the header are comments
	filein.readline()
	filein.readline()
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
		lvec[0,jj]=float(sth0[jj+1])*bohrRadius2angstroem
		lvec[1,jj]=float(sth1[jj+1])*int(sth1[0])*bohrRadius2angstroem  # bohr_radius ?
		lvec[2,jj]=float(sth2[jj+1])*int(sth2[0])*bohrRadius2angstroem
		lvec[3,jj]=float(sth3[jj+1])*int(sth3[0])*bohrRadius2angstroem

	if verbose: print("GridUtils| Load "+fname+" using readNumsUpTo")
	noline = 6+int(sth0[0])
	F = readNumsUpTo(fname,nDim.astype(np.int32).copy(),noline)
	if verbose: print("GridUtils| np.shape(F): ",np.shape(F))
	if verbose: print("GridUtils| nDim: ", nDim)

	FF = np.reshape(F, nDim)
	if not xyz_order:
		FF = FF.transpose((2,1,0)).copy()  # Transposition of the array to have the same order of data as in XSF file

	nDim=[nDim[2],nDim[1],nDim[0]]                          # Setting up the corresponding dimensions.
	head = []
	head.append("BEGIN_BLOCK_DATAGRID_3D \n")
	head.append("g98_3D_unknown \n")
	head.append("DATAGRID_3D_g98Cube \n")
	FF*=Hartree2eV
	return FF,lvec, nDim, head

#================ WSxM output

def saveWSxM_2D(name_file, data, Xs, Ys):
	tmp_data=data.flatten()
	out_data=np.zeros((len(tmp_data),3))
	out_data[:,0]=Xs.flatten()
	out_data[:,1]=Ys.flatten()
	out_data[:,2]=tmp_data	#.copy()
	f=open(name_file,'w')
	print("WSxM file copyright Nanotec Electronica", file=f)
	print("WSxM ASCII XYZ file", file=f)
	print("X[A]  Y[A]  df[Hz]", file=f)
	print("", file=f)
	np.savetxt(f, out_data)
	f.close()

def saveWSxM_3D( prefix, data, extent, slices=None ):
	nDim=np.shape(data)
	if slices is None:
		slices=list(range( nDim[0]))
	xs=np.linspace( extent[0], extent[1], nDim[2] )
	ys=np.linspace( extent[2], extent[3], nDim[1] )
	Xs, Ys = np.meshgrid(xs,ys)
	for i in slices:
		print("slice no: ", i)
		fname = prefix+'_%03d.xyz' %i
		saveWSxM_2D(fname, data[i], Xs, Ys)

#================ Npy

def saveNpy(fname, data, lvec , head=None):
	np.save(fname+'.npy', data)
	np.save(fname+'_vec.npy',lvec)

def loadNpy(fname):
	data = np.load(fname+'.npy')
	lvec = np.load(fname+'_vec.npy')
	return data.copy(), lvec;	#necessary for being 'C_CONTINUOS'

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
	lvec = np.load(fname+'_vec.npy' )
	FF = packVecGrid( Fx, Fy, Fz, FF )
	del Fx,Fy,Fz
	return FF, lvec

def saveVecFieldXsf( fname, FF, lvec, head = XSF_HEAD_DEFAULT ):
	saveXSF(fname+'_x.xsf', FF[:,:,:,0], lvec, head )
	saveXSF(fname+'_y.xsf', FF[:,:,:,1], lvec, head )
	saveXSF(fname+'_z.xsf', FF[:,:,:,2], lvec, head )

def saveVecFieldNpy( fname, FF, lvec , head = XSF_HEAD_DEFAULT ):
	np.save(fname+'_x.npy', FF[:,:,:,0] )
	np.save(fname+'_y.npy', FF[:,:,:,1] )
	np.save(fname+'_z.npy', FF[:,:,:,2] )
	np.save(fname+'_vec.npy', lvec )
	if (head != XSF_HEAD_DEFAULT ):
		print("saving atoms")
		tmp0=head[0]; q=np.zeros(len(tmp0));    #head: [e,[x,y,z],lvec]
		np.save(fname+'_atoms.npy',[tmp0,head[1][0],head[1][1],head[1][2],q]) #atoms: [e, x, y, z, q]

def limit_vec_field( FF, Fmax=100.0 ):
	'''
	remove too large values; preserves direction of vectors
	'''
	FR   = np.sqrt( FF[:,:,:,0]**2  +  FF[:,:,:,1]**2  + FF[:,:,:,2]**2 ).flat
	mask = ( FR > Fmax )
	FF[:,:,:,0].flat[mask] *= Fmax/FR[mask]
	FF[:,:,:,1].flat[mask] *= Fmax/FR[mask]
	FF[:,:,:,2].flat[mask] *= Fmax/FR[mask]

def save_vec_field(fname, data, lvec, data_format="xsf", head = XSF_HEAD_DEFAULT ):
	'''
	Saving scalar fields into xsf, or npy
	'''
	if (data_format=="xsf"):
		saveVecFieldXsf(fname, data, lvec, head = head )
	elif (data_format=="npy"):
		saveVecFieldNpy(fname, data, lvec, head = head )
	else:
		print("I cannot save this format!")


def load_vec_field(fname, data_format="xsf"):
	'''
	Loading Vector fields into xsf, or npy
	'''
	if (data_format=="xsf"):
		data, lvec, ndim, head =loadVecFieldXsf(fname)
	elif (data_format=="npy"):
		data, lvec = loadVecFieldNpy(fname)
		ndim = np.delete(data.shape,3)
	else:
		print("I cannot load this format!")
	return data, lvec, ndim;


# =============== Scalar Fields

def save_scal_field(fname, data, lvec, data_format="xsf", head = XSF_HEAD_DEFAULT ):
	'''
	Saving scalar fields into xsf, or npy
	'''
	if (data_format=="xsf"):
		saveXSF(fname+".xsf", data, lvec, head = head)
	elif (data_format=="npy"):
		saveNpy(fname, data, lvec, head = head)
	else:
		print("I cannot save this format!")


def load_scal_field(fname, data_format="xsf"):
	'''
	Loading scalar fields into xsf, or npy
	'''
	if (data_format=="xsf"):
		data, lvec, ndim, head =loadXSF(fname+".xsf")
	elif (data_format=="npy"):
		data, lvec = loadNpy(fname)
		ndim = data.shape
	elif (data_format=="cube"):
		data,lvec, ndim, head = loadCUBE(fname+".cube")
	else:
		print("I cannot load this format!")
	return data.copy(), lvec, ndim;

# =============== Other Utils

def multArray( F, nx=2,ny=2 ):
	'''
	multiply data array "F" along second two axis (:, :*nx, :*ny )
	it is usefull to visualization of images computed in periodic supercell ( PBC )
	'''
	nF = np.shape(F)
	print("nF: ",nF)
	F_ = np.zeros( (nF[0],nF[1]*ny,nF[2]*nx) )
	for iy in range(ny):
		for ix in range(nx):
			F_[:, iy*nF[1]:(iy+1)*nF[1], ix*nF[2]:(ix+1)*nF[2]  ] = F
	return F_
