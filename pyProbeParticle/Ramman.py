#from unittest.mock import NonCallableMagicMock
import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
from . import cpp_utils

cpp_name='Ramman'
cpp_utils.make("Ramman")
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

array1ui = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS')
array1i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
array2i  = np.ctypeslib.ndpointer(dtype=np.int32,  ndim=2, flags='CONTIGUOUS')
array1d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array2d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array3d  = np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags='CONTIGUOUS')

# ========= C functions

# setEfieldMultipole( int fieldType_, double* coefs ){
lib.setEfieldMultipole.argtypes = [ c_int, array1d ]
lib.setEfieldMultipole.restype  = None
def setEfieldMultipole( coefs ):
    coefs = np.array(coefs)
    nc = len(coefs)
    imode=-1
    if  (nc==1):
        imode=1
    elif(nc==4):
        imode=2
    else: 
        print("ERROR in setEfieldMultipole: len(coefs)=%i | must be 1 or 4 " %nc ); exit()
    lib.setEfieldMultipole( imode, coefs )

#  RammanAmplitudes( int npos, double* tpos, double* As, int na, double* apos, double* alphas, double* mode ){
lib.RammanAmplitudes.argtypes = [ c_int, array2d, array1d, c_int, array2d, array2d, array2d, c_int ]
lib.RammanAmplitudes.restype  = None
def RammanAmplitudes( tpos, apos, alphas, modes, out=None, imode=0 ):
    npos = len(tpos)
    na   = len(apos) 
    if out is None:
        out=np.zeros(npos)
    lib.RammanAmplitudes( npos, tpos, out,   na,  apos, alphas, modes, imode )
    return out

# double RammanDetails( double* tpos, int na, double* apos, double* alphas, double* modes, int imode, double* E_incident, double* E_induced ){
lib.RammanDetails.argtypes = [ array1d, c_int, array2d, array2d, array2d, c_int, array2d, array2d, array2d ]
lib.RammanDetails.restype  = c_double
def RammanDetails( tpos, apos, alphas, modes, imode=0, Einc=None, Eind=None, modeOut=None ):
    tpos=np.array(tpos)
    na   = len(apos) 
    if Einc is None: Einc=np.zeros((na,3))
    if Eind is None: Eind=np.zeros((na,3))
    if modeOut is None: modeOut=np.zeros((na,3))
    Amp = lib.RammanDetails( tpos, na,  apos, alphas, modes, imode, Einc, Eind, modeOut )
    return Amp,Einc, Eind, modeOut

# void EfieldAtPoints( int npos, double* pos, double* Es_ ){
lib.EfieldAtPoints.argtypes = [ c_int, array2d, array2d ]
lib.EfieldAtPoints.restype  = None
def EfieldAtPoints( pos, Es=None ):
    npos=len(pos)
    if Es is None:
        Es=np.zeros((npos,3))
    lib.EfieldAtPoints( npos, pos, Es )
    return Es

# double ProjectPolarizability( const Vec3d& tip_pos, int na, Vec3d * apos, SMat3d* alphas, int idir, int icomp ){
lib.ProjectPolarizability.argtypes = [ c_int, array2d, array1d, c_int, array2d, array2d, c_int, c_int ]
lib.ProjectPolarizability.restype  = None
def ProjectPolarizability( tpos, apos, alphas, out=None, idir=0, icomp=0 ):
    npos = len(tpos)
    na   = len(apos) 
    if out is None:
        out=np.zeros(npos)
    print( " idir, icomp   ", idir, icomp )
    lib.ProjectPolarizability( npos, tpos, out,   na,  apos, alphas, idir, icomp )
    return out

# void setVerbosity(int verbosity_){
lib.setVerbosity.argtypes = [ c_int ]
lib.setVerbosity.restype  = None
def setVerbosity( verbosity ):
    lib.setVerbosity( verbosity )

def error( s ):
    print( "ERROR: "+s )
    exit()

# /home/prokop/Desktop/PROJECT_NOW/PhotonMap_Svec/Raman

def reOrder(A):
    #       0  1  2  3  4  5
    # A    xx xy yy zx zy zz
    # A_   xx yy zz yz xz xy
    na3=A.shape[1]
    na=na3//3
    nxx=A.shape[0]
    A_ = np.empty( (na3,nxx) )
    A_[:,0] = A[0,:]      #  
    A_[:,1] = A[2,:]
    A_[:,2] = A[5,:]
    A_[:,3] = A[4,:]
    A_[:,4] = A[3,:]
    A_[:,5] = A[1,:]

    '''
    # exchange (z,y,x) -> (x,y,z)
    A_=A_.reshape( (na,3,nxx) )
    A =A .reshape( (na,3,nxx) )
    A[:,0,:] = A_[:,2,:]
    A[:,1,:] = A_[:,1,:]
    A[:,2,:] = A_[:,0,:]
    A=A.reshape((na3,nxx))
    '''

    #A[:,]

    return A_

def RunRaman( tpos, wdir='./', imode=0, fname_geom='input.xyz', fname_modes='data_NModes.dat', fname_alphas='data_Dalpha-wrt-X.dat' ):
    from . import atomicUtils as au
    alphas = reOrder( np.genfromtxt(wdir+fname_alphas) )  
    modes  =          np.genfromtxt(wdir+fname_modes).transpose().copy() 

    print( "alphas.shape ", alphas.shape )
    print( "alphas\n", alphas )
    print( "modes\n", modes )
    #AM = np.dot( alphas, modes )
    #print( "AM.shape ", AM.shape )
    #modes  = modes .transpose().copy()   # which order?   expected is (xx,yy,zz, yz,xz,xy)
    #alphas = alphas.transpose().copy()   # which order?   expected is (xx,yy,zz, yz,xz,xy)
    #print( "alphas.shape ", alphas.shape )
    #print( "modes.shape ",  modes.shape  )
    #mode = modes[imode].copy()
    apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)
    #print( " natom ", len(apos), len(apos)*3 )
    na = apos.shape[0]
    if (modes .shape[1]!=na*3): error( "each mode must have 3*N components: natom*3= "+(na*3)+" modes.shape "+str(modes.shape)  )
    if (alphas.shape[0]!=na*3): error( "there must be 3*N atomic polarizability matrices: natom*3= "+(na*3)+" alphas.shape "+str(alphas.shape)  )
    As = RammanAmplitudes( tpos, apos, alphas, modes, imode=imode )   #;print(As)
    return As, apos


def write_to_xyz_vecs( f, enames, apos, vecs, comment="# commnet" ):
    for i in range(na):
        f.write(  "%s %g %g %g    %g %g %g \n" %(enames[i],apos[i,0],apos[i,1],apos[i,2],  vecs[i,0],vecs[i,1],vecs[i,2])   )

def write_xyz_vecs( fname, enames, apos, vecs, comment="# commnet", tpos=None ):
    na=len(apos)
    if tpos is not None:
        na+=1
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("%i\n" %na)
        f.write(comment+"\n")
        if tpos is not None:
            f.write(  "%s %g %g %g    %g %g %g \n" %('U',tpos[0],tpos[1],tpos[2],  0,0,0)   )
        for i in range(len(apos)):
            f.write(  "%s %g %g %g    %g %g %g \n" %(enames[i],apos[i,0],apos[i,1],apos[i,2],  vecs[i,0],vecs[i,1],vecs[i,2])   )

#if __name__ == "__main__":
#    RunRaman( wdir='./home/prokop/Desktop/PROJECT_NOW/PhotonMap_Svec/Raman/Polarizabilities_Sofia/polariz-g16_sofia/2.product_matrices/test_PTCDA/'  )