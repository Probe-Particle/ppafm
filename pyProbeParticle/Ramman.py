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

#  RammanAmplitudes( int npos, double* tpos, double* As, int na, double* apos, double* alphas, double* mode ){
lib.RammanAmplitudes.argtypes = [ c_int, array2d, array1d, c_int, array2d, array2d, array2d ]
lib.RammanAmplitudes.restype  = None
def RammanAmplitudes( tpos, apos, alphas, modes, out=None, imode=0 ):
    npos = len(tpos)
    na   = len(apos) 
    if out is None:
        out=np.zeros(npos)
    lib.RammanAmplitudes( npos, tpos, out,   na,  apos, alphas, modes, imode )
    return out


def error( s ):
    print( "ERROR: "+s )
    exit()

# /home/prokop/Desktop/PROJECT_NOW/PhotonMap_Svec/Raman

def RunRaman( tpos, wdir='./', imode=0, fname_geom='input.xyz', fname_modes='data_NModes.dat', fname_alphas='data_Dalpha-wrt-X.dat' ):
    from . import atomicUtils as au
    alphas = np.genfromtxt(wdir+fname_alphas)
    modes  = np.genfromtxt(wdir+fname_modes)
    #AM = np.dot( alphas, modes )
    #print( "AM.shape ", AM.shape )
    modes  = modes .transpose().copy()   # which order?   expected is (xx,yy,zz, yz,xz,xy)
    alphas = alphas.transpose().copy()   # which order?   expected is (xx,yy,zz, yz,xz,xy)
    #print( "alphas.shape ", alphas.shape )
    #print( "modes.shape ",  modes.shape  )
    #mode = modes[imode].copy()
    apos,Zs,enames,qs = au.loadAtomsNP(wdir+fname_geom)
    #print( " natom ", len(apos), len(apos)*3 )
    na = apos.shape[0]
    if (modes .shape[1]!=na*3): error( "each mode must have 3*N components: natom*3= "+(na*3)+" modes.shape "+str(modes.shape)  )
    if (alphas.shape[0]!=na*3): error( "there must be 3*N atomic polarizability matrices: natom*3= "+(na*3)+" alphas.shape "+str(alphas.shape)  )
    As = RammanAmplitudes( tpos, apos, alphas, modes, imode=imode )   #;print(As)
    return As

#if __name__ == "__main__":
#    RunRaman( wdir='./home/prokop/Desktop/PROJECT_NOW/PhotonMap_Svec/Raman/Polarizabilities_Sofia/polariz-g16_sofia/2.product_matrices/test_PTCDA/'  )