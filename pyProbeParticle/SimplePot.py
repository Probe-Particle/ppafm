
'''
Simple interatomic potential (not forcefield - we have no derivatives)
used to predict probability map where to put new atom, considering atoms which are already there
'''

import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
import sys

from . import atomicUtils as au

if __package__ is None:
    print( " #### DEBUG #### import cpp_utils " )
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
    #from components.core import GameLoopEvents
    import cpp_utils
else:
    print( " #### DEBUG #### from . import cpp_utils " )
    #from ..components.core import GameLoopEvents
    from . import cpp_utils


c_double_p = ctypes.POINTER(c_double)
c_int_p    = ctypes.POINTER(c_int)

def _np_as(arr,atype):
    if arr is None:
        return None
    else: 
        return arr.ctypes.data_as(atype)

cpp_utils.s_numpy_data_as_call = "_np_as(%s,%s)"

# ===== To generate Interfaces automatically from headers call:
header_strings = [
"void init( int natom, int neighPerAtom, double* apos, double* Rcovs ){",
"void eval( int n, double* Es, double* pos_, double Rcov, double RvdW ){",
]
#cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces

libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )

cpp_name='SimplePot'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 

# ========= C functions

#  void init( int natom, int neighPerAtom, double* apos, double* Rcovs ){
lib.init.argtypes  = [c_int, c_int, c_double_p, c_double_p] 
lib.init.restype   =  None
def init( apos, Rcovs=0.7, neighPerAtom = 4 ):
    natom = len(apos)
    if isinstance( Rcovs, float):
        Rcovs = np.ones(natom)*Rcovs
    return lib.init(natom, neighPerAtom, _np_as(apos,c_double_p), _np_as(Rcovs,c_double_p)) 

#  void eval( int n, double* Es, double* pos_, double Rcov, double RvdW ){
lib.eval.argtypes  = [c_int, c_double_p, c_double_p, c_double, c_double] 
lib.eval.restype   =  None
def eval(pos, Es=None, Rcov=0.7, RvdW=1.8 ):
    n = len(ps)
    if Es is None:
        Es = np.empty(n)
    lib.eval(n, _np_as(Es,c_double_p), _np_as(pos,c_double_p), Rcov, RvdW)
    return Es

if __name__ == "__main__":

    def make_ps( extent, dpix=0.1, z0=0.0 ):
        xs=np.arange( extent[0],extent[1], dpix )
        ys=np.arange( extent[2],extent[3], dpix )
        ny,nx=len(ys),len(xs)
        ps = np.zeros( (ny,nx,3) )
        ps[:,:,2] = z0
        ps[:,:,0],ps[:,:,1] = np.meshgrid(xs,ys)
        ps=ps.reshape( (ny*nx,3) )
        return ps,nx,ny

    def make_ps_3D( extent, dpix=0.2, zmin=-2.0,zmax=2.0 ):
        xs=np.arange( extent[0],extent[1], dpix )
        ys=np.arange( extent[2],extent[3], dpix )
        zs=np.arange( zmin,zmax, dpix )
        nz,ny,nx=len(zs),len(ys),len(xs);  # print("nz,ny,nx ", nz,ny,nx)
        ps = np.zeros( (nz,ny,nx,3) )
        #Xs,Ys,Zs = np.meshgrid(xs,ys,zs)
        ps[:,:,:,1],ps[:,:,:,2],ps[:,:,:,0] = np.meshgrid(ys,zs,xs)
        #print( "Xs.shape ", Xs.shape )
        #ps[:,:,0],ps[:,:,1],ps[:,:,2] = np.meshgrid(xs,ys,zs)
        ps=ps.reshape( (nz*ny*nx,3) )
        ps = ps.copy()
        return ps,nx,ny,nz
    
    # ---- Prapere Imaging Grids
    extent=(0.0,20.0,-6.0,6.0)
    b3D = False
    if b3D:
        #extent=(-10.0,10.0,-10.0,10.0)
        ps,nx,ny,nz = make_ps_3D( extent, dpix=0.1 )
    else:
        import matplotlib.pyplot as plt
        ps,nx,ny    = make_ps( extent )
    print( ps.shape )

    # ---- Load Geometry
    xyzs,Zs,elems,qs = au.loadAtomsNP( 'simplePotTest.xyz' )
    #xyzs,Zs,elems,qs = au.loadAtomsNP( 'simplePotTest-.xyz' )
    Rcovs = np.ones(len(xyzs))*0.7

    # ----- Here Call SimplePot
    init( xyzs )     # this will re-allocate auxulary arrays and find neighbors to each atoms (atoms B-C)
    Es = eval( ps )  # evaluates potential for array of points (positions of atom A)

    # ----- plot or store result
    if b3D:
        import pyProbeParticle.GridUtils as GU
        Es = Es.reshape( (nz,ny,nx) )
        GU.saveXSF( 'SimplePot.xsf', Es )
    else:
        Es = Es.reshape( (ny,nx) )
        vmax=1.5
        plt.imshow(Es,origin='image', vmin=-vmax, vmax=vmax, cmap='seismic',extent=extent)
        plt.grid()
        plt.show()
