'''
Simple interatomic potential (not forcefield - we have no derivatives)
used to predict probability map where to put new atom, considering atoms which are already there
'''

import ctypes
from ctypes import c_double, c_int

import numpy as np

from .. import atomicUtils as au
from .. import cpp_utils, io

# Covalent radii of few atoms in Å
# Covalent radii revisited. Dalton Transactions, (21), 2832. doi:10.1039/b801115j
cov_radii = {1: 0.31,
             6: 0.70,
             7: 0.71,
             8: 0.66,
             9: 0.57,
             14: 1.11,
             15: 1.07,
             16: 1.05,
             17: 1.02,
             35: 1.20}

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
"int danglingToArray( double* dangs, double* pmin, double* pmax ){",
"void pickAtomWeighted( int npick, int* ipicks, int nps, double* pos_, double* Ws, double* p0_, double* K_, double kT ){",
"double randomOptAtom( int ntry, double* pos_, double* spread_, double Rcov, double RvdW ){",
"void init_random(int seed){",
"void setGridSize( int* ns_, double* pmin_, double* pmax_){",
"void setGridPointer(double* data){",
]

libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )

cpp_name='SimplePot'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes

# ========= C functions

#  void init_random(int seed){
lib.init_random.argtypes  = [c_int]
lib.init_random.restype   =  None
def init_random(seed):
    return lib.init_random(seed)

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
    n = len(pos)
    if Es is None:
        Es = np.empty(n)
    lib.eval(n, _np_as(Es,c_double_p), _np_as(pos,c_double_p), Rcov, RvdW)
    return Es

#  int danglingToArray( double* dangs, double* pmin, double* pmax ){
lib.danglingToArray.argtypes  = [c_double_p, c_double_p, c_double_p, c_double]
lib.danglingToArray.restype   =  c_int
def danglingToArray(dangs=None, pmin=None, pmax=None, npmax=1000, Rcov=0.3):
    if dangs is None: dangs = np.zeros((npmax,3))
    if pmin is None: pmin=np.array([-1e+300,-1e+300,-1e+300])
    if pmax is None: pmin=np.array([+1e+300,+1e+300,+1e+300])
    n = lib.danglingToArray(_np_as(dangs,c_double_p), _np_as(pmin,c_double_p), _np_as(pmax,c_double_p), Rcov )
    return dangs[:n].copy()

#  void pickAtomWeighted( int npick, int* ipicks, int nps, double* pos_, double* Ws, double* p0_, double* K_, double kT ){
lib.pickAtomWeighted.argtypes  = [c_int, c_int_p, c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_double]
lib.pickAtomWeighted.restype   =  None
def pickAtomWeighted( poss, npick=1, ipicks=None, Ws=None, p0=[0,0,0], Ks=[0.2,0.2,1.0], kT=1.0 ):
    nps = len(poss)
    p0 =np.array(p0).copy()
    Ks =np.array(Ks).copy()
    if Ws     is None: Ws     = np.empty( nps )
    if ipicks is None: ipicks = np.empty( npick, dtype=np.int32 )
    lib.pickAtomWeighted(npick, _np_as(ipicks,c_int_p), nps, _np_as(poss,c_double_p), _np_as(Ws,c_double_p), _np_as(p0,c_double_p), _np_as(Ks,c_double_p), kT)
    return ipicks

#  double randomOptAtom( int ntry, double* pos_, double* spread_, double Rcov, double RvdW ){
lib.randomOptAtom.argtypes  = [c_int, c_double_p, c_double_p, c_double, c_double]
lib.randomOptAtom.restype   =  c_double
def randomOptAtom( pos, spread=[1.0,1.0,1.0], Rcov=0.7, RvdW=1.8, ntry=100 ):
    pos   =np.array(pos).copy()
    spread=np.array(spread).copy()
    lib.randomOptAtom(ntry, _np_as(pos,c_double_p), _np_as(spread,c_double_p), Rcov, RvdW)
    return pos

def genAtoms( p0=[0.0,0.0,0.0], kT=1.0, Ks=[0.2,0.2,1.0], spread=[1.0,1.0,1.0], Rcov=0.7, RvdW=1.8, npick=10, ntry=100 ):
    poss   = danglingToArray( Rcov=Rcov )
    ipicks     = pickAtomWeighted( poss, npick=npick, p0=p0, Ks=Ks, kT=kT )
    #p = randomOptAtom( poss[ipick], spread, Rcov=Rcov, RvdW=RvdW, ntry=ntry )
    ps = poss[ipicks]
    #print( "genAtom p ", p )
    return ps

def genAtomWs( kT=1.0, Rcov=0.7, npick=10, natom=1000 ):
    #poss   = danglingToArray( Rcov=Rcov )
    poss = np.random.rand(natom,3)
    poss[:,0] *= 16; poss[:,0] += 2.0
    poss[:,1] *= 16; poss[:,1] += 2.0
    poss[:,2] *= 10.0; poss[:,2] += -5.0
    Ws         = np.empty( len(poss) )
    print( "Ws.shape ", Ws.shape )
    ipicks     = pickAtomWeighted( poss, Ws=Ws, npick=npick, kT=kT )
    return poss, Ws

#  void setGridSize( int* ns_, double* pmin_, double* pmax_){
lib.setGridSize.argtypes  = [c_int_p, c_double_p, c_double_p]
lib.setGridSize.restype   =  None
def setGridSize(ns, pmin, pmax,z0=-7.0):
    ns=np.array(ns,dtype=np.int32)
    pmin=np.array(pmin); pmin[2]+=z0
    pmax=np.array(pmax); pmax[2]+=z0
    return lib.setGridSize(_np_as(ns,c_int_p), _np_as(pmin,c_double_p), _np_as(pmax,c_double_p))

#  void setGridPointer(double* data){
lib.setGridPointer.argtypes  = [c_double_p]
lib.setGridPointer.restype   =  None
def setGridPointer(data):
    return lib.setGridPointer(_np_as(data,c_double_p))


class SimplePotential:
    def __init__(self, grid_size, grid_window):
        self.grid_size = grid_size
        self.grid_window = grid_window
        self.ps, self.nx, self.ny, self.nz = self.make_ps_3d()
        self.xyzs = None

    def init_molecule(self, xyzs, Zs):
        self.xyzs = xyzs
        covr = []
        for z in Zs:
            covr.append(cov_radii[z])
        rcovs = np.ones(len(self.xyzs))*np.array(covr)
        init(self.xyzs, Rcovs=rcovs)

    def calc_potential(self, z_added=1):
        e = eval(self.ps, Rcov=cov_radii[z_added])
        return e

    def handle_positions(self):
        gw = self.grid_window
        grid_center = np.array([gw[1][0] + gw[0][0], gw[1][1] + gw[0][1]]) / 2
        self.xyzs[:,:2] += grid_center - self.xyzs[:,:2].mean(axis=0)

    def make_ps_3d(self):
        xsi = np.linspace(self.grid_window[0][0], self.grid_window[1][0], self.grid_size[0])
        ysi = np.linspace(self.grid_window[0][1], self.grid_window[1][1], self.grid_size[1])
        zsi = np.linspace(self.grid_window[0][2], self.grid_window[1][2], self.grid_size[2])
        nxi, nyi, nzi = xsi.shape[0], ysi.shape[0], zsi.shape[0]
        psi = np.zeros((nzi, nyi, nxi, 3))
        psi[:, :, :, 1], psi[:, :, :, 2], psi[:, :, :, 0] = np.meshgrid(ysi, zsi, xsi)
        psi = psi.copy().reshape((nzi*nyi*nxi, 3))

        return psi, xsi, ysi, zsi

    def __call__(self, z_added=1):
        return self.calc_potential(z_added)

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
        nz,ny,nx=len(zs),len(ys),len(xs)
        ps = np.zeros( (nz,ny,nx,3) )
        ps[:,:,:,1],ps[:,:,:,2],ps[:,:,:,0] = np.meshgrid(ys,zs,xs)
        ps=ps.reshape( (nz*ny*nx,3) )
        ps = ps.copy()
        return ps,nx,ny,nz

    # ---- Prapere Imaging Grids
    extent=(0.0,20.0,-6.0,6.0)
    b3D = False
    if b3D:
        ps,nx,ny,nz = make_ps_3D( extent, dpix=0.1 )
    else:
        import matplotlib.pyplot as plt
        ps,nx,ny    = make_ps( extent )
    print( ps.shape )

    # ---- Load Geometry
    xyzs, Zs, qs, _ = io.loadXYZ('fail.xyz')
    elems = au.ZsToElems(Zs)
    Rcovs = np.ones(len(xyzs))*0.7

    # ----- Here Call SimplePot
    init( xyzs )     # this will re-allocate auxulary arrays and find neighbors to each atoms (atoms B-C)
    Es = eval( ps )  # evaluates potential for array of points (positions of atom A)
    dangs = danglingToArray()
    print( "dangs\n", dangs.shape, xyzs.shape )
    io.saveXYZ('dangs.xyz', np.concatenate( (xyzs, dangs) ), elems+['H']*len(dangs))

    # ----- plot or store result
    if b3D:
        from ppafm import io
        Es = Es.reshape( (nz,ny,nx) )
        io.saveXSF( 'SimplePot.xsf', Es )
    else:
        Es = Es.reshape( (ny,nx) )
        vmax=1.5
        plt.imshow(Es,origin='upper', vmin=-vmax, vmax=vmax, cmap='seismic',extent=extent)
        plt.grid()
        plt.show()
