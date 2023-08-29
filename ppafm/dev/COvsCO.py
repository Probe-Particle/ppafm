'''

'''

import ctypes
import sys
from ctypes import c_double, c_int

import numpy as np

from .. import cpp_utils

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
    "double* getPoss   ()",
    "double* getAnchors()",
    "double* getForces ()",
    "double* getVels   ()",
    "double* getSTMcoefs ()",

    "double* getLRadials()",
    "double* getKRadials()",
    "double* getKSprings()",
    "double* getREQs    ()",

    "void init(int natom){",
    "double setFF(double R0, double E0, double kR, double kxy, double beta ){",
    "int relaxNsteps( int nsteps, double Fconv, int ialg ){",
    "void scan( int np, double* tip_pos_, double* PP_pos_, double* tip_forces_, double* STMamp_, int nsteps, double Fconv, int ialg ){",
]

libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )

cpp_name='COvsCO'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )
const_eVA_SI = 16.0217662;

# ========= C functions

#  double* getPoss   ()
lib.getPoss   .argtypes  = []
lib.getPoss   .restype   =  c_double_p
def getPoss   (n):
    return np.ctypeslib.as_array( lib.getPoss(), shape=(n,3))

#  double* getAnchors()
lib.getAnchors.argtypes  = []
lib.getAnchors.restype   =  c_double_p
def getAnchors(n):
    return np.ctypeslib.as_array( lib.getAnchors(), shape=(n,3))

#  double* getForces ()
lib.getForces .argtypes  = []
lib.getForces .restype   =  c_double_p
def getForces (n):
    return np.ctypeslib.as_array( lib.getForces(), shape=(n,3))

#  double* getVels   ()
lib.getVels   .argtypes  = []
lib.getVels   .restype   =  c_double_p
def getVels   (n):
    return np.ctypeslib.as_array( lib.getVels(), shape=(n,3))

#  double* getSTMCoefs   ()
lib.getSTMCoefs.argtypes = []
lib.getSTMCoefs.restype  =  c_double_p
def getSTMCoefs(n):
    return np.ctypeslib.as_array( lib.getSTMCoefs(), shape=(n,4))

#  double* getLRadials()
lib.getLRadials.argtypes  = []
lib.getLRadials.restype   =  c_double_p
def getLRadials(n):
    return np.ctypeslib.as_array( lib.getLRadials() , shape=(n,))

#  double* getKRadials()
lib.getKRadials.argtypes  = []
lib.getKRadials.restype   =  c_double_p
def getKRadials(n):
    return np.ctypeslib.as_array( lib.getKRadials() , shape=(n,))

#  double* getKSprings()
lib.getKSprings.argtypes  = []
lib.getKSprings.restype   =  c_double_p
def getKSprings(n):
    return np.ctypeslib.as_array( lib.getKSprings() , shape=(n,3))

#  double* getKSprings()
lib.getPos0s.argtypes  = []
lib.getPos0s.restype   =  c_double_p
def getPos0s(n):
    return np.ctypeslib.as_array( lib.getPos0s() , shape=(n,3))

#  double* getREQs    ()
lib.getREQs    .argtypes  = []
lib.getREQs    .restype   =  c_double_p
def getREQs    (n):
    return np.ctypeslib.as_array(  lib.getREQs    () , shape=(n,3))

#  void init(int natom){
lib.init.argtypes  = [c_int]
lib.init.restype   =  None
def init(natom):
    return lib.init(natom)

#  void getSTM(){
lib.getSTM.argtypes  = []
lib.getSTM.restype   =  c_double
def getSTM():
    return lib.getSTM()

#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double]
lib.setupOpt.restype   =  None
def setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 ):
    lib.setupOpt(dt, damp, f_limit, l_limit)

#  double relaxNsteps( int nsteps, double Fconv, int ialg ){
lib.relaxNsteps.argtypes  = [c_int, c_double, c_int]
lib.relaxNsteps.restype   =  c_int
def relaxNsteps( nsteps=100, Fconv=1e-5, ialg=0):
    return lib.relaxNsteps(nsteps, Fconv, ialg)

#  void scan( int np, double* tip_pos_, double* PP_pos_, double* tip_forces_, double* STMamp, int nsteps, double Fconv, int ialg ){
lib.scan.argtypes  = [c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_int, c_double, c_int]
lib.scan.restype   =  None
def scan( tip_pos, PP_pos=None, tip_forces=None, STMamp=None, STMchans=None, nsteps=100, Fconv=1e-6, ialg=-1 ):
    N = len(tip_pos)
    if PP_pos     is None: PP_pos     = np.zeros(tip_pos.shape)
    if tip_forces is None: tip_forces = np.zeros(tip_pos.shape)
    if STMamp     is None: STMamp     = np.zeros( (tip_pos.shape[0],) )
    if STMchans   is None: STMchans   = np.zeros( (tip_pos.shape[0],5) )
    lib.scan(N, _np_as(tip_pos,c_double_p), _np_as(PP_pos,c_double_p), _np_as(tip_forces,c_double_p),  _np_as(STMamp,c_double_p), _np_as(STMchans,c_double_p),  nsteps, Fconv, ialg)
    return tip_forces, PP_pos, STMamp, STMchans

def scan2D( Xs, Ys, z0, dz=0.2, nz=20, nsteps=100, Fconv=1e-6, ialg=-1 ):
    sh = Xs.shape
    print( " Xs.shape ", Xs.shape )
    tip_pos = np.zeros( (nz,3) )
    tip_pos[:,2] = np.arange( z0, z0-dz*nz, -dz )    ; print( "tip_pos ", tip_pos )
    PP_pos     = np.zeros(tip_pos.shape)
    tip_forces = np.zeros(tip_pos.shape)
    STMamp     = np.zeros( (tip_pos.shape[0],) )
    STMchans     = np.zeros( (tip_pos.shape[0],5) )
    AFM = np.zeros( sh + (len(tip_pos),) )
    STM = np.zeros( sh + (len(tip_pos),) )
    STMch = np.zeros( sh + (len(tip_pos),5) )
    PPz = np.zeros( sh + (len(tip_pos),) )
    print( "AFM.shape ", AFM.shape )
    for ix in range(sh[0]):
        print( "ix ", ix )
        for iy in range(sh[1]):
            print( "ix,iy ", ix, iy )
            tip_pos[:,0] = Xs[ix,iy]
            tip_pos[:,1] = Ys[ix,iy]
            scan( tip_pos, PP_pos=PP_pos, tip_forces=tip_forces, STMamp=STMamp, STMchans=STMchans, nsteps=nsteps, Fconv=Fconv, ialg=ialg )
            AFM[ix,iy,:] = tip_forces[:,2]
            STM[ix,iy,:] = STMamp[:]
            STMch[ix,iy,:,:] = STMchans[:,:]
            PPz[ix,iy,:] = PP_pos[:,2]
    return AFM, STM, PPz, STMch

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # COs in surface lattice coordinates
    COs = [[0,0],[1,0],[0,1],[1,1],[2,0],]
    alat = 4.3 # [A]
    avec = np.array( [0.0,1.0,0.0] )*alat
    bvec = np.array( [np.sqrt(3)/2,-0.5,0.0] )*alat

    COpos = np.zeros( (len(COs)+1,3) )
    for i,p in enumerate(COs):
        COpos[i,:] = avec*p[0] + bvec*p[1]
    COpos[-1,2] = 15.0

    nCO = len(COpos)

    ff = sys.modules[__name__]
    print( " ff.init( nCO-1 ) " )
    ff.init( nCO-1 )
    ff.setupOpt(dt=0.4, damp=0.1, f_limit=10.0, l_limit=0.2 )

    poss     = ff.getPoss    ( nCO )
    anchors  = ff.getAnchors ( nCO )
    STMCoefs = ff.getSTMCoefs( nCO )

    lRadials = ff.getLRadials(nCO)
    kRadials = ff.getKRadials(nCO)
    pos0s    = ff.getPos0s   (nCO)
    kSprings = ff.getKSprings(nCO)
    REQs     = ff.getREQs    (nCO)

    lRadials[:] = 4.0
    kRadials[:] = 30.0/const_eVA_SI
    kSprings[:,0] = 0.5/const_eVA_SI
    kSprings[:,1] = 0.5/const_eVA_SI
    kSprings[:,2] = 0.0
    pos0s   [:] = 0.0
    REQs    [:,0] = 1.6612     # Rii
    REQs    [:,1] = np.sqrt(0.009106)   # Eii
    REQs    [:,2] = 0          # Q

    STMCoefs[:,:] = 0.0
    STMCoefs[:,2] = 1.0  # pz

    poss[:,:]    = COpos[:,:]
    anchors[:,:] = COpos[:,:]
    poss[:nCO-1,2] += 4.0
    poss[ nCO-1,2] -= 4.0

    print( "anchors \n", anchors )
    print( "poss \n",    poss    )

    beta = 1
    Zs  = np.linspace( 14, 8, 60 )
    tip_pos = np.zeros( (60,3) )
    tip_pos[:,2] = Zs
    tip_forces, PPpos, amp_STM, STMchans = ff.scan( tip_pos, nsteps=100, Fconv=1e-6, ialg=0 )

    print( "shape { tip_forces, PPpos, amp_STM, STMchans  }: \n", tip_forces.shape, PPpos.shape, amp_STM.shape, STMchans.shape )

    print( "amp_STM ", amp_STM )
    print( "STMchans ", STMchans )

    A = 10.0
    STMsamp = A*np.exp(  -beta * PPpos[:,2]  )
    STMtot  = amp_STM - STMsamp

    plt.figure(figsize=(5,10))
    plt.subplot(2,1,1)
    plt.plot( Zs, PPpos[:,0],'g', label="X" )
    plt.plot( Zs, tip_forces[:,2]*100.0, label="Fz" )
    plt.grid(); plt.legend()
    plt.subplot(2,1,2)
    plt.plot( Zs, np.log(STMchans[:,0]**2), ':r', label="chan_px" )
    plt.plot( Zs, np.log(STMchans[:,1]**2), ':g', label="chan_py" )
    plt.plot( Zs, np.log(STMchans[:,2]**2), ':b', label="chan_pz" )
    plt.plot( Zs, np.log(STMchans[:,3]**2), ':k', label="chan_s" )
    plt.grid(); plt.legend()
    plt.savefig( 'STM_1D.png', bbox_inches='tight' )
    plt.show()
    plt.close()

    xs = np.linspace( -5, 10.0, 75  )
    ys = np.linspace( -4, 16.0, 100 )
    Xs,Ys = np.meshgrid( xs, ys )
    AFM, STM, PPz, STMchs = ff.scan2D(  Xs, Ys, z0=13.0, dz=0.2, nz=20, ialg=0 )

    beta= 1.0
    STM -= np.exp( -beta*PPz )

    I = STM**2

    print( "plotting .... " )
    for iz in range(AFM.shape[2]):
        fname = "AFM_%03i.png" %iz
        print( "plotting ", fname )
        plt.imshow( AFM[:,:, iz] )
        plt.savefig( fname , bbox_inches='tight' )

        fname = "STM_%03i.png" %iz
        print( "plotting ", fname )
        plt.imshow( I[:,:, iz] )
        plt.savefig( fname , bbox_inches='tight' )

        plt.imshow( STMchs[:,:,iz,0]**2 + STMchs[:,:,iz,1]**2  ); plt.savefig( "STM_xy_%03i.png" %iz , bbox_inches='tight' )
        plt.imshow( STMchs[:,:,iz,2]**2 ); plt.savefig( "STM_pz_%03i.png" %iz , bbox_inches='tight' )
        plt.imshow( STMchs[:,:,iz,3]**2 ); plt.savefig( "STM_s_%03i.png"  %iz , bbox_inches='tight' )

    print( "ALL DONE!!! " )
