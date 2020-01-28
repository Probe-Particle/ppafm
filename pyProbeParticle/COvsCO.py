'''

'''

import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
import sys

if __name__ == '__main__':
    if __package__ is None:
        sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        #from components.core import GameLoopEvents
        import cpp_utils
    else:
        #from ..components.core import GameLoopEvents
        from . import cpp_utils

#from . import cpp_utils
#import cpp_utils

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
    "void init(int natom){",
    "double setFF(double R0, double E0, double kR, double kxy, double beta ){",
    "int relaxNsteps( int nsteps, double Fconv, int ialg ){",
    "void scan( int np, double* tip_pos_, double* PP_pos_, double* tip_forces_, double* STMamp_, int nsteps, double Fconv, int ialg ){",
]
#cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces


libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )

cpp_name='COvsCO'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 



const_eVA_SI = 16.0217662;

# ========= C functions

#  double* getPoss   ()
lib.getPoss   .argtypes  = [] 
lib.getPoss   .restype   =  c_double_p
def getPoss   (n):
    #return lib.getPoss   () 
    return np.ctypeslib.as_array( lib.getPoss(), shape=(n,3))

#  double* getAnchors()
lib.getAnchors.argtypes  = [] 
lib.getAnchors.restype   =  c_double_p
def getAnchors(n):
    #return lib.getAnchors() 
    return np.ctypeslib.as_array( lib.getAnchors(), shape=(n,3))

#  double* getForces ()
lib.getForces .argtypes  = [] 
lib.getForces .restype   =  c_double_p
def getForces (n):
    #return lib.getForces () 
    return np.ctypeslib.as_array( lib.getForces(), shape=(n,3))

#  double* getVels   ()
lib.getVels   .argtypes  = [] 
lib.getVels   .restype   =  c_double_p
def getVels   (n):
    #return lib.getVels   () 
    return np.ctypeslib.as_array( lib.getVels(), shape=(n,3))

#  double* getSTMCoefs   ()
lib.getSTMCoefs.argtypes = [] 
lib.getSTMCoefs.restype  =  c_double_p
def getSTMCoefs(n):
    #return lib.getVels   () 
    return np.ctypeslib.as_array( lib.getSTMCoefs(), shape=(n,4))

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

#  double setFF(double R0, double E0, double kR, double kxy ){
lib.setFF.argtypes  = [c_double, c_double, c_double, c_double, c_double, c_double] 
lib.setFF.restype   =  None
def setFF( R0=1.6612 * 2, E0=0.009106, lR=4.0, kR=30.0/const_eVA_SI, kxy=0.25/const_eVA_SI, beta=-1.0 ):
    return lib.setFF(R0, E0, lR, kR, kxy, beta) 

#  double setupOpt( double dt, double damp, double f_limit, double l_limit ){
lib.setupOpt.argtypes  = [c_double, c_double, c_double, c_double] 
lib.setupOpt.restype   =  c_double
def setupOpt(dt=0.2, damp=0.2, f_limit=10.0, l_limit=0.2 ):
    return lib.setupOpt(dt, damp, f_limit, l_limit) 

#  double relaxNsteps( int nsteps, double Fconv, int ialg ){
lib.relaxNsteps.argtypes  = [c_int, c_double, c_int] 
lib.relaxNsteps.restype   =  c_int
def relaxNsteps( nsteps=100, Fconv=1e-5, ialg=0):
    return lib.relaxNsteps(nsteps, Fconv, ialg) 

#  void scan( int np, double* tip_pos_, double* PP_pos_, double* tip_forces_, double* STMamp, int nsteps, double Fconv, int ialg ){
lib.scan.argtypes  = [c_int, c_double_p, c_double_p, c_double_p, c_double_p, c_int, c_double, c_int] 
lib.scan.restype   =  None
def scan( tip_pos, PP_pos=None, tip_forces=None, STMamp=None, nsteps=100, Fconv=1e-6, ialg=-1 ):
    N = len(tip_pos)
    if PP_pos     is None: PP_pos     = np.zeros(tip_pos.shape)
    if tip_forces is None: tip_forces = np.zeros(tip_pos.shape) 
    if STMamp     is None: STMamp     = np.zeros(tip_pos.shape) 
    lib.scan(N, _np_as(tip_pos,c_double_p), _np_as(PP_pos,c_double_p), _np_as(tip_forces,c_double_p),  _np_as(STMamp,c_double_p),  nsteps, Fconv, ialg)
    return tip_forces, PP_pos

def scan2D( Xs, Ys, z0, dz=0.2, nz=20, nsteps=100, Fconv=1e-6, ialg=-1 ):
    sh = Xs.shape
    print( " Xs.shape ", Xs.shape ) 
    tip_pos = np.zeros( (nz,3) )
    tip_pos[:,2] = np.arange( z0, z0-dz*nz, -dz )    ; print( "tip_pos ", tip_pos )
    PP_pos     = np.zeros(tip_pos.shape)
    tip_forces = np.zeros(tip_pos.shape) 
    STMamp     = np.zeros(tip_pos.shape) 
    #print( "tip_forces ", tip_forces ); exit()
    AFM = np.zeros( sh + (len(tip_pos),) )
    print( "AFM.shape ", AFM.shape )
    for ix in range(sh[0]):
        print( "ix ", ix )
        for iy in range(sh[1]):
            print( "ix,iy ", ix, iy )
            tip_pos[:,0] = Xs[ix,iy]
            tip_pos[:,1] = Ys[ix,iy]
            #tip_pos[-,2] = z0
            scan( tip_pos, PP_pos=PP_pos, tip_forces=tip_forces, STMamp=STMamp, nsteps=nsteps, Fconv=Fconv, ialg=ialg )
            AFM[ix,iy,:] = tip_forces[:,2]
    return AFM

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    COs = [[0,0],[1,0],[0,1],[1,1],[2,0],]
    alat = 4.3
    avec = np.array( [0.0,1.0,0.0] )*alat
    bvec = np.array( [np.sqrt(3)/2,-0.5,0.0] )*alat
    '''
    COpos = np.array([
        [0.0,0.0,0.0],
        #[1.0,0.0,0.0],
        [0.2,0.0,15.0], # PP
    ])
    '''

    COpos = np.zeros( (len(COs)+1,3) )
    for i,p in enumerate(COs):
        COpos[i,:] = avec*p[0] + bvec*p[1]
    COpos[-1,2] = 15.0 

    nCO = len(COpos)

    '''
    plt.plot( COpos[:,0], COpos[:,1], 'o' )
    plt.axis('equal'); plt.grid()
    plt.show(); exit()
    '''

    nz    = 40 
    Zs      = np.zeros( nz )
    amp_STM  = np.zeros( nz )
    PPpos = np.zeros( (nz,3) )  

    ff = sys.modules[__name__]
    print( " ff.init( nCO-1 ) " )
    ff.init( nCO-1 )
    ff.setFF( )
    ff.setupOpt(dt=0.4, damp=0.1, f_limit=10.0, l_limit=0.2 )

    poss     = ff.getPoss    ( nCO )
    anchors  = ff.getAnchors ( nCO )
    STMCoefs = ff.getSTMCoefs( nCO )

    STMCoefs[:,:] = 0.0
    #STMCoefs[:,3] = 1.0 # s 
    STMCoefs[:,2] = 1.0  # pz

    poss[:,:]    = COpos[:,:]
    anchors[:,:] = COpos[:,:]
    poss[:nCO-1,2] += 4.0
    poss[ nCO-1,2] -= 4.0

    print( "anchors \n", anchors )
    print( "poss \n",    poss    ) 

    '''
    tip_pos      = np.zeros( (40,3) )
    tip_pos[:,2] = np.linspace( 10, 6, len(tip_pos) )+4
    print( "tip_pos ", tip_pos )
    tip_forces, PP_pos =  ff.scan( tip_pos, nsteps=100, Fconv=1e-6, ialg=-1 )
    #plt.plot( tip_pos[:,2], tip_forces[:,0], label="X" )
    #plt.plot( tip_pos[:,2], tip_forces[:,1], label="Y" )
    plt.plot( tip_pos[:,2], tip_forces[:,2], label="Z" )
    plt.legend()
    '''


    '''
    #ff.relaxNsteps( nsteps=150, ialg=0 )

    beta = 1.0

    dz = 0.2
    for iz in range(nz):
        print( "iz ====== ", iz )
        ff.relaxNsteps( )
        PPpos[iz,:] = poss   [nCO-1,:].copy()
        Zs   [iz]   = anchors[nCO-1,2]
        # ---- move
        anchors[nCO-1,2] -= dz
        poss   [nCO-1,2] -= dz

        amp_STM[iz] = ff.getSTM()
        #print( " ff.getSTM() ", ff.getSTM() )
        #R = np.sqrt( np.sum( (poss[0,:] - poss[nCO-1,:])**2 ) )
        #current[iz] = np.exp(-beta*R)*100

    #print( current )

    plt.plot( Zs, PPpos[:,0], label="X" )
    #plt.plot( Zs, PPpos[:,1], label="Y" )
    plt.plot( Zs, PPpos[:,2], label="Z" )
    plt.plot(Zs[[0,-1]],PPpos[[0,-1],2], '--k' )
    plt.plot( Zs, (amp_STM[:]   )*50.5,  'g', label="STMamp" )
    plt.plot( Zs, (amp_STM[:]**2)*500.0, 'm', label="I" )
    plt.grid()
    plt.show()
    '''

    xs = np.linspace( -5, 10.0, 75  )
    ys = np.linspace( -4, 16.0, 100 )
    Xs,Ys = np.meshgrid( xs, ys )
    AFM = ff.scan2D(  Xs, Ys, z0=14.0, dz=0.2, nz=20, ialg=0 )

    print( "plotting .... " )
    for iz in range(AFM.shape[2]):
        fname = "AFM_%03i.png" %iz
        print( "plotting ", fname )
        #plt.figure()
        plt.imshow( AFM[:,:, iz] )
        plt.savefig( fname , bbox_inches='tight' )
        #plt.close()
    print( "ALL DONE!!! " )
