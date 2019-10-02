import numpy as np
from   ctypes import c_int, c_double, c_bool, c_float, c_char_p, c_bool, c_void_p
import ctypes
import os
import cpp_utils

c_double_p = ctypes.POINTER(c_double)
c_int_p    = ctypes.POINTER(c_int)

def _np_as(arr,atype):
    if arr is None:
        return None
    else: 
        return arr.ctypes.data_as(atype)

cpp_utils.s_numpy_data_as_call = "_np_as(%s,%s)"

# ===== To generate Interfaces automatically from headers call:
#header_strings = []
# cpp_utils.writeFuncInterfaces( header_strings );        exit()     #   uncomment this to re-generate C-python interfaces

'''
cpp_name='FARFF'
cpp_utils.make(cpp_name)
lib    = ctypes.CDLL(  cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )     # load dynamic librady object using ctypes 
'''

def initViewLib():
    ##LIB_PATH_CPP  = os.path.normpath(LIB_PATH+'../../../'+'/cpp/Build/libs_SDL/FlightView')
    #recompile(LIB_PATH_CPP)
    cpp_name='GLV'
    cpp_utils.make(cpp_name)
    return ctypes.CDLL( cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )

# ========= C functions

class GLView():
    
    def __init__(self, wh=(800,600) ):
        #void init( int w, int h, void* craft1_, void* bursts_){
        
        self.libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
        self.libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )
        #self.libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )
        #self.libGL  = ctypes.CDLL( "/usr/lib32/nvidia-384/libGL.so",   ctypes.RTLD_GLOBAL )
        
        #self.libSDL = ctypes.CDLL( "libSDL2.so", ctypes.RTLD_GLOBAL )
        #self.libGL  = ctypes.CDLL( "libGL.so",   ctypes.RTLD_GLOBAL )

        self.lib = initViewLib()
        #print "========= libView ", libView
        self.lib.init.argtypes = [ c_int, c_int ]
        self.lib.init.restype  = None
        self.lib.init( wh[0], wh[1] )

        #void fly( int n, int nsub, double dt, double* pos_, double* vel_, double* rot_ )
        self.lib.draw.argtypes = []
        self.lib.draw.restype  = None

        self.lib.pre_draw.argtypes = []
        self.lib.pre_draw.restype  = c_bool

        self.lib.post_draw.argtypes = []
        self.lib.post_draw.restype  = c_bool

    def draw(self):
        self.lib.draw()

    def pre_draw(self):
        return self.lib.pre_draw()

    def post_draw(self):
        return  self.lib.post_draw()

if __name__ == "__main__":
    glview = GLView()
    for i in range(100):
        glview.pre_draw()
        glview.post_draw()