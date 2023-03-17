import ctypes
from ctypes import c_bool, c_double, c_int

import numpy as np

from . import cpp_utils

c_double_p = ctypes.POINTER(c_double)
c_int_p    = ctypes.POINTER(c_int)

def _np_as(arr,atype):
    if arr is None:
        return None
    else:
        return arr.ctypes.data_as(atype)

cpp_utils.s_numpy_data_as_call = "_np_as(%s,%s)"

def initViewLib():
    cpp_name='GLV'
    cpp_utils.make(cpp_name)
    return ctypes.CDLL( cpp_utils.CPP_PATH + "/" + cpp_name + cpp_utils.lib_ext )

# ========= C functions

class GLView():

    def __init__(self, wh=(800,600) ):

        self.libSDL = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libSDL2.so", ctypes.RTLD_GLOBAL )
        self.libGL  = ctypes.CDLL( "/usr/lib/x86_64-linux-gnu/libGL.so",   ctypes.RTLD_GLOBAL )

        self.lib = initViewLib()
        self.lib.init.argtypes = [ c_int, c_int ]
        self.lib.init.restype  = None
        self.lib.init( wh[0], wh[1] )

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
