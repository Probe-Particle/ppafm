
import os

import pyopencl as cl
import numpy    as np 

PACKAGE_PATH = os.path.dirname( os.path.realpath( __file__ ) ); print PACKAGE_PATH
CL_PATH      = os.path.normpath( PACKAGE_PATH + '/../cl' )

plats   = cl.get_platforms()
ctx     = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[0])], devices=None)       
queue   = cl.CommandQueue(ctx)

def loadProgram(fname, ctx=ctx, queue=queue):
    f       = open(fname, 'r')
    fstr    = "".join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    return program

def updateBuffer( buff, cl_buff, access=cl.mem_flags ):
    if buff is not None:
        if cl_buff is None:
            cl_buff = cl.Buffer(ctx, access | cl.mem_flags.COPY_HOST_PTR, hostbuf=buff ); return buff.nbytes
        else:
            cl.enqueue_copy( queue, cl_buff, buff )
    return 0
