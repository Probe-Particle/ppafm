
import os

import pyopencl as cl
import numpy    as np 

#PACKAGE_PATH = None
#CL_PATH      = None
#plats      = None
#ctx        = None
#queue      = None

class OCLEnvironment:

    def __init__(self,i_platform=0):
        self.PACKAGE_PATH = os.path.dirname( os.path.realpath( __file__ ) ); 
        print("OCLEnvironment platform[%i]" %i_platform," PACKAGE_PATH: ", self.PACKAGE_PATH)
        self.CL_PATH      = os.path.normpath( self.PACKAGE_PATH + '/../cl' )
        #self.CL_PATH      = os.path.normpath( self.PACKAGE_PATH )
        platforms         = cl.get_platforms()
        print(" i_platform ", i_platform)
        self.platform     = platforms[i_platform]
        self.ctx          = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)], devices=None)
        self.queue        = cl.CommandQueue(self.ctx)

    def loadProgram(self,fname):
        f       = open(fname, 'r')
        fstr    = "".join(f.readlines())
        cl._DEFAULT_INCLUDE_OPTIONS.append( "-I"+self.CL_PATH  )  # this is a bit a hack !!!   not sure how to add include dir properly https://documen.tician.de/pyopencl/runtime_program.html#program
        program = cl.Program(self.ctx, fstr ).build()
        return program

    def updateBuffer(self, buff, cl_buff, access=cl.mem_flags ):
        if buff is not None:
            if cl_buff is None:
                cl_buff = cl.Buffer(self.ctx, access | cl.mem_flags.COPY_HOST_PTR, hostbuf=buff ); 
                return buff.nbytes
            else:
                cl.enqueue_copy( self.queue, cl_buff, buff )
        return 0

    def printInfo(self):
        print("======= DEVICES\n",         self.ctx.get_info(cl.context_info.DEVICES))
        print("======= PROPERTIES\n",      self.ctx.get_info(cl.context_info.PROPERTIES))
        print("======= REFERENCE_COUNT\n", self.ctx.get_info(cl.context_info.REFERENCE_COUNT))

    def printPlatformInfo(self):
        platform = self.platform
        print("===============================================================")
        print(" Platform name:",    platform.name)
        print(" Platform profile:", platform.profile)
        print(" Platform vendor:",  platform.vendor)
        print(" Platform version:", platform.version)
        for device in platform.get_devices():
            print("---------------------------------------------------------------")
            print(" Device name:", device.name)
            print(" type:", cl.device_type.to_string(device.type))
            print(" memory: ", device.global_mem_size//1024//1024, 'MB')
            print(" max clock speed:", device.max_clock_frequency, 'MHz')
            print(" compute units:", device.max_compute_units)
            print("  GLOBAL_MEM_SIZE          = ", device.get_info( cl.device_info.GLOBAL_MEM_SIZE          )/4," float32")
            print("  LOCAL_MEM_SIZE           = ", device.get_info( cl.device_info.LOCAL_MEM_SIZE           )/4," float32")
            print("  MAX_CONSTANT_BUFFER_SIZE = ", device.get_info( cl.device_info.MAX_CONSTANT_BUFFER_SIZE )/4," float32")
            print("  MAX_WORK_GROUP_SIZE      = ", device.get_info( cl.device_info.MAX_WORK_GROUP_SIZE      ))


#def init(i_platform=0):
#    global PACKAGE_PATH #, CL_PATH, plats, ctx,  queue
#    global CL_PATH 
#    global plats 
#    global ctx 
#    global queue
#    PACKAGE_PATH = os.path.dirname( os.path.realpath( __file__ ) ); print PACKAGE_PATH
#    CL_PATH      = os.path.normpath( PACKAGE_PATH + '/../cl' )
#    plats   = cl.get_platforms()
#    ctx     = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[i_platform])], devices=None)
#    #print plats; exit()
#    queue   = cl.CommandQueue(ctx)

#def tryRelease(cl_arr):
#    try:
#        cl_arr.release()
#    except:
#        pass

#def loadProgram(fname, ctx=ctx, queue=queue):
#    f       = open(fname, 'r')
#    fstr    = "".join(f.readlines())
#    #cl._DEFAULT_INCLUDE_OPTIONS.append( "-I "+CL_PATH  )  # this is a bit a hack !!!   not sure how to add include dir properly https://documen.tician.de/pyopencl/runtime_program.html#program
#    program = cl.Program(ctx, fstr ).build()
#    return program


