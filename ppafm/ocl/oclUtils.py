from pathlib import Path

import numpy as np
import pyopencl as cl

from . import field as FFcl
from . import relax as oclr


class OCLEnvironment:
    def __init__(self, i_platform=0):
        platforms = get_platforms()
        self.platform = platforms[i_platform]
        print(f"Initializing an OpenCL environment on {self.platform.name}")

        self.PACKAGE_PATH = Path(__file__).resolve().parent
        self.CL_PATH = self.PACKAGE_PATH / "cl"
        self.ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)], devices=None)
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, fname):
        cl_path = str(self.CL_PATH)
        if self.platform.name != "Portable Computing Language":
            # Older versions of pocl don't handle quotes and spaces properly. This is kind of ugly, but
            # this is needed for the version of pocl running on Github Actions at the moment of writing.
            cl_path = f'"{cl_path}"'
        with open(fname) as f:
            program = cl.Program(self.ctx, f.read()).build(options=["-I", cl_path])
        return program

    def updateBuffer(self, buff, cl_buff, access=cl.mem_flags):
        if buff is not None:
            if cl_buff is None:
                cl_buff = cl.Buffer(self.ctx, access | cl.mem_flags.COPY_HOST_PTR, hostbuf=buff)
                return buff.nbytes
            else:
                cl.enqueue_copy(self.queue, cl_buff, buff)
        return 0

    def printInfo(self):
        # fmt: off
        print("======= DEVICES\n",         self.ctx.get_info(cl.context_info.DEVICES))
        print("======= PROPERTIES\n",      self.ctx.get_info(cl.context_info.PROPERTIES))
        print("======= REFERENCE_COUNT\n", self.ctx.get_info(cl.context_info.REFERENCE_COUNT))
        # fmt: on

    def printPlatformInfo(self):
        # fmt: off
        platform = self.platform
        print("===============================================================")
        print(" Platform name:",    platform.name)
        print(" Platform profile:", platform.profile)
        print(" Platform vendor:",  platform.vendor)
        print(" Platform version:", platform.version)
        for device in platform.get_devices():
            print("---------------------------------------------------------------")
            print(" Device name:",     device.name)
            print(" type:",            cl.device_type.to_string(device.type))
            print(" memory: ",         device.global_mem_size // 1024 // 1024, 'MB')
            print(" max clock speed:", device.max_clock_frequency, 'MHz')
            print(" compute units:",   device.max_compute_units)
            print("  GLOBAL_MEM_SIZE          = ", device.get_info( cl.device_info.GLOBAL_MEM_SIZE          ) / 4, " float32")
            print("  LOCAL_MEM_SIZE           = ", device.get_info( cl.device_info.LOCAL_MEM_SIZE           ) / 4, " float32")
            print("  MAX_CONSTANT_BUFFER_SIZE = ", device.get_info( cl.device_info.MAX_CONSTANT_BUFFER_SIZE ) / 4, " float32")
            print("  MAX_WORK_GROUP_SIZE      = ", device.get_info( cl.device_info.MAX_WORK_GROUP_SIZE      ))
        # fmt: on


def get_platforms():
    try:
        platforms = cl.get_platforms()
    except cl._cl.LogicError:
        raise RuntimeError("Could not find any OpenCL platforms. Check that the OpenCL ICD for your device is installed.")
    return platforms


def init_env(i_platform=0):
    env = OCLEnvironment(i_platform)
    FFcl.init(env)
    oclr.init(env)


def print_platforms():
    platforms = get_platforms()
    for i, plat in enumerate(platforms):
        print(f"Platform {i}: {plat.name}")
