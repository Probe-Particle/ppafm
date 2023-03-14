import os

recompile = True
lib_ext   ='_lib.so'

def work_dir( v__file__ ):
    return os.path.dirname( os.path.realpath( v__file__ ) )

PACKAGE_PATH = work_dir( __file__ )
CPP_PATH     = os.path.normpath( PACKAGE_PATH + '/cpp/' )

print(" PACKAGE_PATH = ", PACKAGE_PATH)
print(" CPP_PATH     = ", CPP_PATH)

def make( what="" ):
    current_directory = os.getcwd()
    os.chdir ( CPP_PATH          )
    if recompile:
        os.system("make clean")
    os.system( "make "+what      )
    os.chdir ( current_directory )
