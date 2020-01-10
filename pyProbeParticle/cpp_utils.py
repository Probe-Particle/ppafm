
import os

recompile = True 
lib_ext   ='_lib.so'

def work_dir( v__file__ ): 
    return os.path.dirname( os.path.realpath( v__file__ ) )

PACKAGE_PATH = work_dir( __file__ )
CPP_PATH     = os.path.normpath( PACKAGE_PATH + '../../cpp/' )

print(" PACKAGE_PATH = ", PACKAGE_PATH)
print(" CPP_PATH     = ", CPP_PATH)

def compile_lib( name,
        #FFLAGS = "-std=c++11 -Og -g -Wall",
        FFLAGS = "-std=c++11 -O3 -ftree-vectorize -unroll-loops -ffast-math",
        LFLAGS = "-I/usr/local/include/SDL2 -lSDL2",
        path   = CPP_PATH,
        clean  = True,
    ):
    lib_name = name+lib_ext
    print(" COMPILATION OF : "+name)
    if path is not None:
        dir_bak = os.getcwd()
        os.chdir( path );
    print(os.getcwd())
    if clean:
        try:
            os.remove( lib_name  )
            os.remove( name+".o" ) 
        except:
            pass 
    os.system("g++ "+FFLAGS+" -c -fPIC "+name+".cpp -o "+name+".o "+LFLAGS )
    os.system("g++ "+FFLAGS+" -shared -Wl,-soname,"+lib_name+" -o "+lib_name+" "+name+".o "+LFLAGS)
    if path is not None:
        os.chdir( dir_bak )

def make( what="" ):
    current_directory = os.getcwd()
    os.chdir ( CPP_PATH          )
    os.system( "make "+what       )
    os.chdir ( current_directory )

def makeclean( ):
    CWD=os.getcwd()
    os.chdir( CPP_PATH )
    os.system("make clean")
    os.chdir(CWD)
