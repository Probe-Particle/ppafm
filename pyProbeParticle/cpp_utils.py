
import os
import ctypes
recompile = True 
lib_ext   ='_lib.so'

c_double_p = ctypes.POINTER(ctypes.c_double)
c_float_p  = ctypes.POINTER(ctypes.c_float)
c_int_p    = ctypes.POINTER(ctypes.c_int)
c_bool_p   = ctypes.POINTER(ctypes.c_bool)



def work_dir( v__file__ ): 
    return os.path.dirname( os.path.realpath( v__file__ ) )

PACKAGE_PATH = work_dir( __file__ )
CPP_PATH     = os.path.normpath( PACKAGE_PATH + '../../cpp/' )

print(" PACKAGE_PATH = ", PACKAGE_PATH)
print(" CPP_PATH     = ", CPP_PATH)

def _np_as(arr,atype):
    """Convert numpy array to C pointer type"""
    if arr is None:
        return None
    elif isinstance(arr, str):
        return arr.encode('utf-8')
    else:
        return arr.ctypes.data_as(atype)


# def compile_lib( name,
#         #FFLAGS = "-std=c++11 -Og -g -Wall",
#         FFLAGS = "-std=c++11 -O3 -ftree-vectorize -unroll-loops -ffast-math",
#         LFLAGS = "-I/usr/local/include/SDL2 -lSDL2",
#         path   = CPP_PATH,
#         clean  = True,
#     ):
#     lib_name = name+lib_ext
#     print(" COMPILATION OF : "+name)
#     if path is not None:
#         dir_bak = os.getcwd()
#         os.chdir( path );
#     print(os.getcwd())
#     if clean:
#         try:
#             os.remove( lib_name  )
#             os.remove( name+".o" ) 
#         except:
#             pass 
#     os.system("g++ "+FFLAGS+" -c -fPIC "+name+".cpp -o "+name+".o "+LFLAGS )
#     os.system("g++ "+FFLAGS+" -shared -Wl,-soname,"+lib_name+" -o "+lib_name+" "+name+".o "+LFLAGS)
#     if path is not None:
#         os.chdir( dir_bak )

def compile_lib(name, FFLAGS="-std=c++20 -fPIC", LFLAGS="", path=None, clean=True, bASAN=False, bDEBUG=True):
    """Compile a C++ library"""
    lib_ext = '.so'
    lib_name = name + lib_ext

    if path is not None:
        dir_bak = os.getcwd()
        os.chdir(path)
    
    print(" COMPILATION OF : " + name)
    print(os.getcwd())
    
    if clean:
        try:
            os.remove(lib_name)
            os.remove(name + ".o")
        except:
            pass
    if bDEBUG:
        FFLAGS += " -g"
    else:
        FFLAGS += " -O3"
    if bASAN:
        # For ASan, we need a specific compilation approach that ensures ASan is linked first
        str1 = f"g++ -std=c++20 -fPIC -g -fsanitize=address -fno-omit-frame-pointer -c -fPIC {name}.cpp -o {name}.o"
        # Use -Wl,--no-as-needed to ensure ASan is linked
        str2 = f"g++ -fsanitize=address -shared -o {lib_name} {name}.o -Wl,--no-as-needed -Wl,-soname,{lib_name}"
        print(str1)
        print(str2)
        os.system(str1)
        os.system(str2)
        return # Skip the standard compilation commands below
    str1 = f"g++ {FFLAGS} -c -fPIC {name}.cpp -o {name}.o"
    str2 = f"g++ {FFLAGS} -shared -Wl,-soname,{lib_name} -o {lib_name} {name}.o {LFLAGS}"
    print(str1)
    print(str2)
    os.system(str1)
    os.system(str2)
    
    if path is not None:
        os.chdir(dir_bak)        



def makeclean( ):
    CWD=os.getcwd()
    os.chdir( CPP_PATH )
    os.system("make clean")
    os.chdir(CWD)

def make( what="" ):
    current_directory = os.getcwd()
    os.chdir ( CPP_PATH          )
    if recompile:
        os.system("make clean")
    os.system( "make "+what      )
    os.chdir ( current_directory )


# ============ automatic C-python interface generation



def parseFuncHeader( s ):
    #args = []
    arg_types = []
    arg_names = []
    i0 = s.find(' ')
    i1 = s.find('(')
    i2 = s.find(')')
    ret_type = s[:i0]
    name     = s[i0+1:i1]
    sargs = s[i1+1:i2]
    largs = sargs.split(',')
    for sarg in largs:
        sarg = sarg.strip()
        i    = sarg.rfind(' ')
        #arg_name=sarg[i+1:]
        #arg_type=sarg[  :i]
        #args.append( (arg_type, arg_name) )
        arg_types.append(sarg[  :i])
        arg_names.append(sarg[i+1:])
    return (name,ret_type,arg_types,arg_names)

def translateTypeName( tname ):
    np = tname.count('*')
    if np > 1 :
        print("Cannot do pointer-to-pointer (**) ", s)
        print("=> exit() ")
        exit()
    else:
        if(np==1):
            i  = tname.find ('*')
            return "c_"+tname[:i]+"_p", True
        else:
            return "c_"+tname     , False

s_numpy_data_as_call = "%s.ctypes.data_as(%s)"

def writePointerCall( name, ttype ):
    if ttype[1]:
        return s_numpy_data_as_call %(name,ttype[0])
    else:
        return name

def writeFuncInterface( parsed ):
    name,ret_type,arg_types,arg_names = parsed
    arg_types_ = [ ]
    #arg_names = [ ]
    if ret_type=="void" : 
        ret_type="None"
    else:
        ret_type=translateTypeName(ret_type)[0]
    sdef_args  = ", ".join( [                    translateTypeName(t)[0] for   t in arg_types                ] )
    scall_args = ", ".join( [ writePointerCall(n,translateTypeName(t))   for n,t in zip(arg_names,arg_types) ] )
    lines = [
        "lib."+name+".argtypes  = ["+ sdef_args + "] ",
        "lib."+name+".restype   =  " + ret_type          ,
        "def "+name+"("+ ", ".join(arg_names) +"):"    ,
        "    return lib."+name+"("+scall_args+")"         ,
    ]
    return "\n".join( lines )

def writeFuncInterfaces( func_headers, debug=False ):
    for s in func_headers:
        parsed = parseFuncHeader( s ); 
        if debug : print("parsed :\n", parsed)
        sgen   = writeFuncInterface( parsed )
        print("\n# ", s)
        print(sgen,"\n\n")






