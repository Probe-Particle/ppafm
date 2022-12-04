#!/usr/bin/python
import sys
import os
import ProbeParticle as PP
import elements
import basUtils
import numpy as np
import GridUtils as GU
import PPPlot as PL
import matplotlib.pyplot as plt
import sys
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")



FFformatList={"xsf":".xsf", "cube":".cube","numpy":".npy"}


try:
    sys.argv[1]
except IndexError:
    print("Please specify a file with coordinates")
    exit(1)






filename = sys.argv[1]

if not os.path.exists(filename):
    print("File {} with coordinates doesn't exist!!! Exiting".format(filename))
    exit(1)


ProjName=filename[:-4]







# Working with parameters file
ParamFilename=ProjName+".ini"

if os.path.exists(ParamFilename):
    PP.loadParams(ParamFilename)
else:
    print("File {} with parameters doesn't exist!!! Using defaults".format(ParamFilename))


cell =np.array([
PP.params['gridA'],
PP.params['gridB'],
PP.params['gridC'],
]).copy()




lvec = PP.params2lvec()
atoms    = basUtils.loadAtoms(filename )

FFparams=None
if os.path.isfile( 'atomtypes.ini' ):
	print(">> LOADING LOCAL atomtypes.ini")
	FFparams=PPU.loadSpecies( 'atomtypes.ini' )
else:
	FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

iZs,Rs,Qs = PP.parseAtoms( atoms, autogeom = False, PBC = True,FFparams=FFparams )



# Lennard Jonnes contribution to the force field
if PP.params['useLJ']:
#checking if files exist:
    exists=True
    for lj_file in ["x", "y", "z"]:
        lj_file=ProjName+"_LJ_F_"+lj_file+".xsf"
        if not os.path.exists(lj_file):
            exists=False

    if exists:
        if query_yes_no( "I have found files containing LJ forcefield. Should I use them (yes) or do you want me to recompute them from scratch (n) ?"):
            todoLJ='read'
        else:
            todoLJ='compute'
    else:
        print("I haven't found files containing LJ forcefields. Therefore I will recompute them from scratch")
        todoLJ='compute'


# Electrostatic contribution to the force field
if PP.params['charge'] != 0.00 :
#checking if files exist
    exists=True
    for el_file in ["x", "y", "z"]:
        print(el_file)
        print(ProjName)
        el_file=ProjName+"_EL_F_"+el_file+".xsf"
        print(el_file)
        if not os.path.exists(el_file):
            exists=False
    if exists:
        if query_yes_no( "I have found files containing electrostatic forcefield. Should I use them (yes) or do you want me to recompute them from scratch (n) ?"):
            todoEL='read'
        else:
            todoEL='compute'
    else:
        print("I haven't found files containing electrostatic forcefields. Therefore I will recompute them from scratch")
        todoEL='compute'





#print iZs




if todoLJ == 'compute':
    FFLJ      = PP.computeLJ( Rs, iZs, FFLJ=None, FFparams=None)
    GU.saveVecFieldXsf(ProjName+"_LJ_F",FFLJ, lvec=[[0.0,0.0,0.0],PP.params['gridA'], PP.params['gridB'],PP.params['gridC']]  )
elif todoLJ == 'read':
    FFLJ, lvec, nDim, head = GU.loadVecFieldXsf( ProjName+"_LJ_F" )

PP.lvec2params( lvec )


xTips,yTips,zTips,lvecScan = PP.prepareScanGrids( )
print(xTips,yTips,zTips)

if todoEL == 'read':
    FFEL, lvec, nDim, head = GU.loadVecFieldXsf( ProjName+"_EL_F" )


Q=PP.params['charge']
K=0.3
PP.setTip( kSpring = np.array((K,K,0.0))/-PP.eVA_Nm )



FF = FFLJ #+ FFEL * Q
PP.setFF_Pointer( FF )

fzs = PP.relaxedScan3D( xTips, yTips, zTips )
