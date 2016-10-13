#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main


import pyProbeParticle                as PPU     
from   pyProbeParticle            import basUtils
from   pyProbeParticle            import elements   
import pyProbeParticle.GridUtils      as GU
#import pyProbeParticle.core          as PPC
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.fieldFFT       as fFFT

HELP_MSG="""Use this program in the following way:
%s -i <filename> 

Supported file fromats are:
   * xyz 
""" %os.path.basename(main.__file__)


from optparse import OptionParser

parser = OptionParser()
parser.add_option( "-i", "--input", action="store", type="string", help="format of input file")
parser.add_option( "-q", "--charge" , action="store_true", default=False, help="Electrostatic forcefield from Q nearby charges ")
parser.add_option( "--noPBC", action="store_false",  help="pbc False", default=True)
parser.add_option( "-E", "--energy", action="store_true",  help="pbc False", default=False)
(options, args) = parser.parse_args()
opt_dict = vars(options)
    
print options

if options.input==None:
    sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)

is_xyz  = options.input.lower().endswith(".xyz")
is_cube = options.input.lower().endswith(".cube")
is_xsf  = options.input.lower().endswith(".xsf")
if not (is_xyz or is_cube or is_xsf ):
    sys.exit("ERROR!!! Unknown format of the input file\n\n"+HELP_MSG)


print " >> OVEWRITING SETTINGS by params.ini  "
PPU.loadParams( 'params.ini' )


lvec=np.zeros((4,3))

lvec[ 1,:  ] =    PPU.params['gridA'].copy() 
lvec[ 2,:  ] =    PPU.params['gridB'].copy()
lvec[ 3,:  ] =    PPU.params['gridC'].copy()
#PPU.params['gridN'] = nDim.copy()

print "--- Compute Lennard-Jones Force-filed ---"
if(is_xyz):
	atoms = basUtils.loadAtoms(options.input, elements.ELEMENT_DICT )
elif(is_cube):
	atoms = basUtils.loadAtomsCUBE(options.input,elements.ELEMENT_DICT)
	lvec  = basUtils.loadCellCUBE(options.input)
	nDim  = basUtils.loadNCUBE(options.input)
	PPU.params['gridN'] = nDim
	PPU.params['gridA'] = lvec[1]
	PPU.params['gridB'] = lvec[2]
	PPU.params['gridC'] = lvec[3]
elif(is_xsf):
	atoms, nDim, lvec = basUtils.loadXSFGeom( options.input )
	PPU.params['gridN'] = nDim
	PPU.params['gridA'] = lvec[1]
	PPU.params['gridB'] = lvec[2]
	PPU.params['gridC'] = lvec[3]
else:
	sys.exit("ERROR!!! Unknown format of geometry system. Supported formats: .xyz, .cube \n\n")




FFparams=None
if os.path.isfile( 'atomtypes.ini' ):
	print ">> LOADING LOCAL atomtypes.ini"  
	FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
iZs,Rs,Qs      = PPH.parseAtoms( atoms, autogeom = False, PBC = options.noPBC )
FFLJ, VLJ      = PPH.computeLJ( Rs, iZs, FFLJ=None, FFparams=FFparams, Vpot=options.energy )

GU.limit_vec_field( FFLJ, Fmax=10.0 ) # remove too large valuesl; keeps the same direction; good for visualization 

print "--- Save  ---"
GU.saveVecFieldXsf( 'FFLJ', FFLJ, lvec)
if options.energy :
	Vmax = 10.0; VLJ[ VLJ>Vmax ] = Vmax
	GU.saveXSF( 'VLJ.xsf', VLJ, lvec)


if opt_dict["charge"]:
    print "Electrostatic Field from xyzq file"
    FFel, VeL = PPH.computeCoulomb( Rs, Qs, FFel=None, Vpot=options.energy  )
    print "--- Save ---"
    GU.saveVecFieldXsf('FFel', FFel, lvec)
    if options.energy :
	Vmax = 10.0; Vel[ Vel>Vmax ] = Vmax
	GU.saveXSF( 'Vel.xsf', Vel, lvec)
