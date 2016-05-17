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
"""+os.path.basename(main.__file__) +""" -i <filename> 

Supported file fromats are:
   * xyz """


from optparse import OptionParser

parser = OptionParser()
parser.add_option( "-i", "--input", action="store", type="string", help="format of input file")
parser.add_option( "-q", "--charge" , action="store_true", default=False, help="Electrostatic forcefield from Q nearby charges ")
parser.add_option( "--noPBC", action="store_false",  help="pbc False", default=True)
(options, args) = parser.parse_args()
opt_dict = vars(options)
    
print options

if options.input==None:
    sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)

is_xyz  = options.input.lower().endswith(".xyz")
is_cube = options.input.lower().endswith(".cube")
if not (is_xyz or is_cube):
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
	n  = basUtils.loadNCUBE(options.input)
	PPU.params['gridN'] = n
	PPU.params['gridA'] = lvec[1]
	PPU.params['gridB'] = lvec[2]
	PPU.params['gridC'] = lvec[3]
else:
	sys.exit("ERROR!!! Unknown format of geometry system. Supported formats: .xyz, .cube \n\n")

FFparams=None
if os.path.isfile( 'atomtypes.ini' ):
	print ">> LOADING LOCAL atomtypes.ini"  
	FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
iZs,Rs,Qs = PPH.parseAtoms( atoms, autogeom = False, PBC = options.noPBC )
FFLJ      = PPH.computeLJ( Rs, iZs, FFLJ=None, FFparams=FFparams )

GU.limit_vec_field( FFLJ, Fmax=100.0 ) # remove too large valuesl; keeps the same direction; good for visualization 

print "--- Save  ---"
GU.saveVecFieldXsf( 'FFLJ', FFLJ, lvec)

if opt_dict["charge"]:
    print "Electrostatic Field from xyzq file"
    FFel = PPH.computeCoulomb( Rs, Qs, FFel=None )
    print "--- Save ---"
    GU.saveVecFieldXsf('FFel', FFel, lvec)
