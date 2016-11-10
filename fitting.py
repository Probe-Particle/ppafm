#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main
from optparse import OptionParser
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from generateLJFF import computeLJFF
from relaxed_scan import perform_relaxation
from pyProbeParticle import basUtils
from generateElFF import computeElFF
import pyProbeParticle.GridUtils as GU
import pyProbeParticle  as PPU     
import pyProbeParticle.HighLevel as PPH


iteration=0

def pm2a(val):
    res=float(val)/100
    return res

def pN2ev_o_a(val):
    res=float(val)*6.241506363094e-4
    return res
def getFzlist(BIGarray,MIN,MAX,points):
#    print "Hello world"
    x=np.linspace(MIN[0],MAX[0],BIGarray.shape[2])
    y=np.linspace(MIN[1],MAX[1],BIGarray.shape[1])
    z=np.linspace(MIN[2],MAX[2],BIGarray.shape[0])
    result=[]
    interp = RegularGridInterpolator((z, y, x), BIGarray)
#    print BIGarray.shape
    for p in points :
        if (p >= MIN).all() and (p <= MAX).all():
            result.append(interp([p[2],p[1],p[0]])[0])
#    print "TEST", interp([MAX[2], current_pos[1],current_pos[0]])
#    print "TEST", interp([8.0, current_pos[1],current_pos[0]])
    return np.array(result)
FFparams=None
if os.path.isfile( 'atomtypes.ini' ):
	print ">> LOADING LOCAL atomtypes.ini"  
	FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
        print FFparams
        elem_dict={}
        for i,ff in enumerate(FFparams):
            elem_dict[ff[3]] = i
else:
    raise ValueError('Please provide the file "atomtypes.ini"')

print " >> OVEWRITING SETTINGS by params.ini  "
PPU.loadParams( 'params.ini',FFparams=FFparams )
scan_min=PPU.params['scanMin']
scan_max=PPU.params['scanMax']
atoms,nDim,lvec=basUtils.loadGeometry("p_eq.xyz", params=PPU.params)
# The function automatically load the geometry from the file of any
# supported format. The desigion about the file format is based on the
# filename extension
PPU.params['gridN'] = nDim
PPU.params['gridA'] = lvec[1]
PPU.params['gridB'] = lvec[2]
PPU.params['gridC'] = lvec[3]
V, lvec_bak, nDim_bak, head = GU.loadCUBE("hartree.cube")
loaded_forces=np.loadtxt("frc_tip.txt",
                         converters={0:pm2a,1:pm2a,2:pm2a},
                         skiprows=2, usecols=(0,1,2,5))
points=loaded_forces[:,:3]
iZs,Rs,Qs=PPH.parseAtoms(atoms, autogeom = False, PBC = PPU.params['PBC'], FFparams=FFparams )

fit_dict={}
def update_atoms(atms=None):
    print "UPDATING ATOMS"
    print atms
    x=[]
    for atm in atms:
        i=elem_dict[atm[0]]
        FFparams[i][0]=float(atm[1])
        x.append(float(atm[1]))
        FFparams[i][1]=float(atm[2])
        x.append(float(atm[2]))
    print "UPDATING : " ,x    
    return x

def set_fit_dict(opt=None):
    i=0
    x=[]
    fit_dict['atom']=[]
    for key,value in opt.iteritems():
            if opt[key] is None:
                continue
            if key is "atom":
                print opt[key]
                x+=update_atoms(value)
                for val in value:
                    print "TYTA",val
                    fit_dict['atom'].append(list(val))
            else:
                fit_dict[key]=opt[key]
                x.append(opt[key])
                i+=1
    return x

def update_fit_dict(x=[]):
    i=0
    for key,value in fit_dict.iteritems():
        if key is "atom":
            for atm in value:
                print atm
                atm[1]=x[i]
                atm[2]=x[i+1]
                i+=2
        else:
            fit_dict[key]=x[i]
            i+=1


def comp_rmsd(x=[]):
    global iteration
    iteration+=1
    update_fit_dict(x) # updating the array with the optimized values
    PPU.apply_options(fit_dict) # setting up all the options according to their
                                # current values
    update_atoms(atms=fit_dict['atom'])
    print FFparams
    FFLJ,VLJ=computeLJFF(iZs,Rs,FFparams)
    FFel,Vpot=computeElFF(iZs,Rs,Qs,V,lvec_bak,nDim_bak,PPU.params['tip'])
    FFboltz=None
    fzs,PPpos,PPdisp,lvecScan=perform_relaxation(FFLJ, FFel,FFboltz,tipspline=None,lvec=lvec)
    Fzlist=getFzlist(BIGarray=fzs, MIN=scan_min, MAX=scan_max, points=points)
    rmsd=np.sum((loaded_forces[:,3]-Fzlist*1.60217733e3)**2) /len(Fzlist)
    with open ("iteration.txt", "a") as myfile:
        myfile.write( "iteration {}: {} rmsd: {}\n".format(iteration, x, rmsd))
    return rmsd

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option( "-q","--charge", action="store", type="float", help="fit tip charge ", default=None)

    parser.add_option( "-s","--sigma", action="store", type="float", help="Fit "
    "the gaussian width of the charge distribution", default=None)

    parser.add_option( "-k","--klat", action="store", type="float", help="Fit "
    "the lateral stiffness of the PP", default=None)

    parser.add_option( "-a","--atom", action="append", type="string",help="Fit "
    "the LJ parameters of the given atom", default=None, nargs=3)
#    parser.add_option( "-a","--atom", action="append",type=l
    (options, args) = parser.parse_args()
    opt_dict = vars(options)
    PPU.apply_options(opt_dict) # setting up all the options according to their
    x=set_fit_dict(opt=opt_dict)
#    print comp_rmsd(x)
    minimize(comp_rmsd,x,method='Nelder-Mead')

