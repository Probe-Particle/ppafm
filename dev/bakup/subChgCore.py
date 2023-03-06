#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import sys

import __main__ as main
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")

from optparse import OptionParser

import ppafm as PPU
import ppafm.fieldFFT as fFFT
from ppafm import io

parser = OptionParser()
parser.add_option( "-s", "--sample", action="store", type="string", default="CHGCAR.xsf", help="sample 3D data-file (.xsf)")
(options, args) = parser.parse_args()

def triLin(abc):
    ia = int(np.floor(abc[0])); fa=abc[0]-ia; #ma=1.0-fa
    ib = int(np.floor(abc[1])); fb=abc[1]-ib; #mb=1.0-fb
    ic = int(np.floor(abc[2])); fc=abc[2]-ic; #mc=1.0-fc
    print("::: ", abc , (ia,ib,ic), fa,fb, fc)
    return (ia,ib,ic), (fa,fb,fc)

def addDeInterp( val, arr, iabc, fabc):
    fa=fabc[0]; ma=1.0-fa
    fb=fabc[1]; mb=1.0-fb
    fc=fabc[2]; mc=1.0-fc
    #print  val, ( ma*mb*mc + ma*mb*fc + ma*fb*mc + ma*fb*fc +
    #      fa*mb*mc  + fa*mb*fc  + fa*fb*mc  +fa*fb*fc)

    arr[iabc[2]  ,iabc[1]  ,iabc[0]  ] += mc*mb*ma*val
    arr[iabc[2]  ,iabc[1]  ,iabc[0]+1] += mc*mb*fa*val
    arr[iabc[2]  ,iabc[1]+1,iabc[0]  ] += mc*fb*ma*val
    arr[iabc[2]  ,iabc[1]+1,iabc[0]+1] += mc*fb*fa*val
    arr[iabc[2]+1,iabc[1]  ,iabc[0]  ] += fc*mb*ma*val
    arr[iabc[2]+1,iabc[1]  ,iabc[0]+1] += fc*mb*fa*val
    arr[iabc[2]+1,iabc[1]+1,iabc[0]  ] += fc*fb*ma*val
    arr[iabc[2]+1,iabc[1]+1,iabc[0]+1] += fc*fb*fa*val

#pixOff    = np.array([-0.5,-0.5,-0.5])
pixOff    = np.array([0.0,0.0,0.0])
valElDict = { 6:4.0, 8:6.0}

atoms,nDim,lvec     = io.loadGeometry( options.sample, params=PPU.params )
atoms_ = np.array(atoms)

rho1, lvec1, nDim1, head1 = io.loadXSF( options.sample )
V = lvec1[1,0]*lvec1[2,1]*lvec1[3,2]
N = nDim1[0]*nDim1[1]*nDim1[2]
dV = (V/N)

print(atoms)
print(atoms_)

'''

lvec_ = lvec[1:]
invLvec = np.linalg.inv( lvec_ )
invLvec[:,0] *= nDim[0]
invLvec[:,1] *= nDim[1]
invLvec[:,2] *= nDim[2]
print invLvec


#rho1[:,:,:] = 0

for ia in range(len(atoms[0])):
    nel = valElDict[atoms[0][ia]]
    xyz = atoms_[1:4,ia]
    #print xyz
    abc = np.dot(invLvec,xyz)
    xyz_ = np.dot(lvec_,abc)
    #print ">> ", xyz, abc, xyz_
    iabc, fabc = triLin(abc + pixOff)
    addDeInterp( -nel/dV, rho1, iabc, fabc)

'''

import ppafm.fieldFFT as ffft

print("sum(RHO), Nelec",  rho1.sum(),  rho1.sum()*dV)
ffft.addCoreDensities( atoms_, valElDict, rho1, lvec1, sigma=0.25 )
print("sum(RHO), Nelec",  rho1.sum(),  rho1.sum()*dV)

# io.saveXSF( "rho_core.xsf", rho1,       lvec1, head=head1 )
io.saveXSF( "rho_diff.xsf", rho1,       lvec1, head=head1 )
