#!/usr/bin/python

'''
Idea is to improve prediction of gemeotry using physical generative model

1) CNN prediction of Bonds&Atoms
2) MM  relaxation (with predicted Bonds&Atoms as external potential)
3) PPM simulation (based on MM geometry)
4) recognize errors (reference AFM data-stack needed)
5) Generation of improved structure (How? Using CNN?)
    -> repeat from (2)

see:  https://mega.nz/#!KLoilKIB!NxxCRQ814xtCXfjy7mPFfmJTOL9TaTHbmPKSxn_0sFs

'''

import sys
import os
import shutil
import time
import random
import matplotlib;
import numpy as np
from enum import Enum

#import matplotlib.pyplot as plt

import pyopencl     as cl

from . import atomicUtils as au
from . import basUtils
from . import common    as PPU
from . import elements
from . import oclUtils     as oclu 
from . import fieldOCL     as FFcl 
from . import RelaxOpenCL  as oclr
from . import HighLevelOCL as hl

from . import GeneratorOCL_LJC #as Imager
from . import FARFF            #as Relaxer

verbose  = 0
bRunTime = False

class Sequence:
    pass

class Critique():

    def __init__(self ):
        pass
    
    def try_improve(self, geomIn, AFMs, AFMRef ):
        AFMdiff = AFMs - AFMRef
        # ToDo : identify are of most difference and make random changes in that area
        geomOut = geomIn.copy()
        return geomOut

class CorrectionLoop():

    rotMat = np.array([[1.,0,0],[0.,1.,0],[0.,0,1.]])

    def __init__(self, relaxator, simulator, critique ):
        self.simulator = simulator
        self.relaxator = relaxator
        self.critique  = critique

    def init(self):
        pass

    def startLoop(self, guess, Zs, qs, atomMap, bondMap, lvecMap ):
        self.guess   = guess
        self.Zs      = qs
        self.qs      = qs
        self.atomMap = atomMap
        self.bondMap = bondMap
        self.mapLvec = lvecMap

    def iteration(self):
        print( "### CorrectionLoop.iteration [1]" )
        self.relaxed    = self.relaxator.preform_relaxation ( self.guess, self.Zs, self.qs, self.mapLvec, self.atomMap, self.bondMap )
        print( "### CorrectionLoop.iteration [2]" )
        AFMs,AuxMap     = self.simulator.perform_imaging( self.relaxed, self.Zs, self.qs, self.rotMat )
        print( "### CorrectionLoop.iteration [3]" )
        Err, self.guess = self.critique.try_improve     ( self.relaxed, AFMs, AFMRef )
        print( "### CorrectionLoop.iteration [4]" )
        return Err
        #Xs,Ys      = simulator.next1( self )

if __name__ == "__main__":
    print( " UNIT_TEST START : CorrectionLoop ... " )
    #import atomicUtils as au

    print("# ------ Init Generator   ")

    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    lvec=None
    simulator  = GeneratorOCL_LJC.Generator( [], [], 1, pixPerAngstrome=5, Ymode='AtomsAndBonds', lvec=lvec  )

    simulator.bOccl = 1   # switch occlusion of atoms 0=off 1=on
    simulator.typeSelection =  [1,6,7,8,16,33]  # select atom types for each output channel
    simulator.nChan = 5      # number of channels, resp. atom size bins
    simulator.Rmin  = 1.4    # minimum atom radius
    simulator.Rstep = 0.2    # size range per bin (resp. cGenerator.nextRotationnel)
    simulator.zmin_xyz = -2.0  # max depth for visible atoGenerator.nextRotation
    simulator.Nmax_xyz = 3    # max number of visible atomGenerator.nextRotation

    simulator.projector.Rpp  = -0.5
    xs = np.linspace(0.0,10.0,100)
    dx = xs[1]-xs[0];
    xs -= dx
    ys = np.exp( -5*xs )

    simulator.projector.Rfunc   = ys.astype(np.float32)
    simulator.projector.invStep = dx
    simulator.projector.Rmax    = xs[-1] - 3*dx
    # --- params randomization 
    #simulator.randomize_enabled    = False
    #simulator.randomize_nz         = True 
    #simulator.randomize_parameters = True
    #simulator.randomize_tip_tilt   = True
    #simulator.randomize_distance   = True
    #simulator.rndQmax     = 0.1    # charge += rndQmax * ( rand()-0.5 )  (negative is off)
    #simulator.rndRmax     = 0.2    # charge += rndRmax * ( rand()-0.5 )  (negative is off)
    #simulator.rndEmax     = 0.5    # charge *= (1 + rndEmax     * ( rand()-0.5 )) (negative is off)
    #simulator.rndAlphaMax = -0.1   # charge *= (1 + rndAlphaMax * ( rand()-0.5 )) (negative is off)
    #simulator.modMolParams = modMolParams_def   # custom function to modify parameters

    #simulator.debugPlots = True
    simulator.distAbove = 7.0
    simulator.distAboveDelta = None  

    simulator.bQZ = True
    simulator.Qs  = np.array([100,-200,100,0]) * -0.2   # Quadrupole Qdz2-1.0[e*A^2]
    simulator.maxTilt0 = 0.5     # asymetric tip  tilted max +/-1.0 Angstreom in random direction

    #simulator.shuffle_rotations = False
    #simulator.shuffle_molecules = False
    #simulator.nBestRotations    = 0

    #simulator.preName    = ""           # appended befroe path filename
    #simulator.postName   = "/pos.xyz"
    simulator.Q = 0.0
    # z-weight exp(-wz*z)
    simulator.wz      = 1.0    # deacay
    simulator.zWeight =  simulator.getZWeights();
    dz=0.1
    dfWeight = PPU.getDfWeight( 10, dz=dz )[0].astype(np.float32)
    simulator.dfWeight = dfWeight

    print( "# ------ Init Relaxator  ")

    relaxator = FARFF.EngineFARFF()

    print( "# ------ Init Critique   ")

    critique = Critique()

    print( "# ------ Init Looper     ")

    looper = CorrectionLoop(relaxator,simulator,critique)
    xyzs,Zs,elems,qs = au.loadAtomsNP("input.xyz")
    atomMap, bondMap, lvecMap = FARFF.makeGridFF( FARFF,  fname_atom='./Atoms.npy', fname_bond='./Bonds.npy',   dx=0.1, dy=0.1 )
    looper.startLoop( xyzs, Zs, qs, atomMap, bondMap, lvecMap )
    ErrConv = 1.0
    print( "# ------ To Loop    ")
    for itr in range(1000):
        Err = looper.iteration()
        if Err > ErrConv:
            break

    #print( "UNIT_TEST is not yet written :-( " )
    print( " UNIT_TEST CorrectionLoop DONE !!! " )
    pass