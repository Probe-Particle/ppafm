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

    def __init__(self, relaxator, simulator, critique ):
        self.simulator = simulator
        self.relaxator = relaxator
        self.critique  = critique

    def init(self):
        pass

    def iteration(self):
        self.relaxed = self.relaxator.preform_relaxation ( self.guess, AtomMap, BondMap )
        AFMs,AuxMap  = self.simulator.perform_imaging    ( self.relaxed )
        Err, self.guess   = self.critique.try_improve         ( self.relaxed, AFMs, AFMRef )
        return Err
        #Xs,Ys      = simulator.next1( self )

if __name__ == "__main__":
    print( " UNIT_TEST START : CorrectionLoop ... " )

    # ------ Init Generator

    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    simulator  = GeneratorOCL_LJC.Generator( [], [], 1, pixPerAngstrome=5, Ymode='AtomsAndBonds', lvec=lvec  )

    # ------ Init Relaxator

    relaxator = FARFF.CEngineFARFF()

    # ------ Init Critique

    critique = Critique()

    # ------ Init Looper

    looper = CorrectionLoop(relaxator,simulator,critique)

    ErrConv = 1.0
    for itr in range(1000):
        Err = looper.iteration()
        if Err > ErrConv:
            break

    #print( "UNIT_TEST is not yet written :-( " )
    print( " UNIT_TEST CorrectionLoop DONE !!! " )
    pass