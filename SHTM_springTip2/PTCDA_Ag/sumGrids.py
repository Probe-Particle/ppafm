#!/usr/bin/python

from xsfutil import *
from basUtils import *
from pylab import *
import sys,os

pauli,lvec, nDim, head = loadXSF('FFpauli_3.xsf')
vdw,lvec, nDim, head   = loadXSF('FFvdw_3.xsf')

LJ = pauli + vdw

saveXSF('FFLJ_3.xsf', head, lvec, LJ )

#show()
