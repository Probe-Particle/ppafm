#!/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np

print(" # ========== make & load  ProbeParticle C++ library ")

import ppafm as PP
from ppafm import io

print(" ============= RUN  ")

F,lvec,nDim,head=io.loadXSF('LOCPOT.xsf')

F4 = 0.25*( F + F[:,:,::-1] + F[:,::-1,:] + F[:,::-1,::-1] )

io.saveXSF('LOCPOT_4sym.xsf', head, lvec, F4 )

plt.show()
