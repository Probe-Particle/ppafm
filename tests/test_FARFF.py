#!/usr/bin/python

import sys
import os
import numpy as np
import time

sys.path.append("../../")
#import pyProbeParticle.basUtils as bu
import pyProbeParticle.atomicUtils as au
import pyProbeParticle.GLView as glv
import pyProbeParticle.FARFF  as fff

if __name__ == "__main__":

    #fff = sys.modules[__name__]
    xyzs,Zs,elems,qs = au.loadAtomsNP("input.xyz")     #; print xyzs

    #fff.insertAtomType(nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW, Epz)

    natom  = len(xyzs)
    ndof   = fff.reallocFF(natom)
    norb   = ndof - natom
    #atypes = fff.getTypes (natom)    ; print "atypes.shape ", atypes.shape
    dofs   = fff.getDofs(ndof)       ; print("dofs.shape ", dofs.shape)
    apos   = dofs[:natom]            ; print("apos.shape ", apos.shape)
    opos   = dofs[natom:]            ; print("opos.shape ", opos.shape)


    #atypes[:] = 0        # use default atom type
    apos[:,:] = xyzs[:,:] #
    #opos[:,:] = np.random.rand( norb, 3 ); print "opos.shape ", opos #   exit()

    cog = np.sum( apos, axis=0 )
    cog*=(1./natom)
    apos -= cog[None,:]

    fff.setupFF(n=natom)   # use default atom type

    atomMapF, bondMapF = fff.makeGridFF( fff )    # prevent GC from deleting atomMapF, bondMapFF

    fff.setupOpt(dt=0.05, damp=0.2, f_limit=100.0, l_limit=0.2 )
    #fff.relaxNsteps(50, Fconv=1e-6, ialg=0)

    glview = glv.GLView()
    for i in range(1000000):
        glview.pre_draw()
        F2err = fff.relaxNsteps(1, Fconv=1e-6, ialg=0)
        print("|F| ", np.sqrt(F2err))
        if glview.post_draw(): break
        time.sleep(.05)

    '''
    def animRelax(i, perFrame=1):
        fff.relaxNsteps(perFrame, Fconv=1e-6, ialg=0)
        return apos,None
    au.makeMovie( "movie.xyz", 100, elems, animRelax )
    '''