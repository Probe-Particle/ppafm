#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append('..')
from pyProbeParticle.basUtils import loadXYZ, saveXYZ, parseCommentASE
from pyProbeParticle.elements import ELEMENTS

def test_xyz():

    N = 20
    test_file = 'io_test.xyz'

    xyzs = 10*np.random.rand(N, 3)
    Zs = np.random.randint(1, 100, N)
    elems = [ELEMENTS[z-1][1] for z in Zs]
    qs = np.random.rand(N)
    comment = 'test comment'

    saveXYZ(test_file, xyzs, elems, qs, comment)
    xyzs_, Zs_, qs_, comment_ = loadXYZ(test_file)

    assert np.allclose(xyzs, xyzs_)
    assert np.allclose(Zs, Zs_)
    assert np.allclose(qs, qs_)
    assert comment == comment_

    saveXYZ(test_file, xyzs, Zs, qs=None)
    _, _, qs_, _ = loadXYZ(test_file)
    assert np.allclose(qs_, np.zeros(N))

    os.remove(test_file)

def test_parse_comment_ase():

    comment = 'Lattice="40.587929240107826 0.0 0.0 0.0 35.15017780893861 0.0 0.0 0.0 42.485492908861346" Properties=species:S:1:pos:R:3:tags:I:1 pbc="T T T"'
    lvec, has_charges = parseCommentASE(comment)
    assert np.allclose(lvec,
        np.array([
            [ 0.0              ,  0.0             ,  0.0              ],
            [40.587929240107826,  0.0             ,  0.0              ],
            [ 0.0              , 35.15017780893861,  0.0              ],
            [ 0.0              ,  0.0             , 42.485492908861346]
        ], dtype=np.float32)
    )
    assert has_charges == False

    comment = 'Properties=species:S:1:pos:R:3:initial_charges:R:1 pbc="F F F"'
    lvec, has_charges = parseCommentASE(comment)
    assert lvec is None
    assert has_charges == True

if __name__ == '__main__':
    test_xyz()
    test_parse_comment_ase()
