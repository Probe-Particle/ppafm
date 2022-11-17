#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append('..')
from pyProbeParticle.basUtils import loadXYZ, saveXYZ
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

if __name__ == '__main__':
    test_xyz()
