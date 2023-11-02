#!/usr/bin/env python3

import os

import numpy as np


def test_xyz():
    from ppafm.elements import ELEMENTS
    from ppafm.io import loadXYZ, saveXYZ

    N = 20
    test_file = "io_test.xyz"

    xyzs = 10 * np.random.rand(N, 3)
    Zs = np.random.randint(1, 100, N)
    elems = [ELEMENTS[z - 1][1] for z in Zs]
    qs = (np.random.rand(N) - 0.5) / N
    comment = "test comment"

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
    from ppafm.io import _getCharges, parseLvecASE

    comment = 'Lattice="40.587929240107826 0.0 0.0 0.0 35.15017780893861 0.0 0.0 0.0 42.485492908861346" Properties=species:S:1:pos:R:3:tags:I:1 pbc="T T T"'
    lvec = parseLvecASE(comment)
    assert np.allclose(lvec, np.array([[0.0, 0.0, 0.0], [40.587929240107826, 0.0, 0.0], [0.0, 35.15017780893861, 0.0], [0.0, 0.0, 42.485492908861346]], dtype=np.float32))

    comment = 'Properties=species:S:1:pos:R:3:initial_charges:R:1 pbc="F F F"'
    lvec = parseLvecASE(comment)
    assert lvec is None

    comment = 'Properties=species:S:1:pos:R:3:tags:I:1:initial_charges:R:1 pbc="F F F"'
    extra_cols = (np.random.rand(10, 2) - 0.5) / 10
    extra_cols_ = [[str(ex[0]), (ex[1])] for ex in extra_cols]
    qs = _getCharges(comment, extra_cols_)
    assert np.allclose(qs, extra_cols[:, 1]), qs

    comment = 'Properties=species:S:1:pos:R:3:initial_charges:R:1:tags:I:1 pbc="F F F"'
    qs = _getCharges(comment, extra_cols_)
    assert np.allclose(qs, extra_cols[:, 0]), qs

    comment = 'Properties=species:S:1:pos:R:3:tags:I:1 pbc="F F F"'
    extra_cols = [[str(v)] for v in np.random.rand(10)]
    qs = _getCharges(comment, extra_cols)
    assert np.allclose(qs, np.zeros(10)), qs


def test_load_aims():
    from ppafm.common import params
    from ppafm.io import loadGeometry

    temp_file = "temp_geom.in"
    geometry_str = """
        lattice_vector 4.0 0.0 0.0
        lattice_vector 1.0 5.0 0.0
        lattice_vector 0.0 0.0 6.0
        atom 1.0 3.0 5.5 C
        atom 2.0 4.0 6.6 O
    """

    with open(temp_file, "w") as f:
        f.write(geometry_str)

    atoms, nDim, lvec = loadGeometry(temp_file, params=params)

    assert np.allclose(
        atoms,
        np.array(
            [
                [6.0, 8.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.5, 6.6],
                [0.0, 0.0],
            ]
        ),
    )
    assert np.allclose(nDim, params["gridN"])
    assert np.allclose(lvec, np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 6.0]]))

    os.remove(temp_file)
