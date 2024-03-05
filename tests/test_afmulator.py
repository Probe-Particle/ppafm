#!/usr/bin/env python3

import os

import numpy as np

from ppafm.ocl.AFMulator import AFMulator


def test_afmulator_save_load():
    afmulator_original = AFMulator(
        pixPerAngstrome=15,
        lvec=np.array([[1, 1, 1], [5, 1, 0], [2, 5, 0], [0, 0, 8]]),
        scan_dim=(200, 200, 60),
        scan_window=((0.0, 0.0, 16.0), (19.9, 19.9, 22.0)),
        iZPP=8,
        df_steps=10,
        tipStiffness=(0.37, 0.37, 0.0, 20.0),
        rho={"dz2": -0.05},
        sigma=0.8,
        A_pauli=15.0,
        B_pauli=1.0,
        tipR0=[0.0, 0.0, 4.0],
        npbc=(1, 1, 1),
        f0Cantilever=30000,
        kCantilever=2000,
    )

    # Save parameters to a file
    params_path = "./params_test.ini"
    afmulator_original.save_params(params_path)

    # Load the same parameters two different ways from the saved file
    afmulator_new1 = AFMulator.from_params(params_path)
    afmulator_new2 = AFMulator()
    afmulator_new2.load_params(params_path)

    # Check that the parameters are the same as they were in the beginning
    for afmulator in (afmulator_new1, afmulator_new2):
        assert np.allclose(afmulator.scan_dim, afmulator_original.scan_dim)
        assert np.allclose(afmulator.scan_window, afmulator_original.scan_window)
        assert np.allclose(afmulator.iZPP, afmulator_original.iZPP)
        assert np.allclose(afmulator.df_steps, afmulator_original.df_steps)
        assert np.allclose(afmulator.scanner.stiffness, afmulator_original.scanner.stiffness)
        assert np.allclose(afmulator.tipR0, afmulator_original.tipR0)
        assert np.allclose(afmulator.f0Cantilever, afmulator_original.f0Cantilever)
        assert np.allclose(afmulator.kCantilever, afmulator_original.kCantilever)
        assert np.allclose(afmulator.npbc, afmulator_original.npbc)
        assert np.allclose(afmulator.lvec, afmulator_original.lvec)
        assert np.allclose(afmulator.pixPerAngstrome, afmulator_original.pixPerAngstrome)
        assert afmulator._rho == afmulator_original._rho
        assert np.allclose(afmulator.sigma, afmulator_original.sigma)
        assert np.allclose(afmulator.A_pauli, afmulator_original.A_pauli)
        assert np.allclose(afmulator.B_pauli, afmulator_original.B_pauli)

    os.remove(params_path)
