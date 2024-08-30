import numpy as np

from ppafm import common


def test_get_df_weight():
    # amplitude = dz = 0.1
    assert np.allclose(common.get_df_weight(0.1), np.array([-5.0, 5.0]))

    # amplitude=1.0, dz=0.2
    w = common.get_df_weight(1.0, dz=0.2)
    assert np.allclose(
        w,
        np.array([-0.35594622, -0.22193265, -0.05447093, 0.05447093, 0.22193265, 0.35594622]),
    )


def test_get_simple_df_weight():
    w = common.get_simple_df_weight(n=5, dz=0.2)
    assert np.allclose(w, np.array([-0.31362841, -0.37274317, 0, 0.37274317, 0.31362841]))


def test_sphere_tangent_space(n=2):
    rots = common.sphereTangentSpace(n)
    assert np.allclose(
        rots,
        np.array(
            [
                [[-0.0, 1.0, 0.0], [-0.5, -0.0, 0.8660254], [0.8660254, 0.0, 0.5]],
                [
                    [-0.67549029, -0.73736888, 0.0],
                    [-0.36868444, 0.33774515, 0.8660254],
                    [-0.63858018, 0.58499175, -0.5],
                ],
            ]
        ),
    )


def test_ppafm_parameters():
    # First, testing the default values
    p_default = common.PpafmParameters()
    assert p_default.PBC == True
    assert p_default.nPBC == [1, 1, 1]
    assert p_default.probeType == "O"
    assert np.isclose(p_default.charge, 0.0)
    assert p_default.ffModel == "LJ"
    assert np.isclose(p_default.r0Probe, [0.0, 0.0, 4.0]).all()
    assert p_default.gridN == [-1, -1, -1]
    assert np.isclose(p_default.klat, 0.5)
    assert np.isclose(p_default.gridA, [20.0, 0.0, 0.0]).all()
    assert np.isclose(p_default.gridB, [0.0, 20.0, 0.0]).all()
    assert np.isclose(p_default.gridC, [0.0, 0.0, 20.0]).all()
    assert np.isclose(p_default.scanStep, 0.1).all()
    assert np.isclose(p_default.scanMin, [0.0, 0.0, 5.0]).all()
    assert np.isclose(p_default.scanMax, [20.0, 20.0, 8.0]).all()
    assert np.isclose(p_default.Amplitude, 1.0)
    assert np.isclose(p_default.aMorse, -1.6)
    assert p_default.vdWDampKind == 2

    # Load the parameters from a file
    p_ini = common.PpafmParameters.from_file("data/test_params.ini")
    assert p_ini.PBC == False
    assert p_ini.nPBC == [2, 3, 4]
    assert p_ini.probeType == "Xe"
    assert np.isclose(p_ini.charge, 0.5)
    assert p_ini.ffModel == "Morse"
    assert np.isclose(p_default.r0Probe, [0.0, 0.0, 4.0]).all()
    assert np.isclose(p_ini.gridN, [10, 20, 30]).all()
    assert np.isclose(p_ini.klat, 1.5)
    assert np.isclose(p_ini.gridA, [5.0, 6.0, 7.0]).all()
    assert np.isclose(p_ini.gridB, [8.0, 9.0, 10.0]).all()
    assert np.isclose(p_ini.gridC, [11.0, 12.0, 13.0]).all()
    assert np.isclose(p_ini.scanStep, [0.1, 0.2, 0.3]).all()
    assert np.isclose(p_ini.scanMin, [1.0, 2.0, 5.0]).all()
    assert np.isclose(p_ini.scanMax, [21.0, 22.0, 9.0]).all()
    assert np.isclose(p_ini.Amplitude, 8.0)
    assert np.isclose(p_ini.aMorse, -2.6)
    assert p_ini.vdWDampKind == 3

    # Dump the parameters to a toml file
    p_ini.to_file("data/test_params.toml")

    # Load the parameters from the toml file and compare to the original
    p_toml = common.PpafmParameters.from_file("data/test_params.toml")
    assert p_ini == p_toml
