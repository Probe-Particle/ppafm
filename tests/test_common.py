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
