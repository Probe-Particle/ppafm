import numpy as np

from ppafm import common


def test_get_df_weight():
    x, y = common.getDfWeight(n=5, dz=0.2)
    assert np.allclose(x, np.array([-0.53836624, -0.12099505, -0.0, 0.12099505, 0.53836624]))
    assert np.allclose(
        y,
        np.array(
            [
                -8.00000000e-01,
                -4.00000000e-01,
                1.11022302e-16,
                4.00000000e-01,
                8.00000000e-01,
            ]
        ),
    )


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
