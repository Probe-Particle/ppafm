#!/usr/bin/python -u

from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

from ppafm import common, core


def test_tip_force():

    spline_ini = """
    0.0   0.4 -0.1
    2.0   0.2 -0.1
    4.0   0.0 -0.1
    5.0  -4.0 -4.0
    10.0 -24.0 -4.0
    """

    tip_spline = core.SplineParameters.from_file(StringIO(spline_ini))

    print("xs: ", tip_spline.np_rff_xs)
    print("ydys: ", tip_spline.np_rff_ydys)
    plt.plot(tip_spline.np_rff_xs, tip_spline.np_rff_ydys[:, 0], "o")
    plt.plot(tip_spline.np_rff_xs, tip_spline.np_rff_ydys[:, 0] + tip_spline.np_rff_ydys[:, 1], ".")

    core.setTip(parameters=common.PpafmParameters())

    fs = np.zeros((60, 3))
    r0 = np.array([0.0, 0.0, 0.5])
    dr = np.array([0.0, 0.0, 0.1])
    R = np.array([0.0, 0.0, 0.0])
    xs = np.array(list(range(len(fs)))) * dr[2] + r0[2]
    # print "xs=",xs

    print(">>>  core.test_force( 1, r0, dr, R, fs )")
    core.test_force(1, r0, dr, R, fs, tip_spline)
    plt.plot(xs, fs[:, 2])

    print(">>>  core.test_force( 2, r0, dr, R, fs )")
    core.test_force(2, r0, dr, R, fs, tip_spline)
    plt.plot(xs, fs[:, 2])

    # print "fs:", fs

    plt.grid()
    plt.savefig("tip_force.png")
    plt.show()


if __name__ == "__main__":
    test_tip_force()
