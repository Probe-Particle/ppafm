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
    # S = np.genfromtxt('TipRSpline.ini')
    S = np.genfromtxt(StringIO(spline_ini), skip_header=1)

    parameters = common.PpafmParameters()

    print("TipRSpline.ini overrides harmonic tip")
    xs = S[:, 0].copy()
    print("xs: ", xs)
    ydys = S[:, 1:].copy()
    print("ydys: ", ydys)
    plt.plot(xs, ydys[:, 0], "o")
    plt.plot(xs, ydys[:, 0] + ydys[:, 1], ".")
    core.setTipSpline(xs, ydys)

    core.setTip(parameters=parameters)

    fs = np.zeros((60, 3))
    r0 = np.array([0.0, 0.0, 0.5])
    dr = np.array([0.0, 0.0, 0.1])
    R = np.array([0.0, 0.0, 0.0])
    xs = np.array(list(range(len(fs)))) * dr[2] + r0[2]
    # print "xs=",xs

    print(">>>  core.test_force( 1, r0, dr, R, fs )")
    core.test_force(1, r0, dr, R, fs)
    plt.plot(xs, fs[:, 2])

    print(">>>  core.test_force( 2, r0, dr, R, fs )")
    core.test_force(2, r0, dr, R, fs)
    plt.plot(xs, fs[:, 2])

    # print "fs:", fs

    plt.grid()
    plt.savefig("tip_force.png")
    plt.show()


if __name__ == "__main__":
    test_tip_force()
