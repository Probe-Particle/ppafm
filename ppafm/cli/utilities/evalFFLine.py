#!/usr/bin/python -u


import matplotlib.pyplot as plt
import numpy as np

import ppafm as PPU
import ppafm.core as PPC
from ppafm import common, io

# ======== setup

# O      14.877000000000001      9.099539999999999      5.729750430000000
# C      14.877000000000001      9.099539999999999      4.583204670000000

# 4,10, 11,59, 18,60
iatoms = [1, 2, 4, 10, 11, 59, 18, 60]

lst = [
    (3,),
    (4, 3),
    (4, 11),
    (4, 12),
    (4, 5),
    (4, 3, 11),
    (4, 11, 12),
    (4, 12, 5),
]
lss = [
    (":", "k"),
    ("-", "r"),
    ("-", "g"),
    ("-", "b"),
    ("-", "c"),
    ("--", "r"),
    ("--", "g"),
    ("--", "b"),
    ("--", "c"),
]

zmin = 3.0
zmax = 12.0
npts = 100

# vminF = 0.0; vminE = 0.0; scy=1.1
vminF = -0.25
vminE = -0.5
scy = 1.0

# ======== functions


def getAtomsPos(atoms, ia):
    ia -= 1
    return np.array([atoms[1][ia], atoms[2][ia], atoms[3][ia]])


def pos2line(pos, zmin=zmin, zmax=zmax, n=npts, label=""):
    return ((pos[0], pos[1], zmin), (pos[0], pos[1], zmax), n, label)


def getLines(lst, atoms):
    lines = []
    for item in lst:
        n = len(item)
        if n == 1:  # on top
            label = "t%i" % item[0]
            lines.append(pos2line(getAtomsPos(atoms, item[0]), label=label))
        elif n == 2:  # bridge
            label = "b%i_%i" % (item[0], item[1])
            lines.append(pos2line((getAtomsPos(atoms, item[0]) + getAtomsPos(atoms, item[1])) / 2.0, label=label))
        elif n == 3:  # hollow
            label = "h%i_%i_%i" % (item[0], item[1], item[2])
            lines.append(pos2line((getAtomsPos(atoms, item[0]) + getAtomsPos(atoms, item[1]) + getAtomsPos(atoms, item[2])) / 3.0, label=label))
    return lines


# ======== main

parameters = common.PpafmParameters()

PPU.loadParams("params.ini")
FFparams = PPU.loadSpecies("atomtypes.ini", parameters=parameters)
elem_dict = PPU.getFFdict(FFparams)
print(elem_dict)

atoms, nDim, lvec = io.loadGeometry("input_plot_mod0.xyz", parameters=parameters)
iZs, Rs, Qs = PPU.parseAtoms(atoms, elem_dict, autogeom=False, PBC=parameters.PBC)

lines = getLines(lst, atoms)
# print lines
# lines = [ ( (0.0,0.0,zmin), (0.0,0.0,zmax), npts, "Ag1"), ]

# print "iZs", iZs; print "Rs", Rs; print "Qs", Qs

parametes.probeType = "Xe"
cLJs = PPU.getAtomsLJ(PPU.atom2iZ(parametes.probeType, elem_dict), iZs, FFparams)

print("cLJs", cLJs)
np.savetxt("cLJs_2D.dat", cLJs)


plt.figure(figsize=(12, 8))

# print "len(atoms)", len(atoms), len(atoms[0])
# print "Rs.shape, cLJs.shape", Rs.shape, cLJs.shape

FEs = PPC.getInPoints_LJ(
    np.array(
        [
            [0.0, 0.0, 0.0],
        ]
    ),
    Rs,
    cLJs,
)


for i, (p1, p2, nps, label) in enumerate(lines):
    ts = np.linspace(0.0, 1.0, nps)
    ps = np.zeros((nps, 3))
    ps[:, 0] = p1[0] + (p2[0] - p1[0]) * ts
    ps[:, 1] = p1[1] + (p2[1] - p1[1]) * ts
    ps[:, 2] = p1[2] + (p2[2] - p1[2]) * ts
    FEs = PPC.getInPoints_LJ(ps, Rs, cLJs)
    plt.subplot(2, 1, 1)
    plt.plot(ps[:, 2], FEs[:, 2], ls=lss[i][0], c=lss[i][1], label=label)
    # vminF = min( vminF, np.nanmin(FEs[:,2]) )
    # plt.subplot(2,1,1); plt.plot( (ps[1:,2]+ps[:-1,2])*0.5, -(FEs[1:,3]-FEs[:-1,3])/(ps[1:,2]-ps[:-1,2]) );
    plt.subplot(2, 1, 2)
    plt.plot(ps[:, 2], FEs[:, 3], ls=lss[i][0], c=lss[i][1], label=label)
    # vminE = min( vminE, np.nanmin(FEs[:,3]) )
    print("vminF, vminE : ", vminF, vminE)

plt.subplot(2, 1, 1)
plt.ylim(vminF * scy, -vminF * scy)
plt.axhline(0, ls="--", c="k")
plt.xlabel("z [A]")
plt.ylabel("fz [eV/A]")
plt.legend()
plt.subplot(2, 1, 2)
plt.ylim(vminE * scy, -vminE * scy)
plt.axhline(0, ls="--", c="k")
plt.xlabel("z [A]")
plt.ylabel("E  [eV]")

plt.savefig("LJcurves.png", bbox_inches="tight")

plt.show()
