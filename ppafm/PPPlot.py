#!/usr/bin/python

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# =========== defaults

default_figsize = (8, 8)
default_cmap = "gray"
default_interpolation = "bicubic"
default_atom_size = 0.10

# =========== Utils


def plotBonds(xyz, bonds):
    for b in bonds:
        i = b[0]
        j = b[1]
        plt.arrow(xyz[1][i], xyz[2][i], xyz[1][j] - xyz[1][i], xyz[2][j] - xyz[2][i], head_width=0.0, head_length=0.0, fc="k", ec="k", lw=1.0, ls="solid")


def plotAtoms(atoms, atomSize=default_atom_size, edge=True, ec="k", color="w"):
    plt.fig = plt.gcf()
    atoms[0]
    xs = atoms[1]
    ys = atoms[2]
    if len(atoms) > 4:
        colors = atoms[4]
    else:
        colors = [color] * 100
    for i in range(len(atoms[1])):
        fc = "#%02x%02x%02x" % colors[i]
        if not edge:
            ec = fc
        circle = plt.Circle((xs[i], ys[i]), atomSize, fc=fc, ec=ec)
        plt.fig.gca().add_artist(circle)


def plotGeom(atoms=None, bonds=None, atomSize=default_atom_size):
    if (bonds is not None) and (atoms is not None):
        plotBonds(atoms, bonds)
    if atoms is not None:
        plotAtoms(atoms, atomSize=atomSize)


def colorize_XY2RG(Xs, Ys):
    r = np.sqrt(Xs**2 + Ys**2)
    vmax = r[5:-5, 5:-5].max()
    Red = 0.5 * Xs / vmax + 0.5
    Green = 0.5 * Ys / vmax + 0.5
    c = np.array((Red, Green, 0.5 * np.ones(np.shape(Red))))  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    c = c.swapaxes(0, 1)
    return c, vmax


def write_plotting_slice(i):
    sys.stdout.write(f"\r plotting slice # {i}")
    sys.stdout.flush()


# =========== plotting functions


def plotImages(
    prefix,
    F,
    slices,
    extent=None,
    zs=None,
    figsize=default_figsize,
    cmap=default_cmap,
    interpolation=default_interpolation,
    vmin=None,
    vmax=None,
    cbar=False,
    atoms=None,
    bonds=None,
    atomSize=default_atom_size,
    symmetric_map=False,
    V0=0.0,
    cbar_label=None,
):
    for ii, i in enumerate(slices):
        # print(" plotting ", i)
        write_plotting_slice(i)
        if symmetric_map:
            limit = max(abs(np.min(F[i] - V0)), abs(np.max(F[i] - V0)))
            vmin = -limit + V0
            vmax = limit + V0
        plt.figure(figsize=figsize)
        plt.imshow(F[i], origin="lower", interpolation=interpolation, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)

        if cbar:
            plt.colorbar(shrink=min(1.0, F[i].shape[0] / F[i].shape[1]), label=cbar_label)
        plotGeom(atoms, bonds, atomSize=atomSize)
        plt.xlabel(r" Tip_x $\AA$")
        plt.ylabel(r" Tip_y $\AA$")
        if zs is None:
            plt.title(r"iz = %i" % i)
        else:
            plt.title(r"Tip_z = %2.2f $\AA$" % zs[i])
        plt.savefig(prefix + "_%3.3i.png" % i, bbox_inches="tight")
        # plt.savefig(prefix + "_%3.3i.svg" % i, bbox_inches="tight")
        plt.close()


def plotVecFieldRG(
    prefix, dXs, dYs, slices, extent=None, zs=None, figsize=default_figsize, interpolation=default_interpolation, atoms=None, bonds=None, atomSize=default_atom_size
):
    for ii, i in enumerate(slices):
        # print(" plotting ", i)
        write_plotting_slice(i)
        plt.figure(figsize=(10, 10))
        HSBs, vmax = colorize_XY2RG(dXs[i], dYs[i])
        plt.imshow(HSBs, extent=extent, origin="lower", interpolation=interpolation)
        plotGeom(atoms, bonds, atomSize=atomSize)
        plt.xlabel(r" Tip_x $\AA$")
        plt.ylabel(r" Tip_y $\AA$")
        if zs is None:
            plt.title(r"iz = %i" % i)
        else:
            plt.title(r"Tip_z = %2.2f $\AA$" % zs[i])
        plt.savefig(prefix + "_%3.3i.png" % i, bbox_inches="tight")
        plt.close()


def plotDistortions(
    prefix,
    X,
    Y,
    slices,
    BG=None,
    by=2,
    extent=None,
    zs=None,
    figsize=default_figsize,
    cmap=default_cmap,
    interpolation=default_interpolation,
    vmin=None,
    vmax=None,
    cbar=False,
    markersize=1.0,
    atoms=None,
    bonds=None,
    atomSize=default_atom_size,
):
    for ii, i in enumerate(slices):
        # print(" plotting ", i)
        write_plotting_slice(i)
        plt.figure(figsize=figsize)
        plt.plot(X[i, ::by, ::by].flat, Y[i, ::by, ::by].flat, "r.", markersize=markersize)
        if BG is not None:
            plt.imshow(BG[i, :, :], origin="lower", interpolation=interpolation, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
            if cbar:
                plt.colorbar()
        plotGeom(atoms, bonds, atomSize=atomSize)
        plt.xlabel(r" Tip_x $\AA$")
        plt.ylabel(r" Tip_y $\AA$")
        if zs is None:
            plt.title(r"iz = %i" % i)
        else:
            plt.title(r"Tip_z = %2.2f $\AA$" % zs[i])
        plt.savefig(prefix + "_%3.3i.png" % i, bbox_inches="tight")
        plt.close()


def plotArrows(
    # not yet tested
    prefix,
    dX,
    dY,
    X,
    Y,
    slices,
    BG=None,
    C=None,
    extent=None,
    zs=None,
    by=2,
    figsize=default_figsize,
    cmap=default_cmap,
    interpolation=default_interpolation,
    vmin=None,
    vmax=None,
    cbar=False,
    atoms=None,
    bonds=None,
    atomSize=default_atom_size,
):
    for ii, i in enumerate(slices):
        # print(" plotting ", i)
        write_plotting_slice(i)
        plt.figure(figsize=figsize)
        plt.quiver(Xs[::by, ::by], Ys[::by, ::by], dX[::by, ::by], dY[::by, ::by], color="k", headlength=10, headwidth=10, scale=15)
        if BG is not None:
            plt.imshow(BG[i, :, :], origin="lower", interpolation=interpolation, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
            if cbar:
                plt.colorbar()
        plotGeom(atoms, bonds, atomSize=atomSize)
        plt.xlabel(r" Tip_x $\AA$")
        plt.ylabel(r" Tip_y $\AA$")
        if zs is None:
            plt.title(r"iz = %i" % i)
        else:
            plt.title(r"Tip_z = %2.2f $\AA$" % zs[i])
        plt.savefig(prefix + "_%3.3i.png" % i, bbox_inches="tight")
        plt.close()


def checkField(F):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(F[F.shape[0] / 2, :, :], interpolation="nearest")
    plt.title("F(y,x)")
    plt.subplot(1, 3, 2)
    plt.imshow(F[:, F.shape[1] / 2, :], interpolation="nearest")
    plt.title("F(z,x)")
    plt.subplot(1, 3, 3)
    plt.imshow(F[:, :, F.shape[2] / 2], interpolation="nearest")
    plt.title("F(z,y)")
    plt.show()


def checkVecField(FF):
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.imshow(FF[FF.shape[0] / 2, :, :, 0], interpolation="nearest")
    plt.title("FF_x(y,x)")
    plt.subplot(3, 3, 2)
    plt.imshow(FF[FF.shape[0] / 2, :, :, 1], interpolation="nearest")
    plt.title("FF_y(y,x)")
    plt.subplot(3, 3, 3)
    plt.imshow(FF[FF.shape[0] / 2, :, :, 2], interpolation="nearest")
    plt.title("FF_z(y,x)")
    plt.subplot(3, 3, 4)
    plt.imshow(FF[:, FF.shape[1] / 2, :, 0], interpolation="nearest")
    plt.title("FF_x(z,x)")
    plt.subplot(3, 3, 5)
    plt.imshow(FF[:, FF.shape[1] / 2, :, 1], interpolation="nearest")
    plt.title("FF_y(z,x)")
    plt.subplot(3, 3, 6)
    plt.imshow(FF[:, FF.shape[1] / 2, :, 2], interpolation="nearest")
    plt.title("FF_z(z,x)")
    plt.subplot(3, 3, 7)
    plt.imshow(FF[:, :, FF.shape[2] / 2, 0], interpolation="nearest")
    plt.title("FF_x(z,y)")
    plt.subplot(3, 3, 8)
    plt.imshow(FF[:, :, FF.shape[2] / 2, 1], interpolation="nearest")
    plt.title("FF_y(z,y)")
    plt.subplot(3, 3, 9)
    plt.imshow(FF[:, :, FF.shape[2] / 2, 2], interpolation="nearest")
    plt.title("FF_z(z,y)")
    plt.savefig("checkfield.png", bbox_inches="tight")


# ================


def makeCmap_Blue1(vals=(0.25, 0.5, 0.75)):
    cdict = {
        "red": ((0.0, 1.0, 1.0), (vals[0], 1.0, 1.0), (vals[1], 1.0, 1.0), (vals[2], 0.0, 0.0), (1.0, 0.0, 0.0)),
        "green": ((0.0, 0.0, 0.0), (vals[0], 1.0, 1.0), (vals[1], 1.0, 1.0), (vals[2], 1.0, 1.0), (1.0, 0.0, 0.0)),
        "blue": ((0.0, 0.0, 0.0), (vals[0], 0.0, 0.0), (vals[1], 1.0, 1.0), (vals[2], 1.0, 1.0), (1.0, 1.0, 1.0)),
    }
    return LinearSegmentedColormap("BlueRed1", cdict)


def makeCmap_Blue2(vals=(0.25, 0.5, 0.75)):
    cdict = {
        "red": ((0.0, 1.0, 1.0), (vals[0], 1.0, 1.0), (vals[1], 0.0, 0.0), (vals[2], 0.0, 0.0), (1.0, 0.0, 0.0)),
        "green": ((0.0, 1.0, 1.0), (vals[0], 0.0, 0.0), (vals[1], 0.0, 0.0), (vals[2], 0.0, 0.0), (1.0, 1.0, 1.0)),
        "blue": ((0.0, 0.0, 0.0), (vals[0], 0.0, 0.0), (vals[1], 0.0, 0.0), (vals[2], 1.0, 1.0), (1.0, 1.0, 1.0)),
    }
    return LinearSegmentedColormap("BlueRed1", cdict)
