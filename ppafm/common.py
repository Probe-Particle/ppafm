#!/usr/bin/python

import os
from argparse import ArgumentParser

import numpy as np

from . import cpp_utils

verbose = 0

# ====================== constants

eVA_Nm = 16.0217662  # [eV/A^2] / [N/m]
CoulombConst = 14.399645
HBAR = 6.58211951440e-16  # [eV.s]
AUMASS = 1.66053904020e-27  # [kg]


# fmt: off
# default parameters of simulation
params = {
    "PBC": True,
    "nPBC": np.array([1, 1, 1]),
    "gridN": np.array([-1, -1, -1]).astype(int),
    "gridO": np.array([0.0, 0.0, 0.0]),
    "gridA": np.array([0.0, 0.0, 0.0]),
    "gridB": np.array([0.0, 0.0, 0.0]),
    "gridC": np.array([0.0, 0.0, 0.0]),
    "FFgrid0": np.array([-1.0, -1.0, -1.0]),
    "FFgridA": np.array([-1.0, -1.0, -1.0]),
    "FFgridB": np.array([-1.0, -1.0, -1.0]),
    "FFgridC": np.array([-1.0, -1.0, -1.0]),
    "moleculeShift": np.array([0.0, 0.0, 0.0]),
    "probeType": "O",
    "charge": 0.00,
    "Apauli": 18.0,
    "Bpauli": 1.0,
    "ffModel": "LJ",
    "Rcore": 0.7,
    "r0Probe": np.array([0.00, 0.00, 4.00]),
    "stiffness": np.array([-1.0, -1.0, -1.0]),
    "klat": 0.5,
    "krad": 20.00,
    "tip": "s",
    "sigma": 0.7,
    "scanStep": np.array([0.10, 0.10, 0.10]),
    "scanMin": np.array([0.0, 0.0, 5.0]),
    "scanMax": np.array([20.0, 20.0, 8.0]),
    "scanTilt": np.array([0.0, 0.0, -0.1]),
    "tiltedScan": False,
    "kCantilever": 1800.0,
    "f0Cantilever": 30300.0,
    "Amplitude": 1.0,
    "plotSliceFrom": 16,
    "plotSliceTo": 22,
    "plotSliceBy": 1,
    "imageInterpolation": "bicubic",
    "colorscale": "gray",
    "colorscale_kpfm": "seismic",
    "ddisp": 0.05,
    "aMorse": -1.6,
    "tip_base": np.array(["None", 0.00]),
    "Rtip": 30.0,
    "permit": 0.00552634959,
    "vdWDampKind": 2,
    "#": None,
}
# fmt: on


class CLIParser(ArgumentParser):
    """
    Subclass from the built-in ArgumentParser with functionality to easily add arguments commonly used in ppafm scripts.
    """

    _cli_args = None
    _params_args = {
        "Amplitude": {
            "short_name": "-a",
            "action": "store",
            "type": float,
            "help": "Oscillation amplitude [Å]",
        },
        "klat": {
            "short_name": "-k",
            "action": "store",
            "type": float,
            "help": "Lateral tip stiffness [N/m]",
        },
        "charge": {
            "short_name": "-q",
            "action": "store",
            "type": float,
            "help": "Probe particle charge [e]",
        },
        "tip": {
            "short_name": "-t",
            "action": "store",
            "type": str,
            "default": "s",
            "help": "Tip model (multipole) {s,pz,dz2,..}",
        },
        "Rcore": {
            "action": "store",
            "type": float,
            "default": params["Rcore"],
            "help": "Width of nuclear charge density blob to achieve charge neutrality [Å]",
        },
        "sigma": {
            "short_name": "-w",
            "action": "store",
            "type": float,
            "help": "Gaussian width for convolution in Electrostatics [Å]",
        },
        "Apauli": {
            "short_name": "-A",
            "action": "store",
            "type": float,
            "default": 1.0,
            "help": "Prefactor A in the density overlap integral.",
        },
        "Bpauli": {
            "short_name": "-B",
            "action": "store",
            "type": float,
            "default": -1.0,
            "help": "Exponent B in the density overlap integral. Negative value is equivalent to B=1.",
        },
        "ffModel": {
            "action": "store",
            "default": "LJ",
            "help": "Force field model ('LJ','Morse','vdW')",
        },
    }
    _extra_args = {
        "input": {
            "short_name": "-i",
            "action": "store",
            "required": True,
            "help": "Input file path. Mandatory. Supported formats are: .xyz, .cube, .xsf.",
        },
        "input_format": {
            "short_name": "-F",
            "action": "store",
            "default": None,
            "help": "Specify the input file format. By default the format is inferred from the file extension.",
        },
        "output_format": {
            "short_name": "-f",
            "action": "store",
            "default": "xsf",
            "help": "Specify the output format. Supported formats are: xsf, npy",
        },
        "noPBC": {
            "action": "store_false",
            "dest": "PBC",
            "default": None,
            "help": "Disable periodic boundary conditions.",
        },
        "energy": {
            "short_name": "-E",
            "action": "store_true",
            "default": False,
            "help": "Compute the potential energy in addition to the force.",
        },
        "krange": {
            "action": "store",
            "type": float,
            "nargs": 3,
            "metavar": ("k_min", "k_max", "n_k"),
            "help": "Do scan for a range of tip stiffnesses. Overrides --klat.",
        },
        "qrange": {
            "action": "store",
            "type": float,
            "nargs": 3,
            "metavar": ("q_min", "q_max", "n_q"),
            "help": "Do scan for a range of tip charges. Overrides --charge.",
        },
        "arange": {
            "action": "store",
            "type": float,
            "nargs": 3,
            "metavar": ("A_min", "A_max", "n_A"),
            "help": "Do scan for a range of amplitudes. Overrides --Amplitude.",
        },
        "Vbias": {
            "short_name": "-V",
            "action": "store",
            "type": float,
            "help": "Applied bias voltage [V].",
        },
        "Vrange": {
            "action": "store",
            "type": float,
            "nargs": 3,
            "metavar": ("V_min", "V_max", "n_V"),
            "help": "Do scan for a range of voltages. Overrides --Vbias.",
        },
    }

    def _check_params_args(self):
        for arg in self._params_args:
            if arg not in params:
                raise ValueError(f"Argument name `{arg}` does not match with any parameter in global parameters dictionary.")

    @property
    def cli_args(self):
        """Dictionary of arguments."""
        if self._cli_args is None:
            self._check_params_args()
            self._cli_args = {**self._params_args, **self._extra_args}
        return self._cli_args

    def add_arguments(self, arg_names):
        """
        Add CLI arguments from predefined dictionary :data:`cli_params`.

        Arguments:
            arg_names: list of str. Names of arguments to add.
        """
        for name in arg_names:
            if name not in self.cli_args:
                raise ValueError(f"Invalid argument name `{name}`")
            arg_dict = self.cli_args[name]
            if "help" not in arg_dict:
                raise ValueError(f"No help message defined for `{name}`")
            if "short_name" in arg_dict:
                arg_names = [arg_dict["short_name"], f"--{name}"]
                del arg_dict["short_name"]
            else:
                arg_names = [f"--{name}"]
            self.add_argument(*arg_names, **arg_dict)


# ==============================
# ============================== Pure python functions
# ==============================


def get_df_weight(Amp, dz=0.1):
    """
    Conversion of vertical force Fz to frequency shift
    Returns the discretized version of the required convolution kernel

    dz = the grid step (distance between the nearest grid points) in the direction in which the tip oscillates
    Amp = peak-to-peak oscillation amplitude (that is, twice the amplitude in the usual sense)

    The integrand for calculating the frequency shift from force is taken from:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)

    The continuous weight function needed for the conversion is w(t) = t/sqrt(1-t**2).
    We want to find an array w[i] (of n+1 elements) - the discrete convolution kernel - such that

    (F*w)[j] = Sum_over(i) F[i+j] w[n+1-i] = -2/(Amp*pi) Integral_from(t=-1)_to(t=+1)_of F(t) w(t) dt

    where Amp is the amplitude and t the normalized coordinate t = z*2/Amp.
    Linear interpolation will be used to define the force field F(t) from its discretized version F[i]:

    F(t) = ( F[i] * (t[i+1] - t) + F[i+1] * (t - t[i]) ) / dt for t[i] < t < t[i+1]

    where dt = dz*2/Amp = t[i+1] - t[i].
    For each i, the interval from t[i] up to t[i+1] contribues to w[n+1-i] the following weight
    (omitting the coefficient of 1/(pi*dz) but including the minus sign in front of the original integrtal)

    Integral_from(t=t[i])_to(t=t[i+1])_of dt*w(t)*( -t[i+1] + t ) = t[i+1]*(f[i+1] - f[i]) + (f2[i+1] - f2[i])

    corresponding to the coefficient multiplying F[i],
    and, additionally, to w[n+1-(i+1)] = w[n-i] the weight

    Integral_from(t=t[i])_to(t=t[i+1])_of dt*w(t)*( t[i] - t ) = -t[i]*(f[i+1] - f[i]) - (f2[i+1] - f2[i])

    corresponding to the coefficient multiplying F[i+1].
    In the above, two auxiliary functions were introduced to express the desired integrals,

    f(t) = Integral_of -w(t)*dt = sqrt(1-t**2)
    f2(t) = Integral_of w(t)*t*dt = (arccos(-t) - sqrt(1-t**2)) / 2

    and their discrete versions f[i] = f(t[i]); f2[i] = f2(t[i]).
    Note: we set f(t) = 0 and f2(t) = pi for t > +1.
    """

    n = int(np.ceil(Amp / dz))
    w = np.zeros(n + 1)
    f = np.zeros(n + 1)
    f2 = np.zeros(n + 1)
    # Note that, because of how the convolution is defined, the weight array w needs to be reversed:
    # The largest index of w[i] corresponds to the smallest index i of force F[j+i].
    # In contrast to the theoretical description above, we already reverse all the auxiliary arrays (t,f,f2,df,df2) in the code,
    # starting from t[n+1-i] -> t[i], so that we do not need to reverse w explicitely in the end.
    t = np.linspace(-1 + 2 * n * dz / Amp, -1, n + 1)
    if n > 1:
        f[1:-1] = np.sqrt(1 - t[1:-1] * t[1:-1])
        f2[1:-1] = (np.arccos(-t[1:-1]) - t[1:-1] * f[1:-1]) / 2.0
    f2[0] = np.pi / 2  # end point for t>=1 must be as if t=1 even if t>1
    df = f[:-1] - f[1:]  # numerical derivative (first derivative integrated over one-step-long intervals) of function f(t)
    df2 = f2[:-1] - f2[1:]  # numerical derivative (first derivative integrated over one-step-long intervals) of f2(t)
    w[1:] = t[:-1] * df + df2  # coefficients to multiply F[i]
    w[:-1] -= t[1:] * df + df2  # coefficients to multiply F[i+1]
    return w / (np.pi * dz)


def get_simple_df_weight(n=10, dz=0.1):
    """
    Conversion of vertical force Fz to frequency shift
    Returns the discretized version of the required convolution kernel

    dz = the grid step (sistance between the nearest grid points) in the direction in which the tip oscillates
    n = number of grid points involved in the convolution (peak-to-peak amplitude = n *dz)

    Simpler version of the above get_df_weight()
    """
    n = int(n)
    if n < 2:
        n = 2
    t = np.linspace(-1 + 0.5 / n, 1 - 0.5 / n, n - 1)
    f = np.sqrt(1 - t * t)
    w = np.zeros(n)
    w[:-1] = f
    w[1:] -= f

    # renormalization
    t = np.linspace(-1, 1, n)
    w0 = np.sum(t * w)  # pi/2 in the large n limit
    return w / (w0 * n * dz)


def Fz2df(
    F,
    dz=0.1,
    k0=params["kCantilever"],
    f0=params["f0Cantilever"],
    amplitude=1.0,
    units=16.0217656,
):
    """
    conversion of vertical force Fz to frequency shift
    according to:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
    """
    W = get_df_weight(amplitude, dz=dz)
    dFconv = np.apply_along_axis(lambda m: np.convolve(m, W, mode="valid"), axis=0, arr=F)
    return dFconv * units * f0 / k0


def Fz2df_tilt(
    F,
    d=params["scanTilt"],
    k0=params["kCantilever"],
    f0=params["f0Cantilever"],
    amplitude=1.0,
    units=16.0217656,
):
    """
    conversion of vertical force Fz to frequency shift
    according to:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
    """
    dr = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    W = get_df_weight(amplitude, dz=dr)
    dFconv_x = np.apply_along_axis(lambda m: np.convolve(m, W, mode="valid"), axis=0, arr=F[:, :, :, 0])
    dFconv_y = np.apply_along_axis(lambda m: np.convolve(m, W, mode="valid"), axis=0, arr=F[:, :, :, 1])
    dFconv_z = np.apply_along_axis(lambda m: np.convolve(m, W, mode="valid"), axis=0, arr=F[:, :, :, 2])
    return (dFconv_x * d[0] + dFconv_y * d[1] + dFconv_z * d[2]) * units * f0 / k0


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def makeRotJitter(n=10, maxAngle=0.3):
    rotJitter = []
    nrn = 50
    rnjit = (np.random.rand(nrn, 4) - 0.5) * 2
    for i in range(nrn):
        axis = rnjit[i, :3] / np.sqrt(np.dot(rnjit[i, :3], rnjit[i, :3]))
        rotJitter.append(rotation_matrix(axis, rnjit[i, 3] * maxAngle))
    return rotJitter


def genRotations(axis, thetas):
    return np.array([rotation_matrix(axis, theta) for theta in thetas])


def sphereTangentSpace(n=100):
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1.0 - 1.0 / n, 1.0 / n - 1.0, n)
    radius = np.sqrt(1.0 - z * z)
    cas = np.cos(theta)
    sas = np.sin(theta)
    rots = np.zeros((n, 3, 3))
    rots[:, 2, 0] = radius * cas
    rots[:, 2, 1] = radius * sas
    rots[:, 2, 2] = z
    rots[:, 0, 0] = -sas
    rots[:, 0, 1] = cas
    rots[:, 1, :] = np.cross(rots[:, 2, :], rots[:, 0, :])
    return rots


def maxAlongDir(atoms, hdir):
    xdir = np.dot(atoms[:, :3], hdir[:, None])
    imin = np.argmax(xdir)
    return imin, xdir[imin][0]


def maxAlongDirEntropy(atoms, hdir, beta=1.0):
    xdir = np.dot(atoms[:, :3], hdir[:, None])
    imin = np.argmax(xdir)
    entropy = np.sum(np.exp(beta * (xdir - xdir[imin])))
    return imin, xdir[imin][0], entropy


# ==============================
# ==============================  server interface file I/O
# ==============================


def autoGridN():
    params["gridN"][0] = round(np.linalg.norm(params["gridA"]) * 10)
    params["gridN"][1] = round(np.linalg.norm(params["gridB"]) * 10)
    params["gridN"][2] = round(np.linalg.norm(params["gridC"]) * 10)
    return params["gridN"]


# overide default parameters by parameters read from a file
def loadParams(fname):
    if verbose > 0:
        print(" >> OVERWRITING SETTINGS by " + fname)
    fin = open(fname)
    for line in fin:
        words = line.split()
        if len(words) >= 2:
            key = words[0]
            if key in params:
                val = params[key]
                if key[0][0] == "#":
                    continue
                if verbose > 0:
                    print(key, " is class ", val.__class__)
                if isinstance(val, bool):
                    word = words[1].strip()
                    if (word[0] == "T") or (word[0] == "t"):
                        params[key] = True
                    else:
                        params[key] = False
                    if verbose > 0:
                        print(key, params[key], ">>", word, "<<")
                elif isinstance(val, float):
                    params[key] = float(words[1])
                    if verbose > 0:
                        print(key, params[key], words[1])
                elif isinstance(val, int):
                    params[key] = int(words[1])
                    if verbose > 0:
                        print(key, params[key], words[1])
                elif isinstance(val, str):
                    params[key] = words[1]
                    if verbose > 0:
                        print(key, params[key], words[1])
                elif isinstance(val, np.ndarray):
                    if val.dtype == float:
                        params[key] = np.array([float(words[1]), float(words[2]), float(words[3])])
                        if verbose > 0:
                            print(key, params[key], words[1], words[2], words[3])
                    elif val.dtype == int:
                        if verbose > 0:
                            print(key)
                        params[key] = np.array([int(words[1]), int(words[2]), int(words[3])])
                        if verbose > 0:
                            print(key, params[key], words[1], words[2], words[3])
                    else:
                        params[key] = np.array([str(words[1]), float(words[2])])
                        if verbose > 0:
                            print(key, params[key], words[1], words[2])
            else:
                raise ValueError(f"Parameter {key} is not known")
    fin.close()
    if params["gridN"][0] <= 0:
        autoGridN()

    params["tip"] = params["tip"].replace('"', "")
    params["tip"] = params["tip"].replace("'", "")
    ### necessary for working even with quotemarks in params.ini
    params["tip_base"][0] = params["tip_base"][0].replace('"', "")
    params["tip_base"][0] = params["tip_base"][0].replace("'", "")
    ### necessary for working even with quotemarks in params.ini


def apply_options(opt):
    if verbose > 0:
        print("!!!! OVERRIDE params !!!! in Apply options:")
    if verbose > 0:
        print(opt)
    for key, value in opt.items():
        if value is None:
            continue
        if key in params:
            params[key] = value
            if key == "klat" and params["stiffness"][0] > 0.0:
                print("Overriding stiffness parameter with klat")
                params["stiffness"] = np.array([-1.0, -1.0, -1.0])  # klat overrides stiffness
            if verbose > 0:
                print(f"Applied: {key} = {value}")


# load atoms species parameters form a file ( currently used to load Lennard-Jones parameters )
def loadSpecies(fname=None):
    if fname is None or not os.path.exists(fname):
        if verbose > 0:
            print("WARRNING: loadSpecies(None) => load default atomtypes.ini")
        fname = cpp_utils.PACKAGE_PATH / "defaults" / "atomtypes.ini"
    if verbose > 0:
        print(" loadSpecies from ", fname)
    # FFparams=np.genfromtxt(fname,dtype=[('rmin',np.float64),('epsilon',np.float64),('atom',int),('symbol', '|S10')],usecols=[0,1,2,3])
    FFparams = np.genfromtxt(
        fname,
        dtype=[
            ("rmin", np.float64),
            ("epsilon", np.float64),
            ("alpha", np.float64),
            ("atom", int),
            ("symbol", "|S10"),
        ],
        usecols=(0, 1, 2, 3, 4),
    )
    return FFparams


# load atoms species parameters form a file ( currently used to load Lennard-Jones parameters )
def loadSpeciesLines(lines):
    params = []
    for l in lines:
        l = l.split()
        if len(l) >= 5:
            # print l
            params.append((float(l[0]), float(l[1]), float(l[2]), int(l[3]), l[4]))
    return np.array(
        params,
        dtype=[
            ("rmin", np.float64),
            ("epsilon", np.float64),
            ("alpha", np.float64),
            ("atom", int),
            ("symbol", "|S10"),
        ],
    )


def autoGeom(Rs, shiftXY=False, fitCell=False, border=3.0):
    """
    set Force-Filed and Scanning supercell to fit optimally given geometry
    then shifts the geometry in the center of the supercell
    """
    zmax = max(Rs[2])
    Rs[2] -= zmax
    if verbose > 0:
        print(" autoGeom subtracted zmax = ", zmax)
    xmin = min(Rs[0])
    xmax = max(Rs[0])
    ymin = min(Rs[1])
    ymax = max(Rs[1])
    if fitCell:
        params["gridA"][0] = (xmax - xmin) + 2 * border
        params["gridA"][1] = 0
        params["gridB"][0] = 0
        params["gridB"][1] = (ymax - ymin) + 2 * border
        params["scanMin"][0] = 0
        params["scanMin"][1] = 0
        params["scanMax"][0] = params["gridA"][0]
        params["scanMax"][1] = params["gridB"][1]
        if verbose > 0:
            print(" autoGeom changed cell to = ", params["scanMax"])
    if shiftXY:
        dx = -0.5 * (xmin + xmax) + 0.5 * (params["gridA"][0] + params["gridB"][0])
        Rs[0] += dx
        dy = -0.5 * (ymin + ymax) + 0.5 * (params["gridA"][1] + params["gridB"][1])
        Rs[1] += dy
        if verbose > 0:
            print(" autoGeom moved geometry by ", dx, dy)


def wrapAtomsCell(Rs, da, db, avec, bvec):
    M = np.array((avec[:2], bvec[:2]))
    invM = np.linalg.inv(M)
    if verbose > 0:
        print(M)
    if verbose > 0:
        print(invM)
    ABs = np.dot(Rs[:, :2], invM)
    if verbose > 0:
        print("ABs.shape", ABs.shape)
    ABs[:, 0] = (ABs[:, 0] + 10 + da) % 1.0
    ABs[:, 1] = (ABs[:, 1] + 10 + db) % 1.0
    Rs[:, :2] = np.dot(ABs, M)


def PBCAtoms(Zs, Rs, Qs, avec, bvec, na=None, nb=None):
    """
    multiply atoms of sample along supercell vectors
    the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
    """
    Zs_ = []
    Rs_ = []
    Qs_ = []
    if na is None:
        na = params["nPBC"][0]
    if nb is None:
        nb = params["nPBC"][1]
    for i in range(-na, na + 1):
        for j in range(-nb, nb + 1):
            for iatom in range(len(Zs)):
                x = Rs[iatom][0] + i * avec[0] + j * bvec[0]
                y = Rs[iatom][1] + i * avec[1] + j * bvec[1]
                Zs_.append(Zs[iatom])
                Rs_.append((x, y, Rs[iatom][2]))
                Qs_.append(Qs[iatom])
    return np.array(Zs_).copy(), np.array(Rs_).copy(), np.array(Qs_).copy()


def PBCAtoms3D(Zs, Rs, Qs, cLJs, lvec, npbc=[1, 1, 1]):
    """
    multiply atoms of sample along supercell vectors
    the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
    """
    Zs_ = []
    Rs_ = []
    Qs_ = []
    cLJs_ = []
    for iatom in range(len(Zs)):
        Zs_.append(Zs[iatom])
        Rs_.append(Rs[iatom])
        Qs_.append(Qs[iatom])
    # fmt: off
    for ia in range(-npbc[0], npbc[0] + 1):
        for ib in range(-npbc[1], npbc[1] + 1):
            for ic in range(-npbc[2], npbc[2] + 1):
                if (ia == 0) and (ib == 0) and (ic == 0):
                    continue
                for iatom in range(len(Zs)):
                    x = Rs[iatom][0] + ia * lvec[0][0] + ib * lvec[1][0] + ic * lvec[2][0]
                    y = Rs[iatom][1] + ia * lvec[0][1] + ib * lvec[1][1] + ic * lvec[2][1]
                    z = Rs[iatom][2] + ia * lvec[0][2] + ib * lvec[1][2] + ic * lvec[2][2]
                    Zs_.append(Zs[iatom])
                    Rs_.append((x, y, z))
                    Qs_.append(Qs[iatom])
                    cLJs_.append(cLJs[iatom, :])
    # fmt: on
    return (
        np.array(Zs_).copy(),
        np.array(Rs_).copy(),
        np.array(Qs_).copy(),
        np.array(cLJs_).copy(),
    )


def findPBCAtoms3D_cutoff(Rs, lvec, Rcut=1.0, corners=None):
    """
    find which atoms with positions 'Rs' and radius 'Rcut' thouch rhombic cell defined by 3x3 matrix 'lvec';
       or more precisely which points 'Rs' belong to a rhombic cell enlarged by margin Rcut on each side
    all assuming that 'Rcut' is smaller than the rhombic cell (in all directions)
    """
    invLvec = np.linalg.inv(lvec)
    abc = np.dot(invLvec, Rs)  # atoms in grid coordinates
    # calculate margin on each side in grid coordinates
    ra = np.sqrt(np.dot(invLvec[0], invLvec[0]))
    mA = ra * Rcut
    rb = np.sqrt(np.dot(invLvec[1], invLvec[1]))
    mB = rb * Rcut
    rc = np.sqrt(np.dot(invLvec[2], invLvec[2]))
    mC = rc * Rcut
    cells = [-1, 0, 1]
    inds = []
    Rs_ = []
    a = abc[0]
    b = abc[1]
    c = abc[2]
    i = 0
    for ia in cells:
        mask_a = (a > (-mA - ia)) & (a < (mA + 1 - ia))
        shift_a = ia * lvec[0, :]
        for ib in cells:
            mask_ab = (b > (-mB - ib)) & (b < (mB + 1 - ib)) & mask_a
            shift_ab = ib * lvec[1, :] + shift_a
            for ic in cells:
                v_shift = ic * lvec[2, :] + shift_ab
                mask = mask_ab & (c > (-mC - ic)) & (c < (mC + 1 - ic))
                inds_abc = np.nonzero(mask)[0]
                if len(inds_abc) == 0:
                    continue
                Rs_abc = Rs[:, inds_abc]
                Rs_abc += v_shift[:, None]
                inds.append(inds_abc)
                Rs_.append(Rs_abc)
                i += 1
    inds = np.concatenate(inds)
    Rs_ = np.hstack(Rs_)

    # fmt: off
    if corners is not None:
        corns = np.array(
            [
                [  -mA,     -mB,    -mC],
                [  -mA,     -mB, 1 + mC],
                [  -mA,  1 + mB,    -mC],
                [  -mA,  1 + mB, 1 + mC],
                [1 + mA,    -mB,    -mC],
                [1 + mA,    -mB, 1 + mC],
                [1 + mA, 1 + mB,     -mC],
                [1 + mA, 1 + mB, 1 + mC],
            ]
        ).transpose()
        corners.append(np.dot(lvec, corns))
    # fmt: on
    return inds, Rs_


def PBCAtoms3D_np(Zs, Rs, Qs, cLJs, REAs, lvec, npbc=[1, 1, 1]):
    """
    multiply atoms of sample along supercell vectors
    the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
    """
    # print( "PBCAtoms3D_np lvec", lvec )
    mx = npbc[0] * 2 + 1
    my = npbc[1] * 2 + 1
    mz = npbc[2] * 2 + 1
    mtot = mx * my * mz
    natom = len(Zs)
    matom = mtot * natom
    Zs_ = np.empty(matom, np.int32)
    xyzs_ = np.empty((matom, 3), np.float32)
    qs_ = np.empty((matom,), np.float32)
    if cLJs is not None:
        cLJs_ = np.empty((matom, 2), np.float32)
    else:
        cLJs_ = None
    if REAs is not None:
        REAs_ = np.empty((matom, 4), np.float32)
    else:
        REAs_ = None
    i0 = 0
    i1 = i0 + natom
    # we want to have cell=(0,0,0) first
    Zs_[i0:i1] = Zs[:]
    xyzs_[i0:i1] = Rs[:, :]
    qs_[i0:i1] = Qs[:]
    if cLJs is not None:
        cLJs_[i0:i1, :] = cLJs[:, :]
    if REAs is not None:
        REAs_[i0:i1, :] = REAs[:, :]
    i0 += natom
    for ia in range(-npbc[0], npbc[0] + 1):
        for ib in range(-npbc[1], npbc[1] + 1):
            for ic in range(-npbc[2], npbc[2] + 1):
                if (ia == 0) and (ib == 0) and (ic == 0):
                    continue
                v_shift = ia * lvec[0, :] + ib * lvec[1, :] + ic * lvec[2, :]
                i1 = i0 + natom
                Zs_[i0:i1] = Zs[:]
                xyzs_[i0:i1] = Rs[:, :] + v_shift[None, :]
                qs_[i0:i1] = Qs[:]
                if cLJs is not None:
                    cLJs_[i0:i1, :] = cLJs[:, :]
                if REAs is not None:
                    REAs_[i0:i1, :] = REAs[:, :]
                i0 += natom
    return Zs_, xyzs_, qs_, cLJs_, REAs_


def multRot(Zs, Rs, Qs, cLJs, rots, cog=(0, 0, 0)):
    """
    multiply atoms of sample along supercell vectors
    the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
    """
    Zs_ = []
    Rs_ = []
    Qs_ = []
    cLJs_ = []
    for rot in rots:
        for iatom in range(len(Zs)):
            xyz = (Rs[iatom][0] - cog[0], Rs[iatom][1] - cog[1], Rs[iatom][2] - cog[2])
            x = xyz[0] * rot[0][0] + xyz[1] * rot[0][1] + xyz[2] * rot[0][2] + cog[0]
            y = xyz[0] * rot[1][0] + xyz[1] * rot[1][1] + xyz[2] * rot[1][2] + cog[1]
            z = xyz[0] * rot[2][0] + xyz[1] * rot[2][1] + xyz[2] * rot[2][2] + cog[2]
            Zs_.append(Zs[iatom])
            Rs_.append((x, y, z))
            Qs_.append(Qs[iatom])
            cLJs_.append(cLJs[iatom, :])
    return (
        np.array(Zs_).copy(),
        np.array(Rs_).copy(),
        np.array(Qs_).copy(),
        np.array(cLJs_).copy(),
    )


def getFFdict(FFparams):
    elem_dict = {}
    for i, ff in enumerate(FFparams):
        if verbose > 0:
            print(i, ff)
        elem_dict[ff[4]] = i + 1
    return elem_dict


def atom2iZ(atm, elem_dict):
    try:
        return int(atm)
    except:
        try:
            return elem_dict[atm.encode()]
        except:
            raise ValueError(f"Did not find atomkind: {atm}")


def atoms2iZs(names, elem_dict):
    return np.array([atom2iZ(name, elem_dict) for name in names], dtype=np.int32)


def parseAtoms(atoms, elem_dict, PBC=True, autogeom=False, lvec=None):
    Rs = np.array([atoms[1], atoms[2], atoms[3]])
    if elem_dict is None:
        if verbose > 0:
            print("WARRNING: elem_dict is None => iZs are zero")
        iZs = np.zeros(len(atoms[0]))
    else:
        iZs = atoms2iZs(atoms[0], elem_dict)
    if autogeom:
        if verbose > 0:
            print("WARRNING: autoGeom shifts atoms")
        autoGeom(Rs, shiftXY=True, fitCell=True, border=3.0)
    Rs = np.transpose(Rs, (1, 0)).copy()
    Qs = np.array(atoms[4])
    if PBC:
        if lvec is not None:
            avec = lvec[1]
            bvec = lvec[2]
        else:
            avec = params["gridA"]
            bvec = params["gridB"]
        iZs, Rs, Qs = PBCAtoms(iZs, Rs, Qs, avec=avec, bvec=bvec)
    return iZs, Rs, Qs


def get_C612(i, j, FFparams):
    """
    compute Lennard-Jones coefitioens C6 and C12 pair of atoms i,j
    """
    Rij = FFparams[i][0] + FFparams[j][0]
    Eij = np.sqrt(FFparams[i][1] * FFparams[j][1])
    return 2 * Eij * (Rij**6), Eij * (Rij**12)


def getAtomsLJ(iZprobe, iZs, FFparams):
    """
    compute Lennard-Jones coefficients C6 and C12 for interaction between atoms in list "iZs" and probe-particle "iZprobe"
    """
    n = len(iZs)
    cLJs = np.zeros((n, 2))
    for i in range(n):
        cLJs[i, 0], cLJs[i, 1] = get_C612(iZprobe - 1, iZs[i] - 1, FFparams)
    return cLJs


def REA2LJ(cREAs, cLJs=None):
    if cLJs is None:
        cLJs = np.zeros((len(cREAs), 2))
    R6 = cREAs[:, 0] ** 6
    cLJs[:, 0] = 2 * cREAs[:, 1] * R6
    cLJs[:, 1] = cREAs[:, 1] * (R6**2)
    return cLJs


def getAtomsREA(iZprobe, iZs, FFparams, alphaFac=-1.0):
    """
    compute Lennard-Jones coefficients C6 and C12 for interaction between atoms in list "iZs" and probe-particle "iZprobe"
    """
    n = len(iZs)
    REAs = np.zeros((n, 4))
    i = iZprobe - 1
    for ii in range(n):
        j = iZs[ii] - 1
        REAs[ii, 0] = FFparams[i][0] + FFparams[j][0]
        REAs[ii, 1] = np.sqrt(FFparams[i][1] * FFparams[j][1])
        REAs[ii, 2] = FFparams[j][2] * alphaFac
    return REAs


def getSampleAtomsREA(iZs, FFparams):
    return np.array([(FFparams[i - 1][0], FFparams[i - 1][1], FFparams[i - 1][2]) for i in iZs])


def getAtomsRE(iZprobe, iZs, FFparams):
    n = len(iZs)
    Rpp = FFparams[iZprobe - 1][0]
    Epp = FFparams[iZprobe - 1][1]
    REs = np.array([(Rpp + FFparams[iZs[i] - 1][0], Epp * FFparams[iZs[i] - 1][1]) for i in range(n)])
    REs[:, 1] = np.sqrt(REs[:, 1])
    return REs


def getAtomsLJ_fast(iZprobe, iZs, FFparams):
    R = np.array([FFparams[i - 1][0] for i in iZs])
    E = np.array([FFparams[i - 1][1] for i in iZs])
    R += FFparams[iZprobe - 1][0]
    E = np.sqrt(E * FFparams[iZprobe - 1][1])
    cLJs = np.zeros((len(E), 2))
    cLJs[:, 0] = E * 2 * R6  # C6  = 2*Eij*(Rij**6 )
    cLJs[:, 1] = cLJs[:, 0] * 0.5 * R6  # C12 =   Eij*(Rij**12)
    return cLJs


# ============= Hi-Level Macros


def prepareScanGrids():
    """
    Defines the grid over which the tip will scan, according to scanMin, scanMax, and scanStep.
    The origin of the grid is going to be shifted (from scanMin) by the bond length between the "Probe Particle"
    and the "Apex", so that while the point of reference on the tip used to interpret scanMin  was the Apex,
    the new point of reference used in the XSF output will be the Probe Particle.
    """
    xTips = np.arange(params["scanMin"][0], params["scanMax"][0] + 0.5 * params["scanStep"][0], params["scanStep"][0])
    yTips = np.arange(params["scanMin"][1], params["scanMax"][1] + 0.5 * params["scanStep"][1], params["scanStep"][1])
    zTips = np.arange(params["scanMin"][2], params["scanMax"][2] + 0.5 * params["scanStep"][2], params["scanStep"][2])
    (xTips[0], xTips[-1], yTips[0], yTips[-1])
    lvecScan = np.array(
        [
            [
                (params["scanMin"] + params["r0Probe"])[0],
                (params["scanMin"] + params["r0Probe"])[1],
                (params["scanMin"] - params["r0Probe"])[2],
            ],
            [(params["scanMax"] - params["scanMin"])[0], 0.0, 0.0],
            [0.0, (params["scanMax"] - params["scanMin"])[1], 0.0],
            [0.0, 0.0, (params["scanMax"] - params["scanMin"])[2]],
        ]
    ).copy()
    return xTips, yTips, zTips, lvecScan


def lvec2params(lvec):
    params["gridA"] = lvec[1, :].copy()
    params["gridB"] = lvec[2, :].copy()
    params["gridC"] = lvec[3, :].copy()


def params2lvec():
    lvec = np.array(
        [
            [0.0, 0.0, 0.0],
            params["gridA"],
            params["gridB"],
            params["gridC"],
        ]
    ).copy
    return lvec


def genFFSampling(lvec, pixPerAngstrome=10):
    nDim = np.array(
        [
            int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[1], lvec[1])))),
            int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[2], lvec[2])))),
            int(round(pixPerAngstrome * np.sqrt(np.dot(lvec[3], lvec[3])))),
            4,
        ],
        np.int32,
    )
    return nDim


def getPos(lvec, nDim=None, pixPerAngstrome=10):
    if nDim is None:
        nDim = genFFSampling(lvec, pixPerAngstrome=pixPerAngstrome)
    dCell = np.array((lvec[1, :] / nDim[2], lvec[2, :] / nDim[1], lvec[3, :] / nDim[0]))
    ABC = np.mgrid[0 : nDim[0], 0 : nDim[1], 0 : nDim[2]]
    X = lvec[0, 0] + ABC[2] * dCell[0, 0] + ABC[1] * dCell[1, 0] + ABC[0] * dCell[2, 0]
    Y = lvec[0, 1] + ABC[2] * dCell[0, 1] + ABC[1] * dCell[1, 1] + ABC[0] * dCell[2, 1]
    Z = lvec[0, 2] + ABC[2] * dCell[0, 2] + ABC[1] * dCell[1, 2] + ABC[0] * dCell[2, 2]
    return X, Y, Z


def getPos_Vec3d(lvec, nDim=None, pixPerAngstrome=10):
    X, Y, Z = getPos(lvec, nDim=nDim, pixPerAngstrome=pixPerAngstrome)
    XYZ = np.empty(X.shape + (3,))
    XYZ[:, :, :, 0] = X
    XYZ[:, :, :, 1] = Y
    XYZ[:, :, :, 2] = Z
    return XYZ
