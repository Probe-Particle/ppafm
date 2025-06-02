#!/usr/bin/python

import sys

import numpy as np

from . import common as PPU
from . import core, cpp_utils
from . import fieldFFT as fFFT
from . import io
from .defaults import d3

verbose = 1

# ===== constants
Fmax_DEFAULT = 10.0
Vmax_DEFAULT = 10.0

# ==== PP Relaxation


def Gauss(Evib, E0, w):
    return np.exp(-0.5 * ((Evib - E0) / w) ** 2)


def symGauss(Evib, E0, w):
    return Gauss(Evib, E0, w) - Gauss(Evib, -E0, w)


def meshgrid3d(xs, ys, zs):
    Xs, Ys, Zs = np.zeros()
    Xs, Ys = np.meshgrid(xs, ys)


def trjByDir(n, d, p0):
    trj = np.zeros((n, 3))
    trj[:, 0] = p0[0] + (np.arange(n)[::-1]) * d[0]
    trj[:, 1] = p0[1] + (np.arange(n)[::-1]) * d[1]
    trj[:, 2] = p0[2] + (np.arange(n)[::-1]) * d[2]
    return trj


def shift_positions(R, s):
    """
    Shifts positions in R by s; returns the result. Needed especially to shift atoms according to the grid origin.

    Arguments:
        R: list of positional vectors or a 2D array. The first index donoting the item (typically an atom)
            the second index determines the coordinate
        s: vector that determines the required shift

    Returns:
        Rs: Rs[ia,i] = R[ia,i] - s[i]
    """
    Rs = np.array(R).copy()
    Rs[:, 0] += s[0]
    Rs[:, 1] += s[1]
    Rs[:, 2] += s[2]
    return Rs


def relaxedScan3D(xTips, yTips, zTips, trj=None, bF3d=False):
    if verbose > 0:
        print(">>BEGIN: relaxedScan3D()")
    if verbose > 0:
        print(" zTips : ", zTips)
    ntips = len(zTips)
    rTips = np.zeros((ntips, 3))
    rs = np.zeros((ntips, 3))
    fs = np.zeros((ntips, 3))
    nx = len(xTips)
    ny = len(yTips)
    nz = len(zTips)
    if bF3d:
        fzs = np.zeros((nx, ny, nz, 3))
    else:
        fzs = np.zeros((nx, ny, nz))
    PPpos = np.zeros((nx, ny, nz, 3))
    if trj is None:
        trj = np.zeros((ntips, 3))
        trj[:, 2] = zTips[::-1]
    for ix, x in enumerate(xTips):
        sys.stdout.write("\033[K")
        sys.stdout.flush()
        sys.stdout.write(f"\rrelax ix: {ix}")
        sys.stdout.flush()
        for iy, y in enumerate(yTips):
            rTips[:, 0] = trj[:, 0] + x
            rTips[:, 1] = trj[:, 1] + y
            rTips[:, 2] = trj[:, 2]
            core.relaxTipStroke(rTips, rs, fs)
            if bF3d:
                fzs[ix, iy, :, 0] = fs[::-1, 0]
                fzs[ix, iy, :, 1] = fs[::-1, 1]
                fzs[ix, iy, :, 2] = fs[::-1, 2]
            else:
                fzs[ix, iy, :] = fs[::-1, 2]
            PPpos[ix, iy, :, 0] = rs[::-1, 0]
            PPpos[ix, iy, :, 1] = rs[::-1, 1]
            PPpos[ix, iy, :, 2] = rs[::-1, 2]
    if verbose > 0:
        print("<<<END: relaxedScan3D()")
    return fzs, PPpos


def relaxedScan3D_omp(xTips, yTips, zTips, trj=None, bF3d=False, tip_spline=None):
    if verbose > 0:
        print(">>BEGIN: relaxedScan3D_omp()")
    if verbose > 0:
        print(" zTips : ", zTips)
    nx = len(xTips)
    ny = len(yTips)
    nz = len(zTips)
    rTips = np.zeros((nx, ny, nz, 3))
    rs = np.zeros((nx, ny, nz, 3))
    fs = np.zeros((nx, ny, nz, 3))
    rTips[:, :, :, 0] = xTips[:, None, None]
    rTips[:, :, :, 1] = yTips[None, :, None]
    rTips[:, :, :, 2] = zTips[::-1][None, None, :]
    core.relaxTipStrokes_omp(rTips, rs, fs, tip_spline=tip_spline)
    rs = rs[:, :, ::-1, :].copy()
    if bF3d:
        fzs = fs[:, :, ::-1, :].copy()
    else:
        fzs = fs[:, :, ::-1, 2].copy()
    if verbose > 0:
        print("<<<END: relaxedScan3D_omp()")
    return fzs, rs


def perform_relaxation(
    lvec,
    FFLJ,
    FFel=None,
    FFpauli=None,
    FFboltz=None,
    FFkpfm_t0sV=None,
    FFkpfm_tVs0=None,
    tip_spline=None,
    bPPdisp=False,
    bFFtotDebug=False,
    parameters=None,
):
    global FF  # We need FF global otherwise it is garbage collected and program crashes inside C++ e.g. in stiffnessMatrix()
    if verbose > 0:
        print(">>>BEGIN: perform_relaxation()")
    xTips, yTips, zTips, lvecScan = PPU.prepareScanGrids(parameters=parameters)
    FF = FFLJ.copy()
    if FFel is not None:
        FF += FFel * parameters.charge
        if verbose > 0:
            print("adding charge:", parameters.charge)
    if FFkpfm_t0sV is not None and FFkpfm_tVs0 is not None:
        FF += (parameters.charge * FFkpfm_t0sV - FFkpfm_tVs0) * parameters.Vbias
        if verbose > 0:
            print("adding charge:", parameters.charge, "and bias:", parameters.Vbias, "V")
    if FFpauli is not None:
        FF += FFpauli * parameters.Apauli
    if FFboltz != None:
        FF += FFboltz
    if bFFtotDebug:
        io.save_vec_field("FFtotDebug", FF, lvec)
    setFF(FF, lvec=lvec, parameters=parameters)
    if (np.array(parameters.stiffness) < 0.0).any():
        parameters.stiffness = np.array([parameters.klat, parameters.klat, parameters.krad])
    if verbose > 0:
        print("stiffness:", parameters.stiffness)
    core.setTip(kSpring=np.array((parameters.stiffness[0], parameters.stiffness[1], 0.0)) / -PPU.eVA_Nm, kRadial=parameters.stiffness[2] / -PPU.eVA_Nm, parameters=parameters)

    # grid origin has to be moved to zero, hence the subtraction of lvec[0,:] from trj and xTip, yTips, zTips
    trj = None
    if parameters.tiltedScan:
        trj = trjByDir(len(zTips), d=parameters.scanTilt, p0=[0.0, 0.0, parameters.scanMin[2] - lvec[0, 2]])
    # fzs, PPpos = relaxedScan3D(xTips - lvec[0, 0], yTips - lvec[0, 1], zTips - lvec[0, 2], trj=trj, bF3d=parameters.tiltedScan)
    fzs, PPpos = relaxedScan3D_omp(xTips - lvec[0, 0], yTips - lvec[0, 1], zTips - lvec[0, 2], trj=trj, bF3d=parameters.tiltedScan, tip_spline=tip_spline)

    # transform probe-particle positions back to the original coordinates
    PPpos[:, :, :, 0] += lvec[0, 0]
    PPpos[:, :, :, 1] += lvec[0, 1]
    PPpos[:, :, :, 2] += lvec[0, 2]

    if bPPdisp:
        PPdisp = PPpos.copy()
        init_pos = np.array(np.meshgrid(xTips, yTips, zTips)) + np.array([parameters.r0Probe[0], parameters.r0Probe[1], -parameters.r0Probe[2]])
        PPdisp -= init_pos
    else:
        PPdisp = None
    if verbose > 0:
        print("<<<END: perform_relaxation()")

    core.deleteFF_Fpointer()

    return fzs, PPpos, PPdisp, lvecScan


# ==== Forcefield grid generation


def setFF(FF=None, computeVpot=False, n=None, lvec=None, parameters=None, verbose=True):

    # Find gridN
    if FF is not None:
        gridN = np.shape(FF)[0:3]
        if parameters is not None:
            parameters.gridN = gridN
    else:
        if n is not None:
            gridN = np.array(n, dtype=np.int32)
        elif parameters is not None:
            if parameters.gridN[2] <= 0:
                gridN = PPU.autoGridN(parameters)
            else:
                gridN = parameters.gridN
        else:
            raise ValueError("FF dimensions not set !!")
        # Create a new array for FF if needed
        FF = np.zeros((gridN[0], gridN[1], gridN[2], 3))
    if verbose:
        print("setFF() gridN: ", gridN)
    core.setGridN(gridN)

    # Set pointer to FF
    if len(FF.shape) == 4 and FF.shape[-1] == 3:
        if verbose:
            print("setFF() Creating a pointer to a vector field")
        core.setFF_Fpointer(FF)
    elif len(FF.shape) == 3 or FF.shape[-1] == 1:
        if verbose:
            print("setFF() Creating a pointer to a scalar field")
        core.setFF_Epointer(FF)
    else:
        raise ValueError("setFF: Array dimensions wrong for both vector and array field !!")

    # Create a scalar (potential) field if required
    if computeVpot:
        if verbose:
            print("setFF() Creating a pointer to a scalar field")
        V = np.zeros((gridN[0], gridN[1], gridN[2]))
        core.setFF_Epointer(V)
    else:
        V = None

    # Set lattice vectors or grid geometry
    if (lvec is None) and (parameters is not None):
        lvec = np.array(
            [
                parameters.gridA,
                parameters.gridB,
                parameters.gridC,
            ],
            dtype=np.float64,
        ).copy()
    lvec = np.array(lvec, dtype=np.float64)
    if lvec.shape == (3, 3):
        lvec = lvec.copy()
    elif lvec.shape == (4, 3):
        lvec = lvec[1:, :].copy()
    else:
        raise ValueError("lvec matrix has a wrong format !!")
    if verbose:
        print("setFF() lvec: ", lvec)
    core.setGridCell(lvec)

    return FF, V


def computeLJ(geomFile, speciesFile, geometry_format=None, save_format=None, computeVpot=False, Fmax=Fmax_DEFAULT, Vmax=Vmax_DEFAULT, ffModel="LJ", parameters=None):
    if verbose > 0:
        print(">>>BEGIN: computeLJ()")
    # --- load species (LJ potential)
    FFparams = PPU.loadSpecies(speciesFile)
    elem_dict = PPU.getFFdict(FFparams)
    # print elem_dict
    # --- load atomic geometry
    atoms, nDim, lvec = io.loadGeometry(geomFile, format=geometry_format, parameters=parameters)
    atomstring = io.primcoords2Xsf(PPU.atoms2iZs(atoms[0], elem_dict), [atoms[1], atoms[2], atoms[3]], lvec)
    if verbose > 0:
        print(parameters.gridN, parameters.gridO, parameters.gridA, parameters.gridB, parameters.gridC)
    iZs, Rs, Qs = PPU.parseAtoms(atoms, elem_dict, autogeom=False, PBC=parameters.PBC, lvec=lvec, parameters=parameters)
    # --- prepare LJ parameters
    iPP = PPU.atom2iZ(parameters.probeType, elem_dict)
    # --- prepare arrays and compute
    FF, V = setFF(None, computeVpot, lvec=lvec, parameters=parameters)
    if verbose > 0:
        print("FFLJ.shape", FF.shape)

    # shift atoms to the coordinate system in which the grid origin is zero
    Rs0 = shift_positions(Rs, -lvec[0])

    if ffModel == "Morse":
        REs = PPU.getAtomsRE(iPP, iZs, FFparams)
        core.getMorseFF(Rs0, REs)  # THE MAIN STUFF HERE
    elif ffModel == "vdW":
        vdWDampKind = parameters.vdWDampKind
        if vdWDampKind == 0:
            cLJs = PPU.getAtomsLJ(iPP, iZs, FFparams)
            core.getVdWFF(Rs0, cLJs)  # THE MAIN STUFF HERE
        else:
            REs = PPU.getAtomsRE(iPP, iZs, FFparams)
            core.getVdWFF_RE(Rs0, REs, kind=vdWDampKind)  # THE MAIN STUFF HERE
    else:
        cLJs = PPU.getAtomsLJ(iPP, iZs, FFparams)
        core.getLennardJonesFF(Rs0, cLJs)  # THE MAIN STUFF HERE
    # --- post porces FFs
    if Fmax is not None:
        if verbose > 0:
            print("Clamp force >", Fmax)
        io.limit_vec_field(FF, Fmax=Fmax)
    if (Vmax is not None) and computeVpot:
        if verbose > 0:
            print("Clamp potential >", Vmax)
        V[V > Vmax] = Vmax  # remove too large values
    # --- save to files ?
    if save_format is not None:
        if verbose > 0:
            print("computeLJ Save ", save_format)
        io.save_vec_field("FF" + ffModel, FF, lvec, data_format=save_format, head=atomstring, atomic_info=(atoms[:4], lvec))
        if computeVpot:
            io.save_scal_field("E" + ffModel, V, lvec, data_format=save_format, head=atomstring, atomic_info=(atoms[:4], lvec))
    if verbose > 0:
        print("<<<END: computeLJ()")
    return FF, V, nDim, lvec


def computeDFTD3(input_file, df_params="PBE", geometry_format=None, save_format=None, compute_energy=False, parameters=None):
    """
    Compute the Grimme DFT-D3 force field and optionally save to a file. See also :meth:`.add_dftd3`.

    Arguments:
        input_file: str. Path to input file. Supported formats are .xyz, .xsf, and .cube.
        save_format: str or None. If not None, then the generated force field is saved to files FFvdW_{x,y,z} in format
            that can be either 'xsf' or 'npy'.
        compute_energy: bool. In addition to force, also compute the energy. The energy is saved to file Evdw if save format
            is not None.
        df_params: str or dict. Functional-specific scaling parameters. Can be a str with the
            functional name or a dict with manually specified parameters.

    Returns:
        FF: np.ndarray of shape (nx, ny, nz, 3). Force field.
        V: np.ndarray of shape (nx, ny, nz) or None. Energy, if compute_energy == True.
        lvec: np.ndarray of shape (4, 3). Origin and lattice vectors of the force field.
    """

    # Load atomic geometry
    atoms, nDim, lvec = io.loadGeometry(input_file, format=geometry_format, parameters=parameters)
    elem_dict = PPU.getFFdict(PPU.loadSpecies())
    iZs, Rs, _ = PPU.parseAtoms(atoms, elem_dict, autogeom=False, PBC=parameters.PBC, lvec=lvec, parameters=parameters)
    iPP = PPU.atom2iZ(parameters.probeType, elem_dict)

    # Compute coefficients for each atom
    df_params = d3.get_df_params(df_params)
    coeffs = core.computeD3Coeffs(Rs, iZs, iPP, df_params)

    # Compute the force field
    FF, V = setFF(None, compute_energy, lvec=lvec, parameters=parameters)
    core.getDFTD3FF(shift_positions(Rs, -lvec[0]), coeffs)

    # Save to file
    if save_format is not None:
        atom_string = io.primcoords2Xsf(PPU.atoms2iZs(atoms[0], elem_dict), atoms[1:4], lvec)
        io.save_vec_field("FFvdW", FF, lvec, data_format=save_format, head=atom_string, atomic_info=(atoms[:4], lvec))
        if compute_energy:
            io.save_scal_field("EvdW", V, lvec, data_format=save_format, head=atom_string, atomic_info=(atoms[:4], lvec))

    return FF, V, lvec


def computeELFF_pointCharge(geomFile, geometry_format=None, tip="s", save_format=None, computeVpot=False, Fmax=Fmax_DEFAULT, Vmax=Vmax_DEFAULT, parameters=None):
    if verbose > 0:
        print(">>>BEGIN: computeELFF_pointCharge()")
    tipKinds = {"s": 0, "pz": 1, "dz2": 2}
    tipKind = tipKinds[tip]
    if verbose > 0:
        print(" ========= get electrostatic forcefiled from the point charges tip=%s %i " % (tip, tipKind))
    # --- load atomic geometry
    FFparams = PPU.loadSpecies()
    elem_dict = PPU.getFFdict(FFparams)
    # print elem_dict

    atoms, nDim, lvec = io.loadGeometry(geomFile, format=geometry_format, parameters=parameters)
    atomstring = io.primcoords2Xsf(PPU.atoms2iZs(atoms[0], elem_dict), [atoms[1], atoms[2], atoms[3]], lvec)
    # --- prepare arrays and compute
    if verbose > 0:
        print(parameters.gridN, parameters.gridA, parameters.gridB, parameters.gridC)
    _, Rs, Qs = PPU.parseAtoms(atoms, elem_dict=elem_dict, autogeom=False, PBC=parameters.PBC, lvec=lvec, parameters=parameters)
    FF, V = setFF(None, computeVpot, lvec=lvec, parameters=parameters)

    # shift atoms to the coordinate system in which the grid origin is zero
    Rs0 = shift_positions(Rs, -lvec[0])

    core.getCoulombFF(Rs0, Qs * PPU.CoulombConst, kind=tipKind)  # THE MAIN STUFF HERE
    # --- post porces FFs
    if Fmax is not None:
        if verbose > 0:
            print("Clamp force >", Fmax)
        io.limit_vec_field(FF, Fmax=Fmax)
    if (Vmax is not None) and computeVpot:
        if verbose > 0:
            print("Clamp potential >", Vmax)
        V[V > Vmax] = Vmax  # remove too large values
    # --- save to files ?
    if save_format is not None:
        if verbose > 0:
            print("computeLJ Save ", save_format)
        io.save_vec_field("FFel", FF, lvec, data_format=save_format, head=atomstring, atomic_info=(atoms[:4], lvec))
        if computeVpot:
            io.save_scal_field("Vel", V, lvec, data_format=save_format, head=atomstring, atomic_info=(atoms[:4], lvec))
    if verbose > 0:
        print("<<<END: computeELFF_pointCharge()")
    return FF, V, nDim, lvec


def computeElFF(V, lvec, nDim, tip, computeVpot=False, tilt=0.0, sigma=None, deleteV=True, parameters=None):
    rho = None
    multipole = None
    if sigma is None:
        sigma = parameters.sigma
    if isinstance(tip, (list, np.ndarray)):
        rho = tip
    elif isinstance(tip, dict):
        multipole = tip
    else:
        if tip in {"s", "px", "py", "pz", "dx2", "dy2", "dz2", "dxy", "dxz", "dyz"}:
            rho = None
            multipole = {tip: 1.0}
        elif tip.endswith(".xsf"):
            rho, lvec_tip, nDim_tip, tiphead = io.loadXSF(tip)
            if any(nDim_tip != nDim):
                sys.exit("Error: Input file for tip charge density has been specified, but the dimensions are incompatible with the Hartree potential file!")
            rho *= -1  # Negative charge density from positive electron density
    Fel_x, Fel_y, Fel_z, Vout = fFFT.potential2forces_mem(V, lvec, nDim, rho=rho, sigma=sigma, multipole=multipole, doPot=computeVpot, tilt=tilt, deleteV=deleteV)
    FFel = io.packVecGrid(Fel_x, Fel_y, Fel_z)
    del Fel_x, Fel_y, Fel_z
    return FFel, Vout


def loadValenceElectronDict():
    valElDict_ = None
    namespace = {}
    try:
        fname_valelec_dict = "valelec_dict.py"
        namespace = {}
        exec(open(fname_valelec_dict).read(), namespace)
        print("   : ", namespace["valElDict"])
        valElDict_ = namespace["valElDict"]
        print("Valence electrons loaded from local file : ", fname_valelec_dict)
    except:
        pass
    if valElDict_ is None:
        namespace = {}
        fname_valelec_dict = cpp_utils.PACKAGE_PATH / "defaults" / "valelec_dict.py"
        exec(open(fname_valelec_dict).read(), namespace)
        valElDict_ = namespace["valElDict"]
        print("Valence electrons loaded from default location : ", fname_valelec_dict)
    if verbose > 0:
        print(" Valence Electron Dict : \n", valElDict_)
    return valElDict_


def _getAtomsWhichTouchPBCcell(Rs, elems, nDim, lvec, Rcut, bSaveDebug, fname=None):
    inds, Rs_ = PPU.findPBCAtoms3D_cutoff(Rs, np.array(lvec[1:]), Rcut=Rcut)  # find periodic images of PBC images of atom of radius Rcut which touch our cell
    elems = [elems[i] for i in inds]  # atomic number of all relevant peridic images of atoms
    if bSaveDebug:
        io.saveGeomXSF(fname + "_TouchCell_debug.xsf", elems, Rs_, lvec[1:], convvec=lvec[1:], bTransposed=True)  # for debugging - mapping PBC images of atoms to the cell
    Rs_ = Rs_.transpose().copy()
    return Rs_, elems


def getAtomsWhichTouchPBCcell(fname, Rcut=1.0, bSaveDebug=True, geometry_format=None, parameters=None):
    atoms, nDim, lvec = io.loadGeometry(fname, format=geometry_format, parameters=parameters)
    Rs = np.array(atoms[1:4])  # get just positions x,y,z
    elems = np.array(atoms[0])
    Rs, elems = _getAtomsWhichTouchPBCcell(Rs, elems, nDim, lvec, Rcut, bSaveDebug, fname)
    return Rs, elems


def subtractCoreDensities(
    rho, lvec, elems=None, Rs=None, fname=None, valElDict=None, Rcore=0.7, bSaveDebugDens=False, bSaveDebugGeom=True, head=io.XSF_HEAD_DEFAULT, parameters=None
):
    nDim = rho.shape
    if fname is not None:
        elems, Rs = getAtomsWhichTouchPBCcell(fname, Rcut=Rcore, bSaveDebug=bSaveDebugDens)
    if valElDict is None:
        valElDict = loadValenceElectronDict()
    print("subtractCoreDensities valElDict ", valElDict)
    print("subtractCoreDensities elems ", elems)
    cRAs = np.array([(-valElDict[elem], Rcore) for elem in elems])
    V = np.linalg.det(lvec[1:])  # volume of triclinic cell
    N = nDim[0] * nDim[1] * nDim[2]
    dV = V / N  # volume of one voxel
    if verbose > 0:
        print("V : ", V, " N: ", N, " dV: ", dV)
    if verbose > 0:
        print("sum(RHO): ", rho.sum(), " Nelec: ", rho.sum() * dV, " voxel volume: ", dV)  # check sum

    # set grid sampling dimension and shape
    setFF(rho, lvec=lvec, parameters=parameters)
    if verbose > 0:
        print(">>> Projecting Core Densities ... ")

    core.getDensityR4spline(shift_positions(Rs, -lvec[0]), cRAs.copy())  # Do the job ( the Projection of atoms onto grid )
    if verbose > 0:
        print("sum(RHO), Nelec: ", rho.sum(), rho.sum() * dV)  # check sum
    if bSaveDebugDens:
        io.saveXSF("rho_subCoreChg.xsf", rho, lvec, head=head)
