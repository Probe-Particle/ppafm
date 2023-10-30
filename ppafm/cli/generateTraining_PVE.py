#!/usr/bin/python -u

import os

import numpy as np

import ppafm.cpp_utils as cpp_utils
import ppafm.fieldFFT as fFFT
from ppafm import io

from .. import common, core
from ..HighLevel import prepareArrays, relaxedScan3D

file_format = "xsf"

# Arguments definition.
common.loadParams("params.ini")

if os.path.isfile("atomtypes.ini"):
    print(">> LOADING LOCAL atomtypes.ini")
    ff_params = common.loadSpecies("atomtypes.ini")
else:
    ff_params = common.loadSpecies(cpp_utils.PACKAGE_PATH / "defaults" / "atomtypes.ini")

elem_dict = common.getFFdict(ff_params)
pp_indexes = common.atom2iZ(common.params["probeType"], elem_dict)

# Load CO tip
drho_tip, lvec_dt, ndim_dt, atomic_info_or_head = io.load_scal_field("drho_tip", data_format=file_format)
rho_tip, lvec_t, ndim_t, atomic_info_or_head = io.load_scal_field("rho_tip", data_format=file_format)

common.params["gridN"] = ndim_t[::-1]
common.params["gridA"] = lvec_t[1]
common.params["gridB"] = lvec_t[2]
common.params["gridC"] = lvec_t[3]  # must be before parseAtoms
print(common.params["gridN"], common.params["gridA"], common.params["gridB"], common.params["gridC"])

force_field, _ = prepareArrays(None, False)

print("FFLJ.shape", force_field.shape)
core.setFF_shape(np.shape(force_field), lvec_t)

base_dir = os.getcwd()
paths = ["out1", "out2"]


for path in paths:
    os.chdir(path)

    # Load data.
    atoms, _, lvec = io.loadGeometry("V.xsf", params=common.params)

    # Generate vdW force field.
    izs, rs, _ = common.parseAtoms(atoms, elem_dict, autogeom=False, PBC=common.params["PBC"])

    force_field[:, :, :, :] = 0
    lj_coefficients = common.getAtomsLJ(pp_indexes, izs, ff_params)
    # print "cLJs",cLJs; np.savetxt("cLJs_3D.dat", cLJs);  exit()
    core.getVdWFF(rs, lj_coefficients)  # THE MAIN STUFF HERE

    # Generate Pauli force field.
    rho1, lvec1, ndim1, atomic_info_or_head = io.load_scal_field("rho", data_format=file_format)
    ff_x, ff_y, ff_z, _ = fFFT.potential2forces_mem(rho1, lvec1, rho1.shape, rho=rho_tip, doForce=True, doPot=False, deleteV=True)
    force_field[:, :, :, 0] = ff_x * common.params["Apauli"]
    force_field[:, :, :, 1] = ff_y * common.params["Apauli"]
    force_field[:, :, :, 2] = ff_z * common.params["Apauli"]

    # Generate Electrostatic force field.
    V_samp, lvec1, ndim1, atomic_info_or_head = io.load_scal_field("V", data_format=file_format)
    ff_x, ff_y, ff_z, _ = fFFT.potential2forces_mem(V_samp, lvec1, V_samp.shape, rho=drho_tip, doForce=True, doPot=False, deleteV=True)
    force_field[:, :, :, 0] = ff_x * common.params["charge"]
    force_field[:, :, :, 1] = ff_y * common.params["charge"]
    force_field[:, :, :, 2] = ff_z * common.params["charge"]

    # Relaxed scan.
    tip_positions_x, tip_positions_y, tip_positions_z, lvec_scan = common.prepareScanGrids()
    core.setTip(kSpring=np.array((common.params["klat"], common.params["klat"], 0.0)) / -common.eVA_Nm)
    fzs, PPpos = relaxedScan3D(tip_positions_x, tip_positions_y, tip_positions_z)

    io.save_scal_field("OutFz", fzs, lvec_scan, data_format=file_format, head=atomic_info_or_head, atomic_info=atomic_info_or_head)

    os.chdir(base_dir)
