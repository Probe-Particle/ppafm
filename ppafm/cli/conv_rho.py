#!/usr/bin/python
import numpy as np

from .. import common, fieldFFT, io


def handle_aeccar(fname, lvec, rho):
    if "AECCAR" in fname:
        rho /= np.abs(np.linalg.det(lvec[1:]))


def handle_negative_density(rho):
    q = rho.sum()
    rho[rho < 0] = 0
    rho *= q / rho.sum()


def main():
    parser = common.CLIParser(
        description="Calculate the density overlap integral for Pauli force field in the full-density based model. "
        "The integral has two parameters A and B, and is of form A*Integral( rho_tip^B * rho_sample^B )"
    )

    # fmt: off
    parser.add_argument('-s', '--sample', action='store', required=True, help='Path to sample charge density (.xsf).')
    parser.add_argument('-t', '--tip',    action='store', required=True, help='Path to tip charge density (.xsf).')
    parser.add_argument('-o', '--output', action='store', default='pauli', help='Name of output energy/force files.')
    parser.add_argument('--saveDebugXsfs', action='store_true', help='Save auxiliary xsf files for debugging.')
    parser.add_argument('--no_negative_check', action='store_true', help='Input density files may contain negative voxels. This is handled by default by setting negative values to zero and rescaling the density so that the total charge is conserved. Setting this option disables the check.' )
    parser.add_arguments(['output_format', 'energy', 'Apauli', 'Bpauli'])
    # fmt: on

    args = parser.parse_args()

    print(">>> Loading sample from ", args.sample, " ... ")
    rho_sample, lvec_sample, n_dim_sample, head_sample = io.loadXSF(args.sample)
    print(">>> Loading tip from ", args.tip, " ... ")
    rho_tip, lvec_tip, n_dim_tip, head_tip = io.loadXSF(args.tip)

    if np.any(n_dim_sample != n_dim_tip):
        raise Exception("Tip and Sample grids have different dimensions! - sample: " + str(n_dim_sample) + " tip: " + str(n_dim_tip))
    if np.any(lvec_sample != lvec_tip):
        raise Exception("Tip and Sample grids have different shapes! - sample: " + str(lvec_sample) + " tip: " + str(lvec_tip))

    handle_aeccar(args.sample, lvec_sample, rho_sample)
    handle_aeccar(args.tip, lvec_tip, rho_tip)

    if args.Bpauli > 0.0:
        print(">>> computing rho^B where B = ", args.Bpauli)
        # NOTE: due to round-off error the density from DFT code is often negative in some voxels which produce NaNs after exponentiation; we need to correct this
        if not args.no_negative_check:
            handle_negative_density(rho_sample)
            handle_negative_density(rho_tip)
        rho_sample[:, :, :] = rho_sample[:, :, :] ** args.Bpauli
        rho_tip[:, :, :] = rho_tip[:, :, :] ** args.Bpauli
        if args.saveDebugXsfs:
            io.save_scal_field("sample_density_pow_%03.3f.xsf" % args.Bpauli, rho_sample, lvec_sample, data_format=args.output_format, head=head_sample)
            io.save_scal_field("tip_density_pow_%03.3f.xsf" % args.Bpauli, rho_tip, lvec_tip, data_format=args.output_format, head=head_tip)

    print(">>> Evaluating convolution E(R) = A*Integral_r ( rho_tip^B(r-R) * rho_sample^B(r) ) using FFT ... ")
    f_x, f_y, f_z, energy = fieldFFT.potential2forces_mem(rho_sample, lvec_sample, n_dim_sample, rho=rho_tip, doForce=True, doPot=True, deleteV=True)

    namestr = args.output
    print(">>> Saving result of convolution to FF_", namestr, "_?.xsf ... ")

    # Density Overlap Model
    if args.energy:
        io.save_scal_field("E" + namestr, energy * args.Apauli, lvec_sample, data_format=args.output_format, head=head_sample)
    force_field = io.packVecGrid(f_x * args.Apauli, f_y * args.Apauli, f_z * args.Apauli)
    io.save_vec_field("FF" + namestr, force_field, lvec_sample, data_format=args.output_format, head=head_sample)


if __name__ == "__main__":
    main()
