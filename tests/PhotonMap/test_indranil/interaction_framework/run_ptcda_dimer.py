#!/usr/bin/env python3
"""
PTCDA Homodimer — Static Site-Energy Shift Validation
=====================================================

Computes Δω₁(R) for a PTCDA homodimer as a function of intermolecular
distance along the Y axis (short molecular axis), using:
  - density_difference_state1.cube  (Δn = n_ES - n_GS)
  - dens.cube  (ground-state electron density n_GS)

Both cube files are from the same PTCDA molecule on the same 80×80×80 grid.

Produces validation plots:
  1. Δω(R) in meV — exact FFT vs multipole
  2. log|Δω| vs log(R) — power-law exponents
  3. Nuclear vs electronic contributions
"""

import os
import sys
import numpy as np
import argparse

# Add parent paths so we can import the module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from site_energy_shift import (
    SiteEnergyShiftCalculator,
    plot_results,
    BOHR_TO_ANG,
    ANG_TO_BOHR,
    HARTREE_TO_EV,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube-dir', type=str, default='/home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge0/wb97x_d3bj')
    parser.add_argument('--diff-cube', type=str, default='density_difference_state1.cube')
    parser.add_argument('--gs-cube', type=str, default='ground_state_density.cube')
    parser.add_argument('--mol2-charge', type=float, default=0.0)
    parser.add_argument('--axis', type=str, default='y', choices=['x', 'y', 'z'])
    parser.add_argument('--scan-kind', type=str, default='displacement', choices=['displacement', 'center', 'gap', 'proj_gap'])
    parser.add_argument('--start', type=float, default=12.0)
    parser.add_argument('--stop', type=float, default=42.0)
    parser.add_argument('--step', type=float, default=2.0)
    parser.add_argument('--bruteforce-at', type=float, default=None)
    parser.add_argument('--bf-npts-dn', type=int, nargs=3, default=[18, 14, 10])
    parser.add_argument('--bf-npts-gs', type=int, nargs=3, default=[18, 18, 18])
    parser.add_argument('--bf-chunk', type=int, default=384)
    parser.add_argument('--bf-rho-cut', type=float, default=0.0)
    parser.add_argument('--bf-r-min', type=float, default=None)
    parser.add_argument('--neutral-dir', type=str, default='/home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge0/wb97x_d3bj')
    parser.add_argument('--anion-dir', type=str, default='/home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge-1/wb97x_d3bj')
    parser.add_argument('--run-4combos', action='store_true')
    parser.add_argument('--out-prefix', type=str, default=os.path.join(script_dir, 'ptcda'))
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.add_argument('--enforce-dn-zero-integral', dest='enforce_dn_zero_integral', action='store_true')
    parser.add_argument('--no-enforce-dn-zero-integral', dest='enforce_dn_zero_integral', action='store_false')
    parser.set_defaults(enforce_dn_zero_integral=True)
    parser.set_defaults(show=True)
    args = parser.parse_args()

    def resolve_paths(mol1_dir, mol2_dir, mol1_cube, mol2_cube):
        p1 = mol1_cube
        p2 = mol2_cube
        if not os.path.isabs(p1):
            p1 = os.path.join(mol1_dir, p1)
        if not os.path.isabs(p2):
            p2 = os.path.join(mol2_dir, p2)
        for p in [p1, p2]:
            if not os.path.isfile(p):
                print(f"ERROR: File not found: {p}")
                sys.exit(1)
        return p1, p2

    # ======================== Setup Calculator ========================
    print("=" * 70)
    print("  PTCDA Homodimer — Static Site-Energy Shift Calculation")
    print("=" * 70)

    def run_case(tag, mol1_dir, mol2_dir, mol1_charge, mol2_charge):
        scan_axis = args.axis
        diff_density_path, gs_density_path = resolve_paths(mol1_dir, mol2_dir, args.diff_cube, args.gs_cube)

        calc = SiteEnergyShiftCalculator(
            diff_density_path=diff_density_path,
            gs_density_path=gs_density_path,
            mol2_charge=mol2_charge,
            enforce_dn_zero_integral=args.enforce_dn_zero_integral,
        )

        distances_ang = np.arange(args.start, args.stop, args.step)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[scan_axis.lower()]
        disps = np.array([
            calc.displacement_from_scan_value(x, axis=scan_axis, scan_kind=args.scan_kind)
            for x in distances_ang
        ], dtype=float)
        max_disp_axis_ang = float(np.max(np.abs(disps[:, axis_idx])) * BOHR_TO_ANG)
        calc.precompute(max_distance_ang=max_disp_axis_ang, scan_axis=scan_axis)

        print("\n" + "=" * 70)
        print(f"  Case: {tag}")
        print("=" * 70)
        results = calc.distance_scan(distances_ang, axis=scan_axis, scan_kind=args.scan_kind)

        mask = (results['R_ang'] > 20.0) & (np.abs(results['shift_eV']) > 1e-15)
        if np.sum(mask) >= 3:
            log_R = np.log(results['R_ang'][mask])
            log_dw = np.log(np.abs(results['shift_eV'][mask]))
            coeffs = np.polyfit(log_R, log_dw, 1)
            exponent = -coeffs[0]
            print(f"\n  Power-law fit (R > 20 Å): |Δω| ~ R^(-{exponent:.2f})")

        if args.bruteforce_at is not None:
            d = calc.displacement_from_scan_value(args.bruteforce_at, axis=scan_axis, scan_kind=args.scan_kind)
            e_bf = calc.compute_shift_bruteforce(
                d,
                npts_dn=tuple(args.bf_npts_dn),
                npts_gs=tuple(args.bf_npts_gs),
                chunk=int(args.bf_chunk),
                rho_cut=float(args.bf_rho_cut),
                r_min=args.bf_r_min,
            )
            e_fft = calc.compute_shift(d)['shift_hartree']
            print(f"\n  Brute-force check at {args.scan_kind}={args.bruteforce_at:.2f} Å:  FFT={e_fft:+.6e} Ha  brute={e_bf:+.6e} Ha  diff={e_fft-e_bf:+.6e} Ha")

        save_prefix = f"{args.out_prefix}_{tag}".replace(' ', '_')
        plot_results(results, title=f"{tag} — Δω₁ scan", save_prefix=save_prefix, show=args.show)

        data_file = f"{save_prefix}_data.txt"
        header = (
            f"# {tag} Static Site-Energy Shift\n"
            f"# Scan axis: {scan_axis}\n"
            f"# Scan kind: {args.scan_kind}\n"
            f"# mol1_charge: {mol1_charge}\n"
            f"# mol2_charge: {mol2_charge}\n"
            f"# diff_density: {diff_density_path}\n"
            f"# gs_density: {gs_density_path}\n"
            "# Columns: scan_ang  R_ang  gap_ang  proj_gap_ang  min_atom_dist_ang  shift_eV  shift_hartree  multipole_eV\n"
        )
        data = np.column_stack([
            results['scan_ang'],
            results['R_ang'],
            results['gap_ang'],
            results['proj_gap_ang'],
            results['min_atom_dist_ang'],
            results['shift_eV'],
            results['shift_hartree'],
            results['multipole_eV'],
        ])
        np.savetxt(data_file, data, header=header, fmt='%16.8e')
        print(f"\n  Data saved to: {data_file}")

        import gc
        del calc
        gc.collect()

        return results

    if args.run_4combos:
        run_case('mol1_neutral__mol2_neutral', args.neutral_dir, args.neutral_dir, 0.0, 0.0)
        run_case('mol1_anion__mol2_anion', args.anion_dir, args.anion_dir, -1.0, -1.0)
        run_case('mol1_neutral__mol2_anion', args.neutral_dir, args.anion_dir, 0.0, -1.0)
        run_case('mol1_anion__mol2_neutral', args.anion_dir, args.neutral_dir, -1.0, 0.0)
        return

    scan_axis = args.axis
    diff_density_path, gs_density_path = resolve_paths(args.cube_dir, args.cube_dir, args.diff_cube, args.gs_cube)

    calc = SiteEnergyShiftCalculator(
        diff_density_path=diff_density_path,
        gs_density_path=gs_density_path,
        mol2_charge=args.mol2_charge,
        enforce_dn_zero_integral=args.enforce_dn_zero_integral,
    )

    distances_ang = np.arange(args.start, args.stop, args.step)

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[scan_axis.lower()]
    disps = np.array([
        calc.displacement_from_scan_value(x, axis=scan_axis, scan_kind=args.scan_kind)
        for x in distances_ang
    ], dtype=float)
    max_disp_axis_ang = float(np.max(np.abs(disps[:, axis_idx])) * BOHR_TO_ANG)
    calc.precompute(max_distance_ang=max_disp_axis_ang, scan_axis=scan_axis)

    print("\n" + "=" * 70)
    print(f"  Distance Scan along {scan_axis.upper()} axis")
    print("=" * 70)
    results = calc.distance_scan(distances_ang, axis=scan_axis, scan_kind=args.scan_kind)

    # ======================== Summary Table ========================
    print("\n" + "=" * 70)
    print("  Summary Table")
    print("=" * 70)
    print(f"{'scan (Å)':>10s}  {'R (Å)':>10s}  {'gap (Å)':>10s}  {'proj_gap (Å)':>12s}  {'d_min (Å)':>10s}  {'Δω (meV)':>12s}  {'Δω_mp (meV)':>12s}")
    print("-" * 80)
    for i in range(len(results['R_ang'])):
        x = results['scan_ang'][i]
        R = results['R_ang'][i]
        gap = results['gap_ang'][i]
        pgap = results.get('proj_gap_ang', results['gap_ang'])[i]
        dmin = results['min_atom_dist_ang'][i]
        dw = results['shift_eV'][i] * 1000
        dw_mp = results['multipole_eV'][i] * 1000
        print(f"{x:10.2f}  {R:10.2f}  {gap:10.2f}  {pgap:12.2f}  {dmin:10.2f}  {dw:+12.6f}  {dw_mp:+12.6f}")

    # ======================== Power-Law Fit ========================
    # Fit log|Δω| = a - n*log(R) at large R to extract decay exponent
    mask = (results['R_ang'] > 20.0) & (np.abs(results['shift_eV']) > 1e-15)
    if np.sum(mask) >= 3:
        log_R = np.log(results['R_ang'][mask])
        log_dw = np.log(np.abs(results['shift_eV'][mask]))
        coeffs = np.polyfit(log_R, log_dw, 1)
        exponent = -coeffs[0]
        print(f"\n  Power-law fit (R > 20 Å): |Δω| ~ R^(-{exponent:.2f})")
        print(f"  Expected for quadrupole-quadrupole: R^(-5)")
        print(f"  Expected for dipole-quadrupole: R^(-4)")
        print(f"  Expected for dipole-dipole: R^(-3)")
    else:
        print("\n  Not enough data points for power-law fit at R > 20 Å")

    # ======================== Plots ========================
    save_prefix = args.out_prefix
    plot_results(results, title="PTCDA Homodimer — Δω₁(R)", save_prefix=save_prefix, show=args.show)

    # ======================== Save numerical data ========================
    data_file = f"{save_prefix}_data.txt"
    header = (
        "# PTCDA Homodimer Static Site-Energy Shift\n"
        f"# Scan axis: {scan_axis}\n"
        f"# Scan kind: {args.scan_kind}\n"
        f"# mol2_charge: {args.mol2_charge}\n"
        f"# diff_density: {diff_density_path}\n"
        f"# gs_density: {gs_density_path}\n"
        "# Columns: scan_ang  R_ang  gap_ang  proj_gap_ang  min_atom_dist_ang  shift_eV  shift_hartree  multipole_eV\n"
    )
    data = np.column_stack([
        results['scan_ang'],
        results['R_ang'],
        results['gap_ang'],
        results['proj_gap_ang'],
        results['min_atom_dist_ang'],
        results['shift_eV'],
        results['shift_hartree'],
        results['multipole_eV'],
    ])
    np.savetxt(data_file, data, header=header, fmt='%16.8e')
    print(f"\n  Data saved to: {data_file}")


if __name__ == '__main__':
    main()
