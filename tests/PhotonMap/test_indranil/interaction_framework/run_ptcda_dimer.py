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
import json

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
    parser.add_argument('--mol1-dir', type=str, default=None)
    parser.add_argument('--mol2-dir', type=str, default=None)
    parser.add_argument('--diff-cube', type=str, default='density_difference_state1.cube')
    parser.add_argument('--gs-cube', type=str, default='ground_state_density.cube')
    parser.add_argument('--interp-kind', type=str, default='linear', choices=['linear', 'cubic'])
    parser.add_argument('--mol2-charge', type=float, default=0.0)
    parser.add_argument('--use-target-charge', type=float, default=None)
    parser.add_argument('--residual-mono-correction', dest='residual_mono_correction', action='store_true')
    parser.add_argument('--no-residual-mono-correction', dest='residual_mono_correction', action='store_false')
    parser.add_argument('--rescale-gs-to-target', dest='rescale_gs_to_target', action='store_true')
    parser.add_argument('--no-rescale-gs-to-target', dest='rescale_gs_to_target', action='store_false')
    parser.add_argument('--validate-potential', dest='validate_potential', action='store_true')
    parser.add_argument('--no-validate-potential', dest='validate_potential', action='store_false')
    parser.add_argument('--potential-cube', type=str, default=None)
    parser.add_argument('--calc-potential-cube', type=str, default=None)
    parser.add_argument('--calc-potential-extrapolate', type=str, default='error', choices=['error', 'multipole'])
    parser.add_argument('--exclude-nuc-radius-bohr', type=float, default=0.5)
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
    parser.add_argument('--anion-dir', type=str, default='/home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/M4_S0.2_optimised_structure_wb97x_d3bj_6_31g__gpu_charge-1')
    parser.add_argument('--run-4combos', action='store_true')
    parser.add_argument('--run-4combos-charge-mode', type=str, default='target', choices=['cube', 'target'])
    parser.add_argument('--out-prefix', type=str, default=os.path.join(script_dir, 'ptcda'))
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.add_argument('--enforce-dn-zero-integral', dest='enforce_dn_zero_integral', action='store_true')
    parser.add_argument('--no-enforce-dn-zero-integral', dest='enforce_dn_zero_integral', action='store_false')
    parser.set_defaults(enforce_dn_zero_integral=True)
    parser.set_defaults(rescale_gs_to_target=False)
    parser.set_defaults(residual_mono_correction=None)
    parser.set_defaults(validate_potential=False)
    parser.set_defaults(show=True)
    args = parser.parse_args()

    mol1_dir_default = args.cube_dir if args.mol1_dir is None else args.mol1_dir
    mol2_dir_default = args.cube_dir if args.mol2_dir is None else args.mol2_dir

    def resolve_charge_settings(target_charge):
        if target_charge is None:
            return {
                'mol2_charge': float(args.mol2_charge),
                'use_target_charge': False,
                'enable_residual_mono_correction': False,
            }
        if args.residual_mono_correction is None:
            enable_corr = True
        else:
            enable_corr = bool(args.residual_mono_correction)
        return {
            'mol2_charge': float(target_charge),
            'use_target_charge': True,
            'enable_residual_mono_correction': bool(enable_corr),
        }

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

    def resolve_potential_cube(mol2_dir):
        if args.potential_cube is None:
            pot_path = os.path.join(mol2_dir, 'electrostatic_potential.cube')
        else:
            pot_path = args.potential_cube
            if not os.path.isabs(pot_path):
                pot_path = os.path.join(mol2_dir, pot_path)
        if not os.path.isfile(pot_path):
            print(f"ERROR: potential cube not found: {pot_path}")
            sys.exit(1)
        return pot_path

    def resolve_calc_potential_cube(mol2_dir):
        if args.calc_potential_cube is None:
            return None
        pot_path = args.calc_potential_cube
        if not os.path.isabs(pot_path):
            pot_path = os.path.join(mol2_dir, pot_path)
        if not os.path.isfile(pot_path):
            print(f"ERROR: calc potential cube not found: {pot_path}")
            sys.exit(1)
        return pot_path

    # ======================== Setup Calculator ========================
    print("=" * 70)
    print("  PTCDA Homodimer — Static Site-Energy Shift Calculation")
    print("=" * 70)

    def run_case(tag, mol1_dir, mol2_dir, mol1_charge, mol2_charge, use_target_charge=False, enable_residual_mono_correction=False):
        scan_axis = args.axis
        diff_density_path, gs_density_path = resolve_paths(mol1_dir, mol2_dir, args.diff_cube, args.gs_cube)

        calc = SiteEnergyShiftCalculator(
            diff_density_path=diff_density_path,
            gs_density_path=gs_density_path,
            mol2_charge=mol2_charge,
            enforce_dn_zero_integral=args.enforce_dn_zero_integral,
            rescale_gs_to_target=args.rescale_gs_to_target,
            use_target_charge=bool(use_target_charge),
            enable_residual_mono_correction=bool(enable_residual_mono_correction),
            calc_potential_cube_path=resolve_calc_potential_cube(mol2_dir),
            calc_potential_extrapolate=str(args.calc_potential_extrapolate),
            interp_kind=args.interp_kind,
        )

        distances_ang = np.arange(args.start, args.stop, args.step)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[scan_axis.lower()]
        disps = np.array([
            calc.displacement_from_scan_value(x, axis=scan_axis, scan_kind=args.scan_kind)
            for x in distances_ang
        ], dtype=float)
        max_disp_axis_ang = float(np.max(np.abs(disps[:, axis_idx])) * BOHR_TO_ANG)
        calc.precompute(max_distance_ang=max_disp_axis_ang, scan_axis=scan_axis)

        q_measured = None if calc.mol2_charge_measured is None else float(calc.mol2_charge_measured)
        q_residual = None if calc.mol2_charge_residual is None else float(calc.mol2_charge_residual)
        q_used = None
        if getattr(calc, 'moments_net', None) is not None:
            q_used = float(calc.moments_net.get('monopole', np.nan))

        if args.validate_potential:
            pot_path = resolve_potential_cube(mol2_dir)
            potval = calc.validate_potential_cube(pot_path, exclude_nuc_radius_bohr=float(args.exclude_nuc_radius_bohr))
            potval_file = f"{args.out_prefix}_{tag}".replace(' ', '_') + "_potval.json"
            with open(potval_file, 'w') as f:
                json.dump(potval, f, indent=2, sort_keys=True)
            print(f"  Potential validation saved to: {potval_file}")

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
            f"# mol2_charge_input: {mol2_charge}\n"
            f"# use_target_charge: {bool(use_target_charge)}\n"
            f"# enable_residual_mono_correction: {bool(enable_residual_mono_correction)}\n"
            f"# mol2_charge_measured: {q_measured}\n"
            f"# mol2_charge_used: {q_used}\n"
            f"# mol2_charge_residual: {q_residual}\n"
            f"# interp_kind: {args.interp_kind}\n"
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
        def charge_for_case(q_target):
            if args.run_4combos_charge_mode == 'cube':
                return resolve_charge_settings(None)
            return resolve_charge_settings(float(q_target))

        charge = charge_for_case(0.0)
        run_case('mol1_neutral__mol2_neutral', args.neutral_dir, args.neutral_dir, 0.0, 0.0, use_target_charge=charge['use_target_charge'], enable_residual_mono_correction=charge['enable_residual_mono_correction'],)
        charge = charge_for_case(-1.0)
        run_case('mol1_anion__mol2_anion', args.anion_dir, args.anion_dir, -1.0, -1.0, use_target_charge=charge['use_target_charge'], enable_residual_mono_correction=charge['enable_residual_mono_correction'],)
        charge = charge_for_case(-1.0)
        run_case('mol1_neutral__mol2_anion', args.neutral_dir, args.anion_dir, 0.0, -1.0, use_target_charge=charge['use_target_charge'], enable_residual_mono_correction=charge['enable_residual_mono_correction'],)
        charge = charge_for_case(0.0)
        run_case('mol1_anion__mol2_neutral', args.anion_dir, args.neutral_dir, -1.0, 0.0, use_target_charge=charge['use_target_charge'], enable_residual_mono_correction=charge['enable_residual_mono_correction'],)
        return

    scan_axis = args.axis
    diff_density_path, gs_density_path = resolve_paths(mol1_dir_default, mol2_dir_default, args.diff_cube, args.gs_cube)

    charge = resolve_charge_settings(args.use_target_charge)

    calc = SiteEnergyShiftCalculator(
        diff_density_path=diff_density_path,
        gs_density_path=gs_density_path,
        mol2_charge=charge['mol2_charge'],
        enforce_dn_zero_integral=args.enforce_dn_zero_integral,
        rescale_gs_to_target=args.rescale_gs_to_target,
        use_target_charge=charge['use_target_charge'],
        enable_residual_mono_correction=charge['enable_residual_mono_correction'],
        calc_potential_cube_path=resolve_calc_potential_cube(mol2_dir_default),
        calc_potential_extrapolate=str(args.calc_potential_extrapolate),
        interp_kind=args.interp_kind,
    )

    distances_ang = np.arange(args.start, args.stop, args.step)

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[scan_axis.lower()]
    disps = np.array([
        calc.displacement_from_scan_value(x, axis=scan_axis, scan_kind=args.scan_kind)
        for x in distances_ang
    ], dtype=float)
    max_disp_axis_ang = float(np.max(np.abs(disps[:, axis_idx])) * BOHR_TO_ANG)
    calc.precompute(max_distance_ang=max_disp_axis_ang, scan_axis=scan_axis)

    q_measured = None if calc.mol2_charge_measured is None else float(calc.mol2_charge_measured)
    q_residual = None if calc.mol2_charge_residual is None else float(calc.mol2_charge_residual)
    q_used = None
    if getattr(calc, 'moments_net', None) is not None:
        q_used = float(calc.moments_net.get('monopole', np.nan))

    if args.validate_potential:
        pot_path = resolve_potential_cube(mol2_dir_default)
        potval = calc.validate_potential_cube(pot_path, exclude_nuc_radius_bohr=float(args.exclude_nuc_radius_bohr))
        potval_file = f"{args.out_prefix}_potval.json"
        with open(potval_file, 'w') as f:
            json.dump(potval, f, indent=2, sort_keys=True)
        print(f"  Potential validation saved to: {potval_file}")

    print("\n" + "=" * 70)
    print(f"  Distance Scan along {scan_axis.upper()} axis")
    print("=" * 70)
    results = calc.distance_scan(distances_ang, axis=scan_axis, scan_kind=args.scan_kind)

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
        f"# mol2_charge_input: {charge['mol2_charge']}\n"
        f"# use_target_charge: {charge['use_target_charge']}\n"
        f"# enable_residual_mono_correction: {charge['enable_residual_mono_correction']}\n"
        f"# mol2_charge_measured: {q_measured}\n"
        f"# mol2_charge_used: {q_used}\n"
        f"# mol2_charge_residual: {q_residual}\n"
        f"# interp_kind: {args.interp_kind}\n"
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


# ======================== Notes (Reproducible Run Commands) ========================
#
# The commands below reproduce the validation matrices used for the report.
# Update the directory paths if your cube locations differ.
#
# Definitions:
#   - cube mode: use cubes "as-is" (no target-charge correction)
#   - target mode: enable --use-target-charge <q>; by default residual correction is enabled
#   - target+noresid: --use-target-charge <q> together with --no-residual-mono-correction
#   - potential validation: --validate-potential [--potential-cube <path>] [--exclude-nuc-radius-bohr <r>]
#   - smoother interpolation: --interp-kind cubic
#
# 1) 4-combo matrix (neutral/anion combinations), cube-as-is (linear interpolation)
# python3 run_ptcda_dimer.py \
#   --run-4combos --run-4combos-charge-mode cube \
#   --start 12 --stop 32 --step 2 --axis y --scan-kind displacement \
#   --interp-kind linear --validate-potential --exclude-nuc-radius-bohr 0.5 \
#   --no-show --out-prefix report_cube_linear
#
# 2) 4-combo matrix, target-charge mode (linear interpolation)
# python3 run_ptcda_dimer.py \
#   --run-4combos --run-4combos-charge-mode target \
#   --start 12 --stop 32 --step 2 --axis y --scan-kind displacement \
#   --interp-kind linear --validate-potential --exclude-nuc-radius-bohr 0.5 \
#   --no-show --out-prefix report_target_linear
#
# 3) 4-combo matrix, target-charge but NO residual monopole correction (linear interpolation)
# python3 run_ptcda_dimer.py \
#   --run-4combos --run-4combos-charge-mode target --no-residual-mono-correction \
#   --start 12 --stop 32 --step 2 --axis y --scan-kind displacement \
#   --interp-kind linear --validate-potential --exclude-nuc-radius-bohr 0.5 \
#   --no-show --out-prefix report_target_linear_noresid
#
# 4) Same three matrices with smoother interpolation
# python3 run_ptcda_dimer.py --run-4combos --run-4combos-charge-mode cube   --interp-kind cubic --validate-potential --exclude-nuc-radius-bohr 0.5 --start 12 --stop 32 --step 2 --axis y --scan-kind displacement --no-show --out-prefix report_cube_cubic
# python3 run_ptcda_dimer.py --run-4combos --run-4combos-charge-mode target --interp-kind cubic --validate-potential --exclude-nuc-radius-bohr 0.5 --start 12 --stop 32 --step 2 --axis y --scan-kind displacement --no-show --out-prefix report_target_cubic
# python3 run_ptcda_dimer.py --run-4combos --run-4combos-charge-mode target --no-residual-mono-correction --interp-kind cubic --validate-potential --exclude-nuc-radius-bohr 0.5 --start 12 --stop 32 --step 2 --axis y --scan-kind displacement --no-show --out-prefix report_target_cubic_noresid
#
# 5) Single hetero run example: Mol1 from anion directory, Mol2 from neutral directory
# python3 run_ptcda_dimer.py \
#   --mol1-dir "$ANION_DIR" --mol2-dir "$NEUTRAL_DIR" \
#   --diff-cube density_difference_state1.cube --gs-cube ground_state_density.cube \
#   --start 12 --stop 32 --step 2 --axis y --scan-kind displacement \
#   --use-target-charge 0 --interp-kind linear \
#   --validate-potential --potential-cube electrostatic_potential.cube --exclude-nuc-radius-bohr 0.5 \
#   --bruteforce-at 12 --no-show --out-prefix single_anion_neutral


# python3 run_ptcda_dimer.py --run-4combos --run-4combos-charge-mode target  --calc-potential-cube electrostatic_potential.cube --calc-potential-extrapolate multipole --start 0.5 --stop 17 --step 1 --axis z --scan-kind displacement --interp-kind linear  --exclude-nuc-radius-bohr 0.5 --out-prefix /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/test_scan