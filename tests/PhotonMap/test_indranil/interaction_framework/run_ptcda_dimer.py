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
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from site_energy_shift import (
    SiteEnergyShiftCalculator,
    plot_results,
    BOHR_TO_ANG,
    ANG_TO_BOHR,
    HARTREE_TO_EV,
)


class SimpleSystem:
    def __init__(self, poss_ang, rots_rad, Ediags_eV):
        self.poss = np.asarray(poss_ang, dtype=float)
        self.rots = np.asarray(rots_rad, dtype=float)
        self.Ediags = np.asarray(Ediags_eV, dtype=float)


def load_molecules_ini(fname):
    data = np.genfromtxt(fname, skip_header=1)
    if len(data.shape) == 1:
        data = np.reshape(data, (1, data.shape[0]))
    poss = np.asarray(data[:, :3], dtype=float)
    rots = np.asarray(data[:, 3], dtype=float) * np.pi / 180.0
    Ediags = np.asarray(data[:, 6], dtype=float)
    return poss, rots, Ediags


def make_system_for_siteshift_scan(mol_ids, mol1_pos_ang, mol2_pos_ang, mol1_rot_rad, mol2_rot_rad):
    mol_ids = [int(x) for x in mol_ids]
    poss = np.zeros((len(mol_ids), 3), dtype=float)
    rots = np.zeros(len(mol_ids), dtype=float)
    for i, mid in enumerate(mol_ids):
        if int(mid) == 1:
            poss[i] = np.asarray(mol1_pos_ang, dtype=float)
            rots[i] = float(mol1_rot_rad)
        elif int(mid) == 2:
            poss[i] = np.asarray(mol2_pos_ang, dtype=float)
            rots[i] = float(mol2_rot_rad)
        else:
            raise ValueError(f"Unsupported mol_id={mid} for scan builder. Expected mol_id 1 or 2.")
    Ediags = np.zeros(len(mol_ids), dtype=float)
    return SimpleSystem(poss, rots, Ediags)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_workflow = parser.add_argument_group('Workflow selection')
    g_workflow.add_argument(
        '--workflow',
        type=str,
        default=None,
        choices=['fft-scan', 'fft-4combos', 'hybrid-eval', 'hybrid-scan'],
        help='Select which standalone workflow to run. If omitted, the script uses legacy behavior: --run-4combos triggers fft-4combos, otherwise fft-scan.',
    )

    g_paths = parser.add_argument_group('Cube paths (FFT / potential-cube workflows)')
    g_paths.add_argument('--cube-dir', type=str, default='/home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge0/wb97x_d3bj', help='Default directory for both molecules if --mol1-dir/--mol2-dir are not set')
    g_paths.add_argument('--mol1-dir', type=str, default=None, help='Directory containing molecule-1 cubes (Δn cube)')
    g_paths.add_argument('--mol2-dir', type=str, default=None, help='Directory containing molecule-2 cubes (n_GS cube and optional potential cube)')
    g_paths.add_argument('--diff-cube', type=str, default='density_difference_state1.cube', help='Δn cube filename for molecule 1 (relative to --mol1-dir unless absolute)')
    g_paths.add_argument('--gs-cube', type=str, default='ground_state_density.cube', help='Ground-state density cube filename for molecule 2 (relative to --mol2-dir unless absolute)')

    g_fft = parser.add_argument_group('FFT / potential-cube settings (SiteEnergyShiftCalculator)')
    g_fft.add_argument('--interp-kind', type=str, default='linear', choices=['linear', 'cubic'], help='Interpolation for potential cube sampling (when --calc-potential-cube is used)')
    g_fft.add_argument('--mol2-charge', type=float, default=0.0, help='Total charge of molecule 2 (used when NOT using --use-target-charge). Units: electron charge')
    g_fft.add_argument('--use-target-charge', type=float, default=None, help='If set, constrain fitted monopole to this target net charge (electron charge)')
    g_fft.add_argument('--residual-mono-correction', dest='residual_mono_correction', action='store_true', help='Enable residual monopole correction in target-charge mode')
    g_fft.add_argument('--no-residual-mono-correction', dest='residual_mono_correction', action='store_false', help='Disable residual monopole correction in target-charge mode')
    g_fft.add_argument('--rescale-gs-to-target', dest='rescale_gs_to_target', action='store_true', help='Rescale molecule-2 ground-state density to match target charge before potential evaluation')
    g_fft.add_argument('--no-rescale-gs-to-target', dest='rescale_gs_to_target', action='store_false', help='Do not rescale ground-state density to match target charge')
    g_fft.add_argument('--calc-potential-cube', type=str, default=None, help='If set, use this external TOTAL electrostatic potential cube instead of FFT Poisson potential (path relative to --mol2-dir unless absolute)')
    g_fft.add_argument('--calc-potential-extrapolate', type=str, default='error', choices=['error', 'multipole'], help='Behavior when sampling outside the external potential cube bounds')
    g_fft.add_argument('--enforce-dn-zero-integral', dest='enforce_dn_zero_integral', action='store_true', help='Enforce ∫Δn dV = 0 by lobe scaling before integration (recommended)')
    g_fft.add_argument('--no-enforce-dn-zero-integral', dest='enforce_dn_zero_integral', action='store_false', help='Do not enforce ∫Δn dV = 0')

    g_validate = parser.add_argument_group('Potential validation diagnostic (optional)')
    g_validate.add_argument('--validate-potential', dest='validate_potential', action='store_true', help='Validate potential cube against FFT-derived reference')
    g_validate.add_argument('--no-validate-potential', dest='validate_potential', action='store_false', help='Disable potential validation')
    g_validate.add_argument('--potential-cube', type=str, default=None, help='Potential cube to validate (defaults to electrostatic_potential.cube in --mol2-dir)')
    g_validate.add_argument('--exclude-nuc-radius-bohr', type=float, default=0.5, help='Exclude points within this radius of nuclei during validation (Bohr)')

    g_scan = parser.add_argument_group('Distance scan settings')
    g_scan.add_argument('--axis', type=str, default='y', choices=['x', 'y', 'z'], help='Scan axis for the dimer displacement')
    g_scan.add_argument('--scan-kind', type=str, default='displacement', choices=['displacement', 'center', 'gap', 'proj_gap'], help='Meaning of scan coordinate: displacement, center distance, surface gap, or projected gap')
    g_scan.add_argument('--start', type=float, default=12.0, help='Start scan value (Å)')
    g_scan.add_argument('--stop', type=float, default=42.0, help='Stop scan value (Å)')
    g_scan.add_argument('--step', type=float, default=2.0, help='Scan step (Å)')

    g_bf = parser.add_argument_group('Brute-force check (optional)')
    g_bf.add_argument('--bruteforce-at', type=float, default=None, help='If set, run a brute-force integral check at this scan coordinate (Å)')
    g_bf.add_argument('--bf-npts-dn', type=int, nargs=3, default=[18, 14, 10], help='Brute-force sampling grid for Δn integration (nx ny nz)')
    g_bf.add_argument('--bf-npts-gs', type=int, nargs=3, default=[18, 18, 18], help='Brute-force sampling grid for n_GS integration (nx ny nz)')
    g_bf.add_argument('--bf-chunk', type=int, default=384, help='Chunk size for brute-force evaluation')
    g_bf.add_argument('--bf-rho-cut', type=float, default=0.0, help='Brute-force density cutoff (skip |ρ| below this threshold)')
    g_bf.add_argument('--bf-r-min', type=float, default=None, help='Brute-force minimum r (Bohr) to regularize 1/r singularities')

    g_reg = parser.add_argument_group('Regression driver (4-combo matrix)')
    g_reg.add_argument('--neutral-dir', type=str, default='/home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge0/wb97x_d3bj', help='Directory for neutral monomer cube set (used by --run-4combos)')
    g_reg.add_argument('--anion-dir', type=str, default='/home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/M4_S0.2_optimised_structure_wb97x_d3bj_6_31g__gpu_charge-1', help='Directory for anion monomer cube set (used by --run-4combos)')
    g_reg.add_argument('--run-4combos', action='store_true', help='Run the 4-combo matrix: neutral/anion combinations (legacy)')
    g_reg.add_argument('--run-4combos-charge-mode', type=str, default='target', choices=['cube', 'target'], help='Charge handling for --run-4combos: cube (as-is) or target (constrain monopole)')

    g_hybrid = parser.add_argument_group('Hybrid site-shift settings (ini-driven cube+RESP)')
    g_hybrid.add_argument('--molecules-ini', type=str, default=None, help='molecules.ini describing aggregate geometry (required for hybrid-eval)')
    g_hybrid.add_argument('--siteshift-cubes', type=str, default=None, help='siteshift_cubes.ini mapping each molecules.ini row to mol_id and Δn cube (and n_GS cube placeholder)')
    g_hybrid.add_argument('--site-shifts', type=str, default=None, help='electrostatics.ini specifying per-molecule potential_cube and resp_charges for hybrid evaluation')
    g_hybrid.add_argument('--siteshift-interp', type=str, default='cubic', choices=['linear', 'cubic'], help='Interpolation kind for potential cube in hybrid evaluator')
    g_hybrid.add_argument('--siteshift-chunk', type=int, default=100000, help='Chunk size for evaluating hybrid potential over many grid points')
    g_hybrid.add_argument('--siteshift-json', type=str, default=None, help='Optional JSON output file (hybrid)')
    g_hybrid.add_argument('--siteshift-terms', type=str, default=None, help='site_shift_terms.ini enabling extra physics terms (exchange, polarization, ct) beyond Coulomb')
    g_hybrid.add_argument('--hybrid-site-index', type=int, default=0, help='For hybrid-scan: which siteshift row index to track/plot as Δω(scan)')
    g_hybrid.add_argument('--hybrid-mol1-rot-deg', type=float, default=0.0, help='For hybrid-scan: mol1 rotation about Z (degrees)')
    g_hybrid.add_argument('--hybrid-mol2-rot-deg', type=float, default=0.0, help='For hybrid-scan: mol2 rotation about Z (degrees)')

    g_out = parser.add_argument_group('Output')
    g_out.add_argument('--out-prefix', type=str, default=os.path.join(script_dir, 'ptcda'), help='Output prefix for plots/data files')
    g_out.add_argument('--show', dest='show', action='store_true', help='Show interactive plots')
    g_out.add_argument('--no-show', dest='show', action='store_false', help='Do not show plots (save only)')
    parser.set_defaults(enforce_dn_zero_integral=True)
    parser.set_defaults(rescale_gs_to_target=False)
    parser.set_defaults(residual_mono_correction=None)
    parser.set_defaults(validate_potential=False)
    parser.set_defaults(show=True)
    args = parser.parse_args()

    if args.workflow is None:
        if bool(args.run_4combos):
            workflow = 'fft-4combos'
        else:
            workflow = 'fft-scan'
    else:
        workflow = str(args.workflow)

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

    def run_hybrid_eval():
        if args.molecules_ini is None:
            raise ValueError("hybrid-eval requires --molecules-ini")
        if args.siteshift_cubes is None:
            raise ValueError("hybrid-eval requires --siteshift-cubes")
        if args.site_shifts is None:
            raise ValueError("hybrid-eval requires --site-shifts")

        import pyProbeParticle.site_shifts as site_shifts

        mol_ini = args.molecules_ini
        if not os.path.isabs(mol_ini):
            mol_ini = os.path.join(script_dir, mol_ini)
        if not os.path.isfile(mol_ini):
            raise ValueError(f"molecules.ini not found: {mol_ini}")
        poss, rots, Ediags = load_molecules_ini(mol_ini)
        system = SimpleSystem(poss, rots, Ediags.copy())

        base_dir = script_dir
        terms_ini = args.siteshift_terms
        if terms_ini is not None:
            if not os.path.isabs(terms_ini):
                terms_ini = os.path.join(script_dir, terms_ini)
            if not os.path.isfile(terms_ini):
                raise ValueError(f"--siteshift-terms file not found: {terms_ini}")

        if terms_ini is not None:
            result = site_shifts.compute_site_shift_terms(
                system,
                args.siteshift_cubes,
                args.site_shifts,
                terms_ini_path=terms_ini,
                base_dir=base_dir,
                interp_kind=str(args.siteshift_interp),
                chunk=int(args.siteshift_chunk),
                save_json_path=args.siteshift_json,
            )
            dw = result['total_eV']
            print("=" * 70)
            print("  Multi-term site-energy shifts from ini-driven framework")
            print("=" * 70)
            print(f"Rows: {len(dw)}")
            for key in ['coulomb_eV', 'exchange_eV', 'polarization_eV', 'ct_eV', 'total_eV']:
                if key in result:
                    print(f"  {key}:")
                    for i, val in enumerate(result[key]):
                        print(f"    i={i:3d}  {val:+.6e} eV")
        else:
            dw = site_shifts.compute_site_shifts_for_system(
                system,
                args.siteshift_cubes,
                args.site_shifts,
                base_dir=base_dir,
                interp_kind=str(args.siteshift_interp),
                chunk=int(args.siteshift_chunk),
                save_json_path=args.siteshift_json,
            )
            print("=" * 70)
            print("  Hybrid site-energy shifts (Δω) from ini-driven cube+RESP")
            print("=" * 70)
            print(f"Rows: {len(dw)}")
            for i, val in enumerate(dw):
                print(f"  i={i:3d}  Δω={val:+.6e} eV")

        data_file = f"{args.out_prefix}_hybrid_eval_data.txt"
        header = "# Columns: i  delta_omega_eV\n"
        data = np.column_stack([np.arange(len(dw), dtype=int), np.asarray(dw, dtype=float)])
        np.savetxt(data_file, data, header=header, fmt=['%6d', '%16.8e'])
        print(f"\n  Data saved to: {data_file}")

    def run_hybrid_scan():
        if args.siteshift_cubes is None:
            raise ValueError("hybrid-scan requires --siteshift-cubes")
        if args.site_shifts is None:
            raise ValueError("hybrid-scan requires --site-shifts")

        import pyProbeParticle.site_shifts as site_shifts

        mol1_dir = mol1_dir_default
        mol2_dir = mol2_dir_default
        diff_density_path, gs_density_path = resolve_paths(mol1_dir, mol2_dir, args.diff_cube, args.gs_cube)

        calc = SiteEnergyShiftCalculator(
            diff_density_path=diff_density_path,
            gs_density_path=gs_density_path,
            mol2_charge=float(args.mol2_charge),
            enforce_dn_zero_integral=bool(args.enforce_dn_zero_integral),
            rescale_gs_to_target=bool(args.rescale_gs_to_target),
            use_target_charge=bool(False),
            enable_residual_mono_correction=bool(False),
            calc_potential_cube_path=resolve_calc_potential_cube(mol2_dir),
            calc_potential_extrapolate=str(args.calc_potential_extrapolate),
            interp_kind=args.interp_kind,
        )

        mol_ids, _, _, _ = site_shifts.load_siteshift_cubes(args.siteshift_cubes, base_dir=script_dir)

        distances_ang = np.arange(args.start, args.stop, args.step)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[str(args.axis).lower()]
        disps = np.array([
            calc.displacement_from_scan_value(x, axis=args.axis, scan_kind=args.scan_kind)
            for x in distances_ang
        ], dtype=float)
        max_disp_axis_ang = float(np.max(np.abs(disps[:, axis_idx])) * BOHR_TO_ANG)
        calc.precompute(max_distance_ang=max_disp_axis_ang, scan_axis=str(args.axis))

        shifts_track = []
        results = {
            'scan_ang': distances_ang,
            'R_ang': np.zeros(len(distances_ang), dtype=float),
            'gap_ang': np.zeros(len(distances_ang), dtype=float),
            'proj_gap_ang': np.zeros(len(distances_ang), dtype=float),
            'min_atom_dist_ang': np.zeros(len(distances_ang), dtype=float),
            'shift_eV': np.zeros(len(distances_ang), dtype=float),
            'shift_hartree': np.zeros(len(distances_ang), dtype=float),
            'multipole_eV': np.zeros(len(distances_ang), dtype=float),
        }

        mol1_rot = float(args.hybrid_mol1_rot_deg) * np.pi / 180.0
        mol2_rot = float(args.hybrid_mol2_rot_deg) * np.pi / 180.0

        for i, x in enumerate(distances_ang):
            d_bohr = calc.displacement_from_scan_value(float(x), axis=args.axis, scan_kind=args.scan_kind)
            d_ang = np.asarray(d_bohr, dtype=float) * BOHR_TO_ANG

            system = make_system_for_siteshift_scan(
                mol_ids,
                mol1_pos_ang=[0.0, 0.0, 0.0],
                mol2_pos_ang=d_ang,
                mol1_rot_rad=mol1_rot,
                mol2_rot_rad=mol2_rot,
            )

            terms_ini = args.siteshift_terms
            if terms_ini is not None:
                if not os.path.isabs(terms_ini):
                    terms_ini = os.path.join(script_dir, terms_ini)
                if not os.path.isfile(terms_ini):
                    raise ValueError(f"--siteshift-terms file not found: {terms_ini}")

            if terms_ini is not None:
                result = site_shifts.compute_site_shift_terms(
                    system,
                    args.siteshift_cubes,
                    args.site_shifts,
                    terms_ini_path=terms_ini,
                    base_dir=script_dir,
                    interp_kind=str(args.siteshift_interp),
                    chunk=int(args.siteshift_chunk),
                    save_json_path=None,
                )
                dw = result['total_eV']
            else:
                dw = site_shifts.compute_site_shifts_for_system(
                    system,
                    args.siteshift_cubes,
                    args.site_shifts,
                    base_dir=script_dir,
                    interp_kind=str(args.siteshift_interp),
                    chunk=int(args.siteshift_chunk),
                    save_json_path=None,
                )

            k = int(args.hybrid_site_index)
            if (k < 0) or (k >= len(dw)):
                raise ValueError(f"--hybrid-site-index out of range: {k} for {len(dw)} rows")
            dw_eV = float(dw[k])
            results['shift_eV'][i] = dw_eV
            results['shift_hartree'][i] = dw_eV / HARTREE_TO_EV
            shifts_track.append(dw_eV)

            geom = calc.compute_shift(d_bohr)
            results['R_ang'][i] = float(geom['R_ang'])
            results['gap_ang'][i] = float(geom['gap_ang'])
            results['proj_gap_ang'][i] = float(geom.get('proj_gap_ang', geom['gap_ang']))
            results['min_atom_dist_ang'][i] = float(geom['min_atom_dist_ang'])

        save_prefix = f"{args.out_prefix}_hybrid".replace(' ', '_')
        plot_results(results, title="Hybrid Δω(scan)", save_prefix=save_prefix, show=args.show)

        data_file = f"{save_prefix}_data.txt"
        header = "# Columns: scan_ang  R_ang  gap_ang  proj_gap_ang  min_atom_dist_ang  shift_eV\n"
        data = np.column_stack([
            results['scan_ang'],
            results['R_ang'],
            results['gap_ang'],
            results['proj_gap_ang'],
            results['min_atom_dist_ang'],
            results['shift_eV'],
        ])
        np.savetxt(data_file, data, header=header, fmt='%16.8e')
        print(f"\n  Data saved to: {data_file}")

        if args.siteshift_json is not None:
            out = {
                'scan_ang': results['scan_ang'].tolist(),
                'shift_eV': results['shift_eV'].tolist(),
                'hybrid_site_index': int(args.hybrid_site_index),
            }
            fout = open(args.siteshift_json, 'w')
            json.dump(out, fout, indent=2, sort_keys=True)
            fout.close()

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

    if workflow == 'hybrid-eval':
        run_hybrid_eval()
        return

    if workflow == 'hybrid-scan':
        run_hybrid_scan()
        return

    if (workflow == 'fft-4combos') or args.run_4combos:
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

# ======================== PhotonMap + Δω diagonal shifts (hybrid cube+RESP) ========================
#
# This example runs PhotonMap exciton diagonalization (Frenkel Hamiltonian) for a 6-row molecules.ini
# system, and applies electrostatic site-energy shifts Δω_i (one per row) to the diagonal energies
# before the Hamiltonian is solved.
#
# Requirements (files in the working directory passed by -w):
#   - molecules.ini              (6 rows: 3 states for Mol1, then 3 states for Mol2)
#   - cubefiles.ini              (transition densities for couplings)
#   - siteshift_cubes.ini        (mol_id + density_difference_stateX.cube per row)
#   - electrostatics.ini         (per mol_id: electrostatic_potential.cube + charges.xyzq)
#
# Run (adjust -w and output as needed):
# python3 /home/indranil/git/ppafm/photonMap.py \
#   -w /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/ \
#   -m molecules.ini -c cubefiles.ini --excitons --volumetric \
#   --siteshift-cubes siteshift_cubes.ini --site-shifts electrostatics.ini \
#   --siteshift-terms site_shift_terms.ini \
#   --siteshift-interp cubic --siteshift-chunk 100000 --siteshift-json site_shifts_multiterm.json \
#   -R 10.0 -Z 6.0 -t s --output out_with_site_shifts
 
# ======================== Multi-term site shifts (Coulomb + add-ons) ========================
#
# This code supports a modular site-energy shift decomposition per site-state (row in molecules.ini):
#
#   Δω_total = Δω_Coulomb + Δω_Exchange + Δω_Polarization + Δω_CT
#
# Where:
#   - Coulomb:      electrostatic interaction of Δρ_A with environment potential (hybrid cube+RESP)
#   - Exchange:     Pauli exchange/steric repulsion via overlap integral Δρ_A · ρ_B^GS
#   - Polarization: mutual polarization of environment molecules via α tensors and electric fields
#   - CT:           charge-transfer mixing via superexchange-like coupling models
#
# Enable/disable terms via --siteshift-terms site_shift_terms.ini.
# If --siteshift-terms is omitted, the workflow computes Coulomb-only shifts (backward compatible).
#
# IMPORTANT (multi-state aggregates):
#   - molecules.ini can contain multiple excited states for the same physical molecule.
#   - These are identified by the same mol_id.
#   - For ALL terms, environment contributions are excluded when env mol_id == site mol_id.
#
# ------------------------------ Required input files ------------------------------
#
# (1) molecules.ini
#   - One row per site-state.
#   - Rows with the same mol_id share the same physical molecule pose (position + rotation).
#
# (2) siteshift_cubes.ini
#   - One row per molecules.ini row.
#   - Columns (space-separated):
#       mol_id   diff_density_cube   gs_density_cube   [virtual_orbital_cube]
#
#   diff_density_cube:
#     - Δρ cube for that excited state (used by Coulomb/exchange/polarization)
#
#   gs_density_cube:
#     - ρ_GS cube for that mol_id (used by exchange)
#     - it is OK to repeat the same GS file on multiple rows
#
#   virtual_orbital_cube (optional 4th column):
#     - used only for CT model = orbital_overlap
#     - typically a LUMO-like orbital cube per site-state
#
# (3) electrostatics.ini
#   - One section per mol_id, e.g. [mol1], [mol2].
#
#   Required keys (Coulomb hybrid cube+RESP potential):
#     potential_cube = path/to/electrostatic_potential.cube
#     resp_charges   = path/to/charges.xyzq
#     resp_center    = cube_atom_mean | none
#     switch_width_ang, cube_margin_ang, beta_match_shell_ang
#
#   charges.xyzq format (space-separated, Angstrom coordinates):
#     x_ang  y_ang  z_ang  q_e
#
#   Optional keys for additional terms:
#     gs_density_cube       = path/to/ground_state_density.cube    (exchange)
#     polarizability_tensor = path/to/polarizability_tensor.txt    (polarization)
#     alpha_units           = bohr3 | ang3                         (polarization)
#     alpha_frame           = local | global                       (polarization)
#     homo_cube             = path/to/HOMO.cube                    (CT hole coupling)
#
# (4) site_shift_terms.ini
#   - Controls which terms are computed and their parameters.
#
#   Minimal example enabling all terms:
#     [terms]
#     enable = coulomb, exchange, polarization, ct
#
#   Exchange parameters:
#     [exchange]
#     k_exch_eV_bohr3 = 7.0
#
#   Polarization parameters (requires polarizability_tensor per mol_id):
#     [polarization]
#     field_method = direct
#     r_min_bohr = 0.5
#
#   CT parameters:
#     [ct]
#     model = orbital_overlap | exp | off
#     k_ct_eV = 1.0
#     E_CT_model = fixed | ip_ea
#     E_CT_eV = 3.5
#
# ------------------------------ Common CLI switches ------------------------------
#
# --siteshift-interp  : cube interpolation kind (linear|cubic)
# --siteshift-chunk   : chunk size for batched evaluation (speed/memory)
# --siteshift-json    : write a JSON report with per-site/per-neighbor breakdown
# --siteshift-terms   : enable multi-term framework using site_shift_terms.ini
#
# ------------------------------ Copy/paste examples ------------------------------
#
# A) Compute multi-term shifts directly (hybrid-eval):
# python3 run_ptcda_dimer.py \
#   --workflow hybrid-eval \
#   --molecules-ini molecules.ini \
#   --siteshift-cubes siteshift_cubes.ini \
#   --site-shifts electrostatics.ini \
#   --siteshift-terms site_shift_terms.ini \
#   --siteshift-interp cubic --siteshift-chunk 100000 \
#   --siteshift-json site_shifts_multiterm.json \
#   --out-prefix ptcda --no-show
#
# B) PhotonMap run applying multi-term Δω_i to exciton diagonals before diagonalization:
# python3 /home/indranil/git/ppafm/photonMap.py \
#   -w /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/ \
#   -m molecules.ini -c cubefiles.ini --excitons --volumetric \
#   --siteshift-cubes siteshift_cubes.ini --site-shifts electrostatics.ini \
#   --siteshift-terms site_shift_terms.ini \
#   --siteshift-interp cubic --siteshift-chunk 100000 \
#   --siteshift-json site_shifts_multiterm.json \
#   -R 10.0 -Z 6.0 -t s --output out_with_multiterm_site_shifts
#
'''
python3 run_ptcda_dimer.py \
  --workflow hybrid-eval \
  --molecules-ini molecules.ini \
  --siteshift-cubes siteshift_cubes.ini \
  --site-shifts electrostatics.ini \
  --siteshift-terms site_shift_terms.ini \
  --siteshift-interp cubic \
  --siteshift-chunk 100000 \
  --siteshift-json site_shifts_multiterm.json \
  --out-prefix ptcda --no-show
  

python3 run_ptcda_dimer.py \
  --workflow hybrid-scan \
  --siteshift-cubes siteshift_cubes.ini \
  --site-shifts electrostatics.ini \
  --siteshift-terms site_shift_terms.ini \
  --siteshift-interp cubic \
  --siteshift-chunk 100000 \
  --hybrid-site-index 0 \
  --axis y --scan-kind displacement --start 12 --stop 32 --step 2 \
  --out-prefix ptcda --no-show  
  
  '''