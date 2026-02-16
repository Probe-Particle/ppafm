

Below is a concise retrospective plus a “how-to” tutorial for smooth, artifact-free interpolation.

## What we did (today, highlights)
- Diagnosed “dotted” vacuum as compact-support kriging with too-small overlap: isolated basis functions cause local bumps.
- Added **adaptive per-point radii** and **nugget** options; also **normalized RBF** (later set aside).
- Added **global kriging evaluation** (dense, no KDTree cutoff) and a **scatter overlay** to compare raw points vs interpolant with identical color scale.
- Ran robustness variants on [HHO-h-p_1-H2O_O](cci:9://file:///home/prokop/git/ppafm/tests/Interpolation/data_Mithun_new/slices_test_autor/HHO-h-p_1-H2O_O:0:0-0:0): k6s18, k6s30_p90, nugget, normalized RBF, and a global baseline (R=25).
- Found best visual quality with **global kriging, nugget=0** ([compare_methods/global_R25_nug0](cci:9://file:///home/prokop/git/ppafm/tests/Interpolation/data_Mithun_new/compare_methods/global_R25_nug0:0:0-0:0)) and also with **local kriging R=8, nugget=0** for the tested system.
- Added batch script [run_all_localR8.sh](cci:7://file:///home/prokop/git/ppafm/tests/Interpolation/run_all_localR8.sh:0:0-0:0) to process all scans; it failed on one scan due to a missing z entry ([HN-hh-CO_C.dat](cci:7://file:///home/prokop/Desktop/CARBSIS/PEOPLE/Mithun/AFM_Tips/afm_scans/results/HN-hh-CO_C.dat:0:0-0:0) missing `p091 z00`). Partial outputs already written before the failure:
  - Volumes: `data_Mithun_new/volumes/local_R8/`
  - Slices (with scatter): `data_Mithun_new/slices/local_R8/`

## Current best method & parameters
- **Interpolator**: Kriging (compact Wendland C2)
- **Mode**: Local (KDTree) is fine if R is large enough. For absolute reference, use global (`--kriging-global 1`) with large R.
- **Good working set** (fast, no nugget): `-k kriging -r 8.0 --kriging-nugget 0.0 --dx 0.10 --dy 0.10 --dz-grid 0.10` plus scatter overlay.
- **Reference baseline** (slow, most robust): `--kriging-global 1 -r 25.0 --kriging-nugget 0.0` (as in [global_R25_nug0](cci:9://file:///home/prokop/git/ppafm/tests/Interpolation/data_Mithun_new/compare_methods/global_R25_nug0:0:0-0:0)).

## Tutorial: how to run cleanly
1) **Clean points (once)**  
   ```bash
   python clean_point_info.py data_Mithun_new/endgroup_points/<TIP>_point_info.txt \
     data_Mithun_new/points_clean/<TIP>_points_clean.txt
   ```
   The batch script already does this if missing.

2) **Pick parameters**  
   - Fast/local: `R=8.0`, nugget=0.0  
   - Reference/global: `--kriging-global 1 -r 25.0`, nugget=0.0  
   - Scatter overlay to verify data fidelity: `--scatter-overlay 1 --scatter-size 10 --scatter-skip 1`

3) **Run a single scan (example)**  
   ```bash
   python interp_zscan_to_grid.py -k kriging \
     -p data_Mithun_new/points_clean/HHO-h-p_1_points_clean.txt \
     -z data_Mithun_new/results/HHO-h-p_1-H2O_O.dat \
     -o data_Mithun_new/volumes/local_R8/HHO-h-p_1-H2O_O.npy \
     -r 8.0 --kriging-nugget 0.0 --dx 0.10 --dy 0.10 --dz-grid 0.10 \
     -a 2.0 -b 6.0 -t 0.1 \
     -f data_Mithun_new/slices/local_R8/HHO-h-p_1-H2O_O \
     --scatter-overlay 1 --scatter-size 10 --scatter-skip 1 -w 0
   ```

4) **Batch all scans (script)**  
   Use [tests/Interpolation/run_all_localR8.sh](cci:7://file:///home/prokop/git/ppafm/tests/Interpolation/run_all_localR8.sh:0:0-0:0) (R=8, nugget=0, scatter).  
   Current failure: [results/HN-hh-CO_C.dat](cci:7://file:///home/prokop/Desktop/CARBSIS/PEOPLE/Mithun/AFM_Tips/afm_scans/results/HN-hh-CO_C.dat:0:0-0:0) missing `p091 z00`.

## Making the batch robust to missing z entries
- Quick fix: wrap the python call in the script to **skip scans that error**:
  ```bash
  python "$INTERP_SCRIPT" ... || {
    echo "[WARN] Skipping $base: interpolation failed (likely missing z entries)" >&2
    continue
  }
  ```
- Or adjust [load_zscan](cci:1://file:///home/prokop/git/ppafm/tests/Interpolation/interp_zscan_to_grid.py:45:0-93:17) to detect missing z and return None; caller skips.
- Or fix the data by adding the missing line to [HN-hh-CO_C.dat](cci:7://file:///home/prokop/Desktop/CARBSIS/PEOPLE/Mithun/AFM_Tips/afm_scans/results/HN-hh-CO_C.dat:0:0-0:0).

## Practical guidance
- If you want “ground truth-ish” smoothness: use **global R=25, nugget=0** ([global_R25_nug0](cci:9://file:///home/prokop/git/ppafm/tests/Interpolation/data_Mithun_new/compare_methods/global_R25_nug0:0:0-0:0)).
- If speed matters and visuals are acceptable: **local R=8, nugget=0** (`local_R8_nug0`).
- Use the scatter overlay to confirm the interpolant passes through raw points without rims.

If you want, I can make the batch script skip incomplete scans and re-run so everything finishes despite missing z entries.