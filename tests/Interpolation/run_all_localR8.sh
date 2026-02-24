#!/usr/bin/env bash
set -euo pipefail

# Base paths
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
#DATA_DIR="$BASE_DIR/data_Mithun_new"
DATA_DIR="$BASE_DIR/data_Mithun_flat"
POINT_INFO_DIR="$DATA_DIR/endgroup_points"
POINTS_CLEAN_DIR="$DATA_DIR/points_clean"
RESULTS_DIR="$DATA_DIR/results"
VOLS_DIR="$DATA_DIR/volumes/"
SLICES_DIR="$DATA_DIR/slices/"

mkdir -p "$POINTS_CLEAN_DIR" "$VOLS_DIR" "$SLICES_DIR"

CLEAN_SCRIPT="$BASE_DIR/clean_point_info.py"
INTERP_SCRIPT="$BASE_DIR/interp_zscan_to_grid.py"

declare -a TIP_NAMES=()
# Prepare clean point files for all tips/endgroups (generated once per tip) and collect names
for info in "$POINT_INFO_DIR"/*_point_info.txt; do
    tip_name=$(basename "$info" _point_info.txt)
    TIP_NAMES+=("$tip_name")
    clean_file="$POINTS_CLEAN_DIR/${tip_name}_points_clean.txt"
    if [[ ! -f "$clean_file" ]]; then
        python "$CLEAN_SCRIPT" "$info" "$clean_file"
    fi
done

echo "Running local kriging (R=8, nugget=0) with scatter overlay for all scans..."

for zfile in "$RESULTS_DIR"/*.dat; do
    base=$(basename "$zfile" .dat)

    # Pick the longest tip name that is a prefix of the base name
    tip_prefix=""
    for tip in "${TIP_NAMES[@]}"; do
        if [[ "$base" == "$tip"* ]]; then
            if [[ ${#tip} -gt ${#tip_prefix} ]]; then
                tip_prefix="$tip"
            fi
        fi
    done

    if [[ -z "$tip_prefix" ]]; then
        echo "[WARN] Skipping $base: no matching tip prefix found" >&2
        continue
    fi

    points_file="$POINTS_CLEAN_DIR/${tip_prefix}_points_clean.txt"

    if [[ ! -f "$points_file" ]]; then
        echo "[WARN] Skipping $base: points file not found ($points_file)" >&2
        continue
    fi

    out_vol="$VOLS_DIR/${base}.npy"
    out_slices="$SLICES_DIR/${base}"
    mkdir -p "$out_slices"

    if python "$INTERP_SCRIPT" -k kriging \
        -p "$points_file" -z "$zfile" \
        -o "$out_vol" \
        -r 8.0 --kriging-nugget 0.0 --dx 0.10 --dy 0.10 --dz-grid 0.10 \
        -a 2.0 -b 6.0 -t 0.1 \
        -f "$out_slices" \
        --scatter-overlay 1 --scatter-size 10 --scatter-skip 1 -w 0; then
        echo "[OK] $base"
    else
        echo "[WARN] Skipping $base: interpolation failed (likely missing z entries)" >&2
        continue
    fi

done

echo "Done. Outputs in $VOLS_DIR and $SLICES_DIR"
