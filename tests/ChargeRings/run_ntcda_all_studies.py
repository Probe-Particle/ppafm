#!/usr/bin/env python3
import subprocess
from pathlib import Path


def run_step(label: str, script: str) -> None:
    script_path = Path(__file__).resolve().parent / script
    print(f"\n=== Running {label}: {script_path} ===")
    result = subprocess.run(["python", str(script_path)], cwd=script_path.parent)
    if result.returncode != 0:
        raise SystemExit(f"Step '{label}' failed with exit code {result.returncode}")


def main() -> None:
    base = Path(__file__).resolve().parent
    print(f"NTCDA all-studies driver in {base}")

    # 1) xV line scans over W, per-geometry
    run_step("xV study", "run_ntcda_xv_study.py")

    # 2) xV summary panels over W for each geometry/solver
    run_step("xV summary", "run_ntcda_xv_summary.py")

    # 3) xy maps over W and VBias for each geometry/solver
    run_step("xy study", "run_ntcda_xy_study.py")

    # 4) xy summary panels over VBias for each W/geometry/solver
    run_step("xy summary", "run_ntcda_xy_summary.py")

    print("\nAll NTCDA studies finished.")
    print("xV summaries: results/NTCDA/<geom>/solver_<mode>/summary.png")
    print("xy summaries: results/NTCDA_xy/<geom>/solver_<mode>/W_<W>/summary_xy.png")


if __name__ == "__main__":
    main()
