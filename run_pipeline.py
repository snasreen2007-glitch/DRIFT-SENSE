"""
run_pipeline.py
==================
Master entry point. Mirrors the step sequence your existing repo's
main.py already uses, extended with the CNN / hybrid stages your
document requires:

  1. generate_dataset.py     -- Section 4
  2. evaluate_pipeline.py    -- Sections 3 (full detector) + 6 + 7
  3. compare_methods.py      -- Section 9
  4. generate_graphs.py      -- figures
  5. final_summary.py        -- fills in the [XX] placeholders

Usage:
    python run_pipeline.py            # full run, 30 samples
    python run_pipeline.py --n 50     # bigger dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"


def run_step(script_name, extra_args=None):
    script_path = SRC / script_name
    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of test samples")
    parser.add_argument("--weights", type=str, default=None,
                         help="path to trained CNN weights (.pt) from train_cnn.py")
    parser.add_argument("--stress_prob", type=float, default=0.15,
                         help="fraction of samples deliberately anchored on the repetitive "
                              "grid pattern (intentional hard cases, Section 7.2)")
    args = parser.parse_args()

    print("=" * 60)
    print("DRIFT-SENSE -- Hybrid AI Navigation-Error Recovery Pipeline")
    print("=" * 60)

    run_step("dataset_generator.py", ["--n", str(args.n), "--out", "data",
                                       "--stress_prob", str(args.stress_prob)])
    eval_args = ["--weights", args.weights] if args.weights else []
    run_step("evaluate_pipeline.py", eval_args)
    cmp_args = ["--weights", args.weights] if args.weights else []
    run_step("compare_methods.py", cmp_args)
    run_step("generate_graphs.py")
    run_step("final_summary.py")

    print("\n" + "=" * 60)
    print("DRIFT-SENSE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results: {PROJECT_ROOT / 'results'}")
    print(f"Figures: {PROJECT_ROOT / 'figures'}")


if __name__ == "__main__":
    main()
