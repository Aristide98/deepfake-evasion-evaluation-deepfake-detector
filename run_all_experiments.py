"""
run_all_experiments.py
----------------------
Master script — runs the full thesis experiment pipeline.

MODELS:
  - Xception  (finetuned via finetune_models.py → xception_finetuned.pt)
  - MesoNet   (finetuned via finetune_models.py → mesonet_finetuned.pt)

PIPELINE:
  Step 1 — Baseline inference on clean FF++ c40 data
  Step 2 — Apply all 5 evasion attack conditions
  Step 3 — Inference on each attacked condition
  Step 4 — Compile master results table (Tables 1, 2, 3)

USAGE:
    # Full pipeline (run after finetune_models.py completes):
    python run_all_experiments.py --data_dir ./data

    # Skip attack generation if already done:
    python run_all_experiments.py --data_dir ./data --skip_attacks

OUTPUT:
    results/
      baseline_xception.csv
      baseline_mesonet.csv
      xception_<attack_condition>.csv   (one per condition)
      mesonet_<attack_condition>.csv
      ALL_RESULTS.csv                   <- your master thesis table
"""

import os
import csv
import argparse
import subprocess
import sys

ATTACK_CONDITIONS = [
    "compression_crf35",
    "compression_crf45",
    "noise_sigma10",
    "noise_sigma25",
    "gamma_06",
    "gamma_16",
]


def run_cmd(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd)


def compile_results(results_dir):
    """Merge all individual CSVs into one master results table."""
    all_rows = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".csv") and fname != "ALL_RESULTS.csv":
            with open(os.path.join(results_dir, fname), newline="") as f:
                for row in csv.DictReader(f):
                    all_rows.append(row)

    if not all_rows:
        print("[WARN] No result CSVs found to compile.")
        return

    master = os.path.join(results_dir, "ALL_RESULTS.csv")
    with open(master, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*65}")
    print("  MASTER RESULTS TABLE")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'Condition':<28} {'Acc%':>6} {'AUC':>6} {'ASR%':>6} {'FPS':>6}")
    print(f"  {'-'*75}")
    for row in all_rows:
        print(f"  {row.get('model',''):<25} "
              f"{row.get('condition',''):<28} "
              f"{row.get('accuracy','')!s:>6} "
              f"{row.get('auc','')!s:>6} "
              f"{row.get('asr','')!s:>6} "
              f"{row.get('fps','')!s:>6}")
    print(f"{'='*65}")
    print(f"\n  Saved to: {master}")
    print(f"  Maps to Tables 1, 2, and 3 in your thesis.")


def main(data_dir, results_dir, attacked_dir,
         weights_xception, weights_mesonet, skip_attacks):

    os.makedirs(results_dir,  exist_ok=True)
    os.makedirs(attacked_dir, exist_ok=True)
    py = sys.executable

    # ── Step 1: Baseline inference ────────────────────────────
    print("\n" + "="*65)
    print("  STEP 1 — BASELINE INFERENCE (clean data)")
    print("="*65)

    xc_args = ["--data_dir", data_dir,
               "--output", os.path.join(results_dir, "baseline_xception.csv")]
    mn_args = ["--data_dir", data_dir,
               "--output", os.path.join(results_dir, "baseline_mesonet.csv")]

    if weights_xception: xc_args += ["--weights", weights_xception]
    if weights_mesonet:  mn_args  += ["--weights", weights_mesonet]

    run_cmd([py, "xception_inference.py"]  + xc_args)
    run_cmd([py, "mesonet_inference.py"]   + mn_args)

    # ── Step 2: Evasion attacks ───────────────────────────────
    if not skip_attacks:
        print("\n" + "="*65)
        print("  STEP 2 — APPLYING EVASION ATTACKS")
        print("="*65)
        run_cmd([py, "evasion_attacks.py",
                 "--data_dir",    data_dir,
                 "--output_base", attacked_dir])
    else:
        print("\n[SKIP] Evasion attacks skipped (--skip_attacks flag set).")

    # ── Step 3: Inference on attacked conditions ──────────────
    print("\n" + "="*65)
    print("  STEP 3 — INFERENCE ON ATTACKED CONDITIONS")
    print("="*65)

    for condition in ATTACK_CONDITIONS:
        cond_dir = os.path.join(attacked_dir, condition)
        if not os.path.isdir(cond_dir):
            print(f"[SKIP] {condition} — folder not found: {cond_dir}")
            continue

        xc_c = ["--data_dir", cond_dir,
                "--output", os.path.join(results_dir, f"xception_{condition}.csv")]
        mn_c = ["--data_dir", cond_dir,
                "--output", os.path.join(results_dir, f"mesonet_{condition}.csv")]

        if weights_xception: xc_c += ["--weights", weights_xception]
        if weights_mesonet:  mn_c  += ["--weights", weights_mesonet]

        run_cmd([py, "xception_inference.py"] + xc_c)
        run_cmd([py, "mesonet_inference.py"]  + mn_c)

    # ── Step 4: Compile master table ──────────────────────────
    print("\n" + "="*65)
    print("  STEP 4 — COMPILING MASTER RESULTS TABLE")
    print("="*65)
    compile_results(results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full thesis experiment pipeline — Xception + MesoNet"
    )
    parser.add_argument("--data_dir",        type=str, default="./data",
                        help="Clean data folder with real/ and fake/ subfolders")
    parser.add_argument("--results_dir",     type=str, default="./results",
                        help="Folder for CSV result files")
    parser.add_argument("--attacked_dir",    type=str, default="./data_attacked",
                        help="Folder for attacked video sets")
    parser.add_argument("--weights_xception",type=str, default="xception_finetuned.pt",
                        help="Path to finetuned Xception weights")
    parser.add_argument("--weights_mesonet", type=str, default="mesonet_finetuned.pt",
                        help="Path to finetuned MesoNet weights")
    parser.add_argument("--skip_attacks",    action="store_true",
                        help="Skip attack generation (use if already done)")
    args = parser.parse_args()

    main(
        data_dir         = args.data_dir,
        results_dir      = args.results_dir,
        attacked_dir     = args.attacked_dir,
        weights_xception = args.weights_xception,
        weights_mesonet  = args.weights_mesonet,
        skip_attacks     = args.skip_attacks,
    )
