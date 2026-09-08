"""
inference_transformer.py
------------------------
Forward-pass inference from already-trained Transformer fold models: regenerate a
run's prediction CSVs without retraining. For run <run-name> it reads

    runs/<run-name>/models/cv_models_transformer/   (fold weights + normalizers)
    runs/<run-name>/models/cv_preds_transformer/    (per-fold val ids)

and writes

    runs/<run-name>/csvs/test_severity_transformer_<run-name>.csv
        all 5 fold models scored on the held-out test set (per-fold + ensemble)
    runs/<run-name>/csvs/oof_validation_severity_transformer.csv
        each fold model scored on its own held-out validation visits, keeping the
        pre-sigmoid logit (the un-saturated OOF severity Figure 3 needs)

The held-out test set / train_val split comes from the demographics 'split'
column; labels come from 'diag'. This is the input make_figures_tables.py consumes.

Usage:
    python inference_transformer.py --dataset ~/NMD_OpenCap_DS_v2 \
        --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained
"""

import os
import argparse
import pandas as pd

from dataset import load_release_samples
import train_transformer as tt   # reuse the model + inference functions


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Regenerate a run's Transformer test + OOF severity CSVs from "
                    "its already-trained fold models (forward pass only, no "
                    "retraining).",
        epilog=(
            "inputs:\n"
            "  --dataset       Dataset root containing\n"
            "                    sub-*/visit-*/ses-*/Kinematics/*.mot\n"
            "  --demographics  participant-info CSV; must contain columns\n"
            "                  subid, visit, diag, split.\n"
            "  --run-name      Name of a trained run under runs/. Reads the fold\n"
            "                  models from runs/<run-name>/models/cv_models_transformer/\n"
            "                  (+ cv_preds_transformer/ val ids) and writes\n"
            "                    runs/<run-name>/csvs/test_severity_transformer_<run-name>.csv\n"
            "                    runs/<run-name>/csvs/oof_validation_severity_transformer.csv\n"
            "                  Train the run first with train_transformer.py.\n\n"
            "example:\n"
            "  python inference_transformer.py --dataset ~/NMD_OpenCap_DS_v2 \\\n"
            "      --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained\n"))
    ap.add_argument("--dataset", required=True, metavar="DIR",
                    help="dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot "
                         "(e.g. ~/NMD_OpenCap_DS_v2)")
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV with columns subid, visit, diag, split, ...")
    ap.add_argument("--run-name", required=True, metavar="NAME",
                    help="name of a trained run; reads/writes under runs/<run-name>/")
    args = ap.parse_args()
    dataset_root     = os.path.expanduser(args.dataset)
    demographics_csv = os.path.expanduser(args.demographics)

    REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR = os.path.join(REPO, "runs", args.run_name)
    # Point the train_transformer inference functions at this run's paths.
    tt.MODEL_DIR = os.path.join(RUN_DIR, "models", "cv_models_transformer")
    tt.PRED_DIR  = os.path.join(RUN_DIR, "models", "cv_preds_transformer")
    tt.OUT_CSV   = os.path.join(RUN_DIR, "csvs", f"test_severity_transformer_{args.run_name}.csv")
    oof_csv      = os.path.join(RUN_DIR, "csvs", "oof_validation_severity_transformer.csv")

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isdir(dataset_root):
        ap.error(f"--dataset: directory not found: {dataset_root}\n"
                 "Expected a dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot.")
    if not os.path.isfile(demographics_csv):
        ap.error(f"--demographics: file not found: {demographics_csv}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, split, ...).")
    if not os.path.isdir(tt.MODEL_DIR):
        ap.error(f"--run-name: no trained fold models at {tt.MODEL_DIR}\n"
                 "Train this run first:\n"
                 f"  python train_transformer.py --dataset {args.dataset} "
                 f"--demographics {args.demographics} --run-name {args.run_name}")

    os.makedirs(os.path.join(RUN_DIR, "csvs"), exist_ok=True)
    print(f"Run '{args.run_name}'  ->  {RUN_DIR}")

    samples = load_release_samples(dataset_root, demographics_csv)
    demo_df = pd.read_csv(demographics_csv)
    split_map = {(str(a), int(b)): str(sp).strip().lower()
                 for a, b, sp in zip(demo_df["subid"], demo_df["visit"], demo_df["split"])}
    is_test = lambda s: split_map.get((str(s["digbi_id"]), int(s["date"])), "train_val") == "test"
    test_samples = [s for s in samples if is_test(s)]
    print(f"Loaded {len(samples)} visits; test = {len(test_samples)} "
          f"({len(set(s['digbi_id'] for s in test_samples))} participants)")

    tt.run_test_inference(test_samples, demo_df)         # -> test_severity_<run>.csv
    tt.compute_oof_continuous(samples, out_csv=oof_csv)  # -> oof_validation_severity_<run>.csv


if __name__ == "__main__":
    main()
