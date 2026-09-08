"""
inference_models.py
--------------------
Forward-pass inference from the trained models: regenerate a run's prediction
CSVs without retraining. For run <run-name> it reads the fold models under
runs/<run-name>/models/ and writes, into runs/<run-name>/csvs/:

    test_severity_transformer_<run-name>.csv   transformer ensemble + per-fold, held-out test
    oof_validation_severity_transformer.csv    transformer out-of-fold (validation) severity
    test_severity_svm_<run-name>.csv           SVM ensemble + per-fold, held-out test
    test_severity_mlp_<run-name>.csv           MLP ensemble + per-fold, held-out test

The transformer fold models carry their weights + normalizers; the SVM/MLP fold
pickles carry their pipeline + task_vocab + normalizer, so all predictions
reproduce from the models alone. The held-out test set comes from the
demographics 'split' column; labels/diagnoses come from participant_info.

Usage:
    python inference_models.py --dataset ~/NMD_OpenCap_DS_v2 \
        --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained
"""

import os
import argparse
import pandas as pd

import train_models as tm   # reuse the model + inference functions


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Regenerate a run's transformer + SVM/MLP prediction CSVs from "
                    "its already-trained fold models (forward pass only, no "
                    "retraining).",
        epilog=(
            "inputs:\n"
            "  --dataset       Dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot\n"
            "  --demographics  participant-info CSV (columns: subid, visit, diag, split)\n"
            "  --run-name      Name of a trained run under runs/. Reads the fold models\n"
            "                  from runs/<run-name>/models/ and writes to\n"
            "                  runs/<run-name>/csvs/:\n"
            "                    test_severity_transformer_<run-name>.csv\n"
            "                    oof_validation_severity_transformer.csv\n"
            "                    test_severity_svm_<run-name>.csv\n"
            "                    test_severity_mlp_<run-name>.csv\n"
            "                  Train the run first with train_models.py.\n\n"
            "example:\n"
            "  python inference_models.py --dataset ~/NMD_OpenCap_DS_v2 \\\n"
            "      --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained\n"))
    ap.add_argument("--dataset", required=True, metavar="DIR",
                    help="dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot")
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV with columns subid, visit, diag, split, ...")
    ap.add_argument("--run-name", required=True, metavar="NAME",
                    help="name of a trained run; reads/writes under runs/<run-name>/")
    args = ap.parse_args()
    dataset_root     = os.path.expanduser(args.dataset)
    demographics_csv = os.path.expanduser(args.demographics)

    REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR = os.path.join(REPO, "runs", args.run_name)
    # Point the train_models inference functions at this run's paths.
    tm.DS_ROOT    = dataset_root
    tm.DEMO_CSV   = demographics_csv
    tm.MODEL_DIR  = os.path.join(RUN_DIR, "models", "cv_models_transformer")
    tm.PRED_DIR   = os.path.join(RUN_DIR, "models", "cv_preds_transformer")
    tm.OUT_CSV    = os.path.join(RUN_DIR, "csvs", f"test_severity_transformer_{args.run_name}.csv")
    tm.OUTPUT_DIR = os.path.join(RUN_DIR, "models")   # -> cv_models_svm / cv_models_mlp
    tm.CSV_DIR    = os.path.join(RUN_DIR, "csvs")     # -> test_severity_{svm,mlp}_<run>.csv
    tm.RUN_TAG    = args.run_name
    oof_csv       = os.path.join(RUN_DIR, "csvs", "oof_validation_severity_transformer.csv")

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isdir(dataset_root):
        ap.error(f"--dataset: directory not found: {dataset_root}\n"
                 "Expected a dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot.")
    if not os.path.isfile(demographics_csv):
        ap.error(f"--demographics: file not found: {demographics_csv}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, split, ...).")
    missing = []
    if not os.path.isdir(tm.MODEL_DIR):
        missing.append("Transformer")
    missing += [name.upper() for name in tm.MODELS
                if not os.path.isdir(os.path.join(tm.OUTPUT_DIR,
                                                   f"cv_models_{name}{tm.VARIANT_SUFFIX}"))]
    if missing:
        ap.error(f"--run-name: no trained fold models for {', '.join(missing)} under "
                 f"{tm.OUTPUT_DIR}\nTrain this run first:\n"
                 f"  python train_models.py --dataset {args.dataset} "
                 f"--demographics {args.demographics} --run-name {args.run_name}")

    os.makedirs(tm.CSV_DIR, exist_ok=True)
    print(f"Run '{args.run_name}'  ->  {RUN_DIR}")

    samples = tm.load_samples()
    demo_df = pd.read_csv(demographics_csv)
    trainval, test = tm.split_test(samples, demo_df)
    print(f"Loaded {len(samples)} visits; test = {len(test)} "
          f"({len(set(s['digbi_id'] for s in test))} participants)")

    # Transformer: held-out test severity + out-of-fold validation severity.
    tm.run_test_inference(test, demo_df)                 # -> test_severity_transformer_<run>.csv
    tm.compute_oof_continuous(samples, out_csv=oof_csv)  # -> oof_validation_severity_transformer.csv

    # SVM / MLP baselines: held-out test severity.
    for name in tm.MODELS:                               # svm, mlp
        tm.infer_from_saved(name)                        # -> test_severity_{name}_<run>.csv


if __name__ == "__main__":
    main()
