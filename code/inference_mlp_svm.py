"""
inference_mlp_svm.py
--------------------
Forward-pass inference from already-trained SVM/MLP fold models: regenerate a
run's test-severity CSVs without retraining. For run <run-name> it reads

    runs/<run-name>/models/cv_models_svm/   (fold pipelines, .pkl)
    runs/<run-name>/models/cv_models_mlp/

and writes

    runs/<run-name>/csvs/test_severity_svm_<run-name>.csv
    runs/<run-name>/csvs/test_severity_mlp_<run-name>.csv

Each fold pickle carries its fitted pipeline + task_vocab + normalizer, so the
predictions reproduce from the models alone. The held-out test set comes from
the demographics 'split' column; diagnosis labels come from participant_info.

Usage:
    python inference_mlp_svm.py --dataset ~/NMD_OpenCap_DS_v2 \
        --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained
"""

import os
import argparse

import train_mlp_svm as tms   # reuse the feature + inference functions


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Regenerate a run's SVM/MLP test-severity CSVs from its "
                    "already-trained fold models (forward pass only, no "
                    "retraining).",
        epilog=(
            "inputs:\n"
            "  --dataset       Dataset root containing\n"
            "                    sub-*/visit-*/ses-*/Kinematics/*.mot\n"
            "  --demographics  participant-info CSV; must contain columns\n"
            "                  subid, visit, diag, split.\n"
            "  --run-name      Name of a trained run under runs/. Reads the fold\n"
            "                  pipelines from runs/<run-name>/models/cv_models_{svm,mlp}/\n"
            "                  and writes\n"
            "                    runs/<run-name>/csvs/test_severity_svm_<run-name>.csv\n"
            "                    runs/<run-name>/csvs/test_severity_mlp_<run-name>.csv\n"
            "                  Train the run first with train_mlp_svm.py.\n\n"
            "example:\n"
            "  python inference_mlp_svm.py --dataset ~/NMD_OpenCap_DS_v2 \\\n"
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
    # Point the train_mlp_svm functions at this run's paths.
    tms.DS_ROOT    = dataset_root
    tms.DEMO_CSV   = demographics_csv
    tms.OUTPUT_DIR = os.path.join(RUN_DIR, "models")   # -> cv_models_svm / cv_models_mlp
    tms.CSV_DIR    = os.path.join(RUN_DIR, "csvs")     # -> test_severity_{svm,mlp}_<run>.csv
    tms.RUN_TAG    = args.run_name

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isdir(dataset_root):
        ap.error(f"--dataset: directory not found: {dataset_root}\n"
                 "Expected a dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot.")
    if not os.path.isfile(demographics_csv):
        ap.error(f"--demographics: file not found: {demographics_csv}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, split, ...).")
    missing = [name for name in tms.MODELS
               if not os.path.isdir(os.path.join(tms.OUTPUT_DIR,
                                                  f"cv_models_{name}{tms.VARIANT_SUFFIX}"))]
    if missing:
        ap.error("--run-name: no trained fold models for "
                 f"{', '.join(m.upper() for m in missing)} under {tms.OUTPUT_DIR}\n"
                 "Train this run first:\n"
                 f"  python train_mlp_svm.py --dataset {args.dataset} "
                 f"--demographics {args.demographics} --run-name {args.run_name}")

    os.makedirs(tms.CSV_DIR, exist_ok=True)
    print(f"Run '{args.run_name}'  ->  {RUN_DIR}")

    for name in tms.MODELS:                      # svm, mlp
        tms.infer_from_saved(name)               # -> test_severity_{name}_<run>.csv


if __name__ == "__main__":
    main()
