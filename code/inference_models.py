"""
inference_models.py
--------------------
Forward-pass inference from the trained models — importable, NO command line.

`run_inference(dataset_root, demographics_csv, run_dir)` reads the trained fold
models under <run_dir>/models/ and writes the prediction CSVs into
<run_dir>/severity_csvs/:

    test_severity_transformer_<run>.csv        transformer ensemble + per-fold, held-out test
    oof_validation_severity_transformer_<run>.csv  transformer out-of-fold (validation) severity
    test_severity_svm_<run>.csv                SVM ensemble + per-fold, held-out test
    test_severity_mlp_<run>.csv                MLP ensemble + per-fold, held-out test

It is called by make_figures_tables.py and make_supplementary_figures_tables.py
(their default, non-``--skip-inference`` mode); it is not run on its own.
"""

import os
import pandas as pd

import train_models as tm   # model + inference functions


def run_inference(dataset_root, demographics_csv, run_dir, include_baselines=True):
    """Generate and save the prediction CSVs for a run.

    Reads the fold models from <run_dir>/models/ and the kinematics from
    dataset_root, and writes the severity CSVs into <run_dir>/severity_csvs/.
    Always writes the transformer test + OOF CSVs (the main-text figures);
    when include_baselines is True (the default) it also scores the SVM/MLP
    baselines (needed only for the supplementary tables). Returns that
    severity_csvs directory. (Never writes to severity_csvs/precomputed.)
    """
    dataset_root     = os.path.expanduser(dataset_root)
    demographics_csv = os.path.expanduser(demographics_csv)
    run_name = os.path.basename(os.path.normpath(run_dir))
    csv_dir  = os.path.join(run_dir, "severity_csvs")
    os.makedirs(csv_dir, exist_ok=True)

    # Point the train_models inference functions at this run's paths.
    tm.DS_ROOT    = dataset_root
    tm.DEMO_CSV   = demographics_csv
    tm.MODEL_DIR  = os.path.join(run_dir, "models", "cv_models_transformer")
    tm.PRED_DIR   = os.path.join(run_dir, "models", "cv_preds_transformer")
    tm.OUTPUT_DIR = os.path.join(run_dir, "models")     # -> cv_models_svm / cv_models_mlp
    tm.OUT_CSV    = os.path.join(csv_dir, f"test_severity_transformer_{run_name}.csv")
    tm.CSV_DIR    = csv_dir                              # -> test_severity_{svm,mlp}_<run>.csv
    tm.RUN_TAG    = run_name
    oof_csv       = os.path.join(csv_dir, f"oof_validation_severity_transformer_{run_name}.csv")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"dataset root not found: {dataset_root}\n"
            "Expected sub-*/visit-*/ses-*/Kinematics/*.mot (the unzipped dataset).")
    if not os.path.isfile(demographics_csv):
        raise FileNotFoundError(f"demographics CSV not found: {demographics_csv}")
    missing = [] if os.path.isdir(tm.MODEL_DIR) else ["Transformer"]
    if include_baselines:
        missing += [n.upper() for n in tm.MODELS
                    if not os.path.isdir(os.path.join(tm.OUTPUT_DIR, f"cv_models_{n}{tm.VARIANT_SUFFIX}"))]
    if missing:
        raise FileNotFoundError(
            f"no trained fold models for {', '.join(missing)} under {tm.OUTPUT_DIR}")

    print(f"[inference] {run_name}: scoring models on {dataset_root}")
    samples = tm.load_release_samples(dataset_root, demographics_csv)
    demo_df = pd.read_csv(demographics_csv)
    trainval, test = tm.split_test(samples, demo_df)
    print(f"[inference] {len(samples)} visits; test = {len(test)} "
          f"({len(set(s['digbi_id'] for s in test))} participants)")

    tm.run_test_inference(test, demo_df)                 # -> test_severity_transformer_<run>.csv
    tm.compute_oof_continuous(samples, out_csv=oof_csv)  # -> oof_validation_severity_transformer_<run>.csv
    if include_baselines:
        for name in tm.MODELS:                           # svm, mlp
            tm.infer_from_saved(name)                    # -> test_severity_{name}_<run>.csv
    return csv_dir


if __name__ == "__main__":
    raise SystemExit(
        "inference_models.py has no command line — it is imported by "
        "make_figures_tables.py / make_supplementary_figures_tables.py.\n"
        "Run those with --dataset (to inference) or --skip-inference (to use the "
        "precomputed CSVs).")
