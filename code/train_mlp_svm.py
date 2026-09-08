"""
train_mlp_svm.py
----------------
The SVM and MLP feature-based baselines, trained and evaluated using ONLY the
public artifacts:

    <dataset root>          -- de-identified kinematics dataset
                               (sub-*/visit-*/ses-*/Kinematics/*.mot)
    participant-info CSV    -- labels (diag) + train/test 'split',
                               keyed on (subid, visit)

Fully independent of the Transformer run: its own 5-fold GroupKFold CV and its
own held-out test set (from the demographics 'split' column). No PII; everything
is keyed on (subid, visit).

This script writes MODELS ONLY (under runs/<run-name>/models/):
    cv_models_svm/  cv_models_mlp/   -- fitted fold pipelines (.pkl)

The test-severity CSVs are produced separately by inference_mlp_svm.py from
these saved fold pickles (each carries its pipeline + task_vocab + normalizer):
    csvs/test_severity_svm_<run>.csv  csvs/test_severity_mlp_<run>.csv
(No cv_preds are written: the reproduction pipeline never reads the SVM/MLP
val/OOF predictions.)

Run:
    python train_mlp_svm.py --dataset ... --demographics ... --run-name ...
    # then: python inference_mlp_svm.py --dataset ... --demographics ... --run-name ...
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader

from dataset import (VisitDataset, collate_visits, build_task_vocab,
                     fit_normalizer, fit_normalizer_per_task,
                     filter_and_clean_tasks, build_participant_dict,
                     flatten_visits_for_model, EXPECTED_COLUMNS)
# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction (formerly mlp_svm_features.py)
# ──────────────────────────────────────────────────────────────────────────────
# Summary-statistic featurization of kinematic time series for the SVM/MLP
# baselines. Behaviour is byte-for-byte equivalent to the original notebook.
#
# FEATURE LAYOUT  (5 * F + 1 = 166 for F = 33 joints)
#     mean[F]   per-joint mean
#     std[F]    per-joint standard deviation
#     range[F]  per-joint max - min
#     rmsv[F]   per-joint RMS of the first difference   (movement speed)
#     iqr[F]    per-joint interquartile range
#     jerk      ONE scalar: mean |second difference| across all joints and frames
#
# Velocity/acceleration are plain per-frame differences (not divided by dt);
# dataset resampling to a uniform rate makes that a constant StandardScaler
# removes. collate_visits() zero-pads and the mask marks valid frames, which is
# honoured so padding does not bias means or derivatives.
from scipy.stats import iqr

# Optional left-right asymmetry block. Off by default: with ADD_ASYMMETRY False
# the featurization is the original notebook version. Enable per-run without
# editing the file:  FEATURE_VARIANT=asym python train_mlp_svm.py
ADD_ASYMMETRY = os.environ.get("FEATURE_VARIANT", "base").lower() == "asym"

# Keep in sync with dataset.EXPECTED_COLUMNS.
JOINT_NAMES = list(EXPECTED_COLUMNS)
PER_JOINT_STATS = ["mean", "std", "range", "rms_velocity", "iqr"]


def bilateral_pairs(joint_names=None):
    """[(base_name, right_index, left_index)] for every _r/_l pair."""
    names = JOINT_NAMES if joint_names is None else joint_names
    idx   = {n: i for i, n in enumerate(names)}
    pairs = []
    for n in names:
        if n.endswith("_r"):
            base = n[:-2]
            if f"{base}_l" in idx:
                pairs.append((base, idx[n], idx[f"{base}_l"]))
    return pairs


N_PAIRS = len(bilateral_pairs())


def visit_feature_dim(n_joints):
    """Width of the vector returned by task_features / aggregate_tasks_to_visit."""
    d = 5 * n_joints + 1
    if ADD_ASYMMETRY:
        d += len(PER_JOINT_STATS) * N_PAIRS
    return d


def feature_names(joint_names=None):
    """Names aligned with the feature vector, for importance reporting."""
    joint_names = JOINT_NAMES if joint_names is None else joint_names
    names = []
    for stat in PER_JOINT_STATS:
        names += [f"{j}__{stat}" for j in joint_names]
    names.append("global__jerk")
    if ADD_ASYMMETRY:
        for stat in PER_JOINT_STATS:
            names += [f"{base}__asym_{stat}" for base, _r, _l in
                      bilateral_pairs(joint_names)]
    return names


def task_features(x_np, m_np):
    """x_np: [L, F] kinematics, m_np: [L] boolean mask (True=valid).
    Returns a 1D feature vector for this task. Trials with < 3 valid frames
    return an all-zero vector."""
    sel = m_np.astype(bool)
    if sel.sum() < 3:  # too short, return zeros to be safe
        F = x_np.shape[1]
        return np.zeros(visit_feature_dim(F), dtype=np.float32)

    x = x_np[sel]                     # [L', F]
    v = np.diff(x, axis=0)            # velocity
    mean = x.mean(0)
    std  = x.std(0)
    rng  = x.max(0) - x.min(0)
    rmsv = np.sqrt((v**2).mean(0))
    iqrv = iqr(x, axis=0, rng=(25, 75))
    jerk = np.abs(np.diff(v, axis=0)).mean() if v.shape[0] > 1 else 0.0
    parts = [mean, std, rng, rmsv, iqrv, [jerk]]

    if ADD_ASYMMETRY:
        # |left - right| for each per-joint statistic, computed per task (before
        # aggregation) and absolute (which side is affected varies by participant).
        stats = {"mean": mean, "std": std, "range": rng,
                 "rms_velocity": rmsv, "iqr": iqrv}
        for stat in PER_JOINT_STATS:
            arr = stats[stat]
            parts.append(np.array([abs(arr[ri] - arr[li])
                                   for _b, ri, li in bilateral_pairs()]))

    return np.concatenate(parts).astype(np.float32)


def aggregate_tasks_to_visit(task_feat_list, mode="max", topk=2):
    """task_feat_list: list of [D] arrays for one visit.
    mode "max" (elementwise max across tasks; partly encodes which tasks were
    completed) or "topk" (mean of the top-k whole tasks by L2 norm)."""
    Z = np.stack(task_feat_list, axis=0) if len(task_feat_list) else None
    if Z is None:
        return None
    if mode == "max":
        return Z.max(axis=0)
    elif mode == "topk":
        k = min(topk, Z.shape[0])
        idx = np.argsort(np.linalg.norm(Z, axis=1))[::-1][:k]
        return Z[idx].mean(axis=0)
    else:
        return Z.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════

VARIANT_SUFFIX = "_asym" if ADD_ASYMMETRY else ""

# ── Config defaults — set from the CLI in __main__ ────────────────────────────
DS_ROOT     = None    # dataset root (sub-*/visit-*/ses-*/Kinematics/*.mot)
DEMO_CSV    = None    # participant-info CSV (subid, visit, diag, split, ...)
OUTPUT_DIR  = None    # runs/<run-name>/models  -> cv_models_svm / cv_models_mlp
CSV_DIR     = None    # runs/<run-name>/csvs    -> test_severity_{svm,mlp}_<run>.csv
RUN_TAG     = None    # run-name tag used in the CSV filenames

K          = 5
NORM_MODE  = "per_task"     # per-task feature normalization for the baselines
AGG_MODE   = "max"
AGG_TOPK   = 2
BATCH_SIZE = 16
SEED       = 42


# ── Model factories (class-balanced, as in the original) ──────────────────────
def make_svm():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=SEED))


def make_mlp():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      max_iter=300, early_stopping=True, validation_fraction=0.1,
                      learning_rate_init=1e-3, random_state=SEED))


MODELS = {"svm": make_svm, "mlp": make_mlp}


def fit_model(name, X, y, rng):
    clf = MODELS[name]()
    if name == "mlp":                       # MLP has no class_weight -> oversample
        w   = compute_sample_weight("balanced", y)
        idx = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
        clf.fit(X[idx], y[idx])
    else:
        clf.fit(X, y)
    return clf


def prob_to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def norm_key(did, date):
    return (str(did).strip(), str(date).strip())


# ── Data loading from the public dataset (keyed subid/visit) ──────────────────
def load_samples():
    """Visit samples from the public dataset (keyed subid/visit), via the shared
    loader in dataset.py."""
    from dataset import load_release_samples
    return load_release_samples(DS_ROOT, DEMO_CSV)


def split_test(samples, demo_df):
    """Held-out test set from the demographics 'split' column ('test' vs
    'train_val') — the same 73-visit test set the released models used, and
    fully independent of any transformer run."""
    split_map = {(str(a), int(b)): str(sp).strip().lower()
                 for a, b, sp in zip(demo_df["subid"], demo_df["visit"], demo_df["split"])}
    is_test = lambda s: split_map.get((str(s["digbi_id"]), int(s["date"])),
                                      "train_val") == "test"
    test     = [s for s in samples if is_test(s)]
    trainval = [s for s in samples if not is_test(s)]
    print(f"Held-out test set: {len(test)} visits "
          f"({len(set(s['digbi_id'] for s in test))} participants)")
    print(f"Train+val pool:    {len(trainval)} visits "
          f"({len(set(s['digbi_id'] for s in trainval))} participants)")
    return trainval, test


def get_folds(trainval):
    """Own participant-grouped 5-fold split (GroupKFold by subject; no leakage,
    no dependency on the transformer's folds)."""
    groups = np.array([s["digbi_id"] for s in trainval], dtype=object)
    return list(GroupKFold(n_splits=K).split(np.zeros(len(groups)), groups=groups))


# ── Featurization ─────────────────────────────────────────────────────────────
def make_loader(samples_list, task_vocab, norm):
    return DataLoader(VisitDataset(samples_list, task_vocab=task_vocab, normalize=norm),
                      batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_visits, num_workers=0)


@torch.no_grad()
def build_visit_matrix(loader):
    Xv, Y, IDs, Dates = [], [], [], []
    for batch in loader:
        X, M, task_ids, visit_idx, labels, digbi_ids, dates = batch
        X = X.cpu().numpy(); M = M.cpu().numpy().astype(bool)
        visit_idx = visit_idx.cpu().numpy(); labels = labels.cpu().numpy()
        B = labels.shape[0]
        per_visit = [[] for _ in range(B)]
        for t in range(X.shape[0]):
            per_visit[int(visit_idx[t])].append(task_features(X[t], M[t]))
        for b in range(B):
            vfeat = aggregate_tasks_to_visit(per_visit[b], mode=AGG_MODE, topk=AGG_TOPK)
            if vfeat is None:
                vfeat = np.zeros(visit_feature_dim(X.shape[2]), dtype=np.float32)
            Xv.append(vfeat); Y.append(labels[b]); IDs.append(digbi_ids[b]); Dates.append(dates[b])
    return (np.stack(Xv).astype(np.float32), np.array(Y).astype(int),
            np.array(IDs, dtype=object), np.array(Dates, dtype=object))


def fit_norm(fold_train):
    return (fit_normalizer_per_task if NORM_MODE == "per_task" else fit_normalizer)(
        fold_train, max_files=None)


# ── Main CV loop (models only) ────────────────────────────────────────────────
def run(trainval, folds):
    """Fit + save the SVM/MLP fold models. Models only — the test-severity CSVs
    are written separately by inference_mlp_svm.py from these saved fold pickles
    (each pickle carries its pipeline + task_vocab + normalizer)."""
    rng = np.random.default_rng(SEED)
    print(f"\nFeature variant: {'base+asym' if ADD_ASYMMETRY else 'base'} "
          f"(dim/task={visit_feature_dim(33)})\nWriting models to: {OUTPUT_DIR}")

    model_dirs = {}
    for name in MODELS:
        model_dirs[name] = os.path.join(OUTPUT_DIR, f"cv_models_{name}{VARIANT_SUFFIX}")
        os.makedirs(model_dirs[name], exist_ok=True)

    for fold, (idx_tr, idx_va) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold}/{K} ---")
        fold_train = [trainval[i] for i in idx_tr]
        fold_val   = [trainval[i] for i in idx_va]
        task_vocab = build_task_vocab(fold_train)
        norm       = fit_norm(fold_train)

        X_tr, y_tr, _, _ = build_visit_matrix(make_loader(fold_train, task_vocab, norm))
        X_va, y_va, _, _ = build_visit_matrix(make_loader(fold_val,   task_vocab, norm))

        for name in MODELS:
            clf = fit_model(name, X_tr, y_tr, rng)
            joblib.dump({"pipeline": clf, "task_vocab": task_vocab, "norm": norm},
                        os.path.join(model_dirs[name], f"{name}_fold{fold}.pkl"))
            p_va = clf.predict_proba(X_va)[:, 1]
            bacc = balanced_accuracy_score(y_va, (p_va >= 0.5).astype(int))
            au = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else np.nan
            print(f"  {name.upper():<4} val AUROC={au:.3f} bACC={bacc:.3f}")

    print("\nTraining complete. Fold models written under:")
    for name in MODELS:
        print(f"  {model_dirs[name]}/")
    print("To generate the test-severity CSVs, run inference_mlp_svm.py with the "
          "same --run-name.")


def _write_test_csv(name, prob, logit, meta):
    """Assemble test_severity_{name}_<run>.csv from per-fold probs/logits (keyed
    subid/visit; diagnosis from participant_info). Used by inference_mlp_svm.py."""
    y_te, te_ids, te_dates = meta
    pi = pd.read_csv(DEMO_CSV)
    diag_map = {(str(a), int(b)): str(dg) for a, b, dg in zip(pi.subid, pi.visit, pi.diag)}
    df = pd.DataFrame({"subid": [str(i).strip() for i in te_ids],
                       "visit": [int(d) for d in te_dates],
                       "true_label": y_te})
    for f in range(1, K + 1):
        df[f"prob_disease_f{f}"]    = prob[f]
        df[f"logit_severity_f{f}"]  = logit[f]
        df[f"predicted_class_f{f}"] = (np.asarray(prob[f]) >= 0.5).astype(int)
    df["prob_disease_mean"]   = np.mean([prob[f]  for f in range(1, K + 1)], axis=0)
    df["logit_severity_mean"] = np.mean([logit[f] for f in range(1, K + 1)], axis=0)
    df["clinical_diagnosis"]  = [diag_map.get((r.subid, r.visit), "NMD") for r in df.itertuples()]
    out_csv = os.path.join(CSV_DIR, f"test_severity_{name}{VARIANT_SUFFIX}_{RUN_TAG}.csv")
    df.to_csv(out_csv, index=False)
    p_ens = df["prob_disease_mean"].to_numpy()
    print(f"\n{name.upper()} test ensemble: AUROC={roc_auc_score(y_te, p_ens):.3f} "
          f"AUPRC={average_precision_score(y_te, p_ens):.3f} "
          f"bACC={balanced_accuracy_score(y_te, (p_ens >= 0.5).astype(int)):.3f}\n  -> {out_csv}")
    return df


def infer_from_saved(name):
    """Regenerate test_severity_{name}_<run>.csv from the SAVED fold models (no
    retraining): each fold's pickle carries its pipeline + task_vocab + norm."""
    model_dir = os.path.join(OUTPUT_DIR, f"cv_models_{name}{VARIANT_SUFFIX}")
    samples = load_samples()
    _, test = split_test(samples, pd.read_csv(DEMO_CSV))
    prob, logit, meta = {}, {}, None
    for f in range(1, K + 1):
        o = joblib.load(os.path.join(model_dir, f"{name}_fold{f}.pkl"))
        X_te, y_te, te_ids, te_dates = build_visit_matrix(
            make_loader(test, o["task_vocab"], o["norm"]))
        p = o["pipeline"].predict_proba(X_te)[:, 1]
        prob[f], logit[f] = p, prob_to_logit(p)
        if meta is None:
            meta = (y_te, te_ids, te_dates)
    return _write_test_csv(name, prob, logit, meta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train the SVM and MLP feature-based baselines (5-fold "
                    "GroupKFold CV) on the de-identified OpenCap dataset. Fully "
                    "independent of the Transformer run.",
        epilog=(
            "inputs:\n"
            "  --dataset       Dataset root containing\n"
            "                    sub-*/visit-*/ses-*/Kinematics/*.mot\n"
            "  --demographics  participant-info CSV; must contain columns\n"
            "                  subid, visit, diag, split.\n"
            "  --run-name      Output identifier. Fold models go to\n"
            "                    runs/<run-name>/models/{cv_models_svm,cv_models_mlp}/\n\n"
            "next step:\n"
            "  This script writes no CSVs. Generate the test-severity CSVs with\n"
            "  inference_mlp_svm.py using the same --run-name.\n\n"
            "example:\n"
            "  python train_mlp_svm.py --dataset ~/NMD_OpenCap_DS_v2 \\\n"
            "      --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained\n"))
    ap.add_argument("--dataset", required=True, metavar="DIR",
                    help="dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot "
                         "(e.g. ~/NMD_OpenCap_DS_v2)")
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV with columns subid, visit, diag, split, ...")
    ap.add_argument("--run-name", required=True, metavar="NAME",
                    help="run identifier; fold models go under runs/<run-name>/models/")
    args = ap.parse_args()

    # Output layout: runs/<run-name>/models. Reassigning these module globals
    # updates what the functions below read. Fully independent — no transformer
    # run required.
    DS_ROOT  = os.path.expanduser(args.dataset)
    DEMO_CSV = os.path.expanduser(args.demographics)
    REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR    = os.path.join(REPO, "runs", args.run_name)
    OUTPUT_DIR = os.path.join(RUN_DIR, "models")   # -> cv_models_svm / cv_models_mlp

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isdir(DS_ROOT):
        ap.error(f"--dataset: directory not found: {DS_ROOT}\n"
                 "Expected a dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot.")
    if not os.path.isfile(DEMO_CSV):
        ap.error(f"--demographics: file not found: {DEMO_CSV}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, split, ...).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Run '{args.run_name}'  ->  {RUN_DIR}")

    samples = load_samples()
    print(f"loaded {len(samples)} visit samples from the release dataset")
    trainval, _test = split_test(samples, pd.read_csv(DEMO_CSV))
    folds = get_folds(trainval)
    run(trainval, folds)
