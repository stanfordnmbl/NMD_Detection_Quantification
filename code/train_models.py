"""
train_models.py
---------------
Train BOTH models used in the paper on the de-identified OpenCap dataset:
the UE-aware VisitTransformer and the SVM / MLP feature-based baselines.

A single 5-fold GroupKFold split (grouped by participant) is computed once and
reused for every model, so the transformer and the baselines are cross-validated
on identical folds. The held-out test set (the demographics `split` column) never
enters cross-validation.

This script writes MODELS ONLY, under runs/<run-name>/models/:
    cv_models_transformer/  cv_preds_transformer/   -- transformer fold models + val ids
    cv_models_svm/  cv_models_mlp/                   -- baseline fold pipelines (.pkl)

The prediction CSVs are produced separately by inference_models.py.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight

from dataset import (VisitDataset, collate_visits, build_task_vocab,
                     fit_normalizer, fit_normalizer_per_task,
                     filter_and_clean_tasks, build_participant_dict,
                     flatten_visits_for_model, load_release_samples)
from transformer_model import VisitTransformer

# ── Constants ──────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
    'pelvis_ty', 'pelvis_tz', 'hip_flexion_r', 'hip_adduction_r',
    'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
    'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l',
    'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l',
    'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 'lumbar_bending',
    'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
    'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
    'elbow_flex_l', 'pro_sup_l'
]

# ── Normalization mode ─────────────────────────────────────────────────────────
# "global"   -- pool all tasks and participants; one mean/std per joint.
# "per_task" -- separate mean/std per joint per task type.
# The paper uses "global".
TF_NORM_MODE = "global"

# The model is always the upper-extremity-aware VisitTransformer: the base
# VisitTransformer architecture trained with UEAwareVisitDataset, which masks the
# pelvis/lower-limb coordinates in the upper-extremity tasks (defined below).

# ── Training config ────────────────────────────────────────────────────────────

DEVICE       = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
K            = 5
TF_BATCH_SIZE   = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 20
PATIENCE     = 5

# Output paths — placeholders set per run by the two entry points (train's
# __main__ and inference_transformer.py) to point at
#   runs/<run-name>/models/cv_models_transformer   (MODEL_DIR)
#   runs/<run-name>/models/cv_preds_transformer    (PRED_DIR)
#   runs/<run-name>/csvs/test_severity_<run>.csv   (OUT_CSV)
# OUT_CSV is written only by run_test_inference (inference_transformer.py);
# training itself writes models only.
MODEL_DIR = None
PRED_DIR  = None
OUT_CSV   = None


# ── Upper-extremity-aware dataset ────────────────────────────────────────────────
# For the upper-extremity tasks (brooke, arm_rom, curls), zero the pelvis and
# lower-limb coordinates AFTER normalization so those tasks contribute only trunk
# and upper-limb coordinates. This removes sitting-vs-standing posture information
# (which lives in the pelvis/lower-limb channels) from tasks that assess upper-limb
# function. All other tasks keep the full 33 coordinates.

_UE_KEEP = {
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",              # trunk
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r",  # upper limb R
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",  # upper limb L
}
# indices to zero = pelvis + lower limb (everything not in the keep set)
_UE_MASK_IDX = [i for i, c in enumerate(EXPECTED_COLUMNS) if c not in _UE_KEEP]


class UEAwareVisitDataset(VisitDataset):
    """VisitDataset that masks pelvis/lower-limb coordinates in the UE tasks."""
    UE_TASKS = {"brooke", "arm_rom", "curls"}

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        id2name = {v: k for k, v in self.task_vocab.items()}
        for i, tid in enumerate(item["task_ids_int"].tolist()):
            if id2name.get(int(tid)) in self.UE_TASKS:
                item["task_tensors"][i][:, _UE_MASK_IDX] = 0.0
        return item


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_loader(samples, task_vocab, norm, shuffle=False):
    ds = UEAwareVisitDataset(samples, task_vocab=task_vocab, normalize=norm)
    return DataLoader(ds, batch_size=TF_BATCH_SIZE, shuffle=shuffle,
                      collate_fn=collate_visits, num_workers=0)


def create_model():
    """Instantiate the (upper-extremity-aware) VisitTransformer."""
    return VisitTransformer(
        input_dim=33, d_model=64, n_heads=4, num_layers=3, dropout=0.3
    )


def create_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# ── Train / eval loops ─────────────────────────────────────────────────────────

def train_epoch(model, optimizer, loader, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        X, M, task_ids, visit_idx, labels, _, _ = batch
        X, M, task_ids, visit_idx, labels = (
            X.to(DEVICE), M.to(DEVICE), task_ids.to(DEVICE),
            visit_idx.to(DEVICE), labels.to(DEVICE).float()
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(X, M, task_ids, visit_idx, B=labels.size(0))
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    for batch in loader:
        X, M, task_ids, visit_idx, labels, _, _ = batch
        X, M, task_ids, visit_idx, labels = (
            X.to(DEVICE), M.to(DEVICE), task_ids.to(DEVICE),
            visit_idx.to(DEVICE), labels.to(DEVICE).float()
        )
        logits = model(X, M, task_ids, visit_idx, B=labels.size(0))
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_labels:
        return np.nan, np.nan, np.nan, np.nan

    y = np.concatenate(all_labels)
    z = np.concatenate(all_logits)
    p = 1 / (1 + np.exp(-z))

    auc   = roc_auc_score(y, p)           if len(np.unique(y)) > 1 else np.nan
    auprc = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan
    bacc  = balanced_accuracy_score(y, (p >= 0.5).astype(int))

    return total_loss / len(loader.dataset), auc, auprc, bacc


@torch.no_grad()
def collect_predictions(model, loader):
    """
    Return (y_true, p_pred, digbi_ids, dates) for all visits in loader.
    digbi_id and date are saved alongside y and p so predictions can always
    be traced back to specific participants and visits.
    """
    model.eval()
    ys, ps, ids, dates = [], [], [], []
    for batch in loader:
        X, M, task_ids, visit_idx, labels, digbi_ids, batch_dates = batch
        X, M, task_ids, visit_idx = (
            X.to(DEVICE), M.to(DEVICE),
            task_ids.to(DEVICE), visit_idx.to(DEVICE)
        )
        logits = model(X, M, task_ids, visit_idx, B=labels.size(0))
        probs  = torch.sigmoid(logits)
        ys.append(labels.numpy())
        ps.append(probs.cpu().numpy())
        ids.extend(digbi_ids)
        dates.extend(batch_dates)

    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.array(ids,   dtype=object),
        np.array(dates, dtype=object),
    )


# ── Main CV loop ───────────────────────────────────────────────────────────────

def run_cross_validation(trainval_samples, folds):
    """
    Five-fold GroupKFold cross-validation on the combined train+val set.

    Saves per-fold (under MODEL_DIR / PRED_DIR):
        cv_models_transformer/visit_transformer_cv_fold{k}.pt  -- best checkpoint
        cv_models_transformer/normalizer_fold{k}.pt            -- normalizer
        cv_preds_transformer/cv_fold{k}_val_ids.npz            -- val subid, visit
        cv_preds_transformer/cv_fold{k}_val_preds.npz          -- y, p, subid, visit

    (train_ids and a pooled cv_oof_preds.npz are intentionally not written.)

    Returns:
        cv_df:  DataFrame of per-fold best metrics
        oof_y:  pooled true labels
        oof_p:  pooled predicted probabilities
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,  exist_ok=True)

    print(f"\nmodel: UE-aware VisitTransformer  |  norm: {TF_NORM_MODE}")
    print(f"  MODEL_DIR = {MODEL_DIR}")
    print(f"  PRED_DIR  = {PRED_DIR}")

    # folds (GroupKFold by participant) are computed once in __main__ and
    # shared with the SVM/MLP baselines so every model uses identical folds.

    cv_rows = []
    oof_y, oof_p, oof_ids, oof_dates, oof_fold = [], [], [], [], []

    print(f"\n==== {K}-fold GroupKFold CV  (group=digbi_id, test untouched) ====")

    for fold, (idx_tr, idx_va) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold}/{K} ---")

        fold_train = [trainval_samples[i] for i in idx_tr]
        fold_val   = [trainval_samples[i] for i in idx_va]

        # Class-weighted loss — upweights CTL (minority) to improve balanced accuracy.
        # pos_weight = n_ctl / n_nmd: each NMD sample contributes proportionally less
        # to the loss so total CTL and NMD contributions are balanced.
        n_nmd     = sum(1 for s in fold_train if s["label"] == 1)
        n_ctl     = sum(1 for s in fold_train if s["label"] == 0)
        pos_weight = torch.tensor([n_ctl / n_nmd], device=DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  Class weights: n_nmd={n_nmd}, n_ctl={n_ctl}, "
              f"pos_weight={pos_weight.item():.3f}")

        # Save the val IDs before training — recoverable even if training fails.
        # (train IDs are not saved: the fold membership is fully defined by the
        # val IDs, and downstream reproduction only reads cv_fold{k}_val_ids.npz.)
        np.savez(
            os.path.join(PRED_DIR, f"cv_fold{fold}_val_ids.npz"),
            subid = np.array([s["digbi_id"] for s in fold_val], dtype=object),
            visit = np.array([int(s["date"]) for s in fold_val]),
        )
        print(f"  Saved val split IDs — val: {len(fold_val)} visits "
              f"(train: {len(fold_train)} visits)")

        # Fit vocab and normalizer on training split only
        task_vocab = build_task_vocab(fold_train)
        if TF_NORM_MODE == "per_task":
            print(f"  Fitting per-task normalizer for fold {fold}...")
            norm = fit_normalizer_per_task(fold_train, max_files=None)
        else:
            norm = fit_normalizer(fold_train, max_files=None)
        torch.save(norm, os.path.join(MODEL_DIR, f"normalizer_fold{fold}.pt"))

        train_loader = make_loader(fold_train, task_vocab, norm, shuffle=True)
        val_loader   = make_loader(fold_val,   task_vocab, norm, shuffle=False)

        model     = create_model().to(DEVICE)
        optimizer = create_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        best_auc, best_state, best_metrics = -1.0, None, None
        bad = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss = train_epoch(model, optimizer, train_loader, criterion)
            va_loss, va_auc, va_auprc, va_bacc = eval_epoch(model, val_loader, criterion)
            scheduler.step(va_auc)

            print(f"  Epoch {epoch:02d} | train={tr_loss:.4f} | "
                  f"val AUROC={va_auc:.3f}  AUPRC={va_auprc:.3f}  bACC={va_bacc:.3f}")

            if va_auc > best_auc:
                best_auc     = va_auc
                best_state   = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_metrics = dict(fold=fold, epoch=epoch, val_loss=va_loss,
                                    val_auc=va_auc, val_auprc=va_auprc, val_bacc=va_bacc)
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch} (best AUROC={best_auc:.3f})")
                    break

        cv_rows.append(best_metrics)

        # Save best model checkpoint
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(MODEL_DIR, f"visit_transformer_cv_fold{fold}.pt")
        torch.save(best_state, ckpt_path)
        print(f"  Saved model → {ckpt_path}")

        # Collect and save OOF predictions with IDs
        fold_y, fold_p, fold_ids, fold_dates = collect_predictions(model, val_loader)
        pred_path = os.path.join(PRED_DIR, f"cv_fold{fold}_val_preds.npz")
        np.savez(
            pred_path,
            y     = fold_y,
            p     = fold_p,
            subid = fold_ids,
            visit = np.array([int(d) for d in fold_dates]),
        )
        print(f"  Saved OOF preds → {pred_path}  "
              f"({len(fold_y)} visits, {len(np.unique(fold_ids))} participants)")

        oof_y.append(fold_y)
        oof_p.append(fold_p)
        oof_ids.append(fold_ids)
        oof_dates.append(fold_dates)
        oof_fold.append(np.full(len(fold_y), fold, dtype=int))

    # (cv_oof_preds.npz is intentionally NOT written: the pooled OOF is fully
    # recoverable from the per-fold cv_fold{k}_val_preds.npz, and nothing in the
    # reproduction pipeline reads a pooled-OOF file. The pooled arrays are still
    # returned below for the in-run CV summary.)

    cv_df = pd.DataFrame(cv_rows)
    print("\n==== CV Summary ====")
    print(cv_df.to_string(index=False))

    return cv_df, np.concatenate(oof_y), np.concatenate(oof_p)


# ── Test set inference ─────────────────────────────────────────────────────────

def run_test_inference(test_samples, demo_df):
    """
    Run all five fold models on the held-out test set.
    Saves per-fold probabilities, predicted classes, logits, and ensemble mean.
    Writes the ensemble test-severity CSV to OUT_CSV (set per run by the caller,
    i.e. inference_transformer.py).

    Args:
        test_samples:  list of sample dicts (keyed subid/visit)
        demo_df:       the demographics table (participant-info), with 'subid',
                       'visit', 'diag' — used for the clinical_diagnosis column.
    """
    # (subid, visit) -> diagnosis, from the demographics table
    diag_map = {(str(a), int(b)): str(dg)
                for a, b, dg in zip(demo_df["subid"], demo_df["visit"], demo_df["diag"])}

    # Save test IDs so the split is always recoverable
    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez(
        os.path.join(PRED_DIR, "test_ids.npz"),
        subid = np.array([s["digbi_id"] for s in test_samples], dtype=object),
        visit = np.array([int(s["date"]) for s in test_samples]),
    )
    print(f"Saved test IDs → {os.path.join(PRED_DIR, 'test_ids.npz')}")

    # Build test loader using fold 1 normalizer
    task_vocab = build_task_vocab(test_samples)
    norm_path  = os.path.join(MODEL_DIR, "normalizer_fold1.pt")
    if os.path.exists(norm_path):
        norm = torch.load(norm_path)
    else:
        print("WARNING: normalizer_fold1.pt not found — fitting on test set (not recommended).")
        norm = fit_normalizer(test_samples)

    test_loader = DataLoader(
        UEAwareVisitDataset(test_samples, task_vocab=task_vocab, normalize=norm),
        batch_size=TF_BATCH_SIZE, shuffle=False,
        collate_fn=collate_visits, num_workers=0
    )

    all_digbi_ids, all_dates, all_labels = [], [], []
    fold_probs, fold_logits = {}, {}

    for fold in range(1, K + 1):
        ckpt  = os.path.join(MODEL_DIR, f"visit_transformer_cv_fold{fold}.pt")
        model = create_model().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        ys, ps, zs, ids, dates = [], [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                X, M, task_ids, visit_idx, labels, digbi_ids, batch_dates = batch
                X, M, task_ids, visit_idx = (
                    X.to(DEVICE), M.to(DEVICE),
                    task_ids.to(DEVICE), visit_idx.to(DEVICE)
                )
                logits = model(X, M, task_ids, visit_idx, B=labels.size(0))
                probs  = torch.sigmoid(logits)
                ys.extend(labels.numpy().tolist())
                ps.extend(probs.cpu().numpy().tolist())
                zs.extend(logits.cpu().numpy().tolist())
                ids.extend(digbi_ids)
                dates.extend(batch_dates)

        fold_probs[fold]  = ps
        fold_logits[fold] = zs

        if fold == 1:
            all_digbi_ids = ids
            all_dates     = dates
            all_labels    = ys

        print(f"Fold {fold} test AUROC = {roc_auc_score(all_labels, ps):.3f}")

    # Assemble output DataFrame (keyed subid/visit)
    df_out = pd.DataFrame({
        "subid":      [str(i) for i in all_digbi_ids],
        "visit":      [int(d) for d in all_dates],
        "true_label": all_labels,
    })

    for fold in range(1, K + 1):
        df_out[f"prob_disease_f{fold}"]    = fold_probs[fold]
        df_out[f"logit_severity_f{fold}"]  = fold_logits[fold]
        df_out[f"predicted_class_f{fold}"] = (np.array(fold_probs[fold]) >= 0.5).astype(int)

    df_out["prob_disease_mean"]   = np.mean([fold_probs[f]  for f in range(1, K+1)], axis=0)
    df_out["logit_severity_mean"] = np.mean([fold_logits[f] for f in range(1, K+1)], axis=0)
    df_out["clinical_diagnosis"]  = [diag_map.get((r.subid, r.visit), "NMD")
                                     for r in df_out.itertuples()]

    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved test predictions → {OUT_CSV}")
    return df_out


# ── Entry point ────────────────────────────────────────────────────────────────

def compute_oof_continuous(samples, out_csv=None):
    """Recompute the leakage-free continuous OOF severity: run each fold model on
    its held-out validation visits (from PRED_DIR/cv_fold{k}_val_ids.npz) and
    record the pre-sigmoid logit — the un-saturated severity used by Figure 3.
    Writes the OOF severity CSV to out_csv (required; set per run by the caller,
    i.e. inference_transformer.py)."""
    task_vocab = build_task_vocab(samples)
    by_key = {(str(s["digbi_id"]).strip(), str(s["date"]).strip()): s for s in samples}
    rows = []
    for fold in range(1, K + 1):
        ids = np.load(os.path.join(PRED_DIR, f"cv_fold{fold}_val_ids.npz"), allow_pickle=True)
        vk  = list(zip([str(x).strip() for x in ids["subid"]],
                       [str(x).strip() for x in ids["visit"]]))
        vs  = [by_key[k] for k in vk if k in by_key]
        norm  = torch.load(os.path.join(MODEL_DIR, f"normalizer_fold{fold}.pt"))
        model = create_model().to(DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, f"visit_transformer_cv_fold{fold}.pt"), map_location=DEVICE))
        model.eval()
        loader = DataLoader(UEAwareVisitDataset(vs, task_vocab=task_vocab, normalize=norm),
                            batch_size=TF_BATCH_SIZE, shuffle=False,
                            collate_fn=collate_visits, num_workers=0)
        with torch.no_grad():
            for X, M, task_ids, visit_idx, labels, did, dt in loader:
                lo = model(X.to(DEVICE), M.to(DEVICE), task_ids.to(DEVICE),
                           visit_idx.to(DEVICE), B=labels.size(0)).cpu().numpy().ravel()
                labels = labels.numpy()
                for b in range(len(did)):
                    rows.append(dict(subid=str(did[b]), visit=int(dt[b]),
                                     label=int(labels[b]), fold=fold, logit=float(lo[b])))
    out = pd.DataFrame(rows)[["subid", "visit", "label", "fold", "logit"]]
    if out_csv is None:
        raise ValueError("compute_oof_continuous: out_csv is required "
                         "(set by train_transformer.__main__ / inference_transformer.py)")
    out.to_csv(out_csv, index=False)
    print(f"Saved continuous OOF severity -> {out_csv}  ({len(out)} rows)")
    return out


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
SVM_NORM_MODE  = "per_task"     # per-task feature normalization for the baselines
AGG_MODE   = "max"
AGG_TOPK   = 2
SVM_BATCH_SIZE = 16
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
def make_feat_loader(samples_list, task_vocab, norm):
    return DataLoader(VisitDataset(samples_list, task_vocab=task_vocab, normalize=norm),
                      batch_size=SVM_BATCH_SIZE, shuffle=False,
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
    return (fit_normalizer_per_task if SVM_NORM_MODE == "per_task" else fit_normalizer)(
        fold_train, max_files=None)


# ── Main CV loop (models only) ────────────────────────────────────────────────
def run_baselines(trainval, folds):
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

        X_tr, y_tr, _, _ = build_visit_matrix(make_feat_loader(fold_train, task_vocab, norm))
        X_va, y_va, _, _ = build_visit_matrix(make_feat_loader(fold_val,   task_vocab, norm))

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
            make_feat_loader(test, o["task_vocab"], o["norm"]))
        p = o["pipeline"].predict_proba(X_te)[:, 1]
        prob[f], logit[f] = p, prob_to_logit(p)
        if meta is None:
            meta = (y_te, te_ids, te_dates)
    return _write_test_csv(name, prob, logit, meta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train the UE-aware VisitTransformer and the SVM/MLP baselines "
                    "on one shared 5-fold GroupKFold split (grouped by participant). "
                    "Writes MODELS ONLY.",
        epilog=(
            "inputs:\n"
            "  --dataset       Dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot\n"
            "  --demographics  participant-info CSV (columns: subid, visit, diag, split)\n"
            "  --run-name      Output identifier. Fold models go under\n"
            "                    runs/<run-name>/models/{cv_models_transformer,\n"
            "                    cv_preds_transformer, cv_models_svm, cv_models_mlp}/\n\n"
            "The transformer's fold split is computed once and reused for the SVM/MLP\n"
            "cross-validation, so all models share identical folds. Only train_val\n"
            "visits are used; test visits are held out. This script writes no CSVs —\n"
            "generate them with inference_models.py.\n\n"
            "example:\n"
            "  python train_models.py --dataset ~/NMD_OpenCap_DS_v2 \\\n"
            "      --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained\n"))
    ap.add_argument("--dataset", required=True, metavar="DIR",
                    help="dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot")
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV with columns subid, visit, diag, split, ...")
    ap.add_argument("--run-name", required=True, metavar="NAME",
                    help="run identifier; fold models go under runs/<run-name>/models/")
    args = ap.parse_args()

    DS_ROOT  = os.path.expanduser(args.dataset)
    DEMO_CSV = os.path.expanduser(args.demographics)
    REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR    = os.path.join(REPO, "runs", args.run_name)
    MODEL_DIR  = os.path.join(RUN_DIR, "models", "cv_models_transformer")
    PRED_DIR   = os.path.join(RUN_DIR, "models", "cv_preds_transformer")
    OUTPUT_DIR = os.path.join(RUN_DIR, "models")   # SVM/MLP dirs made inside run_baselines
    RUN_TAG    = args.run_name

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isdir(DS_ROOT):
        ap.error(f"--dataset: directory not found: {DS_ROOT}\n"
                 "Expected a dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot.")
    if not os.path.isfile(DEMO_CSV):
        ap.error(f"--demographics: file not found: {DEMO_CSV}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, split, ...).")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Run '{args.run_name}'  ->  {RUN_DIR}")

    samples = load_samples()
    print(f"Loaded {len(samples)} visit samples from {DS_ROOT}")
    demo_df = pd.read_csv(DEMO_CSV)
    trainval, test = split_test(samples, demo_df)
    print(f"Split (demographics 'split'): train_val={len(trainval)} visits, "
          f"test={len(test)} visits "
          f"({len(set(s['digbi_id'] for s in test))} test participants)")

    # One GroupKFold split (grouped by participant), shared by all models.
    folds = get_folds(trainval)

    print("\n" + "=" * 66 + "\n  Transformer cross-validation\n" + "=" * 66)
    run_cross_validation(trainval, folds)
    print("\n" + "=" * 66 + "\n  SVM / MLP baselines (same folds)\n" + "=" * 66)
    run_baselines(trainval, folds)

    print("\nTraining complete. Fold models written under runs/"
          f"{args.run_name}/models/.\nGenerate the prediction CSVs with:\n"
          f"  python inference_models.py --dataset {args.dataset} "
          f"--demographics {args.demographics} --run-name {args.run_name}")
