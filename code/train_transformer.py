"""
train_transformer.py
--------------------
Cross-validation training loop for the UE-aware VisitTransformer. Trains the 5
GroupKFold fold models on the train_val split and saves each fold's checkpoint,
normalizer, and held-out validation ids/preds under runs/<run-name>/models/.

This script writes MODELS ONLY. It does not emit the prediction CSVs — the test
and OOF severity CSVs are produced by a separate forward pass:

    python inference_transformer.py --dataset ... --demographics ... --run-name ...

which reads the fold models saved here and writes runs/<run-name>/csvs/. Keeping
the CSVs out of training means predictions can be regenerated from the released
models without retraining, and there is a single code path that writes them.

All npz files include digbi_id and date alongside y and p so predictions can
always be traced back to specific participants and visits.

Saved artifacts per fold (under runs/<run-name>/models/):
    cv_models_transformer/visit_transformer_cv_fold{k}.pt  -- best checkpoint
    cv_models_transformer/normalizer_fold{k}.pt            -- per-fold normalizer
    cv_preds_transformer/cv_fold{k}_val_ids.npz            -- val subid, visit
    cv_preds_transformer/cv_fold{k}_val_preds.npz          -- y, p, subid, visit

(cv_fold{k}_train_ids.npz and cv_oof_preds.npz are intentionally NOT written —
they are redundant with the val-id / per-fold val-pred files and are never read
by the reproduction pipeline, so retraining produces exactly the released file
set.)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

from dataset import VisitDataset, collate_visits, build_task_vocab, fit_normalizer, fit_normalizer_per_task
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
NORM_MODE = "global"

# The model is always the upper-extremity-aware VisitTransformer: the base
# VisitTransformer architecture trained with UEAwareVisitDataset, which masks the
# pelvis/lower-limb coordinates in the upper-extremity tasks (defined below).

# ── Training config ────────────────────────────────────────────────────────────

DEVICE       = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
K            = 5
BATCH_SIZE   = 4
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
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
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

def run_cross_validation(train_samples, val_samples):
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

    print(f"\nmodel: UE-aware VisitTransformer  |  norm: {NORM_MODE}")
    print(f"  MODEL_DIR = {MODEL_DIR}")
    print(f"  PRED_DIR  = {PRED_DIR}")

    trainval_samples = list(train_samples) + list(val_samples)
    groups    = np.array([s["digbi_id"] for s in trainval_samples], dtype=object)
    gkf       = GroupKFold(n_splits=K)

    cv_rows = []
    oof_y, oof_p, oof_ids, oof_dates, oof_fold = [], [], [], [], []

    print(f"\n==== {K}-fold GroupKFold CV  (group=digbi_id, test untouched) ====")

    for fold, (idx_tr, idx_va) in enumerate(
        gkf.split(np.zeros(len(groups)), groups=groups), start=1
    ):
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
        if NORM_MODE == "per_task":
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
        batch_size=BATCH_SIZE, shuffle=False,
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
                            batch_size=BATCH_SIZE, shuffle=False,
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


if __name__ == "__main__":
    import argparse
    from dataset import load_release_samples

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train the UE-aware VisitTransformer (5-fold GroupKFold CV) on "
                    "the de-identified OpenCap dataset. Writes fold MODELS ONLY "
                    "under runs/<run-name>/models/.",
        epilog=(
            "inputs:\n"
            "  --dataset       Dataset root containing\n"
            "                    sub-*/visit-*/ses-*/Kinematics/*.mot\n"
            "  --demographics  participant-info CSV; must contain columns\n"
            "                  subid, visit, diag, split. Only train_val visits\n"
            "                  are used for CV; test visits are held out.\n"
            "  --run-name      Output identifier. Fold checkpoints, normalizers,\n"
            "                  and val ids go under runs/<run-name>/models/.\n\n"
            "next step:\n"
            "  This script writes no CSVs. Generate the test + OOF severity CSVs\n"
            "  with inference_transformer.py using the same --run-name.\n\n"
            "example:\n"
            "  python train_transformer.py --dataset ~/NMD_OpenCap_DS_v2 \\\n"
            "      --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained\n"))
    ap.add_argument("--dataset", required=True, metavar="DIR",
                    help="dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot "
                         "(e.g. ~/NMD_OpenCap_DS_v2)")
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV with columns subid, visit, diag, split, ... "
                         "(e.g. ~/nmd_opencap_participant_info.csv)")
    ap.add_argument("--run-name", required=True, metavar="NAME",
                    help="run identifier; fold models go under runs/<run-name>/models/")
    args = ap.parse_args()
    dataset_root     = os.path.expanduser(args.dataset)
    demographics_csv = os.path.expanduser(args.demographics)

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isdir(dataset_root):
        ap.error(f"--dataset: directory not found: {dataset_root}\n"
                 "Expected a dataset root with sub-*/visit-*/ses-*/Kinematics/*.mot.")
    if not os.path.isfile(demographics_csv):
        ap.error(f"--demographics: file not found: {demographics_csv}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, split, ...).")

    # Output layout: runs/<run-name>/{models, csvs}. Reassigning these module-level
    # names updates the globals the run/inference functions write to.
    REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR    = os.path.join(REPO, "runs", args.run_name)
    MODEL_DIR  = os.path.join(RUN_DIR, "models", "cv_models_transformer")
    PRED_DIR   = os.path.join(RUN_DIR, "models", "cv_preds_transformer")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,  exist_ok=True)
    print(f"Run '{args.run_name}'  ->  {RUN_DIR}")

    # Visit samples from the public dataset (keyed subid/visit).
    samples = load_release_samples(dataset_root, demographics_csv)
    print(f"Loaded {len(samples)} visit samples from {dataset_root}")

    # Held-out test set / train_val split from the demographics 'split' column.
    # (We only train on train_val visits; test visits must never enter CV.)
    demo_df = pd.read_csv(demographics_csv)
    split_map = {(str(a), int(b)): str(sp).strip().lower()
                 for a, b, sp in zip(demo_df["subid"], demo_df["visit"], demo_df["split"])}
    def _is_test(s):
        return split_map.get((str(s["digbi_id"]), int(s["date"])), "train_val") == "test"
    test_samples     = [s for s in samples if _is_test(s)]
    trainval_samples = [s for s in samples if not _is_test(s)]
    print(f"Split (demographics 'split'): train_val={len(trainval_samples)} visits, "
          f"test={len(test_samples)} visits "
          f"({len(set(s['digbi_id'] for s in test_samples))} test participants)")

    cv_df, oof_y, oof_p = run_cross_validation(trainval_samples, [])
    print("\nTraining complete. Fold models + val ids written under "
          f"{MODEL_DIR} / {PRED_DIR}.\nTo generate the test + OOF severity CSVs, run:"
          f"\n  python inference_transformer.py --dataset {args.dataset} "
          f"--demographics {args.demographics} --run-name {args.run_name}")
