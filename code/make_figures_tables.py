"""
make_figures_tables.py
----------------------
Generate the paper's main-text FIGURES (2-5) and TABLES (Table 1 diseases,
Table 2 measures) for a run, and print the figure statistics (AUROC/AUPRC/bACC
with 95% CIs, Cliff's delta, Spearman rho).

READ-ONLY: consumes a run's prediction CSVs + the participant-info CSV; runs no
model inference and never touches the raw dataset.

USAGE:
    python make_figures_tables.py --demographics /path/to/participant_info.csv \\
                                  --run-name pretrained

Writes:
    runs/<run-name>/results/figures/   fig2_performance, fig3_severity, fig4_activlim, fig5_tft
    runs/<run-name>/results/tables/    table1_diseases.csv, table2_measures.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless: build figures to files, never open windows
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, precision_recall_curve
)


def view_sequentially(paths):
    """Show each image in a window; CLOSING it advances to the next (blocking
    plt.show() per image, like the old paper1_code flow). Falls back to opening
    all in the OS default viewer if no interactive matplotlib backend exists."""
    import sys, subprocess
    import matplotlib.image as mpimg
    if not paths:
        return
    for backend in ("MacOSX", "QtAgg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            for p in paths:
                fig = plt.figure()
                try:
                    fig.canvas.manager.set_window_title(os.path.basename(p))
                except Exception:
                    pass
                ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(mpimg.imread(p)); ax.axis("off")
                plt.show()      # blocks until this window is closed; then the next opens
            return
        except Exception:
            continue
    for p in paths:            # no interactive backend — open all in the OS viewer
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", p], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(p)
            else:
                subprocess.run(["xdg-open", p], check=False)
        except Exception:
            pass


def _load_demo(path=None):
    """Read the de-identified participant-info demographics table (keyed
    subid/visit)."""
    return pd.read_csv(path or DEMO_CSV)

# ── Config ─────────────────────────────────────────────────────────────────────
 
K       = 5
METRICS = ["AUROC", "AUPRC", "bACC"]
 
# All four TFT columns — participants must have ALL of these to be included in Fig 4
TFT_COLS = ["tmt_run_time", "tug_cone_time", "tmt_walk_time", "fsts_time"]
 
DIAG_ORDER = ["CTL", "DM", "FSHD", "CMT", "DMD", "scDMD", "SMA", "BMD",
              "CM", "GNE", "LGMD", "SBMA", "ALS"]
PALETTE    = {
    "CTL": "#64748b", "DM": "#3391ff", "FSHD": "#19cc9b",
    "CMT": "#ffb400", "DMD": "#f547d5", "SMA": "#7247b8",
    "BMD": "#ff746C", "CM": "#85b4b7", "GNE": "#cc8770",
    "ALS": "#808000",
    # rare diagnoses present only in the OOF (train_val) split — appear in
    # Supp Fig 2, not the held-out test figures.
    "scDMD": "#e377c2", "LGMD": "#2ca02c", "SBMA": "#8c564b",
}

# Diagnosis corrections. Now empty — the previously-needed DB-689->ALS fix is
# correct in the demographics table (nmd_opencap_participant_info.csv), so no
# post-hoc patch is required. Kept as a dict so callers iterating it are no-ops.
DIAGNOSIS_FIXES = {}


 
# No display relabeling — figures show the raw diagnosis codes (CTL, DM, FSHD, ...)
DISPLAY_NAME = {}


def disp(d):
    """Return the label to show on figures for a diagnosis code."""
    return DISPLAY_NAME.get(str(d), str(d))


VALID_COLOR      = "#4f6fa8" 
TEST_COLOR       = "#2aa876"
X_LABEL_SEVERITY = "Severity score  (SDs above mean healthy control)"
 
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
})
 
rng = np.random.default_rng(0)
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def prob_to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
 
 
def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, seed=42):
    rng    = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n      = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([rng.choice(values, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    return (float(values.mean()),
            float(np.percentile(boot, 100*alpha/2)),
            float(np.percentile(boot, 100*(1-alpha/2))))
 
 
def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    denom = x.size * y.size
    return (gt - lt) / denom if denom else np.nan
 
 
def bootstrap_ci_delta(x, y, n_boot=5000, seed=7, alpha=0.05):
    brng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    deltas = np.array([
        cliffs_delta(brng.choice(x, size=x.size, replace=True),
                     brng.choice(y, size=y.size, replace=True))
        for _ in range(n_boot)
    ])
    return (float(np.median(deltas)),
            float(np.quantile(deltas, alpha/2)),
            float(np.quantile(deltas, 1-alpha/2)))
 
 
def foldnum(col):
    return int(col.split("_f")[-1])
 
 
def p_to_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."
 
 
def style_axes(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("0.65")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=10, colors="0.25", width=0.8)
    if grid_axis == "both":
        ax.grid(True, linewidth=0.6, alpha=0.22)
    else:
        ax.grid(axis=grid_axis, linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)
 
 
def style_axis_corr(ax):
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("0.65")
    ax.tick_params(axis="both", labelsize=7.5, colors="0.45",
                   length=3, width=0.6, direction="out")
    ax.grid(False)
 
 
def panel_label(ax, label, x=-0.05, y=1.02):
    # top-left corner, in line with the (centred) subplot title
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right", color="0.2")
 
 
def add_negative_shade(ax, alpha=0.30):
    x0, _ = ax.get_xlim()
    if x0 < 0:
        ax.axvspan(x0, 0, color="0.9", alpha=alpha, zorder=0)
 
 
def add_threshold_shade(ax, threshold_z, alpha=0.30):
    """Shade left of detection threshold — no border line."""
    x0, _ = ax.get_xlim()
    ax.axvspan(x0, threshold_z, color="0.9", alpha=alpha, zorder=0)
 
 
def add_ctl_mean_shade(ax, alpha=0.30, boundary=0.0):
    """Shade left of `boundary` (default: CTL mean at 0) for visual emphasis."""
    x0, _ = ax.get_xlim()
    if x0 < boundary:
        ax.axvspan(x0, boundary, color="0.9", alpha=alpha, zorder=0)
 
 
def add_sig_bar_h(ax, y1, y2, x, h, text):
    ax.plot([x, x+h, x+h, x], [y1, y1, y2, y2],
            color="0.25", linewidth=1.0, clip_on=False)
    ax.text(x+h, (y1+y2)/2, text, va="center", ha="left",
            fontsize=10, color="0.2")
 
 
def savefig(fig, prefix):
    for ext in ["png", "pdf"]:
        path = os.path.join(FIG_DIR, f"{prefix}.{ext}")
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved -> {path}")
 
 
# ── CTL reference for z-scoring ───────────────────────────────────────────────
 
def compute_ctl_reference(cv_dir=None, verbose=True):
    """
    Compute pooled OOF CTL mean and SD for severity score normalization.
 
    Steps:
        1. Load out-of-fold (OOF) validation predictions from all 5 CV folds
        2. Convert predicted probabilities to logits: log(p / (1-p))
        3. Subset to healthy control participants only (true label y=0)
        4. Compute mean and SD across all CTL OOF logits
 
    severity_z = (logit_severity_mean - ctl_mean) / ctl_std
 
        zero  = mean healthy control score
        1 unit = 1 SD of healthy control logit distribution
        positive = above healthy control mean
        negative = below healthy control mean
 
    Both parameters derived entirely from OOF validation data —
    no test set data is used in normalization.
    """
    cv_dir = CV_PRED_DIR if cv_dir is None else cv_dir
    ctl_logits = []
    for fold in range(1, K + 1):
        d     = np.load(os.path.join(cv_dir, f"cv_fold{fold}_val_preds.npz"),
                        allow_pickle=True)
        y     = np.asarray(d["y"]).reshape(-1)
        p     = np.asarray(d["p"]).reshape(-1)
        logit = prob_to_logit(p)
        ctl_logits.extend(logit[y == 0].tolist())
 
    ctl_mean = float(np.mean(ctl_logits))
    ctl_std  = float(max(np.std(ctl_logits), 1e-6))
 
    if verbose:
        print(f"OOF CTL reference  (n={len(ctl_logits)} CTL visits across {K} folds):")
        print(f"  mean = {ctl_mean:.3f}")
        print(f"  std  = {ctl_std:.3f}")
        print(f"  => severity_z = (logit_severity_mean - {ctl_mean:.3f}) / {ctl_std:.3f}")
    return ctl_mean, ctl_std
 
 
# ── 1) Load and merge data ─────────────────────────────────────────────────────
 
def load_data(ctl_mean, ctl_std, test_csv=None, verbose=True):
    """
    Load test predictions, merge REDCap outcomes, add severity_z column.
    severity_z = (logit_severity_mean - ctl_mean) / ctl_std
        zero  = mean healthy control score (from OOF validation)
        1 unit = 1 SD of healthy control logit distribution (from OOF validation)
    """
    path = TEST_CSV if test_csv is None else test_csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run train_transformer.py for this run first "
            "(it writes the test predictions).")
    df = pd.read_csv(path)

    # De-identified prediction CSVs are keyed on (subid, visit). Older PII-keyed
    # CSVs may still carry digbi_id/date — normalize those only if present.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "digbi_id" in df.columns:
        df["digbi_id"] = df["digbi_id"].astype(str).str.upper().str.strip()
        for _pid, _dx in DIAGNOSIS_FIXES.items():
            df.loc[df["digbi_id"] == _pid, "clinical_diagnosis"] = _dx

    redcap_cols = [
        "subid", "visit",
        "activ_total",
        "tmt_walk_time", "tmt_run_time",
        "tug_cone_time", "fsts_time"
    ]
    redcap = _load_demo(DEMO_CSV)
    # participant-info schema -> the internal TFT column names
    redcap = redcap.rename(columns={
        "diag": "clinical_diagnosis",
        "tft_10mwt": "tmt_walk_time", "tft_10mwrt": "tmt_run_time",
        "tft_tug": "tug_cone_time", "tft_5xsts": "fsts_time",
    })
    redcap = redcap[[c for c in redcap_cols if c in redcap.columns]]

    df_merged = df.merge(redcap, on=["subid", "visit"], how="left")
 
    # Derived columns
    df_merged["run_speed_mps"]  = 10 / df_merged["tmt_run_time"]
    df_merged["walk_speed_mps"] = 10 / df_merged["tmt_walk_time"]
 
    # CTL-normalized severity score
    # zero = OOF CTL mean, 1 unit = 1 SD of OOF CTL logit distribution
    df_merged["severity_z"] = (df_merged["logit_severity_mean"] - ctl_mean) / ctl_std
 
    df_merged["clinical_diagnosis"] = pd.Categorical(
        df_merged["clinical_diagnosis"], categories=DIAG_ORDER, ordered=True
    )
    df_merged = (df_merged
                 .dropna(subset=["clinical_diagnosis", "logit_severity_mean"])
                 .sort_values("clinical_diagnosis")
                 .reset_index(drop=True))
 
    # Report availability
    complete_tft = df_merged[TFT_COLS].notna().all(axis=1).sum()
    if not verbose:
        return df_merged
    print(f"\nMerged dataset: {len(df_merged)} visits")
    print(f"  All 4 TFTs available:  {complete_tft}")
    print(f"  ACTIVLIM available:    {df_merged['activ_total'].notna().sum()}")
    print(f"  severity_z range:      [{df_merged['severity_z'].min():.2f}, "
          f"{df_merged['severity_z'].max():.2f}]  (0 = OOF CTL mean)")
    return df_merged
 
 
# ── 2) Threshold selection and metrics ─────────────────────────────────────────
 
def select_threshold():
    val_y_all, val_p_all = [], []
    for fold in range(1, K+1):
        d = np.load(os.path.join(CV_PRED_DIR, f"cv_fold{fold}_val_preds.npz"),
                    allow_pickle=True)
        val_y_all.append(np.asarray(d["y"]).reshape(-1))
        val_p_all.append(np.asarray(d["p"]).reshape(-1))
    y_oof = np.concatenate(val_y_all)
    p_oof = np.concatenate(val_p_all)
    # Balanced accuracy is reported at a FIXED 0.5 probability threshold. The
    # predicted probabilities are well separated, so bACC is insensitive to the
    # threshold (a data-driven Youden's J cutoff gives near-identical values),
    # and 0.5 avoids tuning a threshold on one distribution (single-fold OOF)
    # and applying it to another (the ensemble test set). AUROC/AUPRC — the
    # primary metrics — are threshold-independent.
    thr_star = 0.5
    return thr_star, y_oof, p_oof
 
 
def compute_fold_metrics(thr_star, y_oof, p_oof, df):
    val_rows = []
    for fold in range(1, K+1):
        d = np.load(os.path.join(CV_PRED_DIR, f"cv_fold{fold}_val_preds.npz"),
                    allow_pickle=True)
        y = np.asarray(d["y"]).reshape(-1)
        p = np.asarray(d["p"]).reshape(-1)
        val_rows.append({
            "fold":  fold,
            "AUROC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "AUPRC": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "bACC":  balanced_accuracy_score(y, (p >= thr_star).astype(int)),
        })
    val_df = pd.DataFrame(val_rows)
 
    y_true    = df["true_label"].astype(int).to_numpy()
    prob_cols = [c for c in df.columns if c.startswith("prob_disease_f")]
    folds_in  = sorted(set(map(foldnum, prob_cols)))
    test_rows = []
    for fold in folds_in:
        p = df[f"prob_disease_f{fold}"].astype(float).to_numpy()
        test_rows.append({
            "fold":  fold,
            "AUROC": roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan,
            "AUPRC": average_precision_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan,
            "bACC":  balanced_accuracy_score(y_true, (p >= thr_star).astype(int)),
        })
    test_df    = pd.DataFrame(test_rows)
    p_test_ens = df["prob_disease_mean"].astype(float).to_numpy()

    return val_df, test_df, y_true, p_test_ens
 
 
# ── 3) Fig 2: Classification performance ──────────────────────────────────────
 
def _metric_ci(y, p, thr, ids=None, n_boot=5000, seed=0):
    """Point estimate + 95% bootstrap CI for each metric on ONE prediction
    vector (ensemble test, or pooled-OOF validation). If ids is given, resample
    clusters (participants); otherwise resample rows (visits). AUROC/AUPRC use
    the continuous score; bACC thresholds it at thr."""
    y = np.asarray(y); p = np.asarray(p)
    def _m(yy, pp):
        return {"AUROC": roc_auc_score(yy, pp),
                "AUPRC": average_precision_score(yy, pp),
                "bACC":  balanced_accuracy_score(yy, (pp >= thr).astype(int))}
    point = _m(y, p)
    rng = np.random.default_rng(seed)
    if ids is not None:
        ids = np.asarray(ids); uid = np.unique(ids)
        idx_by = {u: np.where(ids == u)[0] for u in uid}
    boots = {m: [] for m in METRICS}
    for _ in range(n_boot):
        if ids is not None:
            samp = rng.choice(uid, len(uid), replace=True)
            idx  = np.concatenate([idx_by[u] for u in samp])
        else:
            idx  = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        mm = _m(yy, pp)
        for m in METRICS:
            boots[m].append(mm[m])
    return {m: (point[m], float(np.percentile(boots[m], 2.5)),
                float(np.percentile(boots[m], 97.5))) for m in METRICS}


def plot_fig2(y_oof, p_oof, y_true, p_test_ens, thr_star):
    # Panel a reports the SAME predictors as panel b and the rest of the paper:
    # pooled-OOF validation and the ensemble-averaged test set. CIs are 95%
    # bootstrap intervals resampled over visits (same for validation and test).
    val_ci  = _metric_ci(y_oof,  p_oof,      thr_star, ids=None, seed=1)
    test_ci = _metric_ci(y_true, p_test_ens, thr_star, ids=None, seed=2)
    s = pd.DataFrame([{"Metric": m,
                       "val_mean":  val_ci[m][0],  "val_lo":  val_ci[m][1],  "val_hi":  val_ci[m][2],
                       "test_mean": test_ci[m][0], "test_lo": test_ci[m][1], "test_hi": test_ci[m][2]}
                      for m in METRICS])
    print(f"\n  ensemble / pooled-OOF         Val [95% CI]          Test [95% CI]")
    for _, r in s.iterrows():
        print(f"    {r['Metric']:<6} {r['val_mean']:.3f} [{r['val_lo']:.3f}-{r['val_hi']:.3f}]   "
              f"{r['test_mean']:.3f} [{r['test_lo']:.3f}-{r['test_hi']:.3f}]")
 
    prec_val,  rec_val,  _ = precision_recall_curve(y_oof,  p_oof)
    prec_test, rec_test, _ = precision_recall_curve(y_true, p_test_ens)
    auprc_val  = average_precision_score(y_oof,  p_oof)
    auprc_test = average_precision_score(y_true, p_test_ens)
 
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=200)
    x = np.arange(len(METRICS)); w = 0.34
 
    axA.bar(x-w/2, s["val_mean"],  width=w, color=VALID_COLOR, alpha=0.92,
            label="Validation")
    axA.bar(x+w/2, s["test_mean"], width=w, color=TEST_COLOR,  alpha=0.92,
            label="Held-out test")
    axA.errorbar(x-w/2, s["val_mean"],
                 yerr=[s["val_mean"]-s["val_lo"], s["val_hi"]-s["val_mean"]],
                 fmt="none", ecolor="0.25", elinewidth=1.0, capsize=4, zorder=5)
    axA.errorbar(x+w/2, s["test_mean"],
                 yerr=[s["test_mean"]-s["test_lo"], s["test_hi"]-s["test_mean"]],
                 fmt="none", ecolor="0.25", elinewidth=1.0, capsize=4, zorder=5)
    axA.set_xticks(x); axA.set_xticklabels(METRICS, fontsize=12)
    axA.set_ylim(0, 1.01); axA.set_ylabel("Score", fontsize=10)
    axA.set_title("Classification performance", fontsize=14, y=1.17, color="0.2")
    style_axes(axA, "y")
    axA.grid(False)
    axA.tick_params(axis="y", labelsize=11)
    axA.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=9)
    panel_label(axA, "a", y=1.17)
    # axA.set_title(r"$\bf{a}$" + f"  Validation vs held-out test (mean ± 95% CI)", fontsize=12, color="0.2")
    # remove panel_label(axA, "a")

   
 
    axB.plot(rec_val,  prec_val,  color=VALID_COLOR, linewidth=2.2,
             label=f"Validation (AUPRC={auprc_val:.2f})")
    axB.plot(rec_test, prec_test, color=TEST_COLOR,  linewidth=2.2,
             label=f"Held-out test (AUPRC={auprc_test:.2f})")
    axB.fill_between(rec_val,  prec_val,  color=VALID_COLOR, alpha=0.12)
    axB.fill_between(rec_test, prec_test, color=TEST_COLOR,  alpha=0.12)
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.set_xlabel("Recall", fontsize=10); axB.set_ylabel("Precision", fontsize=10)
    axB.set_title("Precision-Recall curves", fontsize=14, y=1.17, color="0.2")
    style_axes(axB, "both")
    axB.grid(False)
    axB.tick_params(labelsize=11)
    axB.legend(frameon=False, loc="lower left", fontsize=9)
    panel_label(axB, "b", y=1.17)

    # axB.set_title(r"$\bf{b}$  Precision-Recall curves", fontsize=12, color="0.2")
    # remove panel_label(axB, "b")
 
    fig.tight_layout()
    savefig(fig, "fig2_performance")
 
 
# ── 4) Fig 3: Severity score distribution ─────────────────────────────────────
 
def _oof_continuous_severity():
    """Read the leakage-free continuous OOF severity logits for this run
    (subid, visit, label, fold, logit), written by train_transformer.py. No
    inference here — make_figures_tables.py is read-only."""
    if not os.path.exists(OOF_CSV):
        raise FileNotFoundError(
            f"{OOF_CSV} not found — run inference_transformer.py for this run "
            "first (it writes the OOF validation severity).")
    return pd.read_csv(OOF_CSV)


def plot_fig3(df, threshold_z=None):
    """
    Four-panel severity-score distribution.
    Top row (a, b): held-out test set. Bottom row (c, d): out-of-fold (OOF)
    validation, using leakage-free continuous severity logits (each visit scored
    by the single fold model that withheld it). Left column: NMD vs CTL boxplot +
    strip with Cliff's delta [95% CI]; right column: by diagnosis. Both rows are
    normalized to the continuous OOF control mean/SD and share the x-axis.
    threshold_z is accepted for call-signature compatibility but unused.
    """
    oof = _oof_continuous_severity()
    cm = float(oof.loc[oof.label == 0, "logit"].mean())
    cs = float(max(oof.loc[oof.label == 0, "logit"].std(), 1e-6))

    test = df.copy()
    test["severity_z"] = (test["logit_severity_mean"] - cm) / cs

    demo = _load_demo(DEMO_CSV).rename(columns={"diag": "clinical_diagnosis"})
    oof = oof.merge(demo[["subid", "visit", "clinical_diagnosis"]],
                    on=["subid", "visit"], how="left")
    for pid, dx in DIAGNOSIS_FIXES.items():
        oof.loc[oof["digbi_id"] == pid, "clinical_diagnosis"] = dx
    oof["severity_z"] = (oof["logit"] - cm) / cs

    allz = np.concatenate([test["severity_z"].dropna().to_numpy(),
                           oof["severity_z"].dropna().to_numpy()])
    XLO, XHI = allz.min() - 0.4, allz.max() + 0.6
    FILL = {"CTL": "#64748b", "NMD": "0.20"}

    def draw_pair(axA, axB, d, la, lb):
        d = d.dropna(subset=["clinical_diagnosis", "severity_z"])
        ctl = d.loc[d.clinical_diagnosis == "CTL", "severity_z"].to_numpy()
        nmd = d.loc[d.clinical_diagnosis != "CTL", "severity_z"].to_numpy()
        nd = d.loc[d.clinical_diagnosis != "CTL", ["severity_z", "clinical_diagnosis"]]
        pt = cliffs_delta(nmd, ctl)
        _, lo, hi = bootstrap_ci_delta(nmd, ctl)
        print(f"  Cliff's delta ({'held-out test ' if la == 'a' else 'OOF validation'}): "
              f"{pt:.3f} [{lo:.3f}, {hi:.3f}]   (NMD n={len(nmd)}, CTL n={len(ctl)})")
        bp = axA.boxplot([ctl, nmd], vert=False,
                         tick_labels=[f"{disp('CTL')} (n={len(ctl)})", f"NMD (n={len(nmd)})"],
                         widths=0.55, patch_artist=True, showfliers=False,
                         medianprops=dict(color="0.25", linewidth=1.6),
                         boxprops=dict(linewidth=1.2, color="0.35"),
                         whiskerprops=dict(linewidth=1.0, color="0.35"),
                         capprops=dict(linewidth=1.0, color="0.35"))
        for patch, lab in zip(bp["boxes"], ["CTL", "NMD"]):
            patch.set_facecolor(FILL[lab]); patch.set_alpha(0.28); patch.set_edgecolor("0.35")
        axA.scatter(ctl, 1 + rng.uniform(-0.15, 0.15, len(ctl)), s=20, color=FILL["CTL"],
                    alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
        axA.scatter(nd.severity_z, 2 + rng.uniform(-0.15, 0.15, len(nmd)), s=20,
                    c=[PALETTE.get(str(x), "0.2") for x in nd.clinical_diagnosis],
                    alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
        axA.invert_yaxis(); axA.set_xlim(XLO, XHI); add_ctl_mean_shade(axA, 0.3)
        axA.text(0.97, 0.95, f"Cliff's δ = {pt:.2f}\n[{lo:.2f}, {hi:.2f}]",
                 transform=axA.transAxes, ha="right", va="top", fontsize=9.5, color="0.2",
                 bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.92))
        axA.set_title(f"Severity: NMD vs {disp('CTL')}", fontsize=14, pad=4, color="0.2")
        panel_label(axA, la); axA.set_xlabel(X_LABEL_SEVERITY, fontsize=10, color="0.2")
        for sp in ["top", "right", "left"]:
            axA.spines[sp].set_visible(False)
        axA.spines["bottom"].set_color("0.75"); axA.tick_params(labelsize=10, colors="0.35"); axA.grid(False)
        labs = [q for q in DIAG_ORDER if (d.clinical_diagnosis == q).any()]
        for i, q in enumerate(labs, 1):
            vals = d.loc[d.clinical_diagnosis == q, "severity_z"].to_numpy()
            axB.scatter(vals, i + rng.uniform(-0.22, 0.22, len(vals)), s=26, color=PALETTE[q],
                        alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
            med = np.median(vals); q25, q75 = np.quantile(vals, [0.25, 0.75])
            axB.plot([q25, q75], [i, i], color="0.25", lw=1.2, alpha=0.85, zorder=4)
            for xx in (q25, q75):
                axB.plot([xx, xx], [i - 0.1, i + 0.1], color="0.35", lw=1.0, zorder=4)
            axB.plot([med, med], [i - 0.18, i + 0.18], color="0.15", lw=1.6, zorder=4)
        cnt = d.clinical_diagnosis.value_counts().to_dict()
        axB.set_yticks(range(1, len(labs) + 1))
        axB.set_yticklabels([f"{disp(q)} (n={cnt[q]})" for q in labs])
        axB.invert_yaxis(); axB.set_xlim(XLO, XHI); add_ctl_mean_shade(axB, 0.3)
        axB.set_title("Severity by diagnosis", fontsize=14, pad=4, color="0.2")
        panel_label(axB, lb); axB.set_xlabel(X_LABEL_SEVERITY, fontsize=10, color="0.2")
        for sp in ["top", "right", "left"]:
            axB.spines[sp].set_visible(False)
        axB.spines["bottom"].set_color("0.75"); axB.tick_params(labelsize=10, colors="0.35"); axB.grid(False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6),
                             gridspec_kw={"wspace": 0.42, "hspace": 0.30, "left": 0.13})
    draw_pair(axes[0, 0], axes[0, 1], test, "a", "b")
    draw_pair(axes[1, 0], axes[1, 1], oof, "c", "d")
    fig.canvas.draw()
    _cy = lambda ax: (ax.get_position().y0 + ax.get_position().y1) / 2
    fig.text(0.045, _cy(axes[0, 0]),
             f"Held-out test  (n={int(test['clinical_diagnosis'].notna().sum())})",
             rotation=90, va="center", ha="center", fontsize=13, color="0.1", fontweight="bold")
    fig.text(0.045, _cy(axes[1, 0]),
             f"OOF validation  (n={int(oof['clinical_diagnosis'].notna().sum())})",
             rotation=90, va="center", ha="center", fontsize=13, color="0.1", fontweight="bold")
    savefig(fig, "fig3_severity")


# ── 5) Fig 5: TFT correlations ────────────────────────────────────────────────
 
def plot_fig5_tft(df):
    """
    Four-panel scatter: severity_z vs TFTs.
    Sample: visits with ALL four TFTs (listwise deletion across the four timed
    tests), regardless of ACTIVLIM. The ACTIVLIM panel (Fig 4) uses its own
    activ-only sample, so the two convergent-validity figures have different n.
    X-axis: SDs above mean healthy control.
    """
    # All-four-TFT sample — same visits in every TFT panel
    tft_raw_cols = ["tmt_run_time", "tug_cone_time", "tmt_walk_time", "fsts_time"]
    df_complete = df.dropna(
        subset=tft_raw_cols + ["severity_z", "clinical_diagnosis"]).copy()
    df_complete["run_speed_mps"]  = 10 / df_complete["tmt_run_time"]
    df_complete["walk_speed_mps"] = 10 / df_complete["tmt_walk_time"]
    n_total    = len(df)
    n_complete = len(df_complete)
    print(f"  Spearman rho vs severity score  (n={n_complete} visits with all 4 TFTs):")
 
    tft_metrics = [
        ("run_speed_mps",  "10m run speed (m/s)", "10m Run Speed"),
        ("tug_cone_time",  "TUG time (s)",   "TUG Time"),
        ("walk_speed_mps", "10m walk speed (m/s)", "10m Walk Speed"),
        ("fsts_time",      "5xSTS time (s)",       "5xSTS Time"),
    ]
    # Place each annotation in the emptiest corner (off the regression diagonal)
    # so it never overlaps data points. (sx, sy, va, ha)
    stats_pos = [
        (0.97, 0.96, "top", "right"),   # a  10m run speed  (neg slope) -> top-right
        (0.03, 0.96, "top", "left"),    # b  TUG-cone       (pos slope) -> top-left
        (0.97, 0.96, "top", "right"),   # c  10m walk speed (neg slope) -> top-right
        (0.03, 0.96, "top", "left"),    # d  5xSTS          (pos slope) -> top-left
    ]
 
    fig, axes     = plt.subplots(2, 2, figsize=(8.2, 6.2), dpi=200)
    axes          = axes.ravel()
    present_diags = set()
 
    for i, (ax, (col, ylabel, title)) in enumerate(zip(axes, tft_metrics)):
        d = df_complete.copy()
        x = d["severity_z"].to_numpy()
        y = d[col].to_numpy()
 
        r_s, p_s = spearmanr(x, y)
 
        for diag in d["clinical_diagnosis"].unique():
            sub = d[d["clinical_diagnosis"] == diag]
            ax.scatter(sub["severity_z"], sub[col],
                       s=18, alpha=0.90, color=PALETTE.get(str(diag), "0.5"),
                       edgecolors="white", linewidths=0.35, zorder=3)
 
        sns.regplot(data=d, x="severity_z", y=col, ax=ax,
                    scatter=False, ci=95,
                    line_kws=dict(color="0.35", linewidth=0.8, zorder=4))
        for coll in ax.collections:
            if coll.__class__.__name__ == "PolyCollection":
                coll.set_facecolor("0.7"); coll.set_alpha(0.20); coll.set_edgecolor("none")
 
        # add top headroom so the annotation always sits above the data cloud
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.18 * (y1 - y0))
        if i == 0:  # 10m run speed — 0.5-spaced ticks up to 4.0 (drop the unneeded 4.5)
            ax.set_yticks(np.arange(0.5, 4.01, 0.5))

        sx, sy, sva, sha = stats_pos[i]
        p_str = f"p = {p_s:.3f}" if p_s >= 0.001 else "p < 0.001"
        ax.text(sx, sy,
                f"Spearman ρ = {r_s:.2f} ({p_str})",
                transform=ax.transAxes, fontsize=7.5, va=sva, ha=sha, color="0.3")
 
        ax.set_title(title, fontsize=9, pad=4, color="0.2")
        ax.set_xlabel(X_LABEL_SEVERITY, fontsize=7.5, color="0.4")
        ax.set_ylabel(ylabel, fontsize=8, color="0.4")
        style_axis_corr(ax)
        panel_label(ax, ["a", "b", "c", "d"][i])
        present_diags.update(d["clinical_diagnosis"].unique())
        print(f"  {title}: Spearman ρ={r_s:.3f} ({p_str}), n={len(d)}")
 
    legend_elements = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=PALETTE.get(d, "0.5"),
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=5, label=disp(d))
        for d in DIAG_ORDER if d in present_diags
    ]
    fig.legend(handles=legend_elements, frameon=False, fontsize=8,
               title="Diagnosis", title_fontsize=8,
               loc="upper left", bbox_to_anchor=(0.84, 0.88))
    fig.subplots_adjust(right=0.82, wspace=0.35, hspace=0.45)
    savefig(fig, "fig5_tft")
 
 
# ── 6) Fig 4: ACTIVLIM correlation ────────────────────────────────────────────
 
def plot_fig4_activlim(df):
    """
    Single-panel scatter: severity_z vs ACTIVLIM for all participants.
    Both Spearman ρ values (all participants + NMD only) annotated at bottom.
    Regression line fit on all participants.
    X-axis: SDs above mean healthy control.
    """
    # Use the ACTIVLIM Rasch logit measure (activ_logit), which is estimable from
    # partial item responses, so it covers more visits than the complete-case sum.
    # Taken from the production demographics.
    strict = _load_demo(DEMO_CSV)
    strict = strict[["subid", "visit", "activ_logit"]].rename(
        columns={"activ_logit": "activ_val"})
    df = df.merge(strict, on=["subid", "visit"], how="left")

    # ACTIVLIM sample: all visits with an ACTIVLIM (activ_logit) measure,
    # regardless of TFT completeness. (Fig 5 TFT panels use the all-4-TFT
    # sample; the two convergent-validity figures therefore have different n.)
    d_all = df[df["activ_val"].notna()].copy()
    d_nmd = d_all[d_all["clinical_diagnosis"] != "CTL"].copy()

    r_all, p_all = spearmanr(d_all["severity_z"], d_all["activ_val"])
    r_nmd, p_nmd = spearmanr(d_nmd["severity_z"], d_nmd["activ_val"])
 
    p_str_all = "p < 0.001" if p_all < 0.001 else f"p = {p_all:.3f}"
    p_str_nmd = "p < 0.001" if p_nmd < 0.001 else f"p = {p_nmd:.3f}"
 
    print(f"  ACTIVLIM (all):      Spearman ρ={r_all:.3f} ({p_str_all}), n={len(d_all)}")
    print(f"  ACTIVLIM (NMD only): Spearman ρ={r_nmd:.3f} ({p_str_nmd}), n={len(d_nmd)}")
 
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=200)

    # Scatter colored by diagnosis — iterate in DIAG_ORDER for consistent layering
    for diag in [d for d in DIAG_ORDER if d in d_all["clinical_diagnosis"].unique()]:
        sub = d_all[d_all["clinical_diagnosis"] == diag]
        ax.scatter(sub["severity_z"], sub["activ_val"],
                   s=28, alpha=0.90, color=PALETTE.get(diag, "0.5"),
                   edgecolors="white", linewidths=0.4, zorder=3)

    # Regression line on all participants
    sns.regplot(data=d_all, x="severity_z", y="activ_val", ax=ax,
                scatter=False, ci=95,
                line_kws=dict(color="0.25", linewidth=1.0, zorder=4))
    for coll in ax.collections:
        if coll.__class__.__name__ == "PolyCollection":
            coll.set_facecolor("0.6"); coll.set_alpha(0.15); coll.set_edgecolor("none")

    # Zoom out so points aren't crammed against the axes.
    ax.margins(x=0.13, y=0.16)

    # Both rho values inside plot lower left — smaller text to avoid dot overlap
    ax.text(0.03, 0.03,
            f"Spearman ρ = {r_all:.2f} ({p_str_all}) — all\n"
            f"Spearman ρ = {r_nmd:.2f} ({p_str_nmd}) — disease only",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="0.3")
 
    ax.set_xlabel(X_LABEL_SEVERITY, fontsize=10)
    ax.set_ylabel("ACTIVLIM score", fontsize=10)
    style_axis_corr(ax)
 
    # Legend inside the (sparse) upper-right corner — the correlation is
    # negative, so few points sit there, keeping the panel square on save.
    present = d_all["clinical_diagnosis"].unique()
    legend_elements = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=PALETTE.get(d, "0.5"),
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=6, label=disp(d))
        for d in DIAG_ORDER if d in present
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=8,
              title="Diagnosis", title_fontsize=8.5, labelspacing=0.35,
              handletextpad=0.4, borderaxespad=0.4,
              loc="upper right")
 
    fig.tight_layout()
    savefig(fig, "fig4_activlim")


# ── Main ─────────────────────────────────────────────────────────────

def generate_figures():
    print(f"\n{'='*66}")
    print(f"  Figures 2-5   |   models: {os.path.dirname(MODEL_DIR)}")
    print(f"  demographics: {DEMO_CSV}")
    print(f"  figures ->    {FIG_DIR}")
    print(f"{'='*66}")

    # Operating threshold + OOF control reference (from OOF validation preds).
    thr_star, y_oof, p_oof = select_threshold()
    ctl_mean, ctl_std      = compute_ctl_reference(verbose=False)
    df = load_data(ctl_mean, ctl_std, verbose=False)
    logit_thr   = float(np.log(thr_star / (1 - thr_star)))
    threshold_z = (logit_thr - ctl_mean) / ctl_std

    print(f"\n{'-'*66}\n  FIGURE 2  -  Classification performance\n{'-'*66}")
    val_df, test_df, y_true, p_test_ens = compute_fold_metrics(thr_star, y_oof, p_oof, df)
    plot_fig2(y_oof, p_oof, y_true, p_test_ens, thr_star)

    print(f"\n{'-'*66}\n  FIGURE 3  -  Severity score distribution (held-out test + OOF)\n{'-'*66}")
    plot_fig3(df, threshold_z)

    print(f"\n{'-'*66}\n  FIGURE 4  -  Convergent validity: ACTIVLIM\n{'-'*66}")
    plot_fig4_activlim(df)

    print(f"\n{'-'*66}\n  FIGURE 5  -  Convergent validity: timed function tests\n{'-'*66}")
    plot_fig5_tft(df)

    print(f"\n{'='*66}\n  Done. Figures 2-5 saved to {FIG_DIR}\n{'='*66}")



# ══════════════════════════════════════════════════════════════════════════════
#  TABLES (Table 1 diseases, Table 2 measures)
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = None   # runs/<run-name>/results/tables (set in __main__)



# ============================================================================
#  TABLE 1  —  demographics by diagnosis (participant + visit level)
# ============================================================================

# Diagnosis rows, in display order. "CTL" is the data value for the control
# group; it is shown as "TYP" (Typically Developing) in output.
T1_DIAG_ORDER = ["DM", "FSHD", "CMT", "DMD", "scDMD", "GNE", "CM", "SMA",
              "BMD", "LGMD", "SBMA", "ALS", "BM", "UNKNOWN"]

DIAG_FULL_NAMES = {
    "DM":      "Myotonic dystrophy (DM)",
    "FSHD":    "Facioscapulohumeral muscular dystrophy (FSHD)",
    "CMT":     "Charcot-Marie-Tooth disease (CMT)",
    "DMD":     "Duchenne muscular dystrophy (DMD)",
    "scDMD":     "Symptomatic carrier of DMD (scDMD)",
    "GNE":     "GNE myopathy (GNE)",
    "CM":      "Cystinosis myopathy (CM)",
    "SMA":     "Spinal muscular atrophy (SMA)",
    "BMD":     "Becker muscular dystrophy (BMD)",
    "LGMD":    "Limb-girdle muscular dystrophy (LGMD)",
    "SBMA": "Spinobulbar muscular atrophy (SBMA)",
    "ALS":     "Amyotrophic lateral sclerosis (ALS)",
    "BM": "Bethlem myopathy (BM)",
    "UNKNOWN": "Unknown",
}

CTL_VALUE   = "CTL"   # value stored in clinical_diagnosis for controls
CTL_DISPLAY = "CTL"   # how controls are labelled in the output table
SUMMARY_GROUPS = {"Total NMD", f"Total {CTL_DISPLAY}", "Total"}

# sex is normalized to a numeric code in build_table1 (0 = female, 1 = male),
# so the female count works whether the source stores "female"/"male" strings
# (full_demographics_table.csv) or the legacy 0/1 numeric code.
FEMALE_CODE = 0.0


# ── Build the table rows ─────────────────────────────────────────────────────────
def _make_row(label, subset):
    n = len(subset)
    if n == 0:
        return {"Group": label, "N": 0, "Female, n (%)": "0 (0.0%)"}
    n_female = int((subset["sex"] == FEMALE_CODE).sum())
    pct = 100.0 * n_female / n
    return {"Group": label, "N": n, "Female, n (%)": f"{n_female} ({pct:.1f}%)"}


def build_rows(data):
    """Per-group rows plus Total NMD / Total TYP / Total, for one dedup level."""
    rows = []
    for diag in T1_DIAG_ORDER:
        sub = data[data["clinical_diagnosis"] == diag]
        if len(sub) > 0:
            rows.append(_make_row(f"  {DIAG_FULL_NAMES.get(diag, diag)}", sub))

    nmd = data[data["clinical_diagnosis"] != CTL_VALUE]
    ctl = data[data["clinical_diagnosis"] == CTL_VALUE]
    rows.append(_make_row("Total NMD", nmd))
    rows.append(_make_row(f"Total {CTL_DISPLAY}", ctl))
    rows.append(_make_row("Total", data))
    return rows


def build_combined_table(demo):
    """
    Given a demographics dataframe (already restricted to a cohort), return the
    combined participant-LEFT / visit-RIGHT table plus the two totals.
    """
    visit_df       = demo.drop_duplicates(subset=["digbi_id", "date"])
    participant_df = demo.drop_duplicates(subset=["digbi_id"])

    rows_p = build_rows(participant_df)
    rows_v = build_rows(visit_df)

    combined = []
    for rp, rv in zip(rows_p, rows_v):
        assert rp["Group"] == rv["Group"], "Row mismatch between levels!"
        combined.append({
            "Group":    rp["Group"],
            "N_p":      rp["N"],
            "Female_p": rp["Female, n (%)"],
            "N_v":      rv["N"],
            "Female_v": rv["Female, n (%)"],
        })
    return pd.DataFrame(combined), len(participant_df), len(visit_df)


# ── Renderers ────────────────────────────────────────────────────────────────────
def save_csv(tbl, n_p, n_v, path):
    out = tbl.rename(columns={
        "N_p":      f"Participant n (total={n_p})",
        "Female_p": "Participant Female, n (%)",
        "N_v":      f"Visit n (total={n_v})",
        "Female_v": "Visit Female, n (%) [not shown]",
    })
    out.to_csv(path, index=False)
    print("  saved", path)


# ── Main ─────────────────────────────────────────────────────────────────────────
def build_table1():
    # The participant-info CSV IS the modeling cohort. Map its (de-identified)
    # key columns (subid/visit) and diagnosis onto the names the table builders
    # expect; subid/visit are used only as participant/visit dedup keys, so the
    # counts are identical to the pid/date version.
    demo = pd.read_csv(DEMO_CSV).rename(
        columns={"subid": "digbi_id", "visit": "date", "diag": "clinical_diagnosis"})
    demo["digbi_id"]           = demo["digbi_id"].astype(str).str.upper().str.strip()
    demo["date"]               = demo["date"].astype(str).str.strip()
    # Diagnosis codes are matched verbatim against T1_DIAG_ORDER (e.g. "scDMD"),
    # so strip whitespace but do NOT change case.
    demo["clinical_diagnosis"] = demo["clinical_diagnosis"].astype(str).str.strip()
    # Normalize sex to the numeric code (0 = female, 1 = male) so downstream
    # female counts work for both "female"/"male" strings and legacy 0/1 values.
    sx = demo["sex"].astype(str).str.strip().str.lower()
    demo["sex"] = np.where(sx.isin(["female", "f", "0", "0.0"]), 0.0, 1.0)

    tbl, n_p, n_v = build_combined_table(demo)
    print(f"\n=== modeling cohort: {n_p} participants / {n_v} visits ===")
    print(tbl.to_string(index=False))

    stem = os.path.join(OUT_DIR, "table1_diseases")
    save_csv(tbl, n_p, n_v, stem + ".csv")

# ============================================================================
#  TABLE 2  —  measure spreads, NMD vs CTL
# ============================================================================

# Grouped measures. Times (s), not speeds.
SECTIONS = [
    # (section title, summary format, [(column, label), ...])
    ("Demographic measures", "meansd", [
        ("age",        "Age (years)"),
        ("height",     "Height (m)"),
        ("weight",     "Weight (kg)"),
    ]),
    ("Functional measures", "medrange", [
        ("tft_10mwrt", "10 m run time (s)"),
        ("tft_10mwt",  "10 m walk time (s)"),
        ("tft_tug",    "TUG time (s)"),
        ("tft_5xsts",  "5xSTS time (s)"),
        ("brooke",     "Brooke score"),
        ("activ_logit","ACTIVLIM Score"),
    ]),
]
FMT_LABEL = {"meansd": "mean ± SD", "medrange": "median (range)"}
MEAS = [m for _, _, ms in SECTIONS for m in ms]   # flat list for the table rows


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_visit_level(path):
    """Return the visit-level dataframe (one row per visit) with group (NMD/CTL).
    Measures are NOT aggregated per participant — each visit is a data point."""
    df = pd.read_csv(path)
    df["_id"] = df["subid"].astype(str).str.upper().str.strip()
    df["grp"] = np.where(df["diag"].astype(str).str.upper().str.strip() == "CTL",
                         "CTL", "NMD")
    for c, _ in MEAS:
        df[c] = _num(df[c])
    return df


def _fmt_med(v):
    """median (min-max): robust summary for skewed / ceiling-bounded measures."""
    v = _num(v).dropna()
    if len(v) == 0:
        return "-"
    return f"{v.median():.1f} ({v.min():.1f}–{v.max():.1f})"


def _fmt_meansd(v):
    """mean +/- SD: for roughly-symmetric demographics (matches the intro)."""
    v = _num(v).dropna()
    if len(v) == 0:
        return "-"
    return f"{v.mean():.1f} ± {v.std():.1f}"


_FMT = {"meansd": _fmt_meansd, "medrange": _fmt_med}


def build_table2(pp):
    rows = []
    # (Female n (%) is reported in Table 1, so it is omitted here.)
    for section, fmt, meas in SECTIONS:
        rows.append({"Metric": section, "is_section": True,
                     "NMD": FMT_LABEL[fmt], "NMD n": "", "CTL": "", "CTL n": "",
                     "Total": "", "Total n": ""})
        for c, lab in meas:
            r = {"Metric": lab, "is_section": False}
            for g in ["NMD", "CTL"]:
                r[g] = _FMT[fmt](pp.loc[pp.grp == g, c])
                r[f"{g} n"] = int(_num(pp.loc[pp.grp == g, c]).notna().sum())
            # Total = all visits (NMD + CTL combined), visit level.
            r["Total"]   = _FMT[fmt](pp[c])
            r["Total n"] = int(_num(pp[c]).notna().sum())
            rows.append(r)
    tbl = pd.DataFrame(rows)[["Metric", "is_section",
                              "NMD", "NMD n", "CTL", "CTL n", "Total", "Total n"]]
    return tbl


def build_table2_outputs():
    data = load_visit_level(DEMO_CSV)
    nN = data[data.grp == "NMD"]._id.nunique()
    nC = data[data.grp == "CTL"]._id.nunique()
    tbl = build_table2(data)
    # Emit section header rows (empty NMD/CTL) followed by their measures,
    # indented, so the hierarchy is carried in the CSV itself.
    out_rows = []
    for _, r in tbl.iterrows():
        if r["is_section"]:
            out_rows.append({"Metric": r["Metric"], "NMD": "", "CTL": "", "Total": ""})
        else:
            out_rows.append({"Metric": "    " + str(r["Metric"]),
                             "NMD": r["NMD"], "CTL": r["CTL"], "Total": r["Total"]})
    csv_tbl = pd.DataFrame(out_rows, columns=["Metric", "NMD", "CTL", "Total"])
    csv_tbl.to_csv(os.path.join(OUT_DIR, "table2_measures.csv"), index=False)
    print(f"Table 2 (NMD n={nN}, CTL n={nC})\n")
    print(csv_tbl.to_string(index=False))
    print("\nsaved table2_measures.csv")


# ── Main ─────────────────────────────────────────────────────────────────────────
def generate_tables():
    print("#" * 60 + "\n#  TABLE 1 — cohort composition by diagnosis\n" + "#" * 60)
    build_table1()
    print("\n" + "#" * 60 + "\n#  TABLE 2 — measure spreads (NMD vs CTL)\n" + "#" * 60)
    build_table2_outputs()



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate the paper's main-text figures (2-5) AND tables "
                    "(Table 1 diseases, Table 2 measures), and print the figure "
                    "statistics. READ-ONLY: consumes a run's prediction CSVs + the "
                    "participant-info CSV; runs no model inference.",
        epilog=(
            "inputs:\n"
            "  --demographics  participant-info CSV (subid, visit, diag, sex, split,\n"
            "                  + the clinical/measure columns used by figs 4-5 and\n"
            "                  Table 2).\n"
            "  --run-name      Name of a run under runs/. Reads\n"
            "                    runs/<run-name>/csvs/test_severity_transformer_<run-name>.csv\n"
            "                    runs/<run-name>/csvs/oof_validation_severity_transformer.csv\n"
            "                    runs/<run-name>/models/cv_preds_transformer/\n"
            "                  and writes\n"
            "                    runs/<run-name>/results/figures/   (fig2-5)\n"
            "                    runs/<run-name>/results/tables/    (table1_diseases, table2_measures)\n"
            "                  Produce the CSVs first with inference_transformer.py.\n\n"
            "example:\n"
            "  python make_figures_tables.py \\\n"
            "      --demographics ~/nmd_opencap_participant_info.csv --run-name pretrained\n"))
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV (subid, visit, diag, sex, split, + measures)")
    ap.add_argument("--run-name", required=True, metavar="NAME",
                    help="run identifier; reads runs/<run-name>/{csvs,models}, writes "
                         "runs/<run-name>/results/{figures,tables}/")
    args = ap.parse_args()

    REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR = os.path.join(REPO, "runs", args.run_name)

    DEMO_CSV    = os.path.expanduser(args.demographics)
    MODEL_DIR   = os.path.join(RUN_DIR, "models", "cv_models_transformer")
    CV_PRED_DIR = os.path.join(RUN_DIR, "models", "cv_preds_transformer")
    TEST_CSV    = os.path.join(RUN_DIR, "csvs", f"test_severity_transformer_{args.run_name}.csv")
    OOF_CSV     = os.path.join(RUN_DIR, "csvs", "oof_validation_severity_transformer.csv")
    FIG_DIR     = os.path.join(RUN_DIR, "results", "figures")
    OUT_DIR     = os.path.join(RUN_DIR, "results", "tables")

    # Graceful validation: fail with an actionable message, not a traceback.
    if not os.path.isfile(DEMO_CSV):
        ap.error(f"--demographics: file not found: {DEMO_CSV}\n"
                 "Provide the participant-info CSV (columns: subid, visit, diag, sex, split, ...).")
    if not os.path.isdir(RUN_DIR):
        ap.error(f"--run-name: no run directory at {RUN_DIR}\n"
                 "Expected runs/<run-name>/ with a csvs/ subfolder. Train + run "
                 "inference for this run first (train_transformer.py, then "
                 "inference_transformer.py).")
    for label, path in (("test", TEST_CSV), ("OOF", OOF_CSV)):
        if not os.path.isfile(path):
            ap.error(f"--run-name: missing {label} predictions: {path}\n"
                     "Generate this run's CSVs first:\n"
                     f"  python inference_transformer.py --dataset <DATASET> "
                     f"--demographics {args.demographics} --run-name {args.run_name}")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    generate_figures()
    generate_tables()

    # Render the main tables to LaTeX/PDF/PNG automatically (no separate step).
    import create_latex_tables as clt
    clt.render_main_tables(OUT_DIR)

    # View the figures + rendered table images one at a time: each opens in a
    # window and CLOSING it advances to the next (the paper1_code plt.show() flow).
    to_view  = [os.path.join(FIG_DIR, f"{n}.png") for n in
                ("fig2_performance", "fig3_severity", "fig4_activlim", "fig5_tft")]
    to_view += [os.path.join(OUT_DIR, "latex_tables", f"{n}.png") for n in
                ("table1_diseases", "table2_measures")]
    view_sequentially([p for p in to_view if os.path.isfile(p)])
