"""
make_supplementary_figures_tables.py
------------------------------------
Generate the supplementary FIGURE (per-measure distribution histograms) and the
three supplementary TABLES (classification, convergent validity, kinematic
parameters) for the pretrained run.

Two ways to run (choose exactly one of --dataset / --skip-inference):
    # 1) run the models on the dataset, then build the supplementary outputs
    python make_supplementary_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset \\
                                                --demographics /path/to/nmd_opencap_participant_info.csv
    # 2) skip inference; build straight from the shipped precomputed CSVs
    python make_supplementary_figures_tables.py --skip-inference \\
                                                --demographics /path/to/nmd_opencap_participant_info.csv

Mode 1 calls inference_models.py to write the prediction CSVs to
runs/pretrained/severity_csvs/; mode 2 reads them from
runs/pretrained/severity_csvs/precomputed/. Writes:
    runs/pretrained/results/supp_figures/  supplementary_fig1_distributions.{png,pdf}
    runs/pretrained/results/supp_tables/   supplementary_table{1,2,3}_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless: build figures to files, never open windows
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score)


def view_sequentially(paths):
    """Show each image in a window; CLOSING it advances to the next (blocking
    plt.show() per image, like the old paper1_code flow). Falls back to opening
    all in the OS default viewer if no interactive matplotlib backend exists."""
    import sys, subprocess
    import matplotlib.image as mpimg
    if not paths:
        return
    plt.close("all")   # close the Agg figures so switch_backend doesn't warn
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


# Paths are set from the CLI in __main__.
DEMO = None    # participant-info CSV (subid, visit, diag, ...)
FIG_DIR = None  # runs/<run-name>/results/supp_figures

# Same nine measures as Table 2, in the same order.
MEAS = [
    ("age",         "Age (years)"),
    ("height",      "Height (m)"),
    ("weight",      "Weight (kg)"),
    ("tft_10mwrt",  "10 m run time (s)"),
    ("tft_10mwt",   "10 m walk time (s)"),
    ("tft_tug",     "TUG time (s)"),
    ("tft_5xsts",   "5xSTS time (s)"),
    ("brooke",      "Brooke score"),
    ("activ_logit", "ACTIVLIM Score"),
]
COLORS = {"NMD": "#d1495b", "CTL": "#64748b"}


# ── Supp. Fig 1 — measure distribution histograms ─────────────────────────────

def load_visit_level(path):
    """One row per visit, with group (NMD/CTL). Measures are NOT aggregated."""
    df = pd.read_csv(path)
    df["grp"] = np.where(df["diag"].astype(str).str.upper().str.strip() == "CTL",
                         "CTL", "NMD")
    for c, _ in MEAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_supp_fig1_distributions():
    pp = load_visit_level(DEMO)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=200)
    axes = axes.ravel()
    for i, (c, lab) in enumerate(MEAS):
        ax = axes[i]
        nmd = pp.loc[pp.grp == "NMD", c].dropna()
        ctl = pp.loc[pp.grp == "CTL", c].dropna()
        both = pd.concat([nmd, ctl])
        lo, hi = np.nanmin(both), np.nanmax(both)
        nbins = 16 if c == "brooke" else 22
        bins = np.linspace(lo, hi, nbins)
        ax.hist(nmd, bins=bins, density=False, alpha=0.6, color=COLORS["NMD"],
                label=f"NMD ({len(nmd)} visits)", edgecolor="white", linewidth=0.3)
        ax.hist(ctl, bins=bins, density=False, alpha=0.6, color=COLORS["CTL"],
                label=f"CTL ({len(ctl)} visits)", edgecolor="white", linewidth=0.3)
        ax.set_title(lab, fontsize=11, color="0.2", pad=4)
        ax.set_ylabel("Number of visits" if i % 3 == 0 else "", fontsize=9, color="0.4")
        ax.legend(frameon=False, fontsize=7.5, loc="best")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle("Distribution of visit-level demographic and functional measures",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"supplementary_fig1_distributions.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", os.path.join(FIG_DIR, "supplementary_fig1_distributions.png"))
    plt.close(fig)


def generate_supp_figures():
    print("-- Supplementary Fig 1: measure distribution histograms --")
    make_supp_fig1_distributions()



# ══════════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY TABLES (Transformer vs SVM vs MLP + kinematic parameters)
# ══════════════════════════════════════════════════════════════════════════════

OUTDIR = None   # runs/<run-name>/results/supp_tables (set in __main__)

MODELS = ["Transformer", "SVM", "MLP"]
THR    = 0.5   # decision threshold for bACC (common rule across models)

# All three models' predictions are the run's de-identified test CSVs (keyed
# subid/visit); clinical data comes from participant_info. Both are located by
# configure() below. bACC uses the fixed 0.5 threshold above.


# ══════════════════════════════════════════════════════════════════════════════
#  Shared load / align + paired bootstrap (formerly compare_model_stats.py)
# ══════════════════════════════════════════════════════════════════════════════

# Model test-prediction CSVs + clinical source, set per run by configure().
RUNS = {}          # {"Transformer": csv, "SVM": csv, "MLP": csv}
REDCAP_CSV = None  # participant-info CSV (clinical measures, keyed subid/visit)

# Clinical measures for the severity-score comparison: (column, label, flip).
CLINICAL = [
    ("activ_logit",  "ACTIVLIM",      True),
    ("tft_tug",      "TUG-cone",      False),
    ("tft_5xsts",    "5xSTS",         False),
    ("tft_10mwt",    "10m walk time", False),
    ("tft_10mwrt",   "10m run time",  False),
]

K       = 5
N_BOOT  = 5000
SEED    = 17
# Resampling unit for the paired bootstrap. False -> resample VISITS (the unit
# of analysis, matching the main-text figures); True -> resample PARTICIPANTS.
CLUSTER_BY_PARTICIPANT = False


def configure(run_dir, demographics, csv_dir=None):
    """Point RUNS/REDCAP_CSV at a run's prediction CSVs + the demographics CSV.
    RUNS keys the three models to <csv_dir>/test_severity_<model>_<run>.csv;
    csv_dir defaults to runs/<run>/severity_csvs (pass severity_csvs/precomputed
    for the shipped CSVs)."""
    run_name = os.path.basename(os.path.normpath(run_dir))
    csvs = csv_dir if csv_dir else os.path.join(run_dir, "severity_csvs")
    global RUNS, REDCAP_CSV
    RUNS = {
        "Transformer": os.path.join(csvs, f"test_severity_transformer_{run_name}.csv"),
        "SVM":         os.path.join(csvs, f"test_severity_svm_{run_name}.csv"),
        "MLP":         os.path.join(csvs, f"test_severity_mlp_{run_name}.csv"),
    }
    REDCAP_CSV = os.path.expanduser(demographics)
    return RUNS, REDCAP_CSV


def clustered_paired_boot(fn, ids, n_boot=N_BOOT, seed=SEED, cluster=None):
    """Bootstrap the DIFFERENCE fn(idx) between two models. PAIRED: each draw is
    ONE index set evaluated for both models, so correlated errors cancel in the
    difference. Resampling unit follows CLUSTER_BY_PARTICIPANT (or `cluster`):
    visits by default, participants if enabled. Returns
    (point estimate, lo, hi, share of draws with diff > 0)."""
    cluster = CLUSTER_BY_PARTICIPANT if cluster is None else cluster
    ids = np.asarray(ids)
    n   = len(ids)
    rng = np.random.default_rng(seed)
    point = fn(np.arange(n))
    draws = []
    if cluster:
        uniq  = np.unique(ids)
        where = {u: np.flatnonzero(ids == u) for u in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            idx  = np.concatenate([where[u] for u in pick])
            d = fn(idx)
            if d is not None and np.isfinite(d):
                draws.append(d)
    else:
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            d = fn(idx)
            if d is not None and np.isfinite(d):
                draws.append(d)
    draws = np.asarray(draws)
    if draws.size == 0:
        return point, np.nan, np.nan, np.nan
    return (point, float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float((draws > 0).mean()))


def load_all():
    """Load the three models' test CSVs, align them to the shared visits (keyed
    subid/visit), and return (aligned, y, ids, order)."""
    dfs = {}
    for name, path in RUNS.items():
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        d = pd.read_csv(path)
        d["subid"] = d["subid"].astype(str).str.strip()
        d["visit"] = d["visit"].astype(int)
        dfs[name] = d
    if len(dfs) < 2:
        raise SystemExit("Need at least two runs to compare.")
    key = ["subid", "visit"]
    common = None
    for d in dfs.values():
        s = set(map(tuple, d[key].values))
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f"Shared test visits across {len(dfs)} models: {len(common)}")
    aligned = {}
    order = pd.DataFrame(common, columns=key)
    for name, d in dfs.items():
        aligned[name] = order.merge(d, on=key, how="left").reset_index(drop=True)
    ref = aligned[list(aligned)[0]]
    y   = ref["true_label"].astype(int).to_numpy()
    ids = ref["subid"].to_numpy()
    print(f"  participants={len(np.unique(ids))}  "
          f"NMD={int((y==1).sum())} visits  CTL={int((y==0).sum())} visits")
    return aligned, y, ids, order


# ══════════════════════════════════════════════════════════════════════════════
#  Supp. Table 1 & 2 data (Transformer vs SVM vs MLP)
# ══════════════════════════════════════════════════════════════════════════════

def _metric(name, yy, p):
    if name == "AUROC":
        return roc_auc_score(yy, p)
    if name == "AUPRC":
        return average_precision_score(yy, p)
    return balanced_accuracy_score(yy, (p >= THR).astype(int))


def compute_classification(aligned, y, ids):
    # Section 1 — per-fold metric, mean ± SD across the 5 folds
    secA = []
    for m in ["AUROC", "AUPRC", "bACC"]:
        row = []
        for name in MODELS:
            d = aligned[name]
            vals = np.array([_metric(m, y, d[f"prob_disease_f{k}"].astype(float).to_numpy())
                             for k in range(1, K + 1)
                             if f"prob_disease_f{k}" in d.columns])
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")
        secA.append((m, row))

    # Section 2 — difference in the ENSEMBLE metric vs the Transformer
    ens = {n: aligned[n]["prob_disease_mean"].astype(float).to_numpy() for n in MODELS}
    secB = []
    for m in ["AUROC", "AUPRC", "bACC"]:
        row = ["reference"]
        for name in ["SVM", "MLP"]:
            def f(idx, pm=ens[name], pt=ens["Transformer"], m=m):
                yy = y[idx]
                if np.unique(yy).size < 2:
                    return None
                return _metric(m, yy, pm[idx]) - _metric(m, yy, pt[idx])
            d, lo, hi, _ = clustered_paired_boot(f, ids)
            star = "*" if lo * hi > 0 else ""   # CI excludes 0 -> significant
            row.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}")
        secB.append((m, row))

    return [("Test-set performance  (mean ± SD across 5 CV fold models)", secA),
            ("Difference vs Transformer  [95% CI]", secB)]


def compute_convergent(aligned, ids):
    # All clinical measures (TFTs + ACTIVLIM Rasch logit) come straight from the
    # public participant-info table, keyed on (subid, visit). No crosswalk / PII.
    rc = pd.read_csv(REDCAP_CSV)
    rc["subid"] = rc["subid"].astype(str).str.strip()
    rc["visit"] = rc["visit"].astype(int)
    clinical = CLINICAL

    sev = {n: aligned[n][["subid", "visit", "logit_severity_mean"]].merge(
              rc, on=["subid", "visit"], how="left") for n in MODELS}

    secA, secB = [], []
    for col, label, flip in clinical:
        if col not in rc.columns:
            continue
        sgn = -1.0 if flip else 1.0
        ok = sev["Transformer"][col].notna().to_numpy()
        n = int(ok.sum())

        rowA = []
        for name in MODELS:
            s = sev[name]
            r = stats.spearmanr(s.loc[ok, "logit_severity_mean"], s.loc[ok, col]).statistic
            rowA.append(f"{sgn * r:.2f}")
        secA.append((f"{label}  (n={n})", rowA))

        rowB, ids_ok = ["—"], ids[ok]
        for name in ["SVM", "MLP"]:
            sa = sev[name].loc[ok, ["logit_severity_mean", col]].to_numpy()
            sb = sev["Transformer"].loc[ok, ["logit_severity_mean", col]].to_numpy()
            def f(idx, sa=sa, sb=sb, sgn=sgn):
                if np.unique(sa[idx, 1]).size < 3:
                    return None
                ra = stats.spearmanr(sa[idx, 0], sa[idx, 1]).statistic
                rb = stats.spearmanr(sb[idx, 0], sb[idx, 1]).statistic
                return sgn * (ra - rb)
            d, lo, hi, _ = clustered_paired_boot(f, ids_ok)
            star = "*" if lo * hi > 0 else ""   # CI excludes 0 -> significant
            rowB.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}")
        secB.append((label, rowB))

    return [("Spearman ρ with severity score (sign aligned: higher score = worse function)",
             secA),
            ("Δρ vs Transformer  [95% CI]", secB)]


def save_comparison_csv(sections, out_path):
    """Two-section comparison table (model columns) -> CSV."""
    csv_rows = []
    for sec_title, rows in sections:
        for label, vals in rows:
            csv_rows.append([sec_title, label] + list(vals))
    pd.DataFrame(csv_rows, columns=["Section", "Measure"] + MODELS).to_csv(
        out_path, index=False)
    print("saved", out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Supp. Table 3 — kinematic parameters
# ══════════════════════════════════════════════════════════════════════════════

# (opensim_coordinate, human-readable name, segment). The opensim_coordinate is
# checked against dataset.EXPECTED_COLUMNS so the list can't drift.
KINEMATIC = [
    ("pelvis_tilt",      "Pelvis tilt",                         "Pelvis"),
    ("pelvis_list",      "Pelvis list",                         "Pelvis"),
    ("pelvis_rotation",  "Pelvis rotation",                     "Pelvis"),
    ("pelvis_tx",        "Pelvis anterior/posterior translation", "Pelvis"),
    ("pelvis_ty",        "Pelvis vertical translation",         "Pelvis"),
    ("pelvis_tz",        "Pelvis mediolateral translation",     "Pelvis"),
    ("hip_flexion_r",    "Hip flexion (R)",                     "Lower limb"),
    ("hip_adduction_r",  "Hip adduction (R)",                   "Lower limb"),
    ("hip_rotation_r",   "Hip rotation (R)",                    "Lower limb"),
    ("knee_angle_r",     "Knee flexion (R)",                    "Lower limb"),
    ("ankle_angle_r",    "Ankle dorsiflexion (R)",              "Lower limb"),
    ("subtalar_angle_r", "Subtalar angle (R)",                  "Lower limb"),
    ("mtp_angle_r",      "MTP angle (R)",                       "Lower limb"),
    ("hip_flexion_l",    "Hip flexion (L)",                     "Lower limb"),
    ("hip_adduction_l",  "Hip adduction (L)",                   "Lower limb"),
    ("hip_rotation_l",   "Hip rotation (L)",                    "Lower limb"),
    ("knee_angle_l",     "Knee flexion (L)",                    "Lower limb"),
    ("ankle_angle_l",    "Ankle dorsiflexion (L)",              "Lower limb"),
    ("subtalar_angle_l", "Subtalar angle (L)",                  "Lower limb"),
    ("mtp_angle_l",      "MTP angle (L)",                       "Lower limb"),
    ("lumbar_extension", "Lumbar extension",                    "Trunk"),
    ("lumbar_bending",   "Lumbar bending",                      "Trunk"),
    ("lumbar_rotation",  "Lumbar rotation",                     "Trunk"),
    ("arm_flex_r",       "Arm flexion (R)",                     "Upper limb"),
    ("arm_add_r",        "Arm adduction (R)",                   "Upper limb"),
    ("arm_rot_r",        "Arm rotation (R)",                    "Upper limb"),
    ("elbow_flex_r",     "Elbow flexion (R)",                   "Upper limb"),
    ("pro_sup_r",        "Forearm pronation (R)",               "Upper limb"),
    ("arm_flex_l",       "Arm flexion (L)",                     "Upper limb"),
    ("arm_add_l",        "Arm adduction (L)",                   "Upper limb"),
    ("arm_rot_l",        "Arm rotation (L)",                    "Upper limb"),
    ("elbow_flex_l",     "Elbow flexion (L)",                   "Upper limb"),
    ("pro_sup_l",        "Forearm pronation (L)",               "Upper limb"),
]


def save_kinematic_csv(rows, out_path):
    # assert the list matches the model input
    try:
        from dataset import EXPECTED_COLUMNS
        listed = [code for code, _r, _s in rows]
        miss, extra = set(EXPECTED_COLUMNS) - set(listed), set(listed) - set(EXPECTED_COLUMNS)
        if miss or extra:
            print(f"  [warn] kinematic table out of sync: missing={sorted(miss)} extra={sorted(extra)}")
    except Exception as e:
        print(f"  [warn] could not verify against dataset.EXPECTED_COLUMNS: {e}")
    pd.DataFrame([(r[1], r[2]) for r in rows],
                 columns=["Kinematic Parameter", "Segment"]).to_csv(out_path, index=False)
    print("saved", out_path)


def generate_supp_tables():
    aligned, y, ids, order = load_all()

    print("\nSupp. Table 1 — classification (Transformer vs SVM vs MLP)")
    save_comparison_csv(
        compute_classification(aligned, y, ids),
        os.path.join(OUTDIR, "supplementary_table1_model_classification.csv"))

    print("\nSupp. Table 2 — convergent validity (Transformer vs SVM vs MLP)")
    save_comparison_csv(
        compute_convergent(aligned, ids),
        os.path.join(OUTDIR, "supplementary_table2_model_convergent_validity.csv"))

    print(f"\nSupp. Table 3 — kinematic parameters ({len(KINEMATIC)} model inputs)")
    save_kinematic_csv(
        KINEMATIC,
        os.path.join(OUTDIR, "supplementary_table3_kinematic_parameters.csv"))



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate the supplementary figure (measure distributions) and the "
                    "three supplementary tables for the pretrained run.",
        epilog=(
            "two ways to run (choose exactly one of --dataset / --skip-inference):\n"
            "  1) inference:      --dataset <DIR> --demographics <CSV>\n"
            "       runs the trained models on the dataset (via inference_models.py),\n"
            "       saves the prediction CSVs to runs/pretrained/severity_csvs/,\n"
            "       then builds the supplementary figure + tables.\n"
            "  2) skip inference: --skip-inference --demographics <CSV>\n"
            "       skips the models and builds them straight from the shipped CSVs in\n"
            "       runs/pretrained/severity_csvs/precomputed/.\n\n"
            "inputs:\n"
            "  --dataset       dataset root (.../datadir/Neuromuscular_OpenCap_Dataset)\n"
            "  --demographics  participant-info CSV (diag + measures / clinical, keyed subid/visit)\n"
            "  --skip-inference use the precomputed CSVs instead of running the models\n\n"
            "examples:\n"
            "  python make_supplementary_figures_tables.py --dataset datadir/Neuromuscular_OpenCap_Dataset \\\n"
            "      --demographics datadir/nmd_opencap_participant_info.csv\n"
            "  python make_supplementary_figures_tables.py --skip-inference \\\n"
            "      --demographics datadir/nmd_opencap_participant_info.csv\n"))
    ap.add_argument("--demographics", required=True, metavar="CSV",
                    help="participant-info CSV (diag + measures / clinical, keyed subid/visit)")
    ap.add_argument("--dataset", metavar="DIR",
                    help="dataset root (.../datadir/Neuromuscular_OpenCap_Dataset); "
                         "runs inference to generate the prediction CSVs")
    ap.add_argument("--skip-inference", action="store_true",
                    help="skip inference and use the precomputed CSVs in "
                         "runs/pretrained/severity_csvs/precomputed/")
    args = ap.parse_args()

    # Exactly one of --dataset / --skip-inference.
    if bool(args.dataset) == bool(args.skip_inference):
        ap.error("provide exactly one of:\n"
                 "  --dataset <DIR> --demographics <CSV>     (run inference), or\n"
                 "  --skip-inference --demographics <CSV>    (use precomputed CSVs)")

    REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUN_DIR = os.path.join(REPO, "runs", "pretrained")
    DEMO    = os.path.expanduser(args.demographics)
    FIG_DIR = os.path.join(RUN_DIR, "results", "supp_figures")
    OUTDIR  = os.path.join(RUN_DIR, "results", "supp_tables")

    if not os.path.isfile(DEMO):
        ap.error(f"--demographics: file not found: {DEMO}\n"
                 "Provide the participant-info CSV (diag + measures; clinical measures keyed subid/visit).")

    # Prediction CSVs: run inference into severity_csvs/, or use severity_csvs/precomputed/.
    if args.skip_inference:
        CSV_SRC = os.path.join(RUN_DIR, "severity_csvs", "precomputed")
    else:
        import inference_models
        CSV_SRC = inference_models.run_inference(args.dataset, DEMO, RUN_DIR)

    configure(RUN_DIR, DEMO, csv_dir=CSV_SRC)
    missing = [n for n, p in RUNS.items() if not os.path.isfile(p)]
    if missing:
        ap.error(f"missing test-prediction CSVs for {', '.join(missing)} under {CSV_SRC}\n" + (
            "Restore the shipped CSVs, or run with --dataset <DIR> to regenerate them."
            if args.skip_inference else "Inference did not produce the expected CSVs."))

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    generate_supp_figures()
    generate_supp_tables()

    # Render the supplementary tables to LaTeX/PDF/PNG automatically (no separate step).
    import create_latex_tables as clt
    clt.render_supp_tables(OUTDIR)

    # View the supplementary figure + rendered supp-table images one at a time:
    # each opens in a window and CLOSING it advances to the next.
    to_view  = [os.path.join(FIG_DIR, "supplementary_fig1_distributions.png")]
    to_view += [os.path.join(OUTDIR, "latex_tables", f"{n}.png") for n in
                ("supp_table1_classification", "supp_table2_convergent", "supp_table3_kinematic")]
    view_sequentially([p for p in to_view if os.path.isfile(p)])
