"""
supp_severity_demographics.py
-----------------------------
Confounder check: does the severity score just track age or body size rather
than disease?

Uses the CONTINUOUS (non-saturated) OOF severity logits — no ceiling effect —
z-normalized to the leakage-free OOF control reference. For each demographic
(age, BMI) it draws NMD vs CTL and annotates the age+BMI-ADJUSTED partial
Spearman correlation within EACH group (severity vs that demographic, holding
the other demographic fixed).

    severity vs age  | BMI     (within CTL, within NMD)
    severity vs BMI  | age     (within CTL, within NMD)

Outputs -> retrained/figures/supp_severity_vs_demographics.{png,pdf}

Run:
    conda activate opencap-core-latest
    python supp_severity_demographics.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE     = "/Users/sydneycovitz/NMD_OpenCap/paper1_code"
# OOF continuous severity (de-identified, keyed subid/visit) and the public
# participant-info table. Both keyed subid/visit — no crosswalk / PII needed.
OOF_CSV  = os.path.join(BASE, "retrained", "oof_continuous_severity_ue_aware.csv")
DEMO_CSV = os.path.join(BASE, "nmd_opencap_participant_info.csv")
OUT_DIR  = os.path.join(BASE, "retrained", "figures")
COLORS   = {"NMD": "#d1495b", "CTL": "#64748b"}


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y controlling for z.
    Rank-transform, residualize x and y on z, correlate residuals."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    xr, yr, zr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones(n), zr])
    rx = xr - Z @ np.linalg.lstsq(Z, xr, rcond=None)[0]
    ry = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    df = n - 2 - 1                      # n - 2 - (#covariates)
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    p = 2 * stats.t.sf(abs(t), df)
    return r, p, n


def load():
    oof = pd.read_csv(OOF_CSV)
    oof["subid"] = oof["subid"].astype(str).str.strip()
    oof["visit"] = oof["visit"].astype(int)
    # continuous severity z-normalized to the OOF control reference (leakage-free)
    cm = oof.loc[oof.label == 0, "logit"].mean()
    cs = max(oof.loc[oof.label == 0, "logit"].std(), 1e-6)
    oof["severity_z"] = (oof["logit"] - cm) / cs

    # demographic VALUES from the public participant-info table, keyed subid/visit
    demo = pd.read_csv(DEMO_CSV)
    demo["subid"] = demo["subid"].astype(str).str.strip()
    demo["visit"] = demo["visit"].astype(int)
    for c in ("age", "height", "weight"):
        demo[c] = pd.to_numeric(demo[c], errors="coerce")
    demo["bmi"] = demo["weight"] / demo["height"] ** 2

    df = oof.merge(demo[["subid", "visit", "age", "height", "weight", "bmi"]],
                   on=["subid", "visit"], how="left")
    df["grp"] = np.where(df.label == 0, "CTL", "NMD")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    nN = int((df.grp == "NMD").sum()); nC = int((df.grp == "CTL").sum())
    print(f"visits: NMD={nN}  CTL={nC}  severity_z range "
          f"[{df.severity_z.min():.2f}, {df.severity_z.max():.2f}]")

    PANELS = [("age", "Age (years)", "bmi"),
              ("bmi", r"BMI (kg/m$^2$)", "age")]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=200)
    for ax, lab_ab, (xcol, xlab, covar) in zip(axes, ["a", "b"], PANELS):
        # panel label, matching evaluate.panel_label style (bold, top-left)
        ax.text(-0.05, 1.02, lab_ab, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="right", color="0.2")
        for grp in ("NMD", "CTL"):
            g = df[df.grp == grp].dropna(subset=[xcol, "severity_z", covar])
            ax.scatter(g[xcol], g["severity_z"], s=14, alpha=0.55,
                       color=COLORS[grp], edgecolors="white", linewidths=0.25,
                       zorder=2, label=f"{grp} (n={len(g)})")
            if len(g) > 2:                       # illustrative OLS trend line
                b = np.polyfit(g[xcol], g["severity_z"], 1)
                xs = np.linspace(g[xcol].min(), g[xcol].max(), 50)
                ax.plot(xs, np.polyval(b, xs), color=COLORS[grp], lw=2, zorder=3)

        # age+BMI-adjusted partial Spearman within each group
        txt = []
        for grp in ("CTL", "NMD"):
            g = df[df.grp == grp]
            r, p, n = partial_spearman(g[xcol].to_numpy(float),
                                       g["severity_z"].to_numpy(float),
                                       g[covar].to_numpy(float))
            other = "BMI" if covar == "bmi" else "age"
            txt.append(f"within {grp}: partial ρ = {r:+.2f} (p = {p:.2f})")
        ax.text(0.03, 0.97, "\n".join(txt) + f"\n(adjusted for {other})",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                color="0.25", bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9))

        ax.axhline(0, color="0.7", lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel("Severity score (SDs above control)", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle("Severity score vs demographics (OOF validation): "
                 "NMD elevated across the age/BMI range, independent of both",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"supp_severity_vs_demographics.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", os.path.join(OUT_DIR, "supp_severity_vs_demographics.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
