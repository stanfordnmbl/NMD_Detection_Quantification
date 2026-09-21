"""
create_latex_tables.py
----------------------
Render the paper tables to LaTeX. This is the ONLY place LaTeX is generated —
the make_*_tables scripts write the CSV data, then import this module and call
its render functions automatically, so the user never runs this file directly:

    render_main_tables(tables_dir)       # Table 1 diseases, Table 2 measures
        -> called by make_figures_tables.py
    render_supp_tables(supp_tables_dir)  # Supp. Tables 1-3
        -> called by make_supplementary_figures_tables.py

Each render function reads the table CSVs from the given directory and writes,
into <that dir>/latex_tables/:
    <name>.tex   -- standalone, compilable table (booktabs)
    <name>.pdf   -- compiled with tectonic
    <name>.png   -- rasterized from the PDF with pdftoppm (300 dpi)
plus a combined manuscript bundle (tables_latex.tex / supp_tables_latex.tex).

Requires on PATH: tectonic, pdftoppm (poppler) for PDF/PNG; without them only
.tex is written.  (brew install tectonic poppler)
"""

import os
import shutil
import subprocess

import pandas as pd

STANDALONE_PACKAGES = (
    r"\usepackage{booktabs}" "\n"
    r"\usepackage{multirow}" "\n"
    r"\usepackage{makecell}" "\n"
    r"\usepackage{threeparttable}" "\n"
    r"\usepackage[T1]{fontenc}" "\n"
)


# ── small formatting helpers ──────────────────────────────────────────────────

def esc(s):
    """Escape LaTeX specials in a plain-text string."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("–", "--")]:
        s = s.replace(a, b)
    return s


def fmt_text(s):
    """Escape a plain-text label and turn unicode symbols into portable math,
    so the tables compile without a unicode font (pdflatex- and xelatex-safe)."""
    s = str(s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("–", "--")]:
        s = s.replace(a, b)
    s = s.replace("Δρ", r"$\Delta\rho$").replace("Δ", r"$\Delta$")
    s = s.replace("ρ", r"$\rho$").replace("±", r"$\pm$").replace("×", r"$\times$")
    return s


def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def fmt_value(v):
    """Format a numeric / CI / mean±SD data cell for a model-comparison table."""
    v = str(v).strip()
    if v in ("reference",):
        return "ref."
    if v in ("—", "-", "", "nan"):
        return "---"
    star = ""
    if v.endswith("*"):                            # significance mark (CI excludes 0)
        star = r"$^{*}$"; v = v[:-1].strip()
    if "±" in v:
        return "$" + v.replace("±", r"\pm").replace("–", "--") + "$" + star
    if "[" in v:                                   # e.g.  -0.10 [-0.22, +0.01]
        return "$" + v.replace("[", r"\;[").replace("–", "--") + "$" + star
    if "(" in v:                                   # e.g.  6.0 (2.8–24.4)
        return v.replace("–", "--") + star
    return ("$" + v + "$" if _is_num(v) else esc(v)) + star


def fmt_label(s):
    """Tidy a measure / metric label (n= subscript, 5xSTS, 10 m)."""
    import re
    s = " ".join(str(s).split())                   # collapse double spaces
    s = s.replace("5xSTS", r"5$\times$STS")
    s = re.sub(r"\(n=(\d+)\)", r"($n=\1$)", s)      # (n=NN) -> ($n=NN$)
    s = s.replace("10m ", r"10\,m ").replace("10 m ", r"10\,m ")
    return s


def _standalone(body, packages=STANDALONE_PACKAGES):
    return ("\\documentclass[border=6pt]{standalone}\n"
            + packages +
            "\\begin{document}\n" + body + "\n\\end{document}\n")


# ── table bodies (return inner tabular / threeparttable LaTeX) ─────────────────

def table1_body():
    df = pd.read_csv(TABLE1_CSV)
    grp_col = df.columns[0]
    part_n  = df.columns[1]      # "Participant n (total=415)"
    part_f  = df.columns[2]      # "Participant Female, n (%)"
    visit_n = df.columns[3]      # "Visit n (total=675)"
    rule_above = {"Total CTL", "Total"}   # horizontal line before these rows
    bold_rows  = {"Total"}                 # bold only the grand total

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"\textbf{Group} & \makecell[r]{\textbf{Participants}\\\textbf{(n)}} & "
             r"\makecell[r]{\textbf{Female}\\\textbf{n (\%)}} & "
             r"\makecell[r]{\textbf{Visits}\\\textbf{(n)}} \\",
             r"\midrule"]
    for _, r in df.iterrows():
        raw      = str(r[grp_col])
        g        = raw.strip()
        indented = raw != raw.lstrip()     # subgroup rows carry leading spaces
        label    = (r"\hspace{1.5em}" + esc(g)) if indented else esc(g)
        if g in rule_above:
            lines.append(r"\midrule")
        cells = [label, str(r[part_n]), esc(r[part_f]), str(r[visit_n])]
        if g in bold_rows:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table2_body():
    df = pd.read_csv(TABLE2_CSV)
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"\textbf{Metric} & \textbf{NMD} & \textbf{CTL} & \textbf{Total} \\",
             r"\midrule"]
    for _, r in df.iterrows():
        # section header rows carry an empty NMD/CTL/Total; render bold,
        # unindented, and indent the measures beneath them.
        if pd.isna(r["NMD"]) or str(r["NMD"]).strip() == "":
            lines.append(rf"\textbf{{{fmt_label(r['Metric'])}}} & & & \\")
        else:
            lines.append(rf"\quad {fmt_label(r['Metric'])} & "
                         f"{fmt_value(r['NMD'])} & {fmt_value(r['CTL'])} & "
                         f"{fmt_value(r['Total'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table3_body():
    """Supp. table 3: kinematic parameters grouped and indented by segment,
    laid out in two columns across the page (left: Trunk then Pelvis;
    right: Upper limb then Lower limb). Right/left pairs are listed once
    (the "(R)"/"(L)" suffix is dropped)."""
    import re
    df = pd.read_csv(SUPP3_CSV)

    def seg_entries(seg_order):
        entries = []
        for i, seg in enumerate(seg_order):
            if i > 0:
                entries.append("")                     # blank spacer between segments
            entries.append(rf"\textbf{{{fmt_label(seg)}}}")
            seen = set()
            for _, r in df[df["Segment"] == seg].iterrows():
                name = re.sub(r"\s*\((?:R|L)\)$", "", str(r["Kinematic Parameter"]))
                if name in seen:
                    continue
                seen.add(name)
                entries.append(rf"\quad {fmt_label(name)}")
        return entries

    left  = seg_entries(["Trunk", "Lower limb"])
    right = seg_entries(["Upper limb", "Pelvis"])
    n = max(len(left), len(right))
    left  += [""] * (n - len(left))
    right += [""] * (n - len(right))

    lines = [r"\begin{tabular}{@{}l@{\hspace{2.5em}}l@{}}", r"\toprule",
             r"\multicolumn{2}{@{}l}{\textbf{Kinematic parameter}} \\", r"\midrule"]
    for l, r in zip(left, right):
        lines.append(f"{l} & {r} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def comparison_body(csv_path):
    """Supp. tables 1 & 2: Section | Measure | Transformer | SVM | MLP.
    Returns the bare tabular (no caption, no notes)."""
    df = pd.read_csv(csv_path)
    sections = list(dict.fromkeys(df["Section"]))     # preserve order
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Section & Measure & Transformer & SVM & MLP \\", r"\midrule"]
    for si, sec in enumerate(sections):
        block = df[df["Section"] == sec]
        n = len(block)
        # section label as a multirow spanning the block; guard "\\[" (a line
        # beginning with "[" would otherwise be read as an optional-arg to \\).
        sec_tex = r"\makecell[l]{" + r"\\".join(fmt_text(x) for x in _wrap_section(sec)) + "}"
        sec_tex = sec_tex.replace(r"\\[", r"\\{}[")
        for ri, (_, r) in enumerate(block.iterrows()):
            first = rf"\multirow{{{n}}}{{*}}{{{sec_tex}}}" if ri == 0 else ""
            lines.append(f"{first} & {fmt_label(r['Measure'])} & "
                         f"{fmt_value(r['Transformer'])} & "
                         f"{fmt_value(r['SVM'])} & {fmt_value(r['MLP'])} \\\\")
        if si < len(sections) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _wrap_section(sec):
    """Split a long section title over two lines for the multirow cell."""
    sec = str(sec).strip()
    if "(" in sec:                      # break before the parenthetical
        i = sec.index("(")
        return [sec[:i].strip(), sec[i:].strip()]
    if len(sec) > 22 and " " in sec:
        mid = sec.rfind(" ", 0, len(sec) // 2 + 6)
        return [sec[:mid].strip(), sec[mid:].strip()]
    return [sec]


# ── compile helpers ───────────────────────────────────────────────────────────

def compile_pdf_png(name, body, outdir):
    """Write <name>.tex (standalone) into outdir, compile to PDF, rasterize to PNG."""
    tex_path = os.path.join(outdir, name + ".tex")
    with open(tex_path, "w") as f:
        f.write(_standalone(body))
    r = subprocess.run(["tectonic", "-o", outdir, tex_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [tectonic FAILED for {name}]\n{r.stderr[-800:]}")
        return
    pdf_path = os.path.join(outdir, name + ".pdf")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                    pdf_path, os.path.join(outdir, name)], check=True)
    print(f"  saved {name}.tex / .pdf / .png")


def _render(bodies, outdir, bundle_name):
    """Render each (name -> (label, body)) to standalone .tex/.pdf/.png in outdir,
    plus a combined manuscript bundle. tectonic/pdftoppm optional (.tex still written)."""
    os.makedirs(outdir, exist_ok=True)
    have_tectonic = shutil.which("tectonic") and shutil.which("pdftoppm")
    print("Rendering tables ->", outdir)
    for name, (lab, body) in bodies.items():
        if have_tectonic:
            compile_pdf_png(name, body, outdir)
        else:
            with open(os.path.join(outdir, name + ".tex"), "w") as f:
                f.write(_standalone(body))
            print(f"  saved {name}.tex  (tectonic/pdftoppm missing — no PDF/PNG)")

    # Combined manuscript file (\input into the paper). Tables only, no \caption —
    # add captions where the tables are placed in the manuscript.
    combined = [r"% Auto-generated by create_latex_tables.py — do not edit by hand.",
                r"% Requires: \usepackage{booktabs,multirow,makecell}",
                r"% No captions here by design — add \caption{} at each \ref site.",
                ""]
    for name, (lab, body) in bodies.items():
        combined += [r"\begin{table}[htbp]", r"\centering",
                     rf"\label{{{lab}}}", body, r"\end{table}", ""]
    bundle_path = os.path.join(outdir, bundle_name)
    with open(bundle_path, "w") as f:
        f.write("\n".join(combined))
    print("saved combined manuscript file:", bundle_path)


def render_main_tables(tables_dir):
    """Render the main-text tables (Table 1 diseases, Table 2 measures) found in
    tables_dir to LaTeX in tables_dir/latex_tables/. Called automatically by
    make_figures_tables.py. Skips silently if the CSVs are not present."""
    global TABLE1_CSV, TABLE2_CSV
    TABLE1_CSV = os.path.join(tables_dir, "table1_diseases.csv")
    TABLE2_CSV = os.path.join(tables_dir, "table2_measures.csv")
    missing = [p for p in (TABLE1_CSV, TABLE2_CSV) if not os.path.isfile(p)]
    if missing:
        print(f"  [skip latex] missing table CSVs: {missing}")
        return
    bodies = {
        "table1_diseases": ("tab:diseases", table1_body()),
        "table2_measures": ("tab:measures", table2_body()),
    }
    _render(bodies, os.path.join(tables_dir, "latex_tables"), "tables_latex.tex")


def render_supp_tables(supp_tables_dir):
    """Render the supplementary tables (classification, convergent validity,
    kinematic parameters) found in supp_tables_dir to LaTeX in
    supp_tables_dir/latex_tables/. Called automatically by
    make_supplementary_figures_tables.py. Skips silently if the CSVs are absent."""
    global SUPP1_CSV, SUPP2_CSV, SUPP3_CSV
    SUPP1_CSV = os.path.join(supp_tables_dir, "supplementary_table1_model_classification.csv")
    SUPP2_CSV = os.path.join(supp_tables_dir, "supplementary_table2_model_convergent_validity.csv")
    SUPP3_CSV = os.path.join(supp_tables_dir, "supplementary_table3_kinematic_parameters.csv")
    missing = [p for p in (SUPP1_CSV, SUPP2_CSV, SUPP3_CSV) if not os.path.isfile(p)]
    if missing:
        print(f"  [skip latex] missing supp-table CSVs: {missing}")
        return
    bodies = {
        "supp_table1_classification": ("tab:supp-classification", comparison_body(SUPP1_CSV)),
        "supp_table2_convergent":     ("tab:supp-convergent",     comparison_body(SUPP2_CSV)),
        "supp_table3_kinematic":      ("tab:supp-kinematic",      table3_body()),
    }
    _render(bodies, os.path.join(supp_tables_dir, "latex_tables"), "supp_tables_latex.tex")
