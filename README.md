# Detecting and Quantifying Neuromuscular Diseases from Smartphone Video-Derived Kinematics

Code and trained models to reproduce the disease detection and quantification
results from our paper. The repository ships the trained fold models (under
`runs/pretrained/models/`), so the results regenerate **without retraining** —
run inference, then build the figures and tables.

### Example Walkthrough (requires environment management — we recommend miniforge)

## 1. Setup

```
$ cd [path_to_empty_working_directory]
$ git clone <REPO_URL> NMD_Detection_Quantification
$ cd NMD_Detection_Quantification
$ conda env create -f environment.yml -n nmd-opencap
$ conda activate nmd-opencap
```

The working directory is then organized as:

```
NMD_Detection_Quantification
├── README.md
├── environment.yml
├── code                 # the pipeline scripts
└── runs
    └── pretrained
        ├── models       # released, trained fold models
        ├── csvs         # per-visit predictions (written by inference)
        └── results      # figures + tables (written by the make_* scripts)
```

## 2. Downloading the Data

The de-identified dataset and demographics are hosted on Zenodo: <ZENODO_DOI>.
Download them into a `datadir/` folder and unzip the dataset (`zenodo_get` comes
with the environment; or download the two files from the Zenodo page):

```
$ mkdir datadir
$ cd datadir
$ zenodo_get -d <ZENODO_DOI>
$ cd ..
$ unzip datadir/Neuromuscualr_OpenCap_Dataset.zip -d datadir
```

`datadir/` then contains the demographics CSV and the dataset — one folder per
participant, each with per-visit, per-session OpenCap outputs. The pipeline reads
the joint-angle time series in `Kinematics/*.mot`; `Markers/` and `Model/` are the
other OpenCap outputs and are not required to run the code.

```
datadir
├── nmd_opencap_participant_info.csv     # labels + train/test split (subid, visit)
└── Neuromuscualr_OpenCap_Dataset
    ├── sub-001
    │   └── visit-0
    │       ├── ses-1
    │       │   ├── Kinematics            # OpenSim joint angles (model inputs)
    │       │   │   ├── sub-001_visit-0_ses-1_task-curls.mot
    │       │   │   ├── sub-001_visit-0_ses-1_task-jump.mot
    │       │   │   └── sub-001_visit-0_ses-1_task-toe_stand.mot
    │       │   ├── Markers               # 3D marker trajectories (.trc)
    │       │   ├── Model                 # scaled OpenSim model (.osim)
    │       │   └── sub-001_visit-0_ses-1_metadata.yaml
    │       └── ses-2
    │           └── ...
    ├── sub-002
    │   └── ...
    └── sub-415                           # 415 participants
```

## 4. Reproduce the paper results

You can reproduce our main-text results — Figures 2–5 and Tables 1–2 — from the
released models and the demographics CSV:

```
$ DATASET=datadir/Neuromuscualr_OpenCap_Dataset
$ DEMO=datadir/nmd_opencap_participant_info.csv
$ cd code

# Regenerate all model predictions from the released fold models (no retraining)
$ python inference_models.py --dataset $DATASET --demographics $DEMO --run-name pretrained

# Build the main-text figures + tables (each opens for review; close one to see the next)
$ python make_figures_tables.py --demographics $DEMO --run-name pretrained
```

These land under `runs/pretrained/`:

- **`csvs/`** — per-visit predictions: the transformer's held-out test severity
  and out-of-fold (validation) severity, plus the SVM/MLP held-out test severity.
- **`results/figures/`** — **Fig 2** classification performance (AUROC/AUPRC/bACC,
  val vs test), **Fig 3** severity-score distributions (NMD vs CTL, held-out test
  + OOF), **Fig 4** severity vs ACTIVLIM, **Fig 5** severity vs the timed function
  tests.
- **`results/tables/`** — **Table 1** cohort by diagnosis and **Table 2**
  demographic/functional measures (NMD vs CTL); LaTeX/PDF/PNG under
  `latex_tables/`.

`make_figures_tables.py` also prints the figure statistics (AUROC, AUPRC, and
bACC with 95% CIs, Cliff's delta, and Spearman ρ) to the terminal.

## 5. Reproduce the supplementary figures and tables

The supplementary tables compare the transformer against the SVM and MLP
baselines, reusing the predictions generated in step 4. Build the supplementary
outputs:

```
$ python make_supplementary_figures_tables.py --demographics $DEMO --run-name pretrained
```

These land under `runs/pretrained/results/`:

- **`supp_figures/`** — **Supp. Fig 1** per-measure distributions (NMD vs CTL).
- **`supp_tables/`** — **Supp. Tables 1–3**: transformer vs SVM vs MLP
  classification, convergent validity, and the kinematic parameters; LaTeX under
  `latex_tables/`.

## 6. Retrain the models yourself and then run our analyses

To retrain from scratch instead of using the released models, pick a new
`--run-name` (e.g. `myrun`) and run the full pipeline under it. `train_models.py`
writes models only; `inference_models.py` and the `make_*` scripts then
regenerate every CSV, figure, and table for that run.

First point two variables at the data and move into `code/`:

```
$ DATASET=datadir/Neuromuscualr_OpenCap_Dataset
$ DEMO=datadir/nmd_opencap_participant_info.csv
$ cd code
```

`train_models.py` cross-validates the transformer and the SVM/MLP baselines on
one shared 5-fold GroupKFold split (grouped by participant), and uses the same
held-out test set as our results — the split is fixed by the `split` column in
the demographics CSV, so the held-out test visits never enter cross-validation.
Note that the transformer's cross-validation takes ~2.5 h on a single GPU.

```
$ python train_models.py --dataset $DATASET --demographics $DEMO --run-name myrun
```

Inference your freshly trained models to generate the severity scores, then build
the figures and tables:

```
$ python inference_models.py --dataset $DATASET --demographics $DEMO --run-name myrun
$ python make_figures_tables.py               --demographics $DEMO --run-name myrun
$ python make_supplementary_figures_tables.py --demographics $DEMO --run-name myrun
```

Outputs land under `runs/myrun/` with the same layout as the pretrained results
(steps 4–5).

Every script accepts `--help` for its full input description.
