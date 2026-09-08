# Detecting and Quantifying Neuromuscular Diseases from Smartphone Video-Derived Kinematics

Code and trained models to reproduce the disease detection and quantification
results from our paper. The repository ships the trained fold models (under
`runs/pretrained/models/`), so the results regenerate **without retraining** —
run inference, then build the figures and tables.

### Example Walkthrough (requires environment management — we recommend miniforge)

## 1. Setup

```
cd [path_to_empty_working_directory]
git clone <REPO_URL> NMD_Detection_Quantification
cd NMD_Detection_Quantification
conda env create -f environment.yml -n nmd-opencap
conda activate nmd-opencap
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
mkdir datadir
cd datadir
zenodo_get -d <ZENODO_DOI>
cd ..
unzip datadir/Neuromuscualr_OpenCap_Dataset.zip -d datadir
```

`datadir/` then contains the demographics CSV and the dataset — one folder per
participant, each with per-visit, per-session OpenCap outputs. The pipeline reads
the joint-angle time series in `Kinematics/*.mot`; `Markers/` and `Model/` are the
other OpenCap outputs and are not required to run the code.

```
datadir
├── nmd_opencap_participant_info.csv     # labels (diag) + train/test split, keyed (subid, visit)
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

You can reproduce our paper results from the trained models and demographics CSV by running the following commands: 

```
$ DATASET=datadir/Neuromuscualr_OpenCap_Dataset
$ DEMO=datadir/nmd_opencap_participant_info.csv
$ cd code

# a) Regenerate predictions from the released fold models (no retraining)
$ python inference_transformer.py --dataset $DATASET --demographics $DEMO --run-name pretrained
$ python inference_mlp_svm.py     --dataset $DATASET --demographics $DEMO --run-name pretrained

# b) Build the figures + tables (each opens for review; close one to see the next)
$ python make_figures_tables.py               --demographics $DEMO --run-name pretrained
$ python make_supplementary_figures_tables.py --demographics $DEMO --run-name pretrained
```

Everything lands under `runs/pretrained/`:

- **`csvs/`** — per-visit predictions: transformer + SVM/MLP held-out test severity, and the transformer out-of-fold (validation) severity.
- **`results/figures/`** — **Fig 2** classification performance (AUROC/AUPRC/bACC, val vs test), **Fig 3** severity-score distributions (NMD vs CTL, held-out test + OOF), **Fig 4** severity vs ACTIVLIM, **Fig 5** severity vs the timed function tests.
- **`results/tables/`** — **Table 1** cohort by diagnosis, **Table 2** demographic/functional measures (NMD vs CTL); LaTeX/PDF/PNG under `latex_tables/`.
- **`results/supp_figures/`** — **Supp. Fig 1** per-measure distributions.
- **`results/supp_tables/`** — **Supp. Tables 1–3** (model comparison, convergent validity, kinematic parameters); LaTeX under `latex_tables/`.

The scripts also print the figure statistics (AUROC, AUPRC, and bACC with 95% CIs, Cliff's delta, and Spearman ρ) to the terminal, and open each figure and table for review (close one to see the next).



## 6. Retrain the models yourself and then run our analyses

To retrain from scratch instead of using the released models, pick a new `--run-name` (e.g. `myrun`) and run the full pipeline under it. Training writes models only; the inference and `make_*` scripts then regenerate every CSV, figure, and table for that run — the same three steps as above, just with your run name.

```
DATASET=datadir/Neuromuscualr_OpenCap_Dataset
DEMO=datadir/nmd_opencap_participant_info.csv
cd code
```
To re-train our models (using 5-fold Cross Validation with the same held-out-test set that we have in our results enforced ), run the following commands (note that cross-validation for the transformers takes ~2.5h on 1 GPU):

python train_transformer.py --dataset $DATASET --demographics $DEMO --run-name myrun 
python train_mlp_svm.py     --dataset $DATASET --demographics $DEMO --run-name myrun
```

You can now inference your freshly trained models and generate the severity scores as follows:
```
python inference_transformer.py --dataset $DATASET --demographics $DEMO --run-name myrun
python inference_mlp_svm.py     --dataset $DATASET --demographics $DEMO --run-name myrun 
```
Finally, you can build the figures and tables for your new models.

```
python make_figures_tables.py               --demographics $DEMO --run-name myrun
python make_supplementary_figures_tables.py --demographics $DEMO --run-name myrun
```

Outputs land under `runs/myrun/figures` with the same layout as our pre-trained results (see Step 4 above). The transformer uses a 5-fold GroupKFold split grouped by participant; the SVM/MLP baselines are trained independently with their own cross-validation. Held-out test visits (the `split` column from `nmd_opencap_participant_info.csv`) never enter training.

Every script accepts `--help` for its full input description.
