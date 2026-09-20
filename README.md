# Detecting and Quantifying Neuromuscular Diseases from Smartphone Video-Derived Kinematics

This repository contains the code and released models needed to reproduce the disease
detection and quantification results from the following manuscript:

> **Deep learning models detect neuromuscular disease and quantify functional impairment from video-derived biomechanics data**

> Sydney Covitz, Parker S. Ruth, Shelby Vogt-Domke, Carmichael Ong, Tian Tan, Audrey Chun, Sarah Ismail, Lin Karman, Julie Muccini, Shannon Li, Melina Rogers, Jennifer L. Hicks, Scott Uhlrich, John W. Day, Constance de Monts, Tina Duong,  Scott L. Delp

The Example Walkthrough has been tested on Mac (Apple Silicon), Linux, and Windows machines. 

## Example Walkthrough 
*requires environment management (we recommend miniforge or mamba)

### 1. Setup

Install [(miniforge, miniconda,](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html) Python environment manager. Note that if you use `mamba`, swap `conda` for `mamba` in the below code block.

Next, execute the following steps: 

```bash
cd [path/to/empty/working/directory]
git clone git@github.com:stanfordnmbl/NMD_Detection_Quantification.git
cd NMD_Detection_Quantification
conda env create -f environment.yml -n nmd-opencap
conda activate nmd-opencap
```

The working directory will be organized as follows:

```
NMD_Detection_Quantification
├── README.md
├── environment.yml
├── code                     # all scripts
└── runs
    └── pretrained
        ├── models           # released, trained fold models
        ├── severity_csvs    # prediction CSVs (precomputed/ ships with the repo;
        │                    # the inference option writes fresh CSVs here)
        └── results          # figures + tables (written by the make_* scripts)
```

### 2. Downloading the Data

The de-identified Neuromuscular OpenCap Dataset and demographics CSV (nmd_opencap_participant_info.csv) 
are hosted on Zenodo at [this link](https://zenodo.org/records/22309771?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhOGJkYjBjLTM5NjYtNGNmMC1hNjhiLTk0OWZhZTUzOTFhMyIsImRhdGEiOnt9LCJyYW5kb20iOiIwZTMyNWUwMmFiMDRkNDM5YjgxZWI1ZThlODFlZDU3MiJ9.aJ1J91c7oGUjjF51gsIvPU7ynx-liDBoSPOAy0qMfQ-wvkzsoZSa6emVkML0dxpeFJfMAlEhcKgD3ZddChpJMw). 
Download them into a `datadir/` folder and unzip the dataset (`zenodo_get` comes with the environment). 
Alternatively, if you do not want to downoad the entire dataset, you can reproduce our results using the pre-inferenced CSVs 
(see Section 3 Option 2 below) and just download the `nmd_opencap_participant_info.csv` directly from the Zenodo dataset webpage 
and save that to your `datadir`.

```bash
mkdir datadir
cd datadir
zenodo_get -d <ZENODO_DOI>
cd ..
unzip datadir/Neuromuscular_OpenCap_Dataset.zip -d datadir
```

`datadir/` then contains the demographics CSV and the dataset––one folder per
participant, each with per-visit, per-session OpenCap outputs. The pipeline reads
the time series in `Kinematics/*.mot`.

```
datadir
├── nmd_opencap_participant_info.csv     # labels + train/test split (subid, visit)
└── Neuromuscular_OpenCap_Dataset
    ├── sub-001
    │   └── visit-0
    │       ├── ses-1
    │       │   ├── Kinematics            # OpenSim joint coordinates (model inputs)
    │       │   │   ├── sub-001_visit-0_ses-1_task-curls.mot
    │       │   │   ├── sub-001_visit-0_ses-1_task-jump.mot
    │       │   │   └── sub-001_visit-0_ses-1_task-toe_stand.mot
    │       │   ├── Markers               # 3D marker trajectories (.trc)
    │       │   ├── Model                 # scaled OpenSim musculoskeletal model (.osim)
    │       │   └── sub-001_visit-0_ses-1_metadata.yaml
    │       └── ses-2
    │           └── ...
    ├── sub-002
    │   └── ...
    └── sub-415                           # 415 participants
```

> If you only plan to use **Option 2** below (`--skip-inference`), you just need
> `nmd_opencap_participant_info.csv` (not the dataset).

### 3. Reproduce the paper results

In order to reproduce our paper results, you will need to run `make_figures_tables.py` and `make_supplementary_figures_tables.py`.

Both of these scripts accept `--help`, which explains which command line inputs to use, and take `--run-name` to 
select which run under `runs/` to use. Use the **same** run name across training and figure/table
generation. Use `--run-name pretrained` to reproduce our exact paper results with the released models.

There are two ways to run each script — choose one:

**Option 1 — run the models on the dataset.** The script runs the released fold
models over the dataset, saves the prediction CSVs to
`runs/pretrained/severity_csvs/`, then builds the figures and tables:

```bash
python make_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset --demographics /path/to/datadir/nmd_opencap_participant_info.csv --run-name pretrained
python make_supplementary_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset --demographics /path/to/datadir/nmd_opencap_participant_info.csv  --run-name pretrained
```

**Option 2 — skip inference.** The script builds the figures and tables straight
from the CSVs generated from the already inferenced released models, located in 
`runs/pretrained/severity_csvs/precomputed/`. This needs only the demographics CSV 
(not the large dataset download).

```bash
python make_figures_tables.py --skip-inference --demographics /path/to/datadir/nmd_opencap_participant_info.csv  --run-name pretrained
python make_supplementary_figures_tables.py --skip-inference --demographics /path/to/datadir/nmd_opencap_participant_info.csv  --run-name pretrained
```

Either way, each script prints its statistics to the terminal and opens the
figures and tables for review (close one to see the next). Outputs land under
`runs/pretrained/results/`:

- **`figures/`** — **Fig 2** classification performance (AUROC/AUPRC/bACC, val vs test),
  **Fig 3** severity-score distributions (NMD vs CTL, held-out test + OOF),
  **Fig 4** severity vs ACTIVLIM, **Fig 5** severity vs the timed function tests.
- **`tables/`** — **Table 1** cohort by diagnosis and **Table 2** demographic/functional
  measures (NMD vs CTL); LaTeX/PDF/PNG under `latex_tables/`.
- **`supp_figures/`** — **Supp. Fig 1** per-measure distributions (NMD vs CTL).
- **`supp_tables/`** — **Supp. Tables 1–3**: transformer vs SVM vs MLP classification,
  convergent validity, and the kinematic parameters; LaTeX under `latex_tables/`.

### 4. Train the models yourself and then run our analyses

To retrain the models from scratch instead of using the released models, run `train_models.py`
with a new run name (e.g. `retrain_1`). This will run cross-validation on the transformer,
SVM, and MLP on one shared 5-fold GroupKFold split (grouped by participant), and enforces the
same held-out-test set as the one used to train the models we've released. 
freshly trained fold models, then regenerate the results with Option 1 above. 

```bash
python train_models.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset  --demographics /path/to/datadir/nmd_opencap_participant_info.csv ---run-name <run-name>
```

This will take approiximately 2.5h to run on a single GPU and writes the freshly trained models 
to `runs/<run-name>/models`. 

Next, you can inference these models and rebuild every figure and table from your new models using the following:

```bash
python make_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset --demographics /path/to/datadir/nmd_opencap_participant_info.csv --run-name <run-name>
python make_supplementary_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset --demographics /path/to/datadir/nmd_opencap_participant_info.csv ---run-name <run-name>
```
Note that your new run results will be slightly different than our paper figures if you have retrained the models but should be very similar. 

## Citing This Work

We invite you to cite both our [preprint](TODO ADD LINK) and our [Zenodo dataset](https://doi.org/10.5281/zenodo.22309771).

> Covitz, S., et al. Neuromuscular OpenCap Dataset. Zenodo https://doi.org/https://doi.org/10.5281/zenodo.22309771 (2026). 

> Covitz, S., et al. Deep learning models detect neuromuscular disease and quantify functional impairment from video-derived biomechanics data. bioRxiv (2026). doi: [TODO: ADD bioRxiv DOI]