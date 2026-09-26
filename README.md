# Detecting and Quantifying Neuromuscular Diseases from Smartphone Video-Derived Kinematics

This repository contains the code and released models needed to reproduce the disease
detection and quantification results from the following manuscript:

> **Deep learning models detect neuromuscular disease and quantify functional impairment from video-derived biomechanics data**

> Sydney Covitz, Parker S. Ruth, Shelby Vogt-Domke, Carmichael Ong, Tian Tan, Audrey Chun, Sarah Ismail, Lin Karman, Julie Muccini, Shannon Li, Melina Rogers, Scott Uhlrich, John W. Day, Jennifer L. Hicks, Constance de Monts, Tina Duong, Scott L. Delp



## Example Walkthrough 
The following walkthrough has been tested on Mac (Apple Silicon), Linux, and Windows machines. 

### 1. Setup

Install a Python environment manager. We recommend [miniforge or miniconda.](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Next, execute the following steps: 

```bash
mkdir $PWD/nmd_opencap_wkdir
git clone git@github.com:stanfordnmbl/NMD_Detection_Quantification.git
cd NMD_Detection_Quantification
conda env create -f environment.yml -n nmd-opencap
conda activate nmd-opencap
```

The working directory will be organized as follows:

```
NMD_OpenCap                       # root level working directory  
└── NMD_Detection_Quantification  # cloned GitHub repo 
    ├── README.md
    ├── environment.yml
    ├── code                       # all scripts
    └── runs
        └── pretrained
            ├── models             # released, trained fold models
            ├── severity_csvs      # prediction CSVs (precomputed/ ships with the repo;
            │                      # the inference option writes fresh CSVs here)
            └── results            # figures + tables (will get written after you run the make_* scripts)
```

### 2. Download Data

The de-identified Neuromuscular OpenCap Dataset and demographics CSV (nmd_opencap_participant_info.csv) 
are hosted on Zenodo at [this link](https://zenodo.org/records/22309771?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhOGJkYjBjLTM5NjYtNGNmMC1hNjhiLTk0OWZhZTUzOTFhMyIsImRhdGEiOnt9LCJyYW5kb20iOiIwZTMyNWUwMmFiMDRkNDM5YjgxZWI1ZThlODFlZDU3MiJ9.aJ1J91c7oGUjjF51gsIvPU7ynx-liDBoSPOAy0qMfQ-wvkzsoZSa6emVkML0dxpeFJfMAlEhcKgD3ZddChpJMw). 
Download them into a `datadir/` folder and unzip the dataset (`zenodo_get` comes with the environment). 

**If you do not want to download the entire dataset,** you can reproduce our results using the pre-inferenced CSVs. To do this, download only `nmd_opencap_participant_info.csv` directly from the Zenodo dataset webpage, save that file to your `datadir`, and skip to the instructions for **Option 2** in Section 3 below. 


To download the entire dataset, navigate to the `nmd_opencap_wkdir` directory you created above and execute the following: 

```bash
mkdir $PWD/datadir
cd datadir
zenodo_get -d 10.5281/zenodo.22309771
unzip Neuromuscular_OpenCap_Dataset.zip -d .
```

`datadir/` then contains the demographics CSV and the dataset––one folder per
participant, each with per-visit, per-session OpenCap outputs. The pipeline reads
the time series in `Kinematics/*.mot`.

```
nmd_opencap_wkdir                            # root level working directory  
├──NMD_Detection_Quantification              # cloned GitHub repo 
└── datadir                                  # downloads from Zenodo 
    ├── nmd_opencap_participant_info.csv     # demographics and train/test split for each (subid, visit) pair
    ├── table1_diseases.csv                  # per-disease breakdown of participants in the Neuromuscular OpenCap Dataset
    └── Neuromuscular_OpenCap_Dataset        # Dataset root 
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

### 3. Reproduce Paper Results

In order to reproduce our paper results, you will need to run `make_figures_tables.py` and `make_supplementary_figures_tables.py`.

Both of these scripts accept `--help`, which explains which command line inputs to use, and require `--run-name` to 
select which run under `runs/` to use. Use the **same** run name across training and figure/table
generation. Use `--run-name pretrained` to reproduce our exact paper results with the released models.

There are two ways to run each script — choose one:

**Option 1 — inference the trained models and re-create our paper figures/tables (requires the full dataset downloaded and unzipped)**
This option inferences each fold's model with its out-of-fold data, saves the prediction CSVs to `runs/pretrained/severity_csvs/`, 
and builds the figures and tables:

```bash
python make_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset --demographics /path/to/datadir/nmd_opencap_participant_info.csv --run-name pretrained
python make_supplementary_figures_tables.py --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset --demographics /path/to/datadir/nmd_opencap_participant_info.csv  --run-name pretrained
```

**Option 2 `--skip inference (only nmd_opencap_participant_info.csv required).** 
This option builds the figures and tables straight from the CSVs generated from the already inferenced released models, 
which are located in `runs/pretrained/severity_csvs/precomputed/`. This needs only the demographics CSV and allows you 
to reproduce our paper results without having to download and unzip the Neuromuscular OpenCap Dataset. 

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

### 4. Re-train the models yoruself (OPTIONAL)

Note that if you retrain the models yourself, your figures and statistics will differ slightly from those in the paper. Model training is stochastic (random weight initialization, batch shuffling, and GPU nondeterminism), so each run produces a slightly different model. These differences are small, well within the reported 95% confidence intervals, and do not change the conclusions.

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

## Citing This Work

We invite you to cite both our [preprint](TODO ADD LINK) and our [Zenodo dataset](https://doi.org/10.5281/zenodo.22309771).

> Covitz, S., et al. Neuromuscular OpenCap Dataset. *Zenodo https://doi.org/https://doi.org/10.5281/zenodo.22309771 (2026). 

> Covitz, S., et al. Deep learning models detect neuromuscular disease and quantify functional impairment from video-derived biomechanics data. *bioRxiv (2026). doi: [TODO: ADD bioRxiv DOI]