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
nmd_opencap_wkdir                 # root level working directory  
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

`nmd_opencap_wkdir/datadir/` now contains the demographics CSV (`nmd_opencap_participant_info.csv`) and the full, unzipped `Neuromuscular OpenCap Dataset`, so you have everything you need to run our code. See below for information on the dataset structure: 

```
nmd_opencap_wkdir                                           # root level working directory  
├──NMD_Detection_Quantification                             # cloned GitHub repo 
└── datadir                                                 # downloads from Zenodo 
    ├── nmd_opencap_participant_info.csv                    # demographics and train/test split for each (subid, visit) pair
    ├── table1_diseases.csv                                 # per-disease breakdown of participants in the Neuromuscular OpenCap Dataset
    └── Neuromuscular_OpenCap_Dataset                       # Dataset root 
        ├── sub-001
        │   └── visit-0
        │       ├── ses-1
        │       │   ├── Kinematics                          # OpenSim kinematic time series ( machine learning model inputs)
        │       │   │   ├── sub-001_visit-0_ses-1_task-curls.mot
        │       │   │   ├── sub-001_visit-0_ses-1_task-jump.mot
        │       │   │   └── sub-001_visit-0_ses-1_task-toe_stand.mot
        │       │   ├── Markers                              # 3D marker trajectories (.trc)
        │       │   ├── Model                                # scaled OpenSim musculoskeletal model (.osim)
        │       │   └── sub-001_visit-0_ses-1_metadata.yaml  # metadata dictionary detailing the camera information and OpenCap settings used for that session
        │       └── ses-2
        │           └── ...
        ├── sub-002
        │   └── ...
        └── sub-415                           # 415 participants
```

### 3. Reproduce Paper Results

All executable scripts (`make_figures_tables.py`, `make_supplementary_figures_tables.py`, and `train_models.py`) accept a `--help` flag, which explains the command line inputs each script takes. One such required input is `--run-name`, which allows uesers to select the training run from which they would like to generate the figures. Use `--run-name pretrained` to reproduce our exact paper results from the released models.

There are two ways to run each script — choose one:

#### **Option 1: inference the trained models and re-create our paper figures/tables (requires the full dataset downloaded and unzipped)**
This option allows users to inference our trained models cross-validation, create and save the inference results to `nmd_opencap_wkdir/NMD_Detection_Quantification/runs/pretrained/severity_csvs/`, and builds the figures and tables. To run Option 1, execute the following steps (from `nmd_opencap_wkdir`):

```bash
cd NMD_Detection_Quantification/code
python make_figures_tables.py --run-name pretrained --demographics ../../datadir/nmd_opencap_participant_info.csv --dataset ../../datadir/Neuromuscular_OpenCap_Dataset
python make_supplementary_figures_tables.py --run-name pretrained --demographics ../../datadir/nmd_opencap_participant_info.csvnmd_opencap_participant_info.csv --dataset ../../datadir/Neuromuscular_OpenCap_Dataset
```

#### **Option 2: --skip inference (only nmd_opencap_participant_info.csv required).** 
This option builds the figures and tables straight from the CSVs generated from the already inferenced released models, 
which are located in `NMD_Detection_Quantification/runs/pretrained/severity_csvs/precomputed/`. This option requires only the demographics CSV (`nmd_opencap_participant_info.csv`) and allows you to reproduce our paper results without having to download and unzip the Neuromuscular OpenCap Dataset. 
Ton run Option 2, execute the following steps (from `nmd_opencap_wkdir`):

```bash
cd NMD_Detection_Quantification/code
python make_figures_tables.py --run-name pretrained --demographics ../../datadir/nmd_opencap_participant_info.csv  --skip-inference
python make_supplementary_figures_tables.py --run-name pretrained --demographics ../../datadir/nmd_opencap_participant_info.csv --skip-inference 
```
You can now navigate to `NMD_Detection_Quantification/runs/pretrained/results/` to see all figures and tables you have generated, which contains the following: 

- **`figures/`** — **Fig 2** classification performance (AUROC/AUPRC/bACC, val vs test),
  **Fig 3** severity-score distributions (NMD vs CTL, held-out test + OOF),
  **Fig 4** severity vs ACTIVLIM, **Fig 5** severity vs the timed function tests.
- **`tables/`** — **Table 1** cohort by diagnosis and **Table 2** demographic/functional
  measures (NMD vs CTL); LaTeX/PDF/PNG under `latex_tables/`.
- **`supp_figures/`** — **Supp. Fig 1** per-measure distributions (NMD vs CTL).
- **`supp_tables/`** — **Supp. Tables 1–3**: transformer vs SVM vs MLP classification,
  convergent validity, and the kinematic parameters; LaTeX under `latex_tables/`

Regardless of which option you run, each script prints all statistics to the console and automatically displays the
figures and tables for visual review (close one to see the next). All generated figures and tables will always land under
`runs/<run-name>/results/`.

### 4. Re-train the models yoruself (OPTIONAL)

Note that if you retrain the models yourself, your figures and statistics will differ slightly from those in the paper. Model training is stochastic (random weight initialization, batch shuffling, and GPU nondeterminism), so each run produces a slightly different model. These differences are small, well within the reported 95% confidence intervals, and do not change the conclusions.

To retrain the models from scratch instead of using the released models, run `train_models.py`
with a new `--run-name` field (e.g. `retrain_1`). This will run cross-validation on the transformer,
SVM, and MLP on one shared 5-fold GroupKFold split (grouped by participant), enforces the
same validation/held-out-test split we used train the models we've released, and saves all trained models to the `NMD_Detection_Quantification/runs/<run-name>/models`. 
You can then run our `make_*` scripts to generate our same figures and tables using your freshly trained models. To do this, execute the following steps (from `nmd_opencap_wkdir`): 

```bash
cd NMD_Detection_Quantification/code
python train_models.py --run-name <run-name> --demographics ../../datadir/nmd_opencap_participant_info.csv --dataset /path/to/datadir/Neuromuscular_OpenCap_Dataset
```

This will take approiximately 2.5h to run on a single GPU and writes the freshly trained models to `NMD_Detection/Quantification/runs/<run-name>/models`. 

Next, you can inference these models and rebuild every figure and table from your new models using the following:

```bash
python make_figures_tables.py --run-name <run-name> --demographics ../../datadir/nmd_opencap_participant_info.csv --dataset ../../datadir/Neuromuscular_OpenCap_Dataset
python make_supplementary_figures_tables.py --run-name <run-name> --demographics ../../datadir/nmd_opencap_participant_info.csvnmd_opencap_participant_info.csv --dataset ../../datadir/
```

## Citing This Work

We invite you to cite both our preprint and our [Zenodo dataset](https://doi.org/10.5281/zenodo.22309771).

> Covitz, S., et al. Neuromuscular OpenCap Dataset. *Zenodo https://doi.org/https://doi.org/10.5281/zenodo.22309771 (2026). 

> Covitz, S., et al. Deep learning models detect neuromuscular disease and quantify functional impairment from video-derived biomechanics data. *bioRxiv (2026).