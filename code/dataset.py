"""
dataset.py
----------
Data loading, preprocessing, and splitting utilities for NMD kinematic classification.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


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

TASKS_TO_KEEP = [
    '10mwrt', '10mwt', 'arm_rom', 'brooke', 'curls',
    'jump', 'toe_stand', 'tug_line', 'tug_cone'
]


# ── Helper functions ───────────────────────────────────────────────────────────────────

def read_mot(fpath):
    """Read an OpenSim .mot file, skipping the header."""
    with open(fpath, 'r') as f:
        line = f.readline().strip()
        while line.lower() != 'endheader':
            line = f.readline().strip()
        df = pd.read_csv(f, delimiter='\t', header=0)
    return df


def _mot_header_cols(fpath):
    """Return the set of column names of a .mot without loading the data."""
    with open(fpath, 'r') as f:
        line = f.readline()
        while line and line.strip().lower() != 'endheader':
            line = f.readline()
        return set(f.readline().strip().split('\t'))


# ── Signal processing ──────────────────────────────────────────────────────────

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)


def lowpass_filter(data, fs, cutoff=6.0):
    """Zero-phase Butterworth low-pass filter applied column-wise."""
    b, a = butter_lowpass(cutoff, fs)
    return filtfilt(b, a, data, axis=0)


def df_to_tensor(df, fpath):
    """
    Convert a .mot DataFrame to a float32 tensor [L, 33].
    - Infers sampling rate from time column
    - Applies 6 Hz low-pass filter
    - Downsamples 120 Hz -> 60 Hz
    """
    if 'time' in df.columns:
        t = df['time'].values
        t = t[np.isfinite(t)]
        if len(t) >= 2:
            dt = np.diff(t).mean()
            fs = int(round(1.0 / dt)) if dt > 0 else 60
        else:
            fs = 60
    else:
        fs = 60

    if fs not in (60, 120):
        fs = 60 if abs(fs - 60) < abs(fs - 120) else 120

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f".mot file missing columns: {missing}")

    X = df[EXPECTED_COLUMNS].copy()
    for c in EXPECTED_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.interpolate(method='linear', axis=0, limit_direction='both')
    X = X.fillna(0.0)

    data = X.to_numpy(dtype=np.float32, copy=False)

    if data.shape[0] <= 15:
        print(f"WARNING: Very short sequence ({data.shape[0]} frames): {fpath}")

    data = lowpass_filter(data, fs=fs, cutoff=6.0)

    if fs == 120:
        data = data[::2, :]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.ascontiguousarray(data, dtype=np.float32)

    return torch.from_numpy(data)


# ── Label utilities ────────────────────────────────────────────────────────────

def get_label_for_participant(class_df, digbi_id):
    """Return binary label (0=CTL, 1=NMD) for a participant."""
    match = class_df.loc[class_df['digbi_id'] == digbi_id, 'binary_class']
    return int(match.iloc[0])


# ── Data structure builders ────────────────────────────────────────────────────

def build_participant_dict(dataset_dict, class_df):
    """
    Convert a dict of {digbi_id_date: [(mot_path, task_name), ...]}
    into a participant-level dict with visit structure.
    """
    participants = {}
    for visit_key, task_pairs in dataset_dict.items():
        digbi_id, date = visit_key.split('_', 1)
        label = get_label_for_participant(class_df, digbi_id)
        if label == 2:
            continue
        if digbi_id not in participants:
            participants[digbi_id] = {
                "digbi_id": digbi_id,
                "label": label,
                "visits": []
            }
        tasks, task_ids = zip(*task_pairs) if task_pairs else ([], [])
        participants[digbi_id]["visits"].append({
            "date": date,
            "tasks": list(tasks),
            "task_ids": list(task_ids)
        })
    return participants


def flatten_visits_for_model(participants):
    """
    Flatten participant dict into a list of per-visit sample dicts.
    Each sample: {digbi_id, date, label, tasks, task_ids}
    """
    samples = []
    for digbi_id, record in participants.items():
        for visit in record["visits"]:
            samples.append({
                "digbi_id": digbi_id,
                "date": visit["date"],
                "label": record["label"],
                "tasks": visit["tasks"],
                "task_ids": visit["task_ids"]
            })
    return samples


def filter_and_clean_tasks(samples, tasks_to_keep=None):
    """
    Filter samples to only include tasks in tasks_to_keep.
    Renames *sts tasks to '5xsts'. Removes samples with no remaining tasks.
    """
    if tasks_to_keep is None:
        tasks_to_keep = TASKS_TO_KEEP

    cleaned = []
    for s in samples:
        new_tasks, new_task_ids = [], []
        for fpath, task in zip(s['tasks'], s['task_ids']):
            task_clean = task.strip().lower()
            if task_clean in tasks_to_keep:
                new_tasks.append(fpath)
                new_task_ids.append(task_clean)
            elif task.endswith('sts'):
                new_tasks.append(fpath)
                new_task_ids.append('5xsts')
        s = dict(s)
        s['tasks'] = new_tasks
        s['task_ids'] = new_task_ids
        cleaned.append(s)

    cleaned = [s for s in cleaned if len(s['tasks']) > 0]
    return cleaned


def load_release_samples(dataset_root, demographics_csv):
    """Build model-ready visit samples from the public, de-identified OpenCap
    dataset. Expects the released layout

        <dataset_root>/sub-XXX/visit-N/ses-M/Kinematics/
            sub-XXX_visit-N_ses-M_task-<trial>.mot

    Samples are keyed by (subid, visit): the subject id becomes the sample
    'digbi_id' and the integer visit becomes the sample 'date', so all of the
    downstream builders/splitters (which group by 'digbi_id') work unchanged.
    Labels (0 = CTL, 1 = NMD) come from the demographics CSV's 'diag' column,
    matched on 'subid'.
    """
    import os, re, glob

    demo = pd.read_csv(demographics_csv).drop_duplicates('subid')
    class_df = pd.DataFrame({
        'digbi_id':     demo['subid'].astype(str),
        'binary_class': np.where(
            demo['diag'].astype(str).str.upper().str.strip() == 'CTL', 0, 1),
    })

    needed = set(EXPECTED_COLUMNS + ['time'])
    pat = re.compile(r'(sub-\d+)_visit-(\d+)_ses-[^_]+_task-(.+)\.mot$')
    dd = {}
    for p in glob.glob(os.path.join(dataset_root, 'sub-*', 'visit-*', 'ses-*',
                                    'Kinematics', '*.mot')):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        subid, visit, task = m.group(1), m.group(2), m.group(3)
        dd.setdefault(f'{subid}_{visit}', []).append((p, task))

    # Drop any visit whose .mot files lack the model-input columns.
    for k in [k for k, l in dd.items()
              if any(not needed.issubset(_mot_header_cols(mp)) for mp, _ in l)]:
        del dd[k]

    samples = filter_and_clean_tasks(
        flatten_visits_for_model(build_participant_dict(dd, class_df)))
    return samples


# ── Train/val/test split ───────────────────────────────────────────────────────

def split_train_val_test(samples, val_frac=0.10, test_frac=0.10, seed=42):
    """
    Grouped, stratified split by digbi_id.
    - All visits for a participant stay in the same split.
    - Binary label proportions (CTL vs NMD) are stratified at the participant level.
    - Hard assertions verify no participant ID overlap across splits.
    """
    digbi_ids = np.array([s['digbi_id'] for s in samples])
    labels    = np.array([s['label']    for s in samples])
    unique_ids = np.unique(digbi_ids)

    id2label = {}
    for pid in unique_ids:
        idx = np.where(digbi_ids == pid)[0]
        lbls = labels[idx]
        if len(set(lbls)) > 1:
            print(f"Mixed labels for {pid}: {list(lbls)}; using majority.")
        id2label[pid] = int(np.round(lbls.mean()))

    unique_labels = np.array([id2label[pid] for pid in unique_ids])

    train_ids, holdout_ids = train_test_split(
        unique_ids,
        test_size=(val_frac + test_frac),
        random_state=seed,
        stratify=unique_labels
    )

    holdout_labels = np.array([id2label[pid] for pid in holdout_ids])
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=holdout_labels
    )

    def subset(keep_ids):
        keep = set(keep_ids)
        return [s for s in samples if s['digbi_id'] in keep]

    train_samples = subset(train_ids)
    val_samples   = subset(val_ids)
    test_samples  = subset(test_ids)

    # Verify no leakage
    assert set(train_ids).isdisjoint(set(val_ids)),  "Leak: train & val share IDs"
    assert set(train_ids).isdisjoint(set(test_ids)), "Leak: train & test share IDs"
    assert set(val_ids).isdisjoint(set(test_ids)),   "Leak: val & test share IDs"
    print("No participant ID leakage across splits.")

    for name, split in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        ids = sorted(set(s['digbi_id'] for s in split))
        y = [s['label'] for s in split]
        print(f"{name:5s}  participants={len(ids):3d}  visits={len(split):3d}  "
              f"CTL={y.count(0)}  NMD={y.count(1)}")

    return train_samples, val_samples, test_samples


# ── Normalizer ─────────────────────────────────────────────────────────────────

def build_task_vocab(samples):
    """Assign integer IDs to task name strings."""
    vocab = {}
    for s in samples:
        for t in s['task_ids']:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def fit_normalizer(train_samples, max_files=None):
    """
    Global normalizer: compute per-joint mean and std pooled across
    all frames, all tasks, and all participants in the training set.

    Returns:
        dict with keys 'mean' and 'std', each a tensor of shape [33].

    Usage in VisitDataset: pass normalize=norm (a plain dict).
    """
    feats   = []
    counted = 0
    for s in train_samples:
        for fp in s['tasks']:
            x = df_to_tensor(read_mot(fp), fp)
            feats.append(x)
            counted += 1
            if max_files is not None and counted >= max_files:
                break
        if max_files is not None and counted >= max_files:
            break
    X    = torch.cat(feats, dim=0)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp(min=1e-6)
    return {'mean': mean, 'std': std}


def fit_normalizer_per_task(train_samples, max_files=None):
    """
    Per-task normalizer: compute per-joint mean and std separately for
    each task type, using only training data.

    This is more appropriate than global normalization when the model
    receives no explicit task identity signal, since different tasks
    have fundamentally different joint angle ranges. Normalizing within
    each task ensures the model sees joint angles expressed relative to
    what is typical for that specific movement.

    Returns:
        dict mapping task_name -> {'mean': tensor[33], 'std': tensor[33]}
        Includes a '__global__' fallback key for any task seen at inference
        but not present in training.

    Usage in VisitDataset: pass normalize=norm (a per-task dict).
    VisitDataset detects the mode automatically based on dict structure.
    """
    task_feats = {}
    counted    = 0
    for s in train_samples:
        for fp, task_name in zip(s['tasks'], s['task_ids']):
            x = df_to_tensor(read_mot(fp), fp)
            if task_name not in task_feats:
                task_feats[task_name] = []
            task_feats[task_name].append(x)
            counted += 1
            if max_files is not None and counted >= max_files:
                break
        if max_files is not None and counted >= max_files:
            break

    # Per-task statistics
    task_stats = {}
    for task_name, frames in task_feats.items():
        X    = torch.cat(frames, dim=0)
        mean = X.mean(dim=0)
        std  = X.std(dim=0).clamp(min=1e-6)
        task_stats[task_name] = {'mean': mean, 'std': std}
        print(f"  Per-task norm [{task_name}]: "
              f"n_frames={X.shape[0]}, "
              f"mean_abs_mean={mean.abs().mean():.3f}, "
              f"mean_std={std.mean():.3f}")

    # Global fallback for any task unseen during training
    all_frames = torch.cat(
        [torch.cat(v, dim=0) for v in task_feats.values()], dim=0)
    task_stats['__global__'] = {
        'mean': all_frames.mean(dim=0),
        'std':  all_frames.std(dim=0).clamp(min=1e-6),
    }
    print(f"  Global fallback: n_frames={all_frames.shape[0]}")
    return task_stats


# ── Dataset and collate ────────────────────────────────────────────────────────

class VisitDataset(Dataset):
    """
    PyTorch Dataset for per-visit kinematic samples.
    Each item corresponds to one visit (participant/date pair).
    """
    def __init__(self, samples, task_vocab, normalize=None):
        self.samples = samples
        self.task_vocab = task_vocab
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        task_tensors, task_ids_int = [], []

        for fpath, task_name in zip(rec['tasks'], rec['task_ids']):
            x = df_to_tensor(read_mot(fpath), fpath)
            if self.normalize is not None:
                # Auto-detect normalization mode:
                # - Global:   normalize is {'mean': tensor, 'std': tensor}
                # - Per-task: normalize is {task_name: {'mean':..,'std':..}, ...}
                if 'mean' in self.normalize:
                    # Global mode
                    norm = self.normalize
                else:
                    # Per-task mode — fall back to global if task unseen
                    norm = self.normalize.get(
                        task_name,
                        self.normalize.get('__global__', None)
                    )
                if norm is not None:
                    x = (x - norm['mean']) / (norm['std'] + 1e-8)
            task_tensors.append(x)
            task_ids_int.append(self.task_vocab[task_name])

        return {
            'digbi_id':     rec['digbi_id'],
            'date':         rec['date'],
            'label':        torch.tensor(int(rec['label']), dtype=torch.long),
            'task_tensors': task_tensors,
            'task_ids_int': torch.tensor(task_ids_int, dtype=torch.long)
        }


def collate_visits(batch):
    """
    Collate a batch of visit samples into padded tensors.

    Returns:
        tasks_padded:  [T_batch, L_max, 33]  float32
        tasks_mask:    [T_batch, L_max]       bool (True = valid)
        task_ids:      [T_batch]              long
        visit_idx:     [T_batch]              long (0..B-1)
        labels:        [B]                    long
        digbi_ids:     list[str]
        dates:         list[str]
    """
    B = len(batch)
    digbi_ids = [b['digbi_id'] for b in batch]
    dates     = [b['date']     for b in batch]
    labels    = torch.stack([b['label'] for b in batch], dim=0)

    all_tasks, all_task_ids, all_visit_idx = [], [], []
    for i, b in enumerate(batch):
        for x, t_id in zip(b['task_tensors'], b['task_ids_int']):
            all_tasks.append(x)
            all_task_ids.append(t_id)
            all_visit_idx.append(i)

    if len(all_tasks) == 0:
        return (torch.empty(0, 0, 0), torch.empty(0, 0, dtype=torch.bool),
                torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                labels, digbi_ids, dates)

    L_max   = max(t.shape[0] for t in all_tasks)
    F       = all_tasks[0].shape[1]
    T_batch = len(all_tasks)

    tasks_padded = torch.zeros((T_batch, L_max, F), dtype=torch.float32)
    tasks_mask   = torch.zeros((T_batch, L_max),    dtype=torch.bool)

    for i, t in enumerate(all_tasks):
        L = t.shape[0]
        tasks_padded[i, :L, :] = t
        tasks_mask[i, :L]      = True

    task_ids  = torch.stack(all_task_ids, dim=0)
    visit_idx = torch.tensor(all_visit_idx, dtype=torch.long)

    return tasks_padded, tasks_mask, task_ids, visit_idx, labels, digbi_ids, dates