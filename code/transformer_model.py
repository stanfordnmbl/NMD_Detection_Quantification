"""
transformer_model.py
--------------------
Three transformer variants for NMD detection from kinematic time series.
All share the same encoder architecture — only the aggregation strategy
and temporal encoding differ.

VisitTransformer (baseline) - used in paper!
    Mean pooling across tasks. Each task contributes equally to the visit
    representation. Original architecture — saved checkpoints in
    cv_models_baseline/ use this class.

VisitTransformerAttn (attention)
    Learned attention-weighted aggregation across tasks. A small MLP scores
    each task embedding; scores are softmax-normalized within each visit.
    Allows the model to upweight the most informative tasks per patient.
    Saved checkpoints in cv_models_attn/ use this class.

VisitTransformerPE (positional encoding)
    Adds learned positional embeddings to the pooled sequence before the
    transformer encoder, allowing the model to learn temporal patterns
    within each task — e.g. how joint angles evolve through a sit-to-stand
    or gait cycle — rather than treating all timesteps as exchangeable.
    Positional encoding is applied after temporal pooling (on the pooled
    sequence of length L') so positions refer to pooled timesteps.
    Mean pooling across tasks (same as baseline).

    Motivation: The baseline and attention models are order-invariant —
    they cannot distinguish the beginning from the end of a movement
    sequence. Positional encoding is hypothesized to improve convergent
    validity (TFT/ACTIVLIM correlations) by capturing temporal movement
    dynamics that correlate with functional severity, without necessarily
    improving binary discrimination.

    MAX_SEQ_LEN controls the maximum pooled sequence length the positional
    embedding table covers. With pool_stride=4 and typical raw sequences
    of 60–240 frames, pooled sequences are 15–60 timesteps. The default
    of 128 is conservative and covers all expected cases.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Base class — shared encoder logic ─────────────────────────────────────────

class VisitTransformer(nn.Module):
    """
    Encoder-only transformer with mean pooling across tasks.

    Architecture:
        Input [T_batch, L_max, input_dim]
            → Linear projection (input_dim → d_model)
            → Masked temporal average pooling (stride=pool_stride)
            → Transformer encoder (num_layers × n_heads)
            → Masked mean pool over time → z_task [d_model] per task
            → Mean pool across tasks per visit → z_visit [d_model]
            → Linear head → scalar logit [B]
    """

    def __init__(
        self,
        input_dim=33,
        d_model=64,
        n_heads=4,
        num_layers=3,
        dropout=0.3,
        pool_stride=4,
        pool_kernel=None,
        mask_majority=0.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # norm_first=True makes TransformerEncoder skip the nested-tensor fast path
        # and emit a benign UserWarning on every construction — silence just that one.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
            self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, 1)

        self.pool_stride   = int(pool_stride) if pool_stride else 1
        self.pool_kernel   = int(pool_kernel) if pool_kernel else self.pool_stride
        self.mask_majority = float(mask_majority)

    # ── Internal methods (overridden by subclasses) ────────────────────────────

    def _temporal_pool(self, x, mask):
        """
        Masked temporal average pooling with stride.

        Args:
            x:    [T, L, d]  — T tasks in batch, L timesteps, d dims
            mask: [T, L]     — True = valid timestep

        Returns:
            x_pooled:    [T, L', d]
            mask_pooled: [T, L']
        """
        if self.pool_stride <= 1:
            return x, mask

        m    = mask.float()
        x_ch = x.transpose(1, 2)
        eps  = 1e-6

        m_avg = F.avg_pool1d(
            m.unsqueeze(1),
            kernel_size=self.pool_kernel,
            stride=self.pool_stride
        ).squeeze(1)

        x_sum = F.avg_pool1d(
            x_ch * m.unsqueeze(1),
            kernel_size=self.pool_kernel,
            stride=self.pool_stride
        )

        x_pooled    = (x_sum / m_avg.clamp_min(eps).unsqueeze(1)).transpose(1, 2)
        mask_pooled = m_avg > self.mask_majority

        any_valid = mask_pooled.any(dim=1)
        if not any_valid.all():
            mask_pooled[~any_valid, 0] = True

        return x_pooled, mask_pooled

    def _add_positional_encoding(self, x, mask):
        """
        Optional positional encoding applied to the pooled sequence.
        Base class is a no-op — overridden by VisitTransformerPE.

        Args:
            x:    [T, L', d]
            mask: [T, L']

        Returns:
            x:    [T, L', d]  (unchanged in base class)
            mask: [T, L']     (unchanged)
        """
        return x, mask

    def _encode_tasks(self, tasks_padded, tasks_mask):
        """
        Project → temporal pool → (optional PE) → encode → masked mean pool.
        Returns z_task [T, d] — one embedding per task.
        Shared by all subclasses.
        """
        x    = self.input_proj(tasks_padded)      # [T, L, d]
        mask = tasks_mask.bool()

        x, mask = self._temporal_pool(x, mask)               # [T, L', d]
        x, mask = self._add_positional_encoding(x, mask)     # [T, L', d] (PE variant only)

        x = self.encoder(x, src_key_padding_mask=~mask)      # [T, L', d]

        m      = mask.float().unsqueeze(-1)                   # [T, L', 1]
        z_task = (x * m).sum(1) / m.sum(1).clamp_min(1.0)   # [T, d]
        return z_task

    def _aggregate(self, z_task, visit_idx, B):
        """
        Mean pool z_task across tasks belonging to the same visit.
        Returns z_visit [B, d].
        Overridden by VisitTransformerAttn.
        """
        d       = z_task.size(1)
        device  = z_task.device
        z_visit = torch.zeros(B, d, device=device)
        counts  = torch.zeros(B, 1, device=device)
        z_visit.index_add_(0, visit_idx, z_task)
        counts.index_add_(0, visit_idx,
                          torch.ones(z_task.size(0), 1, device=device))
        return z_visit / counts.clamp_min(1.0)

    def forward(self, tasks_padded, tasks_mask, task_ids, visit_idx, B):
        """
        Args:
            tasks_padded: [T, L, input_dim]  zero-padded task sequences
            tasks_mask:   [T, L]             True = valid timestep
            task_ids:     [T]                integer task IDs
            visit_idx:    [T]                visit assignment per task
            B:            int                number of visits in batch

        Returns:
            logits: [B]  pre-sigmoid scalar per visit
        """
        z_task  = self._encode_tasks(tasks_padded, tasks_mask)
        z_visit = self._aggregate(z_task, visit_idx, B)
        return self.cls_head(z_visit).squeeze(-1)


# ── Attention variant ──────────────────────────────────────────────────────────

class VisitTransformerAttn(VisitTransformer):
    """
    VisitTransformer with learned attention-weighted task aggregation.

    Replaces mean pooling across tasks with a small two-layer MLP that
    scores each task embedding. Scores are softmax-normalized within each
    visit so weights sum to 1 per visit independently.

    Only _aggregate is overridden — all other methods are inherited.

    New parameter:
        attn_hidden (int): hidden dim of the task scoring MLP (default 32)
    """

    def __init__(self, attn_hidden=32, **kwargs):
        super().__init__(**kwargs)
        d_model = kwargs.get("d_model", 64)
        self.task_attn = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )

    def _aggregate(self, z_task, visit_idx, B):
        """
        Attention-weighted aggregation: score each task, softmax within
        each visit, return weighted sum.
        """
        scores     = self.task_attn(z_task).squeeze(-1)   # [T]
        scores_exp = torch.zeros_like(scores)

        for b in range(B):
            mask_b = visit_idx == b
            if mask_b.any():
                s = scores[mask_b]
                s = s - s.max()          # numerical stability
                s = torch.exp(s)
                scores_exp[mask_b] = s / s.sum()

        d       = z_task.size(1)
        device  = z_task.device
        z_visit = torch.zeros(B, d, device=device)
        z_visit.index_add_(0, visit_idx,
                           z_task * scores_exp.unsqueeze(-1))
        return z_visit

    def forward(self, tasks_padded, tasks_mask, task_ids, visit_idx, B):
        z_task  = self._encode_tasks(tasks_padded, tasks_mask)
        z_visit = self._aggregate(z_task, visit_idx, B)
        return self.cls_head(z_visit).squeeze(-1)

    @torch.no_grad()
    def forward_with_attn_weights(self, tasks_padded, tasks_mask,
                                  task_ids, visit_idx, B):
        """
        Same as forward() but also returns per-task attention weights.

        Returns:
            logits:       [B]   visit-level logits
            attn_weights: [T]   per-task weight (sums to 1 within each visit)
            task_ids:     [T]
            visit_idx:    [T]
        """
        z_task     = self._encode_tasks(tasks_padded, tasks_mask)
        scores     = self.task_attn(z_task).squeeze(-1)
        scores_exp = torch.zeros_like(scores)

        for b in range(B):
            mask_b = visit_idx == b
            if mask_b.any():
                s = scores[mask_b]
                s = s - s.max()
                s = torch.exp(s)
                scores_exp[mask_b] = s / s.sum()

        d       = z_task.size(1)
        device  = z_task.device
        z_visit = torch.zeros(B, d, device=device)
        z_visit.index_add_(0, visit_idx,
                           z_task * scores_exp.unsqueeze(-1))
        logits = self.cls_head(z_visit).squeeze(-1)
        return logits, scores_exp, task_ids, visit_idx


# ── Positional encoding variant ───────────────────────────────────────────────

class VisitTransformerPE(VisitTransformer):
    """
    VisitTransformer with learned positional embeddings on the pooled sequence.

    The baseline and attention models are order-invariant — masked mean pooling
    over time and the lack of positional information mean the transformer cannot
    distinguish the beginning from the end of a movement sequence. This variant
    adds a learned positional embedding to each position in the pooled sequence
    before the transformer encoder, allowing the model to learn temporal patterns
    within each task.

    Positional encoding is applied AFTER temporal pooling (on the L' pooled
    timesteps, not the original L timesteps). This is the correct placement
    because:
        1. The pooled timesteps are the actual input to the encoder
        2. Applying PE before pooling would have the encodings averaged
           together during pooling, destroying their position-specificity

    With pool_stride=4 and typical sequences of 60-240 frames, pooled
    sequences are 15-60 timesteps. MAX_SEQ_LEN=128 covers all expected cases
    with a comfortable margin.

    Mean pooling across tasks (same as baseline) — only temporal encoding
    differs from the base class.

    New parameter:
        max_seq_len (int): size of the positional embedding table (default 128)
    """

    def __init__(self, max_seq_len=128, **kwargs):
        super().__init__(**kwargs)
        d_model = kwargs.get("d_model", 64)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len   = max_seq_len

    def _add_positional_encoding(self, x, mask):
        """
        Add learned positional embeddings to pooled sequence.

        Args:
            x:    [T, L', d]  pooled sequence
            mask: [T, L']     validity mask

        Returns:
            x:    [T, L', d]  sequence with positional embeddings added
            mask: [T, L']     unchanged
        """
        L_prime = x.size(1)

        if L_prime > self.max_seq_len:
            # Truncate if sequence exceeds table — shouldn't happen with
            # max_seq_len=128 but guard against edge cases
            x    = x[:, :self.max_seq_len, :]
            mask = mask[:, :self.max_seq_len]
            L_prime = self.max_seq_len

        positions = torch.arange(L_prime, device=x.device).unsqueeze(0)  # [1, L']
        pe        = self.pos_embedding(positions)                          # [1, L', d]
        x         = x + pe                                                 # [T, L', d]
        return x, mask

    # forward() is fully inherited from VisitTransformer — no override needed
    # since _add_positional_encoding is called inside _encode_tasks