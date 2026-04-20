"""
Enhanced DIRT+ built on top of the normalized DIRT training/evaluation protocol.

New modules:
1. Causal temporal self-attention on top of TransitionRNN outputs.
2. Exercise-aware query attention for target-oriented history aggregation.
3. Dynamic weighted prediction loss on valid time steps.
4. Frozen stage1 teacher consistency in stage2.
5. Stronger stability / reproducibility defaults.
"""
import ast
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


EPSILON = 1e-8
DEFAULT_LOSS_WEIGHT_CONFIG = {
    'loss_weight_mode': 'rule',
    'step_weight_hidden_dim': 32,
    'step_weight_dropout': 0.1,
    'step_weight_use_teacher_confidence': 1,
    'step_weight_use_position_feature': 1,
    'step_weight_min': 0.5,
    'step_weight_max': 3.0,
    'focal_gamma': 2.0,
    'focal_alpha': None,
    'use_temporal_self_attention': 1,
    'use_query_guided_attention': 1
}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def config_ws(ws, config_dict):
    config_dict = dict(config_dict)
    for key, value in DEFAULT_LOSS_WEIGHT_CONFIG.items():
        config_dict.setdefault(key, value)
    os.makedirs(ws, exist_ok=True)
    os.makedirs(os.path.join(ws, 'snapshot'), exist_ok=True)
    with open(os.path.join(ws, 'model_config.txt'), 'w', encoding='utf8') as o_f:
        json.dump(config_dict, o_f, indent=4)


def read_ws_config(ws):
    config_path = os.path.join(ws, 'model_config.txt')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'The work space has not been configured yet: {ws}')
    with open(config_path, 'r', encoding='utf8') as i_f:
        ws_config = json.load(i_f)
    for key, value in DEFAULT_LOSS_WEIGHT_CONFIG.items():
        ws_config.setdefault(key, value)
    return ws_config


def reset_metric_logs(ws):
    for log_name in ['results.txt', 'metrics.jsonl']:
        with open(os.path.join(ws, log_name), 'w', encoding='utf8'):
            pass


def append_metric_record(ws, record):
    with open(os.path.join(ws, 'metrics.jsonl'), 'a', encoding='utf8') as o_f:
        o_f.write(json.dumps(record, ensure_ascii=True) + '\n')


def append_text_line(file_path, line):
    with open(file_path, 'a', encoding='utf8') as o_f:
        o_f.write(line + '\n')


def build_experiment_name(args):
    if getattr(args, 'exp_name', None):
        return args.exp_name
    loss_mode = args.loss_weight_mode
    stage1_lr_str = f'{args.stage1_lr if args.stage1_lr is not None else args.lr:.4g}'.replace('.', 'p')
    stage2_lr_str = f'{args.stage2_lr if args.stage2_lr is not None else args.lr * 0.5:.4g}'.replace('.', 'p')
    range_str = f'{args.step_weight_min:.2f}_{args.step_weight_max:.2f}'.replace('.', 'p')
    focal_str = f'{args.focal_gamma:.2f}'.replace('.', 'p')
    return (
        f'{loss_mode}_s1lr{stage1_lr_str}_s2lr{stage2_lr_str}_'
        f'range{range_str}_tc{int(args.step_weight_use_teacher_confidence)}_'
        f'pos{int(args.step_weight_use_position_feature)}_fg{focal_str}'
    )


def emit_final_experiment_summary(ws, summary_record):
    append_metric_record(ws, summary_record)
    summary_line = (
        f"[final-summary] attr={summary_record['attr_name']}, best_stage={summary_record['best_stage']}, "
        f"best_epoch={summary_record['best_epoch']}, validation_auc={summary_record['validation_auc']}, "
        f"validation_acc={summary_record['validation_acc']}, validation_rmse={summary_record['validation_rmse']}, "
        f"test_auc={summary_record['test_auc']}, test_acc={summary_record['test_acc']}, "
        f"test_rmse={summary_record['test_rmse']}, loss_weight_mode={summary_record['loss_weight_mode']}, "
        f"step_weight_mean={summary_record['step_weight_mean']}, step_weight_std={summary_record['step_weight_std']}, "
        f"step_weight_min={summary_record['step_weight_min']}, step_weight_max={summary_record['step_weight_max']}, "
        f"focal_factor_mean={summary_record['focal_factor_mean']}"
    )
    append_text_line(os.path.join(ws, 'results.txt'), summary_line)
    print(summary_line)


def emit_root_experiment_summary(ws_root, exp_name, summary_records):
    summary_file = os.path.join(ws_root, 'experiment_summary.txt')
    header = f'========== experiment={exp_name} =========='
    append_text_line(summary_file, header)
    for record in summary_records:
        line = (
            f"attr={record['attr_name']}, best_stage={record['best_stage']}, best_epoch={record['best_epoch']}, "
            f"validation_auc={record['validation_auc']}, validation_acc={record['validation_acc']}, "
            f"validation_rmse={record['validation_rmse']}, test_auc={record['test_auc']}, "
            f"test_acc={record['test_acc']}, test_rmse={record['test_rmse']}, "
            f"loss_weight_mode={record['loss_weight_mode']}, step_weight_mean={record['step_weight_mean']}, "
            f"step_weight_std={record['step_weight_std']}, step_weight_min={record['step_weight_min']}, "
            f"step_weight_max={record['step_weight_max']}, focal_factor_mean={record['focal_factor_mean']}"
        )
        append_text_line(summary_file, line)
    append_text_line(summary_file, '')


def find_best_epoch(ws, stage, dtype='validation', key='auc'):
    metrics_path = os.path.join(ws, 'metrics.jsonl')
    records = []
    with open(metrics_path, 'r', encoding='utf8') as i_f:
        for line in i_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('stage') == stage and record.get('dtype') == dtype:
                records.append(record)

    if not records:
        raise ValueError(f'No records found for dtype={dtype}, stage={stage}.')
    best_record = max(records, key=lambda item: (item[key], -item['epoch']))
    return int(best_record['epoch']), float(best_record[key])


def find_best_epoch_by_validation_priority(ws, stage, dtype='validation'):
    metrics_path = os.path.join(ws, 'metrics.jsonl')
    records = []
    with open(metrics_path, 'r', encoding='utf8') as i_f:
        for line in i_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('stage') == stage and record.get('dtype') == dtype:
                records.append(record)
    if not records:
        raise ValueError(f'No records found for dtype={dtype}, stage={stage}.')
    best_record = max(
        records,
        key=lambda item: (
            float(item.get('auc', 0.0)),
            float(item.get('acc', 0.0)),
            -float(item.get('rmse', float('inf'))),
            -int(item.get('epoch', 0))
        )
    )
    return (
        int(best_record['epoch']),
        float(best_record.get('auc', 0.0)),
        float(best_record.get('acc', 0.0)),
        float(best_record.get('rmse', 0.0))
    )


def build_sequence_mask(log_lens, max_log, device):
    if not torch.is_tensor(log_lens):
        log_lens = torch.as_tensor(log_lens, device=device)
    else:
        log_lens = log_lens.to(device)
    time_ids = torch.arange(max_log - 1, device=device).unsqueeze(0)
    return time_ids < (log_lens.unsqueeze(1) - 1)


def safe_roc_auc_score(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    finite_mask = np.isfinite(labels) & np.isfinite(preds)
    labels = labels[finite_mask]
    preds = preds[finite_mask]
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5
    return roc_auc_score(labels, preds)


def sanitize_prediction_tensor(pred_tensor):
    pred_tensor = torch.nan_to_num(pred_tensor, nan=0.5, posinf=1.0 - EPSILON, neginf=EPSILON)
    pred_tensor = torch.clamp(pred_tensor, min=EPSILON, max=1.0 - EPSILON)
    return pred_tensor


def sanitize_hidden_tensor(hidden_tensor):
    return torch.nan_to_num(hidden_tensor, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_prediction_arrays(pred_all, pred_label_all):
    pred_all = np.asarray(pred_all, dtype=np.float64)
    pred_label_all = np.asarray(pred_label_all, dtype=np.float64)
    finite_mask = np.isfinite(pred_all) & np.isfinite(pred_label_all)
    pred_all = pred_all[finite_mask]
    pred_label_all = pred_label_all[finite_mask]
    return pred_all, pred_label_all


def summarize_masked_tensor(values, valid_mask):
    valid_values = values[valid_mask > 0]
    if valid_values.numel() == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    return {
        'mean': float(valid_values.mean().item()),
        'std': float(valid_values.std(unbiased=False).item()) if valid_values.numel() > 1 else 0.0,
        'min': float(valid_values.min().item()),
        'max': float(valid_values.max().item())
    }


def get_optimizer_lr(optimizer):
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]['lr'])


def get_consistency_weight(epoch, max_weight, warmup_epochs, warmup_start_ratio):
    if max_weight <= 0.0:
        return 0.0
    warmup_epochs = max(int(warmup_epochs), 1)
    warmup_start_ratio = float(np.clip(warmup_start_ratio, 0.0, 1.0))
    progress = min(float(epoch + 1) / float(warmup_epochs), 1.0)
    current_ratio = warmup_start_ratio + (1.0 - warmup_start_ratio) * progress
    return float(max_weight * current_ratio)


def build_stage_scheduler(optimizer, ws_config):
    if not ws_config.get('use_stage_scheduler', 1):
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=float(ws_config['scheduler_factor']),
        patience=int(ws_config['scheduler_patience']),
        min_lr=float(ws_config['scheduler_min_lr'])
    )


class IRT(nn.Module):
    def __init__(self, exer_n):
        super().__init__()
        self.e_difficulty = nn.Embedding.from_pretrained(torch.zeros(exer_n, 1), freeze=False)
        self.e_discrimination = nn.Embedding.from_pretrained(torch.zeros(exer_n, 1), freeze=False)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, ability, exer_id):
        difficulty = torch.sigmoid(self.e_difficulty(exer_id))
        discrimination = torch.sigmoid(self.e_discrimination(exer_id))
        prob_correct = torch.sigmoid(discrimination * (ability - difficulty) * 1.7)
        prob_wrong = 1.0 - prob_correct
        return torch.cat((prob_wrong, prob_correct), dim=-1)


class TransitionRNN(nn.Module):
    def __init__(self, input_size, stu_ho_dim=50, rnn_type='gru', batch_size=32):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.stu_ho_dim = stu_ho_dim
        self.batch_size = batch_size
        self.register_buffer('rnn_hidden', None)
        self.register_buffer('initial_rnn_hidden', None)
        self.full_1 = nn.Linear(self.input_size, self.stu_ho_dim)
        self.full_2 = nn.Linear(self.stu_ho_dim, self.stu_ho_dim)
        self.inst_norm_1 = nn.InstanceNorm1d(batch_size, affine=True, track_running_stats=False)
        self.inst_norm_2 = nn.InstanceNorm1d(batch_size, affine=True, track_running_stats=False)
        self.init_hidden()
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.stu_ho_dim, hidden_size=self.stu_ho_dim, num_layers=1)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.stu_ho_dim, hidden_size=self.stu_ho_dim, num_layers=1, nonlinearity='tanh')
        else:
            self.rnn = nn.LSTM(input_size=self.stu_ho_dim, hidden_size=self.stu_ho_dim, num_layers=1)

    def init_hidden(self):
        device = next(self.parameters()).device
        if self.initial_rnn_hidden is None:
            if self.rnn_type in ['gru', 'rnn']:
                self.initial_rnn_hidden = torch.zeros(1, self.batch_size, self.stu_ho_dim, device=device)
            else:
                h = torch.zeros(1, self.batch_size, self.stu_ho_dim, device=device)
                c = torch.zeros(1, self.batch_size, self.stu_ho_dim, device=device)
                self.initial_rnn_hidden = (h, c)
        if self.rnn_type in ['gru', 'rnn']:
            self.rnn_hidden = self.initial_rnn_hidden.clone()
        else:
            self.rnn_hidden = tuple(hidden.clone() for hidden in self.initial_rnn_hidden)

    def forward(self, input_x):
        input_x = self.inst_norm_1(self.full_1(input_x))
        input_x = sanitize_hidden_tensor(input_x)
        h_2toT, _ = self.rnn(input_x, self.rnn_hidden)
        h_2toT = torch.tanh(self.inst_norm_2(self.full_2(h_2toT)))
        h_2toT = sanitize_hidden_tensor(h_2toT)
        return h_2toT


class Decoder(nn.Module):
    def __init__(self, stu_ho_dim, stu_lo_dim):
        super().__init__()
        self.layer = nn.Linear(stu_ho_dim, stu_lo_dim)

    def forward(self, input_ho_state):
        return torch.sigmoid(self.layer(input_ho_state))


class CausalTemporalSelfAttention(nn.Module):
    # New enhancement: causal self-attention above the original RNN backbone.
    def __init__(
        self,
        hidden_dim,
        num_heads=2,
        dropout=0.1,
        use_multihead_temporal_attn=0,
        multihead_temporal_num_heads=2,
        multihead_attn_dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_multihead_temporal_attn = bool(use_multihead_temporal_attn)
        self.num_heads = int(multihead_temporal_num_heads if self.use_multihead_temporal_attn else 1)
        if hidden_dim % self.num_heads != 0:
            raise ValueError('hidden_dim must be divisible by temporal attention num_heads.')
        attn_dropout = multihead_attn_dropout if self.use_multihead_temporal_attn else dropout
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=self.num_heads, dropout=attn_dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, hidden_seq, key_padding_mask):
        valid_float = key_padding_mask.unsqueeze(-1).float()
        safe_hidden = hidden_seq * valid_float
        has_valid_token = key_padding_mask.any(dim=1)
        if not has_valid_token.any():
            return safe_hidden

        seq_len = hidden_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_seq.device, dtype=torch.bool),
            diagonal=1
        )
        attn_input = safe_hidden[has_valid_token]
        attn_key_padding_mask = key_padding_mask[has_valid_token]
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            key_padding_mask=~attn_key_padding_mask
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        attn_out = self.norm(attn_input + self.dropout(attn_out))
        attn_out = attn_out * attn_key_padding_mask.unsqueeze(-1).float()
        attn_out = sanitize_hidden_tensor(attn_out)

        full_attn_out = torch.zeros_like(safe_hidden)
        full_attn_out[has_valid_token] = attn_out
        return full_attn_out


class ExerciseAwareQuery(nn.Module):
    # New enhancement: query depends on target exercise and target knowledge vector.
    def __init__(self, exer_n, knowledge_n, hidden_dim):
        super().__init__()
        self.exer_embedding = nn.Embedding(exer_n, hidden_dim)
        self.knowledge_proj = nn.Linear(knowledge_n, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, next_exer_ids, next_knowledge_relevancies):
        exer_emb = self.exer_embedding(next_exer_ids)
        knowledge_emb = self.knowledge_proj(next_knowledge_relevancies)
        query = torch.tanh(self.query_proj(torch.cat([exer_emb, knowledge_emb], dim=-1)))
        return self.norm(query)


class QueryGuidedAttention(nn.Module):
    # New enhancement: target-guided aggregation over valid historical hidden states.
    def __init__(
        self,
        hidden_dim,
        dropout=0.1,
        use_temporal_bias=1,
        temporal_bias_mode='linear',
        temporal_bias_init=0.05,
        use_exercise_aware_decay=0,
        exercise_aware_decay_mode='query_linear',
        exercise_aware_decay_scale=1.0,
        exercise_aware_decay_min=0.0,
        exercise_aware_decay_max=0.2,
        use_multihead_query_attn=0,
        multihead_query_num_heads=2,
        multihead_attn_dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_multihead_query_attn = bool(use_multihead_query_attn)
        self.num_heads = int(multihead_query_num_heads if self.use_multihead_query_attn else 1)
        if hidden_dim % self.num_heads != 0:
            raise ValueError('hidden_dim must be divisible by query attention num_heads.')
        self.head_dim = hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(multihead_attn_dropout if self.use_multihead_query_attn else dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.use_temporal_bias = bool(use_temporal_bias)
        self.temporal_bias_mode = temporal_bias_mode
        if self.use_temporal_bias:
            self.temporal_bias_strength = nn.Parameter(torch.tensor(float(temporal_bias_init)))
        else:
            self.register_parameter('temporal_bias_strength', None)
        self.use_exercise_aware_decay = bool(use_exercise_aware_decay)
        self.exercise_aware_decay_mode = exercise_aware_decay_mode
        self.exercise_aware_decay_scale = float(exercise_aware_decay_scale)
        self.exercise_aware_decay_min = float(exercise_aware_decay_min)
        self.exercise_aware_decay_max = float(exercise_aware_decay_max)
        if self.use_exercise_aware_decay:
            self.decay_proj = nn.Linear(hidden_dim, 1)
        else:
            self.decay_proj = None
        self.last_decay_gamma_stats = None

    def _build_temporal_bias(self, seq_len, device, dtype):
        if not self.use_temporal_bias or seq_len <= 0:
            return None
        distances = torch.arange(seq_len - 1, -1, -1, device=device, dtype=dtype)
        if self.temporal_bias_mode == 'linear':
            distance_feature = distances
        elif self.temporal_bias_mode == 'log':
            distance_feature = torch.log1p(distances)
        else:
            raise ValueError(f'Unsupported temporal_bias_mode: {self.temporal_bias_mode}')
        penalty_strength = torch.nn.functional.softplus(self.temporal_bias_strength).to(dtype=dtype)
        return -penalty_strength * distance_feature.view(1, 1, seq_len)

    def _build_exercise_aware_decay(self, query, seq_len, device, dtype):
        if not self.use_exercise_aware_decay or seq_len <= 0:
            self.last_decay_gamma_stats = None
            return None
        if self.exercise_aware_decay_mode != 'query_linear':
            raise ValueError(f'Unsupported exercise_aware_decay_mode: {self.exercise_aware_decay_mode}')
        gamma = torch.sigmoid(self.decay_proj(query)) * self.exercise_aware_decay_scale
        gamma = torch.clamp(gamma, min=self.exercise_aware_decay_min, max=self.exercise_aware_decay_max)
        distances = torch.arange(seq_len - 1, -1, -1, device=device, dtype=dtype).view(1, 1, seq_len)
        decay_bias = -gamma.to(dtype=dtype) * distances
        self.last_decay_gamma_stats = (
            float(gamma.mean().item()),
            float(gamma.std(unbiased=False).item())
        )
        return decay_bias

    def forward(self, hidden_seq, query, valid_mask):
        batch_size, _, hidden_dim = hidden_seq.size()
        has_history = valid_mask.any(dim=1)
        full_context = torch.zeros(batch_size, hidden_dim, device=hidden_seq.device, dtype=hidden_seq.dtype)
        if not has_history.any():
            return full_context

        safe_hidden = hidden_seq[has_history]
        safe_query = query[has_history]
        safe_mask = valid_mask[has_history]
        projected_query = self.q_proj(safe_query).view(-1, safe_query.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        projected_key = self.k_proj(safe_hidden).view(-1, safe_hidden.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        projected_value = self.v_proj(safe_hidden).view(-1, safe_hidden.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(projected_query, projected_key.transpose(-1, -2)) * self.scale
        temporal_bias = self._build_temporal_bias(safe_hidden.size(1), safe_hidden.device, safe_hidden.dtype)
        if temporal_bias is not None:
            scores = scores + temporal_bias.unsqueeze(1)
        decay_bias = self._build_exercise_aware_decay(safe_query, safe_hidden.size(1), safe_hidden.device, safe_hidden.dtype)
        if decay_bias is not None:
            scores = scores + decay_bias.unsqueeze(1)
        scores = scores.masked_fill(~safe_mask.unsqueeze(1).unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn * safe_mask.unsqueeze(1).unsqueeze(1).float()
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(EPSILON)
        attn = attn / denom
        context = torch.matmul(attn, projected_value)
        context = context.transpose(1, 2).contiguous().view(-1, safe_query.size(1), self.hidden_dim)
        context = self.out_proj(context)
        context = self.norm(safe_query + self.dropout(context))
        context = sanitize_hidden_tensor(context)
        full_context[has_history] = context.squeeze(1)
        return full_context

    def get_last_decay_gamma_stats(self):
        if self.last_decay_gamma_stats is None:
            return None
        return self.last_decay_gamma_stats


class TemporalFusionLayer(nn.Module):
    # New enhancement: stable gated fusion of RNN, self-attention, and query-guided context.
    def __init__(
        self,
        hidden_dim,
        dropout=0.1,
        use_adaptive_fusion=0,
        adaptive_fusion_mode='time_step_softmax',
        adaptive_fusion_hidden_dim=None
    ):
        super().__init__()
        fusion_hidden_dim = adaptive_fusion_hidden_dim or hidden_dim
        self.use_adaptive_fusion = bool(use_adaptive_fusion)
        self.adaptive_fusion_mode = adaptive_fusion_mode
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.mix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.rnn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate_hidden = nn.Linear(hidden_dim * 3, fusion_hidden_dim)
        self.fusion_gate_out = nn.Linear(fusion_hidden_dim, 3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_gate_means = None

    def forward(self, rnn_hidden, self_attn_hidden, query_context):
        expanded_context = self.context_proj(query_context)
        fusion_input = torch.cat([rnn_hidden, self_attn_hidden, expanded_context], dim=-1)
        if self.use_adaptive_fusion:
            if self.adaptive_fusion_mode != 'time_step_softmax':
                raise ValueError(f'Unsupported adaptive_fusion_mode: {self.adaptive_fusion_mode}')
            fusion_gate_hidden = torch.tanh(self.fusion_gate_hidden(fusion_input))
            fusion_gate_logits = self.fusion_gate_out(fusion_gate_hidden)
            fusion_gate = torch.softmax(fusion_gate_logits, dim=-1)
            projected_rnn = self.rnn_proj(rnn_hidden)
            projected_attn = self.attn_proj(self_attn_hidden)
            projected_query = self.query_proj(expanded_context)
            fused = (
                fusion_gate[..., 0:1] * projected_rnn +
                fusion_gate[..., 1:2] * projected_attn +
                fusion_gate[..., 2:3] * projected_query
            )
            fused = self.norm(rnn_hidden + self.dropout(fused))
            fused = sanitize_hidden_tensor(fused)
            self.last_gate_means = fusion_gate.detach().mean(dim=(0, 1))
            return fused

        gate = torch.sigmoid(self.gate(fusion_input))
        candidate = torch.tanh(self.mix(fusion_input))
        fused = gate * candidate + (1.0 - gate) * rnn_hidden
        fused = self.norm(rnn_hidden + self.dropout(fused))
        fused = sanitize_hidden_tensor(fused)
        self.last_gate_means = None
        return fused

    def get_last_gate_means(self):
        if self.last_gate_means is None:
            return None
        return self.last_gate_means.clone()


class LearnableStepWeightNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.1, min_weight=0.5, max_weight=3.0):
        super().__init__()
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.input_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.mlp(self.input_norm(features)).squeeze(-1)
        weights = torch.nn.functional.softplus(logits)
        if self.max_weight > self.min_weight:
            weights = self.min_weight + (self.max_weight - self.min_weight) * torch.sigmoid(logits)
        else:
            weights = weights + self.min_weight
        return torch.clamp(weights, min=self.min_weight, max=self.max_weight)


class DIRTPlusModel(nn.Module):
    def __init__(
        self,
        stu_ho_dim,
        rnn_type,
        attr_idx,
        max_log,
        knowledge_n,
        exer_n,
        batch_size,
        attn_heads=2,
        dropout=0.1,
        use_multihead_temporal_attn=0,
        use_multihead_query_attn=0,
        multihead_temporal_num_heads=2,
        multihead_query_num_heads=2,
        multihead_attn_dropout=0.1,
        use_temporal_self_attention=1,
        use_query_guided_attention=1,
        use_temporal_bias=1,
        temporal_bias_mode='linear',
        temporal_bias_init=0.05,
        use_exercise_aware_decay=0,
        exercise_aware_decay_mode='query_linear',
        exercise_aware_decay_scale=1.0,
        exercise_aware_decay_min=0.0,
        exercise_aware_decay_max=0.2,
        use_adaptive_fusion=0,
        adaptive_fusion_mode='time_step_softmax',
        adaptive_fusion_hidden_dim=None,
        step_weight_hidden_dim=32,
        step_weight_dropout=0.1,
        step_weight_min=0.5,
        step_weight_max=3.0
    ):
        super().__init__()
        self.stu_ho_dim = stu_ho_dim
        self.rnn_type = rnn_type
        self.attr_idx = attr_idx
        self.stu_lo_dim = 1
        self.seq_len = max_log
        self.exer_n = exer_n
        self.knowledge_n = knowledge_n
        self.batch_size = batch_size
        self.use_temporal_self_attention = bool(use_temporal_self_attention)
        self.use_query_guided_attention = bool(use_query_guided_attention)

        if self.attr_idx == 1:
            self.transition_rnn = TransitionRNN(exer_n * 2, stu_ho_dim, rnn_type, batch_size)
        else:
            self.transition_rnn = TransitionRNN(knowledge_n * 2, stu_ho_dim, rnn_type, batch_size)
        self.temporal_self_attention = CausalTemporalSelfAttention(
            stu_ho_dim,
            num_heads=attn_heads,
            dropout=dropout,
            use_multihead_temporal_attn=use_multihead_temporal_attn,
            multihead_temporal_num_heads=multihead_temporal_num_heads,
            multihead_attn_dropout=multihead_attn_dropout
        )
        self.exercise_aware_query = ExerciseAwareQuery(exer_n, knowledge_n, stu_ho_dim)
        self.query_guided_attention = QueryGuidedAttention(
            stu_ho_dim,
            dropout=dropout,
            use_multihead_query_attn=use_multihead_query_attn,
            multihead_query_num_heads=multihead_query_num_heads,
            multihead_attn_dropout=multihead_attn_dropout,
            use_temporal_bias=use_temporal_bias,
            temporal_bias_mode=temporal_bias_mode,
            temporal_bias_init=temporal_bias_init,
            use_exercise_aware_decay=use_exercise_aware_decay,
            exercise_aware_decay_mode=exercise_aware_decay_mode,
            exercise_aware_decay_scale=exercise_aware_decay_scale,
            exercise_aware_decay_min=exercise_aware_decay_min,
            exercise_aware_decay_max=exercise_aware_decay_max
        )
        self.temporal_fusion = TemporalFusionLayer(
            stu_ho_dim,
            dropout=dropout,
            use_adaptive_fusion=use_adaptive_fusion,
            adaptive_fusion_mode=adaptive_fusion_mode,
            adaptive_fusion_hidden_dim=adaptive_fusion_hidden_dim
        )
        self.decoder = Decoder(stu_ho_dim, self.stu_lo_dim)
        self.irt = IRT(exer_n)
        self.step_weight_feature_dim = 10
        self.learnable_step_weight_net = LearnableStepWeightNet(
            input_dim=self.step_weight_feature_dim,
            hidden_dim=step_weight_hidden_dim,
            dropout=step_weight_dropout,
            min_weight=step_weight_min,
            max_weight=step_weight_max
        )

    def init_hidden(self):
        self.transition_rnn.init_hidden()

    def train(self, mode=1):
        if mode not in [0, 1, 2]:
            raise ValueError('mode must be one of [0, 1, 2].')
        if mode == 0:
            super().train(False)
            return

        super().train(True)
        if mode == 1:
            for param in self.parameters():
                param.requires_grad_(True)
            return

        for param in self.parameters():
            param.requires_grad_(False)
        for module in [
            self.transition_rnn,
            self.temporal_self_attention,
            self.exercise_aware_query,
            self.query_guided_attention,
            self.temporal_fusion,
            self.decoder,
            self.learnable_step_weight_net
        ]:
            for param in module.parameters():
                param.requires_grad_(True)

    def _build_rnn_input(self, exer_id, knowledge_relevancy, corrs, stage):
        device = exer_id.device
        seq_steps = self.seq_len - 1
        if self.attr_idx == 1:
            rnn_input = torch.zeros(self.batch_size, seq_steps, 2 * self.exer_n, device=device)
            step_exer = exer_id[:, :-1]
            step_corr = corrs[:, :-1].long()
            scatter_idx = step_corr * self.exer_n + step_exer
            rnn_input.scatter_(2, scatter_idx.unsqueeze(-1), 1.0)
            return rnn_input

        if stage == 1 or self.attr_idx == 2:
            emb = knowledge_relevancy[:, :-1]
        elif self.attr_idx == 3:
            e_difficulty = torch.sigmoid(self.irt.e_difficulty(exer_id[:, :-1]).detach())
            emb = knowledge_relevancy[:, :-1] * e_difficulty
        else:
            e_difficulty = torch.sigmoid(self.irt.e_difficulty(exer_id[:, :-1]).detach())
            e_discrimination = torch.sigmoid(self.irt.e_discrimination(exer_id[:, :-1]).detach())
            emb = knowledge_relevancy[:, :-1] * e_difficulty * e_discrimination

        rnn_input = torch.zeros(self.batch_size, seq_steps, 2 * self.knowledge_n, device=device)
        step_corr = corrs[:, :-1].float().unsqueeze(-1)
        rnn_input[:, :, :self.knowledge_n] = emb * (1.0 - step_corr)
        rnn_input[:, :, self.knowledge_n:] = emb * step_corr
        return rnn_input

    def _encode_states(self, exer_id, knowledge_relevancy, corrs, log_lens, stage, return_features=False):
        rnn_input = self._build_rnn_input(exer_id, knowledge_relevancy, corrs, stage)
        rnn_output = self.transition_rnn(rnn_input.transpose(0, 1)).transpose(0, 1)
        rnn_output = sanitize_hidden_tensor(rnn_output)
        valid_mask = build_sequence_mask(log_lens, self.seq_len, rnn_output.device)
        if self.use_temporal_self_attention:
            self_attn_output = self.temporal_self_attention(rnn_output, valid_mask)
            self_attn_output = sanitize_hidden_tensor(self_attn_output)
            history_source = self_attn_output
            fusion_attn_input = self_attn_output
        else:
            self_attn_output = torch.zeros_like(rnn_output)
            history_source = rnn_output
            fusion_attn_input = self_attn_output

        next_exer_ids = exer_id[:, 1:]
        next_knowledge_relevancies = knowledge_relevancy[:, 1:]
        next_exer_emb = sanitize_hidden_tensor(self.exercise_aware_query.exer_embedding(next_exer_ids))
        if self.use_query_guided_attention:
            query = self.exercise_aware_query(next_exer_ids, next_knowledge_relevancies)
            query = sanitize_hidden_tensor(query)
            query_context_list = []
            seq_steps = rnn_output.size(1)
            for step_idx in range(seq_steps):
                step_hidden = history_source[:, :step_idx + 1, :]
                step_mask = valid_mask[:, :step_idx + 1]
                step_query = query[:, step_idx:step_idx + 1, :]
                step_context = self.query_guided_attention(step_hidden, step_query, step_mask)
                query_context_list.append(step_context)
            query_context = torch.stack(query_context_list, dim=1)
            query_context = sanitize_hidden_tensor(query_context)
        else:
            query = torch.zeros_like(next_exer_emb)
            query_context = torch.zeros_like(rnn_output)

        fused_hidden = self.temporal_fusion(rnn_output, fusion_attn_input, query_context)
        fused_hidden = fused_hidden * valid_mask.unsqueeze(-1).float()
        fused_hidden = sanitize_hidden_tensor(fused_hidden)
        pred_lo_states = self.decoder(fused_hidden)
        pred_lo_states = torch.clamp(sanitize_hidden_tensor(pred_lo_states), min=EPSILON, max=1.0 - EPSILON)
        if return_features:
            feature_dict = {
                'rnn_hidden': rnn_output,
                'self_attn_hidden': self_attn_output,
                'query_context': query_context,
                'fused_hidden': fused_hidden,
                'query': query,
                'exercise_embedding': next_exer_emb
            }
            return pred_lo_states, valid_mask, feature_dict
        return pred_lo_states, valid_mask

    def forward(self, stu_id, exer_id, knowledge_relevancy, corrs, log_lens, stage, return_features=False):
        encode_outputs = self._encode_states(
            exer_id, knowledge_relevancy, corrs, log_lens, stage, return_features=return_features
        )
        if return_features:
            pred_lo_states, valid_mask, feature_dict = encode_outputs
        else:
            pred_lo_states, valid_mask = encode_outputs
        pred_out = self.irt(pred_lo_states, exer_id[:, 1:])
        pred_out = sanitize_prediction_tensor(pred_out)
        pred_out = pred_out * valid_mask.unsqueeze(-1).float()
        if return_features:
            return pred_out, pred_lo_states, valid_mask, feature_dict
        return pred_out, pred_lo_states, valid_mask

    def get_state(self, stu_id, exer_id, knowledge_relevancy, corrs, log_lens, stage, return_features=False):
        with torch.no_grad():
            encode_outputs = self._encode_states(
                exer_id, knowledge_relevancy, corrs, log_lens, stage, return_features=return_features
            )
        if return_features:
            pred_lo_states, valid_mask, feature_dict = encode_outputs
            detached_features = {key: value.detach() for key, value in feature_dict.items()}
            return pred_lo_states.detach(), valid_mask.detach(), detached_features
        pred_lo_states, valid_mask = encode_outputs
        return pred_lo_states.detach(), valid_mask.detach()


class DataLoader(object):
    def __init__(self, ws_config, cross_idx):
        self.batch_size = ws_config['batch_size']
        self.knowledge_dim = ws_config['knowledge_n']
        self.max_log = ws_config['max_log']
        self.data = []
        self.ptr = 0
        file_name = f"data/{ws_config['data']}/train_{cross_idx}.json"
        with open(file_name, encoding='utf8') as i_f:
            self.data = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None
        next_ptr = min(self.ptr + self.batch_size, len(self.data))
        batch_len = next_ptr - self.ptr
        stu_ids, log_lens, exer_ids, corrs = [], [], [], []
        knowledge_relevancies = np.zeros((self.batch_size, self.max_log, self.knowledge_dim), dtype=np.float32)
        for i in range(self.ptr, next_ptr):
            stu_i = self.data[i]
            stu_ids.append(stu_i[0] - 1)
            log_len = stu_i[1]
            log_lens.append(log_len)
            stu_exer_ids, stu_corrs = [], []
            stu_knowledge_relevancies = np.zeros((self.max_log, self.knowledge_dim), dtype=np.float32)
            for j in range(log_len):
                log_j = stu_i[2][j]
                stu_exer_ids.append(log_j[1] - 1)
                stu_corrs.append(log_j[2])
                for skill in log_j[3]:
                    stu_knowledge_relevancies[j][skill - 1] = 1.0
            stu_exer_ids += [0] * (self.max_log - log_len)
            stu_corrs += [0] * (self.max_log - log_len)
            exer_ids.append(stu_exer_ids)
            corrs.append(stu_corrs)
            knowledge_relevancies[i - self.ptr] = stu_knowledge_relevancies
        if batch_len < self.batch_size:
            pad_len = self.batch_size - batch_len
            stu_ids += [0] * pad_len
            log_lens += [0] * pad_len
            exer_ids += [[0] * self.max_log] * pad_len
            corrs += [[0] * self.max_log] * pad_len
        self.ptr = next_ptr
        return (
            batch_len,
            np.array(stu_ids),
            np.array(log_lens),
            torch.LongTensor(exer_ids),
            torch.tensor(knowledge_relevancies),
            torch.LongTensor(corrs)
        )

    def is_end(self):
        return self.ptr >= len(self.data)

    def reset(self):
        self.ptr = 0
        random.shuffle(self.data)


class ValTestDataLoader(object):
    def __init__(self, ws_config, cross_idx, dtype='validation'):
        self.batch_size = ws_config['batch_size']
        self.knowledge_dim = ws_config['knowledge_n']
        self.max_log = ws_config['max_log']
        self.data = []
        self.ptr = 0
        if dtype == 'validation':
            file_name = f"data/{ws_config['data']}/val_{cross_idx}.json"
        else:
            file_name = f"data/{ws_config['data']}/test.json"
        with open(file_name, encoding='utf8') as i_f:
            self.data = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None
        next_ptr = min(self.ptr + self.batch_size, len(self.data))
        batch_len = next_ptr - self.ptr
        stu_ids, log_lens, exer_ids, corrs = [], [], [], []
        knowledge_relevancies = np.zeros((self.batch_size, self.max_log, self.knowledge_dim), dtype=np.float32)
        for i in range(self.ptr, next_ptr):
            stu_i = self.data[i]
            stu_ids.append(stu_i[0] - 1)
            log_len = stu_i[1]
            log_lens.append(log_len)
            stu_exer_ids, stu_corrs = [], []
            stu_knowledge_relevancies = np.zeros((self.max_log, self.knowledge_dim), dtype=np.float32)
            for j in range(log_len):
                log_j = stu_i[2][j]
                stu_exer_ids.append(log_j[1] - 1)
                stu_corrs.append(log_j[2])
                for skill in log_j[3]:
                    stu_knowledge_relevancies[j][skill - 1] = 1.0
            stu_exer_ids += [0] * (self.max_log - log_len)
            stu_corrs += [0] * (self.max_log - log_len)
            exer_ids.append(stu_exer_ids)
            corrs.append(stu_corrs)
            knowledge_relevancies[i - self.ptr] = stu_knowledge_relevancies
        if batch_len < self.batch_size:
            pad_len = self.batch_size - batch_len
            stu_ids += [0] * pad_len
            log_lens += [0] * pad_len
            exer_ids += [[0] * self.max_log] * pad_len
            corrs += [[0] * self.max_log] * pad_len
        self.ptr = next_ptr
        return (
            batch_len,
            np.array(stu_ids),
            np.array(log_lens),
            torch.LongTensor(exer_ids),
            torch.tensor(knowledge_relevancies),
            torch.LongTensor(corrs)
        )

    def is_end(self):
        return self.ptr >= len(self.data)

    def reset(self):
        self.ptr = 0


def compute_dynamic_weighted_loss(pred_out, corrs, valid_mask, next_knowledge_relevancies):
    # Backward-compatible rule weighting entry.
    pred_out = sanitize_prediction_tensor(pred_out)
    pred_prob = pred_out[:, :, 1]
    pred_loss = compute_base_bce_loss(pred_prob, corrs[:, 1:].float())
    knowledge_density = next_knowledge_relevancies.sum(dim=-1)
    knowledge_density = knowledge_density / knowledge_density.max(dim=1, keepdim=True).values.clamp_min(1.0)
    uncertainty = 1.0 - torch.abs(pred_prob.detach() - 0.5) * 2.0
    weights = 1.0 + 0.35 * knowledge_density + 0.35 * uncertainty
    weights = torch.clamp(weights, min=0.5, max=3.0)

    valid_mask_float = valid_mask.float()
    effective_weights = weights * valid_mask_float
    valid_steps_per_sample = valid_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized_weights = effective_weights / effective_weights.sum(dim=1, keepdim=True).clamp_min(EPSILON)
    normalized_weights = normalized_weights * valid_steps_per_sample

    unweighted_pred_loss = (pred_loss * valid_mask_float).sum() / valid_mask_float.sum().clamp_min(1.0)
    weighted_pred_loss = (pred_loss * normalized_weights).sum() / valid_mask_float.sum().clamp_min(1.0)
    return weighted_pred_loss, unweighted_pred_loss, normalized_weights.detach()


def compute_base_bce_loss(pred_prob, target):
    pred_prob = torch.clamp(pred_prob, min=EPSILON, max=1.0 - EPSILON)
    return -(target * torch.log(pred_prob) + (1.0 - target) * torch.log(1.0 - pred_prob))


def compute_focal_bce_loss(pred_prob, target, gamma=2.0, alpha=None):
    pred_prob = torch.clamp(pred_prob, min=EPSILON, max=1.0 - EPSILON)
    bce_loss = compute_base_bce_loss(pred_prob, target)
    pt = target * pred_prob + (1.0 - target) * (1.0 - pred_prob)
    focal_factor = torch.pow(torch.clamp(1.0 - pt, min=0.0), gamma)
    if alpha is not None and 0.0 <= float(alpha) <= 1.0:
        alpha_factor = target * float(alpha) + (1.0 - target) * (1.0 - float(alpha))
        focal_factor = focal_factor * alpha_factor
    return bce_loss * focal_factor, focal_factor.detach(), bce_loss


def compute_teacher_confidence_feature(teacher_pred_out):
    teacher_pred_out = sanitize_prediction_tensor(teacher_pred_out)
    teacher_prob = teacher_pred_out[:, :, 1].detach()
    return 2.0 * torch.abs(teacher_prob - 0.5)


def normalize_step_weights(weights, valid_mask):
    valid_mask_float = valid_mask.float()
    effective_weights = torch.clamp(weights, min=0.0) * valid_mask_float
    valid_steps_per_sample = valid_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized_weights = effective_weights / effective_weights.sum(dim=1, keepdim=True).clamp_min(EPSILON)
    normalized_weights = normalized_weights * valid_steps_per_sample
    normalized_weights = normalized_weights * valid_mask_float
    return normalized_weights


def build_step_weight_features(
    fused_hidden,
    exercise_embedding,
    pred_prob,
    valid_mask,
    teacher_confidence=None,
    use_teacher_confidence=1,
    use_position_feature=1
):
    valid_mask_float = valid_mask.float()
    hidden_abs_mean = fused_hidden.abs().mean(dim=-1)
    hidden_mean = fused_hidden.mean(dim=-1)
    hidden_norm = torch.linalg.norm(fused_hidden, dim=-1) / np.sqrt(max(fused_hidden.size(-1), 1))
    exercise_abs_mean = exercise_embedding.abs().mean(dim=-1)
    exercise_mean = exercise_embedding.mean(dim=-1)
    exercise_norm = torch.linalg.norm(exercise_embedding, dim=-1) / np.sqrt(max(exercise_embedding.size(-1), 1))
    uncertainty = 1.0 - torch.abs(pred_prob.detach() - 0.5) * 2.0
    feature_list = [
        hidden_mean,
        hidden_abs_mean,
        hidden_norm,
        exercise_mean,
        exercise_abs_mean,
        exercise_norm,
        pred_prob.detach(),
        uncertainty
    ]
    if use_teacher_confidence and teacher_confidence is not None:
        teacher_feature = teacher_confidence.detach()
    else:
        teacher_feature = torch.zeros_like(pred_prob)
    feature_list.append(teacher_feature)

    features = torch.stack(feature_list, dim=-1)
    if use_position_feature:
        seq_steps = pred_prob.size(1)
        position_ids = torch.arange(seq_steps, device=pred_prob.device, dtype=pred_prob.dtype)
        position_feature = position_ids.unsqueeze(0) / max(seq_steps - 1, 1)
        features = torch.cat([features, position_feature.unsqueeze(-1).expand(pred_prob.size(0), -1, 1)], dim=-1)
    else:
        zero_position = torch.zeros_like(pred_prob).unsqueeze(-1)
        features = torch.cat([features, zero_position], dim=-1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features * valid_mask_float.unsqueeze(-1)


def build_rule_step_weights(pred_prob, next_knowledge_relevancies):
    knowledge_density = next_knowledge_relevancies.sum(dim=-1)
    knowledge_density = knowledge_density / knowledge_density.max(dim=1, keepdim=True).values.clamp_min(1.0)
    uncertainty = 1.0 - torch.abs(pred_prob.detach() - 0.5) * 2.0
    weights = 1.0 + 0.35 * knowledge_density + 0.35 * uncertainty
    return torch.clamp(weights, min=0.5, max=3.0)


def compute_masked_correlation(x, y, valid_mask):
    valid_values = valid_mask > 0
    if valid_values.sum() <= 1:
        return 0.0
    x = x[valid_values].float()
    y = y[valid_values].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom.item() <= EPSILON:
        return 0.0
    return float((x * y).mean().item() / denom.item())


def compute_step_weighted_pred_loss(
    pred_out,
    corrs,
    valid_mask,
    next_knowledge_relevancies,
    loss_weight_mode='rule',
    step_weight_net=None,
    fused_hidden=None,
    exercise_embedding=None,
    teacher_confidence=None,
    step_weight_use_teacher_confidence=1,
    step_weight_use_position_feature=1,
    focal_gamma=2.0,
    focal_alpha=None
):
    pred_out = sanitize_prediction_tensor(pred_out)
    target = corrs[:, 1:].float()
    pred_prob = pred_out[:, :, 1]
    valid_mask_float = valid_mask.float()
    raw_bce_loss = compute_base_bce_loss(pred_prob, target)

    use_focal = loss_weight_mode in ['focal', 'learnable_focal']
    use_learnable = loss_weight_mode in ['learnable', 'learnable_focal']
    use_plain = loss_weight_mode == 'plain'

    if use_focal:
        pred_loss, focal_factor, _ = compute_focal_bce_loss(
            pred_prob, target, gamma=focal_gamma, alpha=focal_alpha
        )
    else:
        pred_loss = raw_bce_loss
        focal_factor = torch.ones_like(pred_prob)

    if use_plain:
        raw_step_weights = torch.ones_like(pred_prob)
    elif use_learnable:
        if step_weight_net is None or fused_hidden is None or exercise_embedding is None:
            raise ValueError('Learnable step weighting requires step_weight_net, fused_hidden, and exercise_embedding.')
        step_features = build_step_weight_features(
            fused_hidden=fused_hidden,
            exercise_embedding=exercise_embedding,
            pred_prob=pred_prob,
            valid_mask=valid_mask,
            teacher_confidence=teacher_confidence,
            use_teacher_confidence=step_weight_use_teacher_confidence,
            use_position_feature=step_weight_use_position_feature
        )
        raw_step_weights = step_weight_net(step_features)
    else:
        raw_step_weights = build_rule_step_weights(pred_prob, next_knowledge_relevancies)

    normalized_step_weights = normalize_step_weights(raw_step_weights, valid_mask)
    weighted_pred_loss = (pred_loss * normalized_step_weights).sum() / valid_mask_float.sum().clamp_min(1.0)
    unweighted_pred_loss = (raw_bce_loss * valid_mask_float).sum() / valid_mask_float.sum().clamp_min(1.0)

    step_weight_stats = summarize_masked_tensor(normalized_step_weights, valid_mask_float)
    focal_stats = summarize_masked_tensor(focal_factor, valid_mask_float)
    teacher_confidence_stats = summarize_masked_tensor(
        teacher_confidence if teacher_confidence is not None else torch.zeros_like(pred_prob),
        valid_mask_float
    )
    loss_stats = {
        'loss_weight_mode': loss_weight_mode,
        'valid_steps': int(valid_mask_float.sum().item()),
        'step_weight_mean': step_weight_stats['mean'],
        'step_weight_std': step_weight_stats['std'],
        'step_weight_min': step_weight_stats['min'],
        'step_weight_max': step_weight_stats['max'],
        'focal_factor_mean': focal_stats['mean'],
        'focal_factor_std': focal_stats['std'],
        'focal_factor_min': focal_stats['min'],
        'focal_factor_max': focal_stats['max'],
        'teacher_confidence_mean': teacher_confidence_stats['mean'],
        'teacher_confidence_weight_corr': compute_masked_correlation(
            teacher_confidence if teacher_confidence is not None else torch.zeros_like(pred_prob),
            normalized_step_weights.detach(),
            valid_mask_float
        ) if teacher_confidence is not None else 0.0
    }
    return weighted_pred_loss, unweighted_pred_loss, normalized_step_weights.detach(), focal_factor.detach(), loss_stats


def build_confidence_consistency_weights(teacher_pred_out, valid_mask, mode='prob_margin', eps=1e-8):
    teacher_pred_out = sanitize_prediction_tensor(teacher_pred_out)
    teacher_prob = teacher_pred_out[:, :, 1].detach()
    if mode == 'prob_margin':
        confidence_weights = 2.0 * torch.abs(teacher_prob - 0.5)
    else:
        raise ValueError(f'Unsupported confidence_consistency_mode: {mode}')

    valid_mask_float = valid_mask.float()
    effective_weights = confidence_weights * valid_mask_float
    valid_steps_per_sample = valid_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized_weights = effective_weights / effective_weights.sum(dim=1, keepdim=True).clamp_min(float(eps))
    normalized_weights = normalized_weights * valid_steps_per_sample
    normalized_weights = normalized_weights * valid_mask_float
    raw_stats = summarize_masked_tensor(effective_weights, valid_mask_float)
    normalized_stats = summarize_masked_tensor(normalized_weights, valid_mask_float)
    return normalized_weights.detach(), raw_stats, normalized_stats


def compute_consistency_loss(
    student_states,
    teacher_states,
    valid_mask,
    teacher_pred_out=None,
    use_confidence_consistency=0,
    confidence_consistency_mode='prob_margin',
    confidence_consistency_eps=1e-8
):
    # New enhancement: stage2 teacher-student consistency on explicit low-order states.
    valid_mask_float = valid_mask.float()
    smooth_l1 = torch.nn.functional.smooth_l1_loss(student_states, teacher_states, reduction='none')
    per_step_smooth_l1 = smooth_l1.mean(dim=-1)

    if use_confidence_consistency:
        if teacher_pred_out is None:
            raise ValueError('teacher_pred_out is required when use_confidence_consistency=1.')
        confidence_weights, raw_confidence_stats, normalized_confidence_stats = build_confidence_consistency_weights(
            teacher_pred_out=teacher_pred_out,
            valid_mask=valid_mask,
            mode=confidence_consistency_mode,
            eps=confidence_consistency_eps
        )
    else:
        confidence_weights = valid_mask_float
        raw_confidence_stats = {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0
        }
        normalized_confidence_stats = {
            'mean': 1.0,
            'std': 0.0,
            'min': 1.0,
            'max': 1.0
        }

    consistency_loss = (per_step_smooth_l1 * confidence_weights).sum() / valid_mask_float.sum().clamp_min(1.0)
    return consistency_loss, confidence_weights.detach(), raw_confidence_stats, normalized_confidence_stats


def compute_state_consistency_loss(
    student_state_repr,
    teacher_state_repr,
    valid_mask,
    loss_type='smooth_l1',
    confidence_weights=None
):
    valid_mask_float = valid_mask.float()
    if loss_type == 'smooth_l1':
        per_dim_loss = torch.nn.functional.smooth_l1_loss(student_state_repr, teacher_state_repr, reduction='none')
    elif loss_type == 'mse':
        per_dim_loss = torch.nn.functional.mse_loss(student_state_repr, teacher_state_repr, reduction='none')
    else:
        raise ValueError(f'Unsupported state_consistency_loss: {loss_type}')
    per_step_loss = per_dim_loss.mean(dim=-1)
    if confidence_weights is None:
        confidence_weights = valid_mask_float
    state_loss = (per_step_loss * confidence_weights).sum() / valid_mask_float.sum().clamp_min(1.0)
    state_confidence_mean = (confidence_weights.sum() / valid_mask_float.sum().clamp_min(1.0)).item()
    return state_loss, float(state_confidence_mean)


def evaluate_predictions(pred_all, pred_label_all):
    pred_all, pred_label_all = sanitize_prediction_arrays(pred_all, pred_label_all)
    pred_auc = safe_roc_auc_score(pred_label_all, pred_all)
    pred_rmse = np.sqrt(np.mean((pred_all - pred_label_all) ** 2)) if pred_all.size else 0.0
    pred_binary = (pred_all >= 0.5).astype(int)
    pred_acc = float((pred_binary == pred_label_all).mean()) if pred_all.size else 0.0
    return pred_auc, pred_rmse, pred_acc


def val_test(ws, model, epoch_i, stage, dtype, cross_idx, extra_metrics=None):
    ws_config = read_ws_config(ws)
    data_loader = ValTestDataLoader(ws_config, cross_idx, dtype=dtype)
    device = next(model.parameters()).device
    model.train(0)

    pred_all, pred_label_all = [], []
    pred_loss_sum = 0.0
    valid_steps_sum = 0.0

    while not data_loader.is_end():
        batch_len, stu_ids, log_lens, exer_ids, knowledge_relevancies, corrs = data_loader.next_batch()
        del batch_len, stu_ids
        exer_ids = exer_ids.to(device)
        knowledge_relevancies = knowledge_relevancies.to(device)
        corrs = corrs.to(device)
        log_lens_tensor = torch.LongTensor(log_lens).to(device)

        with torch.no_grad():
            model.init_hidden()
            pred_out, _, valid_mask = model.forward(
                None, exer_ids, knowledge_relevancies, corrs, log_lens_tensor, stage
            )
            pred_out = sanitize_prediction_tensor(pred_out)
            target = corrs[:, 1:].float()
            pred_prob = pred_out[:, :, 1]
            pred_loss = -(target * torch.log(pred_prob) + (1.0 - target) * torch.log(1.0 - pred_prob))
            pred_loss_sum += (pred_loss * valid_mask.float()).sum().item()
            valid_steps_sum += valid_mask.float().sum().item()

            valid_pred = pred_prob[valid_mask].detach().cpu().numpy()
            valid_label = target[valid_mask].detach().cpu().numpy()
            pred_all.extend(valid_pred.tolist())
            pred_label_all.extend(valid_label.tolist())

    pred_auc, pred_rmse, pred_acc = evaluate_predictions(pred_all, pred_label_all)
    avg_pred_loss = pred_loss_sum / max(valid_steps_sum, 1.0)

    log_line = (
        f'[{dtype}] stage={stage}, epoch={epoch_i}: '
        f'auc={pred_auc}, rmse={pred_rmse}, acc={pred_acc}, pred_loss={avg_pred_loss}'
    )
    print(log_line)
    with open(os.path.join(ws, 'results.txt'), 'a', encoding='utf8') as o_f:
        o_f.write(log_line + '\n')

    metric_record = {
        'dtype': dtype,
        'stage': int(stage),
        'epoch': int(epoch_i),
        'auc': float(pred_auc),
        'rmse': float(pred_rmse),
        'acc': float(pred_acc),
        'pred_loss': float(avg_pred_loss)
    }
    if extra_metrics:
        metric_record.update(extra_metrics)
    append_metric_record(ws, metric_record)
    return metric_record


def create_model(ws_config):
    return DIRTPlusModel(
        stu_ho_dim=ws_config['stu_ho_dim'],
        rnn_type=ws_config['rnn_type'],
        attr_idx=ws_config['attr_idx'],
        max_log=ws_config['max_log'],
        knowledge_n=ws_config['knowledge_n'],
        exer_n=ws_config['exer_n'],
        batch_size=ws_config['batch_size'],
        attn_heads=ws_config['attn_heads'],
        dropout=ws_config['dropout'],
        use_multihead_temporal_attn=ws_config['use_multihead_temporal_attn'],
        use_multihead_query_attn=ws_config['use_multihead_query_attn'],
        multihead_temporal_num_heads=ws_config['multihead_temporal_num_heads'],
        multihead_query_num_heads=ws_config['multihead_query_num_heads'],
        multihead_attn_dropout=ws_config['multihead_attn_dropout'],
        use_temporal_self_attention=ws_config['use_temporal_self_attention'],
        use_query_guided_attention=ws_config['use_query_guided_attention'],
        use_temporal_bias=ws_config['use_temporal_bias'],
        temporal_bias_mode=ws_config['temporal_bias_mode'],
        temporal_bias_init=ws_config['temporal_bias_init'],
        use_exercise_aware_decay=ws_config['use_exercise_aware_decay'],
        exercise_aware_decay_mode=ws_config['exercise_aware_decay_mode'],
        exercise_aware_decay_scale=ws_config['exercise_aware_decay_scale'],
        exercise_aware_decay_min=ws_config['exercise_aware_decay_min'],
        exercise_aware_decay_max=ws_config['exercise_aware_decay_max'],
        use_adaptive_fusion=ws_config['use_adaptive_fusion'],
        adaptive_fusion_mode=ws_config['adaptive_fusion_mode'],
        adaptive_fusion_hidden_dim=ws_config['adaptive_fusion_hidden_dim'],
        step_weight_hidden_dim=ws_config['step_weight_hidden_dim'],
        step_weight_dropout=ws_config['step_weight_dropout'],
        step_weight_min=ws_config['step_weight_min'],
        step_weight_max=ws_config['step_weight_max']
    )


def get_temporal_fusion_gate_means(model):
    gate_means = model.temporal_fusion.get_last_gate_means()
    if gate_means is None:
        return None
    return {
        'fusion_gate_rnn_mean': float(gate_means[0].item()),
        'fusion_gate_attn_mean': float(gate_means[1].item()),
        'fusion_gate_query_mean': float(gate_means[2].item())
    }


def get_decay_gamma_stats(model):
    decay_stats = model.query_guided_attention.get_last_decay_gamma_stats()
    if decay_stats is None:
        return None
    return {
        'decay_gamma_mean': float(decay_stats[0]),
        'decay_gamma_std': float(decay_stats[1])
    }


def train_stage1(ws, cross_idx, device='cpu', lr=0.002, n_epochs=10, grad_clip=5.0):
    ws_config = read_ws_config(ws)
    model = create_model(ws_config).to(device)
    data_loader = DataLoader(ws_config, cross_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = build_stage_scheduler(optimizer, ws_config)

    epoch_times = []
    for epoch in range(n_epochs):
        start = time.time()
        model.train(1)
        data_loader.reset()

        running_pred_loss = 0.0
        running_total_loss = 0.0
        batch_count = 0
        epoch_pred_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_fusion_gate_rnn_sum = 0.0
        epoch_fusion_gate_attn_sum = 0.0
        epoch_fusion_gate_query_sum = 0.0
        epoch_decay_gamma_mean_sum = 0.0
        epoch_decay_gamma_std_sum = 0.0
        epoch_step_weight_mean_sum = 0.0
        epoch_step_weight_std_sum = 0.0
        epoch_step_weight_min_sum = 0.0
        epoch_step_weight_max_sum = 0.0
        epoch_focal_factor_mean_sum = 0.0
        epoch_focal_factor_std_sum = 0.0
        epoch_teacher_confidence_mean_sum = 0.0
        running_step_weight_mean = 0.0
        running_step_weight_std = 0.0
        running_focal_factor_mean = 0.0
        running_valid_steps = 0.0

        while not data_loader.is_end():
            _, stu_ids, log_lens, exer_ids, knowledge_relevancies, corrs = data_loader.next_batch()
            del stu_ids
            batch_count += 1

            exer_ids = exer_ids.to(device)
            knowledge_relevancies = knowledge_relevancies.to(device)
            corrs = corrs.to(device)
            log_lens_tensor = torch.LongTensor(log_lens).to(device)

            model.init_hidden()
            optimizer.zero_grad()
            pred_out, _, valid_mask, student_features = model.forward(
                None, exer_ids, knowledge_relevancies, corrs, log_lens_tensor, stage=1, return_features=True
            )
            weighted_pred_loss, raw_pred_loss, _, _, loss_stats = compute_step_weighted_pred_loss(
                pred_out=pred_out,
                corrs=corrs,
                valid_mask=valid_mask,
                next_knowledge_relevancies=knowledge_relevancies[:, 1:],
                loss_weight_mode=ws_config['loss_weight_mode'],
                step_weight_net=model.learnable_step_weight_net,
                fused_hidden=student_features['fused_hidden'],
                exercise_embedding=student_features['exercise_embedding'],
                teacher_confidence=None,
                step_weight_use_teacher_confidence=ws_config['step_weight_use_teacher_confidence'],
                step_weight_use_position_feature=ws_config['step_weight_use_position_feature'],
                focal_gamma=ws_config['focal_gamma'],
                focal_alpha=ws_config['focal_alpha']
            )
            total_loss = weighted_pred_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_pred_loss += raw_pred_loss.item()
            running_total_loss += total_loss.item()
            epoch_pred_loss_sum += raw_pred_loss.item()
            epoch_total_loss_sum += total_loss.item()
            epoch_step_weight_mean_sum += loss_stats['step_weight_mean']
            epoch_step_weight_std_sum += loss_stats['step_weight_std']
            epoch_step_weight_min_sum += loss_stats['step_weight_min']
            epoch_step_weight_max_sum += loss_stats['step_weight_max']
            epoch_focal_factor_mean_sum += loss_stats['focal_factor_mean']
            epoch_focal_factor_std_sum += loss_stats['focal_factor_std']
            epoch_teacher_confidence_mean_sum += loss_stats['teacher_confidence_mean']
            running_step_weight_mean += loss_stats['step_weight_mean']
            running_step_weight_std += loss_stats['step_weight_std']
            running_focal_factor_mean += loss_stats['focal_factor_mean']
            running_valid_steps += loss_stats['valid_steps']
            gate_metrics = get_temporal_fusion_gate_means(model)
            if gate_metrics is not None:
                epoch_fusion_gate_rnn_sum += gate_metrics['fusion_gate_rnn_mean']
                epoch_fusion_gate_attn_sum += gate_metrics['fusion_gate_attn_mean']
                epoch_fusion_gate_query_sum += gate_metrics['fusion_gate_query_mean']
            decay_metrics = get_decay_gamma_stats(model)
            if decay_metrics is not None:
                epoch_decay_gamma_mean_sum += decay_metrics['decay_gamma_mean']
                epoch_decay_gamma_std_sum += decay_metrics['decay_gamma_std']
            if batch_count % 10 == 0:
                print(
                    f'[{epoch}, {batch_count:5d}] '
                    f'pred_loss={running_pred_loss / 10:.3f}, '
                    f'total_loss={running_total_loss / 10:.3f}, '
                    f'lr={get_optimizer_lr(optimizer):.6f}, '
                    f'valid_steps={int(running_valid_steps / 10)}, '
                    f'batch_step_weight_mean={running_step_weight_mean / 10:.4f}, '
                    f'batch_step_weight_std={running_step_weight_std / 10:.4f}, '
                    f'batch_step_weight_min={loss_stats["step_weight_min"]:.4f}, '
                    f'batch_step_weight_max={loss_stats["step_weight_max"]:.4f}, '
                    f'batch_focal_factor_mean={running_focal_factor_mean / 10:.4f}'
                )
                running_pred_loss = 0.0
                running_total_loss = 0.0
                running_step_weight_mean = 0.0
                running_step_weight_std = 0.0
                running_focal_factor_mean = 0.0
                running_valid_steps = 0.0

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print('time:', epoch_time)
        epoch_train_record = {
            'dtype': 'train',
            'stage': 1,
            'epoch': int(epoch),
            'lr': get_optimizer_lr(optimizer),
            'pred_loss': float(epoch_pred_loss_sum / max(batch_count, 1)),
            'total_loss': float(epoch_total_loss_sum / max(batch_count, 1)),
            'consistency_loss': 0.0,
            'loss_weight_mode': ws_config['loss_weight_mode'],
            'step_weight_mean': float(epoch_step_weight_mean_sum / max(batch_count, 1)),
            'step_weight_std': float(epoch_step_weight_std_sum / max(batch_count, 1)),
            'step_weight_min': float(epoch_step_weight_min_sum / max(batch_count, 1)),
            'step_weight_max': float(epoch_step_weight_max_sum / max(batch_count, 1)),
            'focal_gamma': float(ws_config['focal_gamma']),
            'focal_alpha': None if ws_config['focal_alpha'] is None else float(ws_config['focal_alpha']),
            'focal_factor_mean': float(epoch_focal_factor_mean_sum / max(batch_count, 1)),
            'focal_factor_std': float(epoch_focal_factor_std_sum / max(batch_count, 1)),
            'teacher_confidence_mean': float(epoch_teacher_confidence_mean_sum / max(batch_count, 1)),
            'use_multihead_temporal_attn': int(ws_config['use_multihead_temporal_attn']),
            'use_multihead_query_attn': int(ws_config['use_multihead_query_attn']),
            'multihead_temporal_num_heads': int(ws_config['multihead_temporal_num_heads']),
            'multihead_query_num_heads': int(ws_config['multihead_query_num_heads']),
            'use_adaptive_fusion': int(ws_config['use_adaptive_fusion']),
            'adaptive_fusion_mode': ws_config['adaptive_fusion_mode'],
            'use_exercise_aware_decay': int(ws_config['use_exercise_aware_decay']),
            'exercise_aware_decay_mode': ws_config['exercise_aware_decay_mode']
        }
        if ws_config['use_adaptive_fusion']:
            epoch_train_record.update(
                {
                    'fusion_gate_rnn_mean': float(epoch_fusion_gate_rnn_sum / max(batch_count, 1)),
                    'fusion_gate_attn_mean': float(epoch_fusion_gate_attn_sum / max(batch_count, 1)),
                    'fusion_gate_query_mean': float(epoch_fusion_gate_query_sum / max(batch_count, 1))
                }
            )
        if ws_config['use_exercise_aware_decay']:
            epoch_train_record.update(
                {
                    'decay_gamma_mean': float(epoch_decay_gamma_mean_sum / max(batch_count, 1)),
                    'decay_gamma_std': float(epoch_decay_gamma_std_sum / max(batch_count, 1))
                }
            )
        append_metric_record(ws, epoch_train_record)
        with open(os.path.join(ws, 'results.txt'), 'a', encoding='utf8') as o_f:
            log_line = (
                f"[train] stage=1, epoch={epoch}: pred_loss={epoch_train_record['pred_loss']}, "
                f"total_loss={epoch_train_record['total_loss']}, consistency_loss=0.0, "
                f"lr={epoch_train_record['lr']}, "
                f"loss_weight_mode={epoch_train_record['loss_weight_mode']}, "
                f"step_weight_mean={epoch_train_record['step_weight_mean']}, "
                f"step_weight_std={epoch_train_record['step_weight_std']}, "
                f"step_weight_min={epoch_train_record['step_weight_min']}, "
                f"step_weight_max={epoch_train_record['step_weight_max']}, "
                f"focal_factor_mean={epoch_train_record['focal_factor_mean']}, "
                f"use_multihead_temporal_attn={epoch_train_record['use_multihead_temporal_attn']}, "
                f"use_multihead_query_attn={epoch_train_record['use_multihead_query_attn']}, "
                f"use_adaptive_fusion={epoch_train_record['use_adaptive_fusion']}, "
                f"use_exercise_aware_decay={epoch_train_record['use_exercise_aware_decay']}"
            )
            if ws_config['use_adaptive_fusion']:
                log_line += (
                    f", fusion_gate_rnn_mean={epoch_train_record['fusion_gate_rnn_mean']}, "
                    f"fusion_gate_attn_mean={epoch_train_record['fusion_gate_attn_mean']}, "
                    f"fusion_gate_query_mean={epoch_train_record['fusion_gate_query_mean']}"
                )
            if ws_config['use_exercise_aware_decay']:
                log_line += (
                    f", decay_gamma_mean={epoch_train_record['decay_gamma_mean']}, "
                    f"decay_gamma_std={epoch_train_record['decay_gamma_std']}"
                )
            o_f.write(log_line + '\n')
        val_metrics = val_test(
            ws,
            model,
            epoch,
            stage=1,
            dtype='validation',
            cross_idx=cross_idx,
            extra_metrics={
                'train_stage': 1,
                'lr': get_optimizer_lr(optimizer),
                'loss_weight_mode': ws_config['loss_weight_mode'],
                'focal_gamma': float(ws_config['focal_gamma']),
                'focal_alpha': None if ws_config['focal_alpha'] is None else float(ws_config['focal_alpha']),
                'use_multihead_temporal_attn': int(ws_config['use_multihead_temporal_attn']),
                'use_multihead_query_attn': int(ws_config['use_multihead_query_attn']),
                'multihead_temporal_num_heads': int(ws_config['multihead_temporal_num_heads']),
                'multihead_query_num_heads': int(ws_config['multihead_query_num_heads']),
                'use_adaptive_fusion': int(ws_config['use_adaptive_fusion']),
                'adaptive_fusion_mode': ws_config['adaptive_fusion_mode'],
                'use_exercise_aware_decay': int(ws_config['use_exercise_aware_decay']),
                'exercise_aware_decay_mode': ws_config['exercise_aware_decay_mode']
            }
        )
        lr_before_scheduler = get_optimizer_lr(optimizer)
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
        lr_after_scheduler = get_optimizer_lr(optimizer)
        scheduler_log_line = (
            f"[scheduler] stage=1, epoch={epoch}: monitor_auc={val_metrics['auc']}, "
            f"lr_before={lr_before_scheduler}, lr_after={lr_after_scheduler}"
        )
        print(scheduler_log_line)
        with open(os.path.join(ws, 'results.txt'), 'a', encoding='utf8') as o_f:
            o_f.write(scheduler_log_line + '\n')
        append_metric_record(
            ws,
            {
                'dtype': 'scheduler',
                'stage': 1,
                'epoch': int(epoch),
                'monitor_auc': float(val_metrics['auc']),
                'lr_before': float(lr_before_scheduler),
                'lr_after': float(lr_after_scheduler),
                'scheduler_enabled': int(scheduler is not None)
            }
        )
        torch.save(model.state_dict(), os.path.join(ws, 'snapshot', f'stage1-{epoch}'))
    print('epoch_times:', str(epoch_times))


def train_stage2(
    ws,
    cross_idx,
    stage1_epoch,
    n_epochs=5,
    device='cpu',
    lr=0.002,
    grad_clip=5.0,
    lambda_consistency=0.2,
    consistency_warmup_epochs=3,
    consistency_warmup_start_ratio=0.2,
    use_confidence_consistency=0,
    confidence_consistency_mode='prob_margin',
    confidence_consistency_eps=1e-8,
    use_state_consistency=0,
    state_consistency_target='fused_hidden',
    state_consistency_loss='smooth_l1',
    lambda_state_consistency=0.05,
    use_confidence_for_state_consistency=1
):
    ws_config = read_ws_config(ws)
    if ws_config['attr_idx'] not in [3, 4]:
        raise ValueError('stage2 is only applicable to DIRT_3 / DIRT_4.')

    snapshot_path = os.path.join(ws, 'snapshot', f'stage1-{stage1_epoch}')
    model = create_model(ws_config)
    model.load_state_dict(torch.load(snapshot_path, map_location='cpu'))
    model = model.to(device)

    teacher_model = create_model(ws_config)
    teacher_model.load_state_dict(torch.load(snapshot_path, map_location='cpu'))
    teacher_model = teacher_model.to(device)
    teacher_model.train(0)
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    data_loader = DataLoader(ws_config, cross_idx)
    optimizer = optim.Adam(
        [
            {'params': model.transition_rnn.parameters()},
            {'params': model.temporal_self_attention.parameters()},
            {'params': model.exercise_aware_query.parameters()},
            {'params': model.query_guided_attention.parameters()},
            {'params': model.temporal_fusion.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.learnable_step_weight_net.parameters()}
        ],
        lr=lr
    )
    scheduler = build_stage_scheduler(optimizer, ws_config)

    epoch_times = []
    for epoch in range(n_epochs):
        start = time.time()
        model.train(2)
        data_loader.reset()
        current_consistency_weight = get_consistency_weight(
            epoch,
            max_weight=lambda_consistency,
            warmup_epochs=consistency_warmup_epochs,
            warmup_start_ratio=consistency_warmup_start_ratio
        )

        running_pred_loss = 0.0
        running_total_loss = 0.0
        running_consistency_loss = 0.0
        running_raw_confidence_mean = 0.0
        running_normalized_confidence_mean = 0.0
        batch_count = 0
        epoch_pred_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_consistency_loss_sum = 0.0
        epoch_raw_confidence_mean_sum = 0.0
        epoch_raw_confidence_std_sum = 0.0
        epoch_raw_confidence_min_sum = 0.0
        epoch_raw_confidence_max_sum = 0.0
        epoch_normalized_confidence_mean_sum = 0.0
        epoch_normalized_confidence_std_sum = 0.0
        epoch_normalized_confidence_min_sum = 0.0
        epoch_normalized_confidence_max_sum = 0.0
        epoch_fusion_gate_rnn_sum = 0.0
        epoch_fusion_gate_attn_sum = 0.0
        epoch_fusion_gate_query_sum = 0.0
        epoch_decay_gamma_mean_sum = 0.0
        epoch_decay_gamma_std_sum = 0.0
        epoch_state_consistency_loss_sum = 0.0
        epoch_state_confidence_mean_sum = 0.0
        epoch_step_weight_mean_sum = 0.0
        epoch_step_weight_std_sum = 0.0
        epoch_step_weight_min_sum = 0.0
        epoch_step_weight_max_sum = 0.0
        epoch_focal_factor_mean_sum = 0.0
        epoch_focal_factor_std_sum = 0.0
        epoch_teacher_weight_corr_sum = 0.0
        running_step_weight_mean = 0.0
        running_step_weight_std = 0.0
        running_focal_factor_mean = 0.0
        running_valid_steps = 0.0

        while not data_loader.is_end():
            _, stu_ids, log_lens, exer_ids, knowledge_relevancies, corrs = data_loader.next_batch()
            del stu_ids
            batch_count += 1

            exer_ids = exer_ids.to(device)
            knowledge_relevancies = knowledge_relevancies.to(device)
            corrs = corrs.to(device)
            log_lens_tensor = torch.LongTensor(log_lens).to(device)

            model.init_hidden()
            optimizer.zero_grad()
            pred_out, student_states, valid_mask, student_features = model.forward(
                None, exer_ids, knowledge_relevancies, corrs, log_lens_tensor, stage=2, return_features=True
            )

            with torch.no_grad():
                teacher_model.init_hidden()
                teacher_pred_out, teacher_states, _, teacher_features = teacher_model.forward(
                    None, exer_ids, knowledge_relevancies, corrs, log_lens_tensor, stage=1, return_features=True
                )
            teacher_confidence = None
            if ws_config['step_weight_use_teacher_confidence']:
                teacher_confidence = compute_teacher_confidence_feature(teacher_pred_out)
            weighted_pred_loss, raw_pred_loss, _, _, loss_stats = compute_step_weighted_pred_loss(
                pred_out=pred_out,
                corrs=corrs,
                valid_mask=valid_mask,
                next_knowledge_relevancies=knowledge_relevancies[:, 1:],
                loss_weight_mode=ws_config['loss_weight_mode'],
                step_weight_net=model.learnable_step_weight_net,
                fused_hidden=student_features['fused_hidden'],
                exercise_embedding=student_features['exercise_embedding'],
                teacher_confidence=teacher_confidence,
                step_weight_use_teacher_confidence=ws_config['step_weight_use_teacher_confidence'],
                step_weight_use_position_feature=ws_config['step_weight_use_position_feature'],
                focal_gamma=ws_config['focal_gamma'],
                focal_alpha=ws_config['focal_alpha']
            )
            consistency_loss, confidence_weights, raw_confidence_stats, normalized_confidence_stats = compute_consistency_loss(
                student_states,
                teacher_states,
                valid_mask,
                teacher_pred_out=teacher_pred_out,
                use_confidence_consistency=use_confidence_consistency,
                confidence_consistency_mode=confidence_consistency_mode,
                confidence_consistency_eps=confidence_consistency_eps
            )
            state_consistency_value = torch.tensor(0.0, device=device)
            state_confidence_mean = 0.0
            if use_state_consistency:
                if state_consistency_target not in student_features or state_consistency_target not in teacher_features:
                    raise ValueError(f'Unsupported state_consistency_target: {state_consistency_target}')
                state_confidence_weights = confidence_weights if use_confidence_for_state_consistency else None
                state_consistency_value, state_confidence_mean = compute_state_consistency_loss(
                    student_features[state_consistency_target],
                    teacher_features[state_consistency_target],
                    valid_mask,
                    loss_type=state_consistency_loss,
                    confidence_weights=state_confidence_weights
                )
            total_loss = (
                weighted_pred_loss +
                current_consistency_weight * consistency_loss +
                lambda_state_consistency * state_consistency_value
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_pred_loss += raw_pred_loss.item()
            running_total_loss += total_loss.item()
            running_consistency_loss += consistency_loss.item()
            running_raw_confidence_mean += raw_confidence_stats['mean']
            running_normalized_confidence_mean += normalized_confidence_stats['mean']
            running_step_weight_mean += loss_stats['step_weight_mean']
            running_step_weight_std += loss_stats['step_weight_std']
            running_focal_factor_mean += loss_stats['focal_factor_mean']
            running_valid_steps += loss_stats['valid_steps']
            epoch_pred_loss_sum += raw_pred_loss.item()
            epoch_total_loss_sum += total_loss.item()
            epoch_consistency_loss_sum += consistency_loss.item()
            epoch_raw_confidence_mean_sum += raw_confidence_stats['mean']
            epoch_raw_confidence_std_sum += raw_confidence_stats['std']
            epoch_raw_confidence_min_sum += raw_confidence_stats['min']
            epoch_raw_confidence_max_sum += raw_confidence_stats['max']
            epoch_normalized_confidence_mean_sum += normalized_confidence_stats['mean']
            epoch_normalized_confidence_std_sum += normalized_confidence_stats['std']
            epoch_normalized_confidence_min_sum += normalized_confidence_stats['min']
            epoch_normalized_confidence_max_sum += normalized_confidence_stats['max']
            epoch_state_consistency_loss_sum += state_consistency_value.item()
            epoch_state_confidence_mean_sum += state_confidence_mean
            epoch_step_weight_mean_sum += loss_stats['step_weight_mean']
            epoch_step_weight_std_sum += loss_stats['step_weight_std']
            epoch_step_weight_min_sum += loss_stats['step_weight_min']
            epoch_step_weight_max_sum += loss_stats['step_weight_max']
            epoch_focal_factor_mean_sum += loss_stats['focal_factor_mean']
            epoch_focal_factor_std_sum += loss_stats['focal_factor_std']
            epoch_teacher_weight_corr_sum += loss_stats['teacher_confidence_weight_corr']
            gate_metrics = get_temporal_fusion_gate_means(model)
            if gate_metrics is not None:
                epoch_fusion_gate_rnn_sum += gate_metrics['fusion_gate_rnn_mean']
                epoch_fusion_gate_attn_sum += gate_metrics['fusion_gate_attn_mean']
                epoch_fusion_gate_query_sum += gate_metrics['fusion_gate_query_mean']
            decay_metrics = get_decay_gamma_stats(model)
            if decay_metrics is not None:
                epoch_decay_gamma_mean_sum += decay_metrics['decay_gamma_mean']
                epoch_decay_gamma_std_sum += decay_metrics['decay_gamma_std']
            if batch_count % 10 == 0:
                print(
                    f'[{epoch}, {batch_count:5d}] '
                    f'pred_loss={running_pred_loss / 10:.3f}, '
                    f'consistency_loss={running_consistency_loss / 10:.3f}, '
                    f'total_loss={running_total_loss / 10:.3f}, '
                    f'consistency_weight={current_consistency_weight:.4f}, '
                    f'raw_confidence_mean={running_raw_confidence_mean / 10:.4f}, '
                    f'normalized_confidence_mean={running_normalized_confidence_mean / 10:.4f}, '
                    f'valid_steps={int(running_valid_steps / 10)}, '
                    f'batch_step_weight_mean={running_step_weight_mean / 10:.4f}, '
                    f'batch_step_weight_std={running_step_weight_std / 10:.4f}, '
                    f'batch_step_weight_min={loss_stats["step_weight_min"]:.4f}, '
                    f'batch_step_weight_max={loss_stats["step_weight_max"]:.4f}, '
                    f'batch_focal_factor_mean={running_focal_factor_mean / 10:.4f}, '
                    f'batch_raw_confidence_std={raw_confidence_stats["std"]:.4f}, '
                    f'batch_raw_confidence_min={raw_confidence_stats["min"]:.4f}, '
                    f'batch_raw_confidence_max={raw_confidence_stats["max"]:.4f}, '
                    f'batch_normalized_confidence_std={normalized_confidence_stats["std"]:.4f}, '
                    f'lr={get_optimizer_lr(optimizer):.6f}'
                )
                running_pred_loss = 0.0
                running_total_loss = 0.0
                running_consistency_loss = 0.0
                running_raw_confidence_mean = 0.0
                running_normalized_confidence_mean = 0.0
                running_step_weight_mean = 0.0
                running_step_weight_std = 0.0
                running_focal_factor_mean = 0.0
                running_valid_steps = 0.0

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print('time:', epoch_time)
        epoch_train_record = {
            'dtype': 'train',
            'stage': 2,
            'epoch': int(epoch),
            'lr': get_optimizer_lr(optimizer),
            'pred_loss': float(epoch_pred_loss_sum / max(batch_count, 1)),
            'total_loss': float(epoch_total_loss_sum / max(batch_count, 1)),
            'consistency_loss': float(epoch_consistency_loss_sum / max(batch_count, 1)),
            'lambda_consistency': float(lambda_consistency),
            'consistency_weight': float(current_consistency_weight),
            'use_confidence_consistency': int(use_confidence_consistency),
            'confidence_consistency_mode': confidence_consistency_mode,
            'loss_weight_mode': ws_config['loss_weight_mode'],
            'step_weight_mean': float(epoch_step_weight_mean_sum / max(batch_count, 1)),
            'step_weight_std': float(epoch_step_weight_std_sum / max(batch_count, 1)),
            'step_weight_min': float(epoch_step_weight_min_sum / max(batch_count, 1)),
            'step_weight_max': float(epoch_step_weight_max_sum / max(batch_count, 1)),
            'focal_gamma': float(ws_config['focal_gamma']),
            'focal_alpha': None if ws_config['focal_alpha'] is None else float(ws_config['focal_alpha']),
            'focal_factor_mean': float(epoch_focal_factor_mean_sum / max(batch_count, 1)),
            'focal_factor_std': float(epoch_focal_factor_std_sum / max(batch_count, 1)),
            'confidence_mean': float(epoch_raw_confidence_mean_sum / max(batch_count, 1)),
            'raw_confidence_mean': float(epoch_raw_confidence_mean_sum / max(batch_count, 1)),
            'raw_confidence_std': float(epoch_raw_confidence_std_sum / max(batch_count, 1)),
            'raw_confidence_min': float(epoch_raw_confidence_min_sum / max(batch_count, 1)),
            'raw_confidence_max': float(epoch_raw_confidence_max_sum / max(batch_count, 1)),
            'normalized_confidence_mean': float(epoch_normalized_confidence_mean_sum / max(batch_count, 1)),
            'normalized_confidence_std': float(epoch_normalized_confidence_std_sum / max(batch_count, 1)),
            'normalized_confidence_min': float(epoch_normalized_confidence_min_sum / max(batch_count, 1)),
            'normalized_confidence_max': float(epoch_normalized_confidence_max_sum / max(batch_count, 1)),
            'teacher_confidence_mean': float(epoch_raw_confidence_mean_sum / max(batch_count, 1))
            if ws_config['step_weight_use_teacher_confidence'] else 0.0,
            'teacher_confidence_weight_corr': float(epoch_teacher_weight_corr_sum / max(batch_count, 1)),
            'use_multihead_temporal_attn': int(ws_config['use_multihead_temporal_attn']),
            'use_multihead_query_attn': int(ws_config['use_multihead_query_attn']),
            'multihead_temporal_num_heads': int(ws_config['multihead_temporal_num_heads']),
            'multihead_query_num_heads': int(ws_config['multihead_query_num_heads']),
            'use_state_consistency': int(use_state_consistency),
            'state_consistency_target': state_consistency_target,
            'state_consistency_loss': state_consistency_loss,
            'lambda_state_consistency': float(lambda_state_consistency),
            'state_consistency_loss_value': float(epoch_state_consistency_loss_sum / max(batch_count, 1)),
            'state_confidence_mean': float(epoch_state_confidence_mean_sum / max(batch_count, 1)),
            'use_adaptive_fusion': int(ws_config['use_adaptive_fusion']),
            'adaptive_fusion_mode': ws_config['adaptive_fusion_mode'],
            'use_exercise_aware_decay': int(ws_config['use_exercise_aware_decay']),
            'exercise_aware_decay_mode': ws_config['exercise_aware_decay_mode']
        }
        if ws_config['use_adaptive_fusion']:
            epoch_train_record.update(
                {
                    'fusion_gate_rnn_mean': float(epoch_fusion_gate_rnn_sum / max(batch_count, 1)),
                    'fusion_gate_attn_mean': float(epoch_fusion_gate_attn_sum / max(batch_count, 1)),
                    'fusion_gate_query_mean': float(epoch_fusion_gate_query_sum / max(batch_count, 1))
                }
            )
        if ws_config['use_exercise_aware_decay']:
            epoch_train_record.update(
                {
                    'decay_gamma_mean': float(epoch_decay_gamma_mean_sum / max(batch_count, 1)),
                    'decay_gamma_std': float(epoch_decay_gamma_std_sum / max(batch_count, 1))
                }
            )
        append_metric_record(ws, epoch_train_record)
        with open(os.path.join(ws, 'results.txt'), 'a', encoding='utf8') as o_f:
            log_line = (
                f"[train] stage=2, epoch={epoch}: pred_loss={epoch_train_record['pred_loss']}, "
                f"total_loss={epoch_train_record['total_loss']}, "
                f"consistency_loss={epoch_train_record['consistency_loss']}, "
                f"state_consistency_loss={epoch_train_record['state_consistency_loss_value']}, "
                f"consistency_weight={epoch_train_record['consistency_weight']}, "
                f"use_confidence_consistency={epoch_train_record['use_confidence_consistency']}, "
                f"loss_weight_mode={epoch_train_record['loss_weight_mode']}, "
                f"step_weight_mean={epoch_train_record['step_weight_mean']}, "
                f"step_weight_std={epoch_train_record['step_weight_std']}, "
                f"step_weight_min={epoch_train_record['step_weight_min']}, "
                f"step_weight_max={epoch_train_record['step_weight_max']}, "
                f"focal_factor_mean={epoch_train_record['focal_factor_mean']}, "
                f"raw_confidence_mean={epoch_train_record['raw_confidence_mean']}, "
                f"raw_confidence_std={epoch_train_record['raw_confidence_std']}, "
                f"raw_confidence_min={epoch_train_record['raw_confidence_min']}, "
                f"raw_confidence_max={epoch_train_record['raw_confidence_max']}, "
                f"normalized_confidence_mean={epoch_train_record['normalized_confidence_mean']}, "
                f"normalized_confidence_std={epoch_train_record['normalized_confidence_std']}, "
                f"teacher_confidence_weight_corr={epoch_train_record['teacher_confidence_weight_corr']}, "
                f"use_multihead_temporal_attn={epoch_train_record['use_multihead_temporal_attn']}, "
                f"use_multihead_query_attn={epoch_train_record['use_multihead_query_attn']}, "
                f"use_state_consistency={epoch_train_record['use_state_consistency']}, "
                f"lr={epoch_train_record['lr']}, "
                f"use_adaptive_fusion={epoch_train_record['use_adaptive_fusion']}, "
                f"use_exercise_aware_decay={epoch_train_record['use_exercise_aware_decay']}"
            )
            if ws_config['use_adaptive_fusion']:
                log_line += (
                    f", fusion_gate_rnn_mean={epoch_train_record['fusion_gate_rnn_mean']}, "
                    f"fusion_gate_attn_mean={epoch_train_record['fusion_gate_attn_mean']}, "
                    f"fusion_gate_query_mean={epoch_train_record['fusion_gate_query_mean']}"
                )
            if ws_config['use_exercise_aware_decay']:
                log_line += (
                    f", decay_gamma_mean={epoch_train_record['decay_gamma_mean']}, "
                    f"decay_gamma_std={epoch_train_record['decay_gamma_std']}"
                )
            o_f.write(log_line + '\n')
        val_metrics = val_test(
            ws,
            model,
            epoch,
            stage=2,
            dtype='validation',
            cross_idx=cross_idx,
            extra_metrics={
                'train_stage': 2,
                'lambda_consistency': float(lambda_consistency),
                'consistency_weight': float(current_consistency_weight),
                'use_confidence_consistency': int(use_confidence_consistency),
                'confidence_consistency_mode': confidence_consistency_mode,
                'loss_weight_mode': ws_config['loss_weight_mode'],
                'focal_gamma': float(ws_config['focal_gamma']),
                'focal_alpha': None if ws_config['focal_alpha'] is None else float(ws_config['focal_alpha']),
                'use_multihead_temporal_attn': int(ws_config['use_multihead_temporal_attn']),
                'use_multihead_query_attn': int(ws_config['use_multihead_query_attn']),
                'multihead_temporal_num_heads': int(ws_config['multihead_temporal_num_heads']),
                'multihead_query_num_heads': int(ws_config['multihead_query_num_heads']),
                'use_state_consistency': int(use_state_consistency),
                'state_consistency_target': state_consistency_target,
                'state_consistency_loss': state_consistency_loss,
                'lambda_state_consistency': float(lambda_state_consistency),
                'use_adaptive_fusion': int(ws_config['use_adaptive_fusion']),
                'adaptive_fusion_mode': ws_config['adaptive_fusion_mode'],
                'use_exercise_aware_decay': int(ws_config['use_exercise_aware_decay']),
                'exercise_aware_decay_mode': ws_config['exercise_aware_decay_mode'],
                'lr': get_optimizer_lr(optimizer)
            }
        )
        lr_before_scheduler = get_optimizer_lr(optimizer)
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
        lr_after_scheduler = get_optimizer_lr(optimizer)
        scheduler_log_line = (
            f"[scheduler] stage=2, epoch={epoch}: monitor_auc={val_metrics['auc']}, "
            f"lr_before={lr_before_scheduler}, lr_after={lr_after_scheduler}"
        )
        print(scheduler_log_line)
        with open(os.path.join(ws, 'results.txt'), 'a', encoding='utf8') as o_f:
            o_f.write(scheduler_log_line + '\n')
        append_metric_record(
            ws,
            {
                'dtype': 'scheduler',
                'stage': 2,
                'epoch': int(epoch),
                'monitor_auc': float(val_metrics['auc']),
                'lr_before': float(lr_before_scheduler),
                'lr_after': float(lr_after_scheduler),
                'scheduler_enabled': int(scheduler is not None)
            }
        )
        torch.save(model.state_dict(), os.path.join(ws, 'snapshot', f'stage2-{epoch}'))
    print('epoch_times:', str(epoch_times))


def test(ws, cross_idx, epoch_model=None, stage=2, device='cpu'):
    if epoch_model is None:
        raise ValueError('test() requires an explicit epoch_model so testing only runs once.')
    ws_config = read_ws_config(ws)
    model = create_model(ws_config)
    snapshot_path = os.path.join(ws, 'snapshot', f'stage{stage}-{int(epoch_model)}')
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f'Snapshot not found: {snapshot_path}')
    model.load_state_dict(torch.load(snapshot_path, map_location='cpu'))
    model = model.to(device)
    return val_test(ws, model, int(epoch_model), stage, 'test', cross_idx)


def find_metric_record_by_epoch(ws, stage, epoch, dtype='train'):
    metrics_path = os.path.join(ws, 'metrics.jsonl')
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r', encoding='utf8') as i_f:
        for line in i_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                record.get('stage') == int(stage) and
                record.get('dtype') == dtype and
                record.get('epoch') == int(epoch)
            ):
                return record
    return None


def run_dirt_plus_experiment(ws, device, cross_idx, stage1_epochs, stage2_epochs):
    ws_config = read_ws_config(ws)
    attr_idx = ws_config['attr_idx']
    if attr_idx not in [1, 2, 3, 4]:
        raise ValueError('attr_idx must be one of [1, 2, 3, 4].')
    if ws_config['stage2_lr'] >= ws_config['stage1_lr']:
        raise ValueError('stage2_lr must be smaller than stage1_lr for stable stage-wise fine-tuning.')

    reset_metric_logs(ws)
    train_stage1(
        ws,
        cross_idx,
        device=device,
        lr=ws_config['stage1_lr'],
        n_epochs=stage1_epochs,
        grad_clip=ws_config['grad_clip']
    )
    best_stage1_epoch, best_stage1_auc, best_stage1_acc, best_stage1_rmse = find_best_epoch_by_validation_priority(
        ws, stage=1, dtype='validation'
    )
    print(
        f'best stage1 epoch={best_stage1_epoch}, '
        f'validation_auc={best_stage1_auc}, validation_acc={best_stage1_acc}, validation_rmse={best_stage1_rmse}'
    )

    if attr_idx in [1, 2]:
        test_metrics = test(ws, cross_idx, epoch_model=best_stage1_epoch, stage=1, device=device)
        train_record = find_metric_record_by_epoch(ws, stage=1, epoch=best_stage1_epoch, dtype='train') or {}
        summary_record = {
            'dtype': 'final_summary',
            'stage': 1,
            'attr_idx': int(attr_idx),
            'attr_name': f'DIRT_{attr_idx}',
            'best_stage': 1,
            'best_epoch': int(best_stage1_epoch),
            'validation_auc': float(best_stage1_auc),
            'validation_acc': float(best_stage1_acc),
            'validation_rmse': float(best_stage1_rmse),
            'test_auc': float(test_metrics['auc']),
            'test_acc': float(test_metrics['acc']),
            'test_rmse': float(test_metrics['rmse']),
            'loss_weight_mode': ws_config['loss_weight_mode'],
            'step_weight_mean': float(train_record.get('step_weight_mean', 0.0)),
            'step_weight_std': float(train_record.get('step_weight_std', 0.0)),
            'step_weight_min': float(train_record.get('step_weight_min', 0.0)),
            'step_weight_max': float(train_record.get('step_weight_max', 0.0)),
            'focal_factor_mean': float(train_record.get('focal_factor_mean', 1.0))
        }
        emit_final_experiment_summary(ws, summary_record)
        return summary_record

    train_stage2(
        ws,
        cross_idx,
        stage1_epoch=best_stage1_epoch,
        n_epochs=stage2_epochs,
        device=device,
        lr=ws_config['stage2_lr'],
        grad_clip=ws_config['grad_clip'],
        lambda_consistency=ws_config['lambda_consistency'],
        consistency_warmup_epochs=ws_config['consistency_warmup_epochs'],
        consistency_warmup_start_ratio=ws_config['consistency_warmup_start_ratio'],
        use_confidence_consistency=ws_config['use_confidence_consistency'],
        confidence_consistency_mode=ws_config['confidence_consistency_mode'],
        confidence_consistency_eps=ws_config['confidence_consistency_eps'],
        use_state_consistency=ws_config['use_state_consistency'],
        state_consistency_target=ws_config['state_consistency_target'],
        state_consistency_loss=ws_config['state_consistency_loss'],
        lambda_state_consistency=ws_config['lambda_state_consistency'],
        use_confidence_for_state_consistency=ws_config['use_confidence_for_state_consistency']
    )
    best_stage2_epoch, best_stage2_auc, best_stage2_acc, best_stage2_rmse = find_best_epoch_by_validation_priority(
        ws, stage=2, dtype='validation'
    )
    print(
        f'best stage2 epoch={best_stage2_epoch}, '
        f'validation_auc={best_stage2_auc}, validation_acc={best_stage2_acc}, validation_rmse={best_stage2_rmse}'
    )
    test_metrics = test(ws, cross_idx, epoch_model=best_stage2_epoch, stage=2, device=device)
    train_record = find_metric_record_by_epoch(ws, stage=2, epoch=best_stage2_epoch, dtype='train') or {}
    summary_record = {
        'dtype': 'final_summary',
        'stage': 2,
        'attr_idx': int(attr_idx),
        'attr_name': f'DIRT_{attr_idx}',
        'best_stage': 2,
        'best_epoch': int(best_stage2_epoch),
        'validation_auc': float(best_stage2_auc),
        'validation_acc': float(best_stage2_acc),
        'validation_rmse': float(best_stage2_rmse),
        'test_auc': float(test_metrics['auc']),
        'test_acc': float(test_metrics['acc']),
        'test_rmse': float(test_metrics['rmse']),
        'loss_weight_mode': ws_config['loss_weight_mode'],
        'step_weight_mean': float(train_record.get('step_weight_mean', 0.0)),
        'step_weight_std': float(train_record.get('step_weight_std', 0.0)),
        'step_weight_min': float(train_record.get('step_weight_min', 0.0)),
        'step_weight_max': float(train_record.get('step_weight_max', 0.0)),
        'focal_factor_mean': float(train_record.get('focal_factor_mean', 1.0))
    }
    emit_final_experiment_summary(ws, summary_record)
    return summary_record


def run_all_dirt_plus_experiments(
    ws_root,
    device,
    base_ws_config,
    cross_idx,
    stage1_epochs,
    stage2_epochs,
    exp_name='default',
    attr_indices=None
):
    summary_records = []
    attr_indices = attr_indices or [1, 2, 3, 4]
    for attr_idx in attr_indices:
        ws = os.path.join(ws_root, f'DIRT_{attr_idx}')
        ws_config = dict(base_ws_config)
        ws_config['attr_idx'] = attr_idx
        config_ws(ws, ws_config)
        print(f'========== Running DIRT+_{attr_idx} at {ws} ==========')
        summary_record = run_dirt_plus_experiment(ws, device, cross_idx, stage1_epochs, stage2_epochs)
        if summary_record is not None:
            summary_records.append(summary_record)
    emit_root_experiment_summary(ws_root, exp_name, summary_records)


def parse_args():
    parser = argparse.ArgumentParser(description='DIRT+ independent enhanced implementation.')
    parser.add_argument('--data', type=str, default='assist2009')
    parser.add_argument('--cross_idx', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--stage1_lr', type=float, default=None)
    parser.add_argument('--stage2_lr', type=float, default=None)
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stu_ho_dim', type=int, default=50)
    parser.add_argument('--rnn_type', type=str, default='gru')
    parser.add_argument('--attn_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_multihead_temporal_attn', type=int, default=0)
    parser.add_argument('--use_multihead_query_attn', type=int, default=0)
    parser.add_argument('--multihead_temporal_num_heads', type=int, default=2)
    parser.add_argument('--multihead_query_num_heads', type=int, default=2)
    parser.add_argument('--multihead_attn_dropout', type=float, default=0.1)
    parser.add_argument('--use_temporal_self_attention', type=int, default=1)
    parser.add_argument('--use_query_guided_attention', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--lambda_consistency', type=float, default=0.2)
    parser.add_argument('--consistency_warmup_epochs', type=int, default=3)
    parser.add_argument('--consistency_warmup_start_ratio', type=float, default=0.2)
    parser.add_argument('--use_confidence_consistency', type=int, default=1)
    parser.add_argument('--confidence_consistency_mode', type=str, default='prob_margin')
    parser.add_argument('--confidence_consistency_eps', type=float, default=1e-8)
    parser.add_argument('--use_stage_scheduler', type=int, default=1)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=1)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-5)
    parser.add_argument('--use_temporal_bias', type=int, default=1)
    parser.add_argument('--temporal_bias_mode', type=str, default='linear')
    parser.add_argument('--temporal_bias_init', type=float, default=0.05)
    parser.add_argument('--use_exercise_aware_decay', type=int, default=1)
    parser.add_argument('--exercise_aware_decay_mode', type=str, default='query_linear')
    parser.add_argument('--exercise_aware_decay_scale', type=float, default=1.0)
    parser.add_argument('--exercise_aware_decay_min', type=float, default=0.0)
    parser.add_argument('--exercise_aware_decay_max', type=float, default=0.2)
    parser.add_argument('--use_adaptive_fusion', type=int, default=1)
    parser.add_argument('--adaptive_fusion_mode', type=str, default='time_step_softmax')
    parser.add_argument('--adaptive_fusion_hidden_dim', type=int, default=50)
    parser.add_argument('--use_state_consistency', type=int, default=0)
    parser.add_argument('--state_consistency_target', type=str, default='fused_hidden')
    parser.add_argument('--state_consistency_loss', type=str, default='smooth_l1')
    parser.add_argument('--lambda_state_consistency', type=float, default=0.05)
    parser.add_argument('--use_confidence_for_state_consistency', type=int, default=1)
    parser.add_argument('--loss_weight_mode', type=str, default='rule')
    parser.add_argument('--step_weight_hidden_dim', type=int, default=32)
    parser.add_argument('--step_weight_dropout', type=float, default=0.1)
    parser.add_argument('--step_weight_use_teacher_confidence', type=int, default=1)
    parser.add_argument('--step_weight_use_position_feature', type=int, default=1)
    parser.add_argument('--step_weight_min', type=float, default=0.5)
    parser.add_argument('--step_weight_max', type=float, default=3.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--attr_indices', type=str, default='1,2,3,4')
    parser.add_argument('--ws_root', type=str, default='ws/dirt_plus/assist09')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.loss_weight_mode not in ['plain', 'rule', 'learnable', 'focal', 'learnable_focal']:
        raise ValueError('loss_weight_mode must be one of [plain, rule, learnable, focal, learnable_focal].')
    if args.step_weight_max < args.step_weight_min:
        raise ValueError('step_weight_max must be >= step_weight_min.')
    attr_indices = [int(item.strip()) for item in args.attr_indices.split(',') if item.strip()]
    if not attr_indices or any(item not in [1, 2, 3, 4] for item in attr_indices):
        raise ValueError('attr_indices must be a comma-separated subset of [1,2,3,4].')
    set_global_seed(args.seed)
    exp_name = build_experiment_name(args)
    stage1_lr = args.stage1_lr if args.stage1_lr is not None else args.lr
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr * 0.5

    with open(f'data/{args.data}/data_config.txt', encoding='utf8') as i_f:
        data_config = ast.literal_eval(i_f.readline())

    ws_config_dict = {
        'stu_ho_dim': args.stu_ho_dim,
        'rnn_type': args.rnn_type,
        'attr_idx': 1,
        'data': args.data,
        'cross_idx': args.cross_idx,
        'stage1_epochs': args.stage1_epochs,
        'stage2_epochs': args.stage2_epochs,
        'batch_size': args.batch_size,
        'max_log': data_config['max_log'],
        'exer_n': data_config['exer_n'],
        'knowledge_n': data_config['knowledge_n'],
        'student_n': data_config['student_n'],
        'attn_heads': args.attn_heads,
        'dropout': args.dropout,
        'use_multihead_temporal_attn': args.use_multihead_temporal_attn,
        'use_multihead_query_attn': args.use_multihead_query_attn,
        'multihead_temporal_num_heads': args.multihead_temporal_num_heads,
        'multihead_query_num_heads': args.multihead_query_num_heads,
        'multihead_attn_dropout': args.multihead_attn_dropout,
        'use_temporal_self_attention': args.use_temporal_self_attention,
        'use_query_guided_attention': args.use_query_guided_attention,
        'stage1_lr': stage1_lr,
        'stage2_lr': stage2_lr,
        'grad_clip': args.grad_clip,
        'lambda_consistency': args.lambda_consistency,
        'consistency_warmup_epochs': args.consistency_warmup_epochs,
        'consistency_warmup_start_ratio': args.consistency_warmup_start_ratio,
        'use_confidence_consistency': args.use_confidence_consistency,
        'confidence_consistency_mode': args.confidence_consistency_mode,
        'confidence_consistency_eps': args.confidence_consistency_eps,
        'use_stage_scheduler': args.use_stage_scheduler,
        'scheduler_factor': args.scheduler_factor,
        'scheduler_patience': args.scheduler_patience,
        'scheduler_min_lr': args.scheduler_min_lr,
        'use_temporal_bias': args.use_temporal_bias,
        'temporal_bias_mode': args.temporal_bias_mode,
        'temporal_bias_init': args.temporal_bias_init,
        'use_exercise_aware_decay': args.use_exercise_aware_decay,
        'exercise_aware_decay_mode': args.exercise_aware_decay_mode,
        'exercise_aware_decay_scale': args.exercise_aware_decay_scale,
        'exercise_aware_decay_min': args.exercise_aware_decay_min,
        'exercise_aware_decay_max': args.exercise_aware_decay_max,
        'use_adaptive_fusion': args.use_adaptive_fusion,
        'adaptive_fusion_mode': args.adaptive_fusion_mode,
        'adaptive_fusion_hidden_dim': args.adaptive_fusion_hidden_dim,
        'use_state_consistency': args.use_state_consistency,
        'state_consistency_target': args.state_consistency_target,
        'state_consistency_loss': args.state_consistency_loss,
        'lambda_state_consistency': args.lambda_state_consistency,
        'use_confidence_for_state_consistency': args.use_confidence_for_state_consistency,
        'loss_weight_mode': args.loss_weight_mode,
        'step_weight_hidden_dim': args.step_weight_hidden_dim,
        'step_weight_dropout': args.step_weight_dropout,
        'step_weight_use_teacher_confidence': args.step_weight_use_teacher_confidence,
        'step_weight_use_position_feature': args.step_weight_use_position_feature,
        'step_weight_min': args.step_weight_min,
        'step_weight_max': args.step_weight_max,
        'focal_gamma': args.focal_gamma,
        'focal_alpha': args.focal_alpha,
        'exp_name': exp_name,
        'seed': args.seed
    }

    final_ws_root = os.path.join(args.ws_root, exp_name)
    run_all_dirt_plus_experiments(
        ws_root=final_ws_root,
        device=args.device,
        base_ws_config=ws_config_dict,
        cross_idx=args.cross_idx,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        exp_name=exp_name,
        attr_indices=attr_indices
    )
