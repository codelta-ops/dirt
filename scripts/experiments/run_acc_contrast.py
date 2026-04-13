import argparse
import os
import subprocess
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_SCRIPT = os.path.join("models", "dirt_plus.py")


EXPERIMENTS: List[Dict] = [
    {
        'name': 'acc_control_tc0_range08_20',
        'purpose': 'Current ACC-best control group.',
        'args': {
            'loss_weight_mode': 'learnable',
            'step_weight_hidden_dim': 32,
            'step_weight_dropout': 0.10,
            'step_weight_use_teacher_confidence': 0,
            'step_weight_use_position_feature': 1,
            'step_weight_min': 0.8,
            'step_weight_max': 2.0,
            'stage1_lr': 0.002,
            'stage2_lr': 0.0008,
            'lambda_consistency': 0.2,
            'consistency_warmup_epochs': 4
        }
    },
    {
        'name': 'acc_tc0_range08_20_s2lr0007',
        'purpose': 'Use a milder lower stage2 learning rate to balance validation and test.',
        'args': {
            'loss_weight_mode': 'learnable',
            'step_weight_hidden_dim': 32,
            'step_weight_dropout': 0.10,
            'step_weight_use_teacher_confidence': 0,
            'step_weight_use_position_feature': 1,
            'step_weight_min': 0.8,
            'step_weight_max': 2.0,
            'stage1_lr': 0.002,
            'stage2_lr': 0.0007,
            'lambda_consistency': 0.2,
            'consistency_warmup_epochs': 4
        }
    },
    {
        'name': 'acc_tc0_range08_20_lambda015',
        'purpose': 'Slightly reduce consistency strength to test whether test generalization improves.',
        'args': {
            'loss_weight_mode': 'learnable',
            'step_weight_hidden_dim': 32,
            'step_weight_dropout': 0.10,
            'step_weight_use_teacher_confidence': 0,
            'step_weight_use_position_feature': 1,
            'step_weight_min': 0.8,
            'step_weight_max': 2.0,
            'stage1_lr': 0.002,
            'stage2_lr': 0.0008,
            'lambda_consistency': 0.15,
            'consistency_warmup_epochs': 4
        }
    },
    {
        'name': 'acc_tc0_range08_20_warmup5',
        'purpose': 'Extend consistency warmup slightly for a gentler stage2 transition.',
        'args': {
            'loss_weight_mode': 'learnable',
            'step_weight_hidden_dim': 32,
            'step_weight_dropout': 0.10,
            'step_weight_use_teacher_confidence': 0,
            'step_weight_use_position_feature': 1,
            'step_weight_min': 0.8,
            'step_weight_max': 2.0,
            'stage1_lr': 0.002,
            'stage2_lr': 0.0008,
            'lambda_consistency': 0.2,
            'consistency_warmup_epochs': 5
        }
    }
]


def format_value(value):
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def build_command(args, exp_config):
    cmd = [
        args.python_bin,
        args.script,
        '--data', args.data,
        '--cross_idx', str(args.cross_idx),
        '--device', args.device,
        '--stage1_epochs', str(args.stage1_epochs),
        '--stage2_epochs', str(args.stage2_epochs),
        '--ws_root', args.ws_root,
        '--exp_name', exp_config['name'],
        '--attr_indices', args.attr_indices
    ]
    for key, value in exp_config['args'].items():
        cmd.extend([f'--{key}', format_value(value)])
    return cmd


def to_shell_line(cmd):
    return ' '.join(cmd)


def build_experiment_label(exp_config):
    arg_items = []
    for key, value in exp_config['args'].items():
        arg_items.append(f'{key}={format_value(value)}')
    return f'{exp_config["name"]} | ' + ', '.join(arg_items)


def print_experiment_header(prefix, exp_config, idx=None, total=None):
    if idx is not None and total is not None:
        print(f'{prefix} [{idx}/{total}] {build_experiment_label(exp_config)}')
    else:
        print(f'{prefix} {build_experiment_label(exp_config)}')


def print_commands(args):
    for exp in EXPERIMENTS:
        print_experiment_header('# Experiment', exp)
        print(f'# Purpose: {exp["purpose"]}')
        print(to_shell_line(build_command(args, exp)))
        print()


def run_commands(args):
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        cmd = build_command(args, exp)
        print_experiment_header('>>> Running', exp, idx=idx, total=len(EXPERIMENTS))
        print(f'>>> Purpose: {exp["purpose"]}')
        print(to_shell_line(cmd))
        subprocess.run(cmd, check=True)


def read_summary_file(summary_path):
    if not os.path.exists(summary_path):
        return []
    with open(summary_path, 'r', encoding='utf8') as i_f:
        return [line.rstrip('\n') for line in i_f if line.strip()]


def print_summaries(args):
    for exp in EXPERIMENTS:
        summary_path = os.path.join(args.ws_root, exp['name'], 'experiment_summary.txt')
        print(f'===== {build_experiment_label(exp)} =====')
        if not os.path.exists(summary_path):
            print(f'missing summary: {summary_path}')
            print()
            continue
        for line in read_summary_file(summary_path):
            print(line)
        print()


def parse_args():
    parser = argparse.ArgumentParser(description='ACC-focused contrast runner for dirt_plus.')
    parser.add_argument('--mode', type=str, default='print', choices=['print', 'run', 'summary'])
    parser.add_argument('--python_bin', type=str, default='python')
    parser.add_argument('--script', type=str, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument('--data', type=str, default='assist2009')
    parser.add_argument('--cross_idx', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=5)
    parser.add_argument('--attr_indices', type=str, default='3,4')
    parser.add_argument('--ws_root', type=str, default='ws/acc_search')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if args.mode == 'print':
        print_commands(args)
    elif args.mode == 'run':
        run_commands(args)
    else:
        print_summaries(args)
