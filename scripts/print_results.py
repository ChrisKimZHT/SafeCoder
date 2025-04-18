import os
import json
import argparse

from safecoder.metric import FuncEval, SecEval, MMLUEval, TruthfulQAEval

EVAL_CHOICES = [
    'human_eval',
    'mbpp',
    'trained',
    'trained-new',
    'trained-joint',
    'not-trained',
    'mmlu',
    'tqa',
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True)
    parser.add_argument('--detail', action='store_true', default=False)
    parser.add_argument('--eval_type', type=str, choices=EVAL_CHOICES, default='trained')
    parser.add_argument('--split', type=str, choices=['val', 'test', 'all', 'validation', 'intersec', 'diff'], default='test')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    args = parser.parse_args()

    if args.n_shots is None:
        if args.eval_type == 'multiple_choice':
            args.n_shots = 5
        elif args.eval_type == 'mmlu':
            args.n_shots = 5

    return args

def main():
    args = get_args()
    if args.eval_type in ('human_eval', 'mbpp'):
        e = FuncEval(os.path.join(args.experiments_dir, args.eval_type, args.eval_name))
    elif args.eval_type == 'mmlu':
        e = MMLUEval(os.path.join(
            args.experiments_dir, 'mmlu_eval', args.eval_name, args.eval_type, args.split, f'result_{args.n_shots}_{args.seed}.csv'
        ))
    elif args.eval_type == 'tqa':
        e = TruthfulQAEval(os.path.join(
            args.experiments_dir, 'truthfulqa_eval', args.eval_name, 'multiple_choice', args.split, f'result_{args.n_shots}_{args.seed}.csv'
        ))
    else:
        e = SecEval(os.path.join(args.experiments_dir, 'sec_eval', args.eval_name), args.split, args.eval_type)
    e.pretty_print(args.detail)

if __name__ == '__main__':
    main()