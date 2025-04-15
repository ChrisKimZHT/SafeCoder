import os
import re
import sys
import torch
import numpy
import random
import shutil
import argparse
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

from safecoder.utils import set_seed, load_model
from safecoder.human_eval.problem_yaml import Problem
from safecoder.constants import PRETRAINED_MODELS, CHAT_MODELS, PROMPT_NO_INPUT, INSTRUCTION

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, required=True, choices=['human_eval', 'mbpp'])
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='codegen-350m')

    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=40)
    parser.add_argument('--num_samples_per_gen', type=int, default=10)

    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=1)

    # MyResoner specific arguments
    parser.add_argument('--use-my-reasoner', action='store_true')
    parser.add_argument('--use-system-prompt', action='store_true')
    parser.add_argument('--api-base', type=str, default='http://127.0.0.1:9997/v1')
    parser.add_argument('--token', type=str, default='none')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max-completion-tokens', type=int, default=16384)
    parser.add_argument("--max-workers", type=int, default=8)

    args = parser.parse_args()

    if not args.use_my_reasoner:
        assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.experiments_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args

args = get_args()

def extract_docstr(prompt):
    delim = '\"\"\"'
    assert delim in prompt

    output = prompt[prompt.find(delim)+len(delim):]
    output = output[:output.find(delim)]
    output = output.replace('\n    ', '\n').strip()

    return output

def extract_funcsig(prompt):
    delim = '\"\"\"'
    return prompt[:prompt.find(delim)].strip()

def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion

def query_model(client, messages: list) -> tuple[str, str]:
    completion = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_completion_tokens,
        max_completion_tokens=args.max_completion_tokens,
        stream=True
    )

    think_content = ""
    answer_content = ""

    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
            think_content += delta.reasoning_content
        elif hasattr(delta, "content") and delta.content != None:
            answer_content += delta.content

    return think_content, answer_content

def extract_markdown(md):
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, md, re.DOTALL)
    return matches

def sample_one(client, messages: list) -> tuple[str, str] | None:
    try:
        _, answer_content = query_model(client, messages)
    except Exception as e:
        print(f"Error querying model: {e}")
        traceback.print_exc()
        return None

    if answer_content == "":
        print("Empty answer content")
        return None

    code_blocks = extract_markdown(answer_content)
    if len(code_blocks) == 0:
        print("No code blocks found")
        return None

    code_block = max(code_blocks, key=len)

    return code_block


def main():
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        sys.exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    if args.use_my_reasoner:
        client = OpenAI(api_key=args.token, base_url=args.api_base)
        system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
At the same time, please wrap the generated code in markdown code blocks ``` ```.
i.e., <think> reasoning process here </think><answer> answer here ``` code here ``` </answer>"""
        user_prompt_prefix = "Completing the following function snippet, then output the full code in a ``` code block."

    else:
        tokenizer, model = load_model(args.model_name, args)
        model.eval()

    is_pretrained = args.model_name in PRETRAINED_MODELS
    is_chat = args.model_name in CHAT_MODELS

    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        orig_prompt = problem.prompt.strip()

        # this is my reasoner specific
        if args.use_my_reasoner:
            messages = []
            if args.use_system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            
            prompt = f"{user_prompt_prefix}\n\n```python\n{orig_prompt}\n```"
            messages.append({'role': 'user', 'content': prompt})
        elif is_chat:
            if args.model_name == 'octocoder':
                template = 'Question: {instruction}\n\nAnswer: '
                prompt = template.format_map({'instruction': INSTRUCTION.format_map({'language': 'Python', 'prompt': extract_docstr(orig_prompt)})})
                prompt += extract_funcsig(orig_prompt)
            else:
                prompt = PROMPT_NO_INPUT[:PROMPT_NO_INPUT.rfind('\n\n')].format_map({'instruction': INSTRUCTION.format_map({'language': 'Python', 'prompt': extract_docstr(orig_prompt)})})
                messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': extract_funcsig(orig_prompt)}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt = prompt.removeprefix('<s>').removesuffix('</s> ').removesuffix(' </s>')
        elif is_pretrained:
            prompt = orig_prompt
        else:
            prompt = PROMPT_NO_INPUT.format_map({'instruction': INSTRUCTION.format_map({'language': 'Python', 'prompt': extract_docstr(orig_prompt)})})
            prompt += extract_funcsig(orig_prompt)

        # this is my reasoner specific
        if args.use_my_reasoner:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # futures = [executor.submit(sample_one, client, messages) for _ in range(args.num_samples)]
                futures = []
                for _ in range(args.num_samples):
                    futures.append(executor.submit(sample_one, client, messages))
                    if len(futures) < args.max_workers:
                        time.sleep(0.1)
                with tqdm(total=len(futures), dynamic_ncols=True) as pbar:
                    for future in as_completed(futures):
                        res = future.result()
                        if res is not None:
                            problem.completions.append(res)
                        pbar.update(1)
        else:
            # print(prompt)
            # print('='*150)
            inputs = tokenizer(prompt.strip(), return_tensors='pt').to(model.device)
            seed = args.seed
            for i in range(args.num_samples // args.num_samples_per_gen):
                set_seed(seed+i)
                with torch.no_grad():
                    if hasattr(model.config, 'n_positions'):
                        n_ctx = model.config.n_positions
                    elif hasattr(model.config, 'max_position_embeddings'):
                        n_ctx = model.config.max_position_embeddings
                    else:
                        n_ctx = 32000 # some arbitrary large context, risky as it could lead to errors
                    max_gen_len = max(0, min(n_ctx - 1 - len(inputs['input_ids'][0]), args.max_gen_len))
                    samples = model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=args.num_samples_per_gen,
                        temperature=args.temp,
                        max_new_tokens=max_gen_len,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                for sample in samples.tolist():
                    # print(tokenizer.decode(sample))
                    # print('*'*150)
                    completion = sample[inputs['input_ids'].shape[1]:]
                    if tokenizer.eos_token_id in completion:
                        completion = completion[:completion.index(tokenizer.eos_token_id)]
                    completion = tokenizer.decode(completion)
                    completion = trim_code(completion, problem.stop_tokens)
                    # print(completion)
                    # print('='*150)
                    problem.completions.append(completion)
        with problem_yaml_path.open('w') as f:
            f.write(Problem.dump(problem))

if __name__ == '__main__':
    main()