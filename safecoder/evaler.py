import os
import re
import abc
import openai
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .utils import load_model, set_seed, try_parse
from .constants import PROMPT_NO_INPUT, INSTRUCTION, LANGUAGE_MAPS, PRETRAINED_MODELS
from .constants import SECURE_PROMPTING_GENERIC, SECURE_PROMPTING_SPECIFIC, CWE_DESCRIPTIONS


def truncate_after(completion, trunc_str):
    return completion[:completion.find(trunc_str) + len(trunc_str)]


def truncate_before(completion, trunc_str):
    return completion[:completion.find(trunc_str)].rstrip()


def truncate_after_last(completion, trunc_str):
    return completion[:completion.rfind(trunc_str) + len(trunc_str)]


def truncate_before_last(completion, trunc_str):
    return completion[:completion.rfind(trunc_str)]


class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = load_model(args.model_name, args)

    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed(self.args.seed+i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[:completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + '\n'
                if info['language'] != 'go' and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, non_parsed_srcs

    @abc.abstractclassmethod
    def preprocess(self, file_context, func_context, info):
        raise NotImplementedError()

    def postprocess(self, completion, info):
        if info['language'] == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                if '\n    #' in completion:
                    completion = truncate_before_last(completion, '\n    #')
        elif info['language'] in ['c', 'cpp']:
            if '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif ';\n' in completion:
                completion = truncate_after_last(completion, ';\n') + '\n}'
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'go':
            if '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'js':
            if '\n});' in completion: # for app function definitions
                completion = truncate_after(completion, '\n});')
            elif re.search(r'\n}(?!;)', completion) is not None: # normal function end
                match = re.search(r'\n}(?!;)', completion)
                completion = completion[:match.end()]
            elif '\n//' in completion:
                completion = truncate_before_last(completion, '\n//').rstrip()
            elif '\n/*' in completion:
                completion = truncate_before_last(completion, '\n/*').rstrip()
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'jsx':
            # only for cwe-200 0-jsx
            if '\n' in completion:
                completion = truncate_before(completion, '\n')
        elif info['language'] == 'rb':
            if '\n    end' in completion:
                completion = truncate_after(completion, '\n    end') + '\nend'
            elif '\nend' in completion:
                completion = truncate_after(completion, '\nend')
            elif '    #' in completion:
                completion = truncate_before_last(completion, '    #').rstrip('\n') + '\nend'
                if '\nend' not in completion: completion += '\nend'
            else:
                completion = completion
        elif info['language'] == 'java':
            if '\n    }' in completion:
                completion = truncate_after(completion, '\n    }') + '\n}'
            elif '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif ';\n' in completion:
                completion = truncate_after_last(completion, ';\n') + '\n    }' + '\n}'
            elif '    //' in completion:
                completion = truncate_before_last(completion, '    //').rstrip('\n') + '\n}'
                if '\n}' not in completion: completion += '\n}'
            elif '    /*' in completion:
                completion = truncate_before_last(completion, '    /*').rstrip('\n') + '\n}'
                if '\n}' not in completion: completion += '\n}'
            else:
                completion = completion
        else:
            raise NotImplementedError('Postprocessing for {language} is not implemented yet'.format(language=info['language']))

        if 'postprocess' in info:
            scope = {'completion': completion}
            exec(info['postprocess'], scope)
            completion = scope['completion']

        return completion

class EvalerCodePLM(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context

class EvalerCodeFT(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info['language']]
        if self.args.sec_prompting == 'generic':
            instruction = SECURE_PROMPTING_GENERIC.format_map({'language': lang, 'prompt': info['description']})
        elif self.args.sec_prompting == 'specific':
            instruction = SECURE_PROMPTING_SPECIFIC.format_map({'language': lang, 'prompt': info['description'], 'cwe': info['cwe'], 'cwe_desc': CWE_DESCRIPTIONS[info['cwe']]})
        else:
            instruction = INSTRUCTION.format_map({'language': lang, 'prompt': info['description']})
        prompt = PROMPT_NO_INPUT.format_map({'instruction': instruction})
        prompt += file_context + func_context
        return prompt

class EvalerChat(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info['language']]

        if self.args.sec_prompting == 'generic':
            instruction = SECURE_PROMPTING_GENERIC.format_map({'language': lang, 'prompt': info['description']})
        elif self.args.sec_prompting == 'specific':
            instruction = SECURE_PROMPTING_SPECIFIC.format_map({'language': lang, 'prompt': info['description'], 'cwe': info['cwe'], 'cwe_desc': CWE_DESCRIPTIONS[info['cwe']]})
        else:
            instruction = INSTRUCTION.format_map({'language': lang, 'prompt': info['description']})

        if self.args.model_name == 'octocoder':
            template = 'Question: {instruction}\n\nAnswer: \n'
            prompt = template.format_map({'instruction': instruction})
            prompt += file_context + func_context
        else:
            if self.args.model_name == 'deepseek':
                prompt = instruction
            else:
                prompt = PROMPT_NO_INPUT[:PROMPT_NO_INPUT.rfind('\n\n')].format_map({'instruction': instruction})
            messages = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': file_context+func_context}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = prompt.removeprefix('<s>').removesuffix('</s> ').removesuffix(' </s>').removesuffix('\n<|EOT|>\n')
        return prompt

class EvalerOpenAI(EvalerBase):
    def __init__(self, args):
        self.args = args
        self.model = args.model_name
        self.client = openai.OpenAI()

    def _extract_markdown(self, md):
        pattern = r'```.*?\n(.*?)```'
        matches = re.findall(pattern, md, re.DOTALL)
        return matches

    def sample(self, file_context, func_context, info):
        lang = info['language']

        if self.args.sec_prompting == 'generic':
            instruction = SECURE_PROMPTING_GENERIC.format_map({'language': lang, 'prompt': info['description']})
        elif self.args.sec_prompting == 'specific':
            instruction = SECURE_PROMPTING_SPECIFIC.format_map({'language': lang, 'prompt': info['description'], 'cwe': info['cwe'], 'cwe_desc': CWE_DESCRIPTIONS[info['cwe']]})
        else:
            instruction = INSTRUCTION.format_map({'language': lang, 'prompt': info['description']})
        prompt = PROMPT_NO_INPUT.format_map({'instruction': instruction})
        prompt += file_context+func_context

        srcs = []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                n=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                seed=self.args.seed+i
            )
            for choice in response.choices:
                completion = choice.text
                completion = self.postprocess(completion, info)
                srcs.append(file_context+func_context+completion)

        output_srcs, non_parsed_srcs = [], []
        for src in srcs:
            if info['language'] != 'go' and try_parse(src, info) != 0:
                non_parsed_srcs.append(src)
            else:
                output_srcs.append(src)

        return output_srcs, non_parsed_srcs
    

class EvalerMyReasoner(EvalerBase):
    def __init__(self, args):
        self.system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
At the same time, please wrap the generated code in markdown code blocks ``` ```.
i.e., <think> reasoning process here </think><answer> answer here ``` code here ``` </answer>"""
        self.user_template = "Please complete the code according to the task description and the provided code snippet.\n# Task description:\n\n{description}\n\n# Code snippet:\n\n```{language}\n{snippet}\n```"
        self.args = args
        self.model = args.model_name
        self.client = openai.OpenAI(api_key=args.token, base_url=args.api_base)

    def _extract_markdown(self, md):
        pattern = r'```.*?\n(.*?)```'
        matches = re.findall(pattern, md, re.DOTALL)
        return matches

    def _query_model(self, user_message: str) -> tuple[str, str]:
        messages = []
        if self.args.use_system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})

        completion = self.client.chat.completions.create(
            model=self.args.model_name,
            messages=messages,
            temperature=self.args.temperature,
            max_tokens=self.args.max_completion_tokens,
            max_completion_tokens=self.args.max_completion_tokens,
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

    def _sample_one(self, prompt: str, info) -> tuple[str, str] | None:
        try:
            _, answer_content = self._query_model(prompt)
        except Exception as e:
            print(f"Error querying model: {e}")
            traceback.print_exc()
            return None

        if answer_content == "":
            print("Empty answer content")
            return None

        code_blocks = self._extract_markdown(answer_content)
        if len(code_blocks) == 0:
            print("No code blocks found")
            return None

        code_block = max(code_blocks, key=len)

        return code_block

    def sample(self, file_context, func_context, info):
        lang = info['language']

        prompt = self.user_template.format_map({'description': info['description'], 'snippet': file_context + func_context, 'language': lang})

        srcs = []
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            futures = [executor.submit(self._sample_one, prompt, info) for _ in range(self.args.num_samples)]
            with tqdm(total=len(futures), dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        srcs.append(res)
                    pbar.update(1)

        output_srcs, non_parsed_srcs = [], []
        for src in srcs:
            if info['language'] != 'go' and try_parse(src, info) != 0:
                non_parsed_srcs.append(src)
            else:
                output_srcs.append(src)

        return output_srcs, non_parsed_srcs
