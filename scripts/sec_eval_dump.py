import argparse
import csv
import json
import os
import re


def listdir_only_dirs(path: str) -> list[str]:
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def process_one_run(name: str, parsed_results: list, non_parsed_results: list, codeql_results: list) -> list:
    vulner_sample_id = []

    for codeql_result in codeql_results:
        # assuming csv format is correct, or it may crash
        file_name = codeql_result[4]
        file_id = int(re.search(r"(\d+)", file_name).group(1))
        vulner_sample_id.append(file_id)

    processed_result = []

    for i, result in enumerate(parsed_results):
        processed_result.append({
            "type": "vulnerable" if i in vulner_sample_id else "secure",
            "name": name,
            **result
        })

    for result in non_parsed_results:
        processed_result.append({
            "type": "error",
            "name": name,
            **result
        })

    return processed_result


def main() -> None:
    eval_types = listdir_only_dirs(args.input)

    output_file = open(args.output, "w")

    for eval_type in eval_types:
        eval_dir = os.path.join(args.input, eval_type)
        cwes = listdir_only_dirs(eval_dir)

        for cwe in cwes:
            cwe_dir = os.path.join(eval_dir, cwe)
            runs = listdir_only_dirs(cwe_dir)

            for run in runs:
                run_dir = os.path.join(cwe_dir, run)

                with open(os.path.join(run_dir, "parsed_results.jsonl"), "r") as f:
                    parsed_results = [json.loads(line) for line in f]

                with open(os.path.join(run_dir, "non_parsed_results.jsonl"), "r") as f:
                    non_parsed_results = [json.loads(line) for line in f]

                with open(os.path.join(run_dir, "codeql.csv"), "r") as f:
                    codeql_results = list(csv.reader(f))

                results = process_one_run(f"{cwe}-{run}", parsed_results, non_parsed_results, codeql_results)
                for result in results:
                    output_file.write(json.dumps(result) + "\n")

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="sec_eval_dump.jsonl")
    args = parser.parse_args()
    main()
