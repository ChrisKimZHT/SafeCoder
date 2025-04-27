import argparse
import os
import yaml


def main():
    yaml_files = os.listdir(args.dir)
    yaml_files = [f for f in yaml_files if f.endswith(".yaml")]
    yaml_files = [f for f in yaml_files if "results" not in f]

    length = {}
    for yaml_file in yaml_files:
        with open(os.path.join(args.dir, yaml_file), "r") as f:
            data = yaml.safe_load(f)
        completion_count = len(data["completions"])
        if completion_count < args.threshold:
            print(f"File {yaml_file} has only {completion_count} completions.")

        length[completion_count] = length.get(completion_count, 0) + 1

    print("Completion count distribution:")
    length = dict(sorted(length.items()))
    last = -1
    for k, v in length.items():
        if last < args.threshold and k >= args.threshold:
            print("-" * 10)
        last = k
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=20)
    args = parser.parse_args()
    main()
