"""
Run repeated paired evaluations over multiple evaluation RNG seeds.

Each evaluation script must accept:
    -c / --checkpoint
    -o / --output_dir
    -d / --device
    --eval_seed

Example:
python run_multi_seed_eval.py \
    -c data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
    --output-root data/eval_precision_seed \
    --job fp32=eval_seed.py \
    --job v0=eval_fp16_v0_seed.py \
    --job v1=eval_fp16_v1_seed.py \
    --seed-start 1000 \
    --num-seeds 20 \
    -d cuda:0 \
    --baseline fp32
"""

import argparse
import datetime
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def parse_job(job_spec):
    """
    Parse NAME=SCRIPT.

    Example:
        fp32=eval_seed.py
        v1=eval_fp16_v1_seed.py
    """
    if "=" not in job_spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --job value: {job_spec!r}. Expected NAME=SCRIPT."
        )

    name, script = job_spec.split("=", 1)
    name = name.strip()
    script = script.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            f"Invalid --job value: {job_spec!r}. Job name is empty."
        )

    if not script:
        raise argparse.ArgumentTypeError(
            f"Invalid --job value: {job_spec!r}. Script path is empty."
        )

    return name, Path(script)


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def compute_stats(values):
    """
    Compute summary statistics for one scalar metric.

    values:
        List of (eval_seed, metric_value).
    """
    numeric_values = [float(value) for _, value in values]
    count = len(numeric_values)

    result = {
        "n": count,
        "mean": statistics.fmean(numeric_values),
        "std_sample": (
            statistics.stdev(numeric_values)
            if count >= 2
            else None
        ),
        "std_population": statistics.pstdev(numeric_values),
        "median": statistics.median(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
        "values_by_seed": {
            str(seed): value
            for seed, value in values
        },
    }

    return result


def read_log(log_path):
    with open(log_path, "r") as f:
        return json.load(f)


def build_method_summary(method_name, method_dir, expected_seeds):
    """
    Read all available eval_log.json files under:

        method_dir/seed_<seed>/eval_log.json

    and create aggregate statistics for all scalar fields.
    """
    runs = []
    missing_seeds = []
    invalid_logs = []
    metric_values = {}

    for eval_seed in expected_seeds:
        log_path = method_dir / f"seed_{eval_seed}" / "eval_log.json"

        if not log_path.exists():
            missing_seeds.append(eval_seed)
            continue

        try:
            log = read_log(log_path)
        except (OSError, json.JSONDecodeError) as exc:
            invalid_logs.append({
                "eval_seed": eval_seed,
                "log_path": str(log_path),
                "error": str(exc),
            })
            continue

        reported_seed = log.get("eval_seed")

        run_entry = {
            "requested_eval_seed": eval_seed,
            "reported_eval_seed": reported_seed,
            "log_path": str(log_path),
            "log": log,
        }
        runs.append(run_entry)

        for key, value in log.items():
            if key == "eval_seed":
                continue

            if is_finite_number(value):
                metric_values.setdefault(key, []).append(
                    (eval_seed, float(value))
                )

    metrics = {
        key: compute_stats(values)
        for key, values in sorted(metric_values.items())
    }

    return {
        "method": method_name,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "expected_seeds": expected_seeds,
        "completed_seed_count": len(runs),
        "missing_seeds": missing_seeds,
        "invalid_logs": invalid_logs,
        "metrics": metrics,
        "runs": runs,
    }


def write_json(path, payload):
    """
    Write through a temporary file so an interrupted process does not leave
    a partially-written summary JSON file.
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    temp_path.replace(path)


def update_all_method_summaries(jobs, output_root, seeds):
    summaries = {}

    for method_name, _ in jobs:
        method_dir = output_root / method_name

        summary = build_method_summary(
            method_name=method_name,
            method_dir=method_dir,
            expected_seeds=seeds,
        )

        summary_path = method_dir / "multi_seed_summary.json"
        write_json(summary_path, summary)

        summaries[method_name] = summary

    return summaries


def build_paired_comparison(baseline_name, summaries):
    """
    Compare every method to baseline on matching eval_seed values.

    delta = method_value - baseline_value
    """
    if baseline_name not in summaries:
        return {
            "error": (
                f"Baseline {baseline_name!r} is not in the completed jobs."
            )
        }

    baseline_metrics = summaries[baseline_name]["metrics"]
    comparisons = {}

    for method_name, summary in summaries.items():
        if method_name == baseline_name:
            continue

        method_metrics = summary["metrics"]
        metric_deltas = {}

        common_metric_names = sorted(
            set(baseline_metrics.keys()) & set(method_metrics.keys())
        )

        for metric_name in common_metric_names:
            baseline_values = baseline_metrics[metric_name]["values_by_seed"]
            method_values = method_metrics[metric_name]["values_by_seed"]

            common_seeds = sorted(
                set(baseline_values.keys()) & set(method_values.keys()),
                key=int,
            )

            deltas = []

            for seed_text in common_seeds:
                delta = (
                    float(method_values[seed_text])
                    - float(baseline_values[seed_text])
                )
                deltas.append((int(seed_text), delta))

            if deltas:
                metric_deltas[metric_name] = compute_stats(deltas)

        comparisons[method_name] = {
            "delta_definition": f"{method_name} - {baseline_name}",
            "metrics": metric_deltas,
        }

    return {
        "baseline": baseline_name,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more evaluation scripts repeatedly with paired "
            "evaluation RNG seeds, then aggregate eval_log.json files."
        )
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="Checkpoint passed to every evaluation script.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Root output directory. Each job writes to "
            "OUTPUT_ROOT/<job_name>/seed_<eval_seed>/."
        ),
    )
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        type=parse_job,
        metavar="NAME=SCRIPT",
        help=(
            "Evaluation job. Repeat this option for fp32, v0, v1, etc. "
            "Example: --job fp32=eval_seed.py"
        ),
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="Device passed to every evaluation script.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="First eval_seed when using --num-seeds. Default: 1000.",
    )

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of consecutive seeds to evaluate. Default: 10.",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit eval_seed list. Overrides --seed-start.",
    )

    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Method name used for paired differences. "
            "Default: first --job name."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument appended to every evaluation command. "
            "Repeat when needed."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining jobs after an evaluation command fails.",
    )

    args = parser.parse_args()

    jobs = args.job
    job_names = [name for name, _ in jobs]

    if len(set(job_names)) != len(job_names):
        parser.error("Every --job name must be unique.")

    for _, script_path in jobs:
        if not script_path.is_file():
            parser.error(f"Evaluation script does not exist: {script_path}")

    if args.seeds is not None:
        seeds = args.seeds
    else:
        if args.num_seeds <= 0:
            parser.error("--num-seeds must be greater than zero.")

        seeds = list(
            range(args.seed_start, args.seed_start + args.num_seeds)
        )

    if len(set(seeds)) != len(seeds):
        parser.error("Each eval_seed must be unique.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_name = args.baseline or job_names[0]

    if baseline_name not in job_names:
        parser.error(
            f"Baseline {baseline_name!r} is not one of: {job_names}"
        )

    experiment_config = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "seeds": seeds,
        "baseline": baseline_name,
        "jobs": [
            {
                "name": name,
                "script": str(script),
            }
            for name, script in jobs
        ],
        "extra_args": args.extra_arg,
    }
    write_json(
        output_root / "multi_seed_experiment_config.json",
        experiment_config,
    )

    for method_name, script_path in jobs:
        method_dir = output_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        for eval_seed in seeds:
            seed_dir = method_dir / f"seed_{eval_seed}"
            log_path = seed_dir / "eval_log.json"

            # Safe resume behavior:
            # a finished run is never overwritten or re-run automatically.
            if log_path.exists():
                print(
                    f"[SKIP] method={method_name}, eval_seed={eval_seed}: "
                    f"{log_path} already exists."
                )
                continue

            if seed_dir.exists():
                raise RuntimeError(
                    f"Output directory already exists but has no eval_log.json: "
                    f"{seed_dir}\n"
                    "Resolve or remove this incomplete directory manually "
                    "before rerunning."
                )

            command = [
                sys.executable,
                str(script_path),
                "-c",
                args.checkpoint,
                "-o",
                str(seed_dir),
                "-d",
                args.device,
                "--eval_seed",
                str(eval_seed),
                *args.extra_arg,
            ]

            print()
            print("=" * 80)
            print(f"Running method={method_name}, eval_seed={eval_seed}")
            print("Command:", " ".join(command))
            print("=" * 80)

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[FAILED] method={method_name}, "
                    f"eval_seed={eval_seed}, returncode={exc.returncode}"
                )

                # Even after an error, write summaries for completed work.
                summaries = update_all_method_summaries(
                    jobs=jobs,
                    output_root=output_root,
                    seeds=seeds,
                )
                comparison = build_paired_comparison(
                    baseline_name=baseline_name,
                    summaries=summaries,
                )
                write_json(
                    output_root
                    / f"paired_comparison_to_{baseline_name}.json",
                    comparison,
                )

                if not args.continue_on_error:
                    raise

            # Update summaries after every completed evaluation.
            summaries = update_all_method_summaries(
                jobs=jobs,
                output_root=output_root,
                seeds=seeds,
            )
            comparison = build_paired_comparison(
                baseline_name=baseline_name,
                summaries=summaries,
            )
            write_json(
                output_root / f"paired_comparison_to_{baseline_name}.json",
                comparison,
            )

    # Final aggregation, also covers all-skipped resume runs.
    summaries = update_all_method_summaries(
        jobs=jobs,
        output_root=output_root,
        seeds=seeds,
    )
    comparison = build_paired_comparison(
        baseline_name=baseline_name,
        summaries=summaries,
    )
    write_json(
        output_root / f"paired_comparison_to_{baseline_name}.json",
        comparison,
    )

    print()
    print("All requested evaluations have finished.")
    print(f"Output root: {output_root}")
    print(
        "Per-method summaries: "
        "OUTPUT_ROOT/<method>/multi_seed_summary.json"
    )
    print(
        "Paired comparison: "
        f"OUTPUT_ROOT/paired_comparison_to_{baseline_name}.json"
    )


if __name__ == "__main__":
    main()
