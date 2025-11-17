"""Utility script to retrain a single PPO config from the combination set and save a checkpoint."""

import argparse
import sys
from pathlib import Path

import ray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_experiments.combination.combination_experiments import (
    get_combination_experiments,
    run_experiment,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train a specific combination-experiment config and save its checkpoint."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="aggressive_exploration",
        help="Name of the experiment in get_combination_experiments() (default: aggressive_exploration)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of PPO iterations per trial (default: 10)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials to run (default: 1)",
    )
    parser.add_argument(
        "--disable-ray-init",
        action="store_true",
        help="Skip automatic ray.init() (useful if the caller already initialized Ray)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    experiments = get_combination_experiments()
    target = next((exp for exp in experiments if exp["name"] == args.experiment), None)
    if target is None:
        raise SystemExit(f"{args.experiment} config not found in combination experiments.")

    ray_was_initialized = ray.is_initialized()
    if not ray_was_initialized and not args.disable_ray_init:
        ray.init(ignore_reinit_error=True)

    try:
        result = run_experiment(
            target,
            num_trials=args.trials,
            num_iterations=args.iterations,
            save_checkpoints=True,
        )
    finally:
        if not ray_was_initialized and not args.disable_ray_init:
            ray.shutdown()

    if not result:
        print(f"[WARN] No successful trials were completed for {args.experiment}.")
        return

    checkpoints = [trial.get("checkpoint_path") for trial in result["trials"] if trial.get("checkpoint_path")]
    if not checkpoints:
        print("[WARN] Training finished but no checkpoints were produced. Enable save_checkpoints to debug.")
        return

    print(f"\n=== {args.experiment} Checkpoints ===")
    for idx, ckpt in enumerate(checkpoints, 1):
        print(f"Trial {idx}: {ckpt}")
    print(f"Latest checkpoint: {checkpoints[-1]}")


if __name__ == "__main__":
    main()
