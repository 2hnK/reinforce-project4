"""
단일 실험 빠른 실행 스크립트
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_experiments.combination.combination_experiments import (
    get_combination_experiments,
    run_experiment,
)
import ray

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='Experiment name')
args = parser.parse_args()

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

experiments = get_combination_experiments()
target_exp = None

for exp in experiments:
    if exp['name'] == args.name:
        target_exp = exp
        break

if not target_exp:
    print(f"실험 '{args.name}'을 찾을 수 없습니다")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"실행: {target_exp['name']}")
print(f"{'='*80}")

result = run_experiment(target_exp, num_trials=1, num_iterations=10, save_checkpoints=True)

if result:
    print(f"\n✅ 완료! 최종 보상: {result['statistics']['final_reward_mean']:.2f}")
else:
    print(f"\n❌ 실패")

ray.shutdown()
