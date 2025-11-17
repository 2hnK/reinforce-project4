"""
나머지 4개 실험 (baseline_conservative, stable_conservative, balanced_high_momentum, kl_focused) 빠르게 실행
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_experiments.combination.combination_experiments import (
    get_combination_experiments,
    run_experiment,
)
import ray

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

experiments = get_combination_experiments()

# baseline_default와 aggressive_exploration은 이미 실행했으므로 제외
target_names = [
    'baseline_conservative',
    'stable_conservative', 
    'balanced_high_momentum',
    'kl_focused'
]

for exp in experiments:
    if exp['name'] in target_names:
        print(f"\n{'='*80}")
        print(f"실행 중: {exp['name']}")
        print(f"{'='*80}")
        
        # 1 trial, 10 iterations로 빠르게 실행
        result = run_experiment(exp, num_trials=1, num_iterations=10, save_checkpoints=True)
        
        if result:
            print(f"\n✅ {exp['name']} 완료!")
            print(f"   최종 보상: {result['statistics']['final_reward_mean']:.2f}")
        else:
            print(f"\n❌ {exp['name']} 실패")

print("\n" + "="*80)
print("모든 실험 완료!")
print("="*80)

ray.shutdown()
