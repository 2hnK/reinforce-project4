"""
20227128 김지훈

파라미터 조합 실험 - 신뢰성 있는 조합 테스트

단일 파라미터 실험 결과를 바탕으로 문헌 검증된 조합 실험 수행

참고 문헌:
    - PPO 원논문 (Schulman et al., 2017)
    - OpenAI Spinning Up
    - RLlib 공식 문서
    - CleanRL MuJoCo 벤치마크

환경 설정:
    - 기본 num_env_runners=10, 균형 실험은 16까지 확장
    - num_envs_per_env_runner=5 (RLlib 권장 4-8 범위)
    - 총 50~80개 환경 동시 실행 (PPO 논문 32-64 envs 권장 상단 대비 여유분 확보)
    - GPU 사용 (가능 시, 없으면 CPU 자동 폴백)
    - 근거: Schulman et al. 2017, CleanRL MuJoCo benchmark
    - 예상 SPS: 25,000-35,000
    - 예상 효율: 50-60%
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_experiments.resource_monitoring import (
    ResourceMonitor,
    aggregate_resource_stats,
)

try:
    import torch  # type: ignore
except ImportError:  # torch optional
    torch = None  # type: ignore

GPU_AVAILABLE = bool(torch) and torch.cuda.is_available()  # type: ignore
NUM_GPUS_PER_LEARNER = 1 if GPU_AVAILABLE else 0
RAY_RESULTS_DIR = PROJECT_ROOT / "ray_results"
RAY_RESULTS_DIR.mkdir(exist_ok=True)

# RLlib PPO 기본값을 한 번만 추출해둔다.
_PPO_DEFAULT = PPOConfig()
PPO_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    'clip_param': _PPO_DEFAULT.clip_param,
    'vf_clip_param': _PPO_DEFAULT.vf_clip_param,
    'entropy_coeff': _PPO_DEFAULT.entropy_coeff,
    'use_kl_loss': getattr(_PPO_DEFAULT, 'use_kl_loss', False),
    'kl_coeff': _PPO_DEFAULT.kl_coeff,
    'kl_target': _PPO_DEFAULT.kl_target,
    'grad_clip': _PPO_DEFAULT.grad_clip,
    'gamma': _PPO_DEFAULT.gamma,
    'use_gae': _PPO_DEFAULT.use_gae,
    'use_critic': _PPO_DEFAULT.use_critic,
}


def _optional_value(config: Dict[str, Any], key: str):
    if key in config:
        return config[key]
    return PPO_OPTIONAL_DEFAULTS[key]


def _load_overrides(overrides_arg: str | None) -> Dict[str, Any]:
    """JSON 문자열 또는 파일 경로로부터 override dict를 로드한다."""
    if not overrides_arg:
        return {}

    candidate = Path(overrides_arg)
    if candidate.exists():
        with open(candidate, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.loads(overrides_arg)

    if not isinstance(data, dict):
        raise ValueError("Baseline overrides는 dict 형태여야 합니다.")
    return data


def _parse_args():
    parser = argparse.ArgumentParser(description="Run PPO hyperparameter combination experiments")
    parser.add_argument(
        "--baseline-overrides",
        type=str,
        default=None,
        help="JSON string or path to JSON file with additional baseline parameters",
    )
    parser.add_argument(
        "--auto-yes",
        action="store_true",
        help="Skip interactive confirmation",
    )
    return parser.parse_args()


def get_baseline_config(extra_overrides: Dict[str, Any] | None = None):
    """베이스라인 설정 (고정 파라미터)

    extra_overrides를 통해 원하는 파라미터만 간단히 추가할 수 있다.
    """
    config = {
        'lambda_': 0.95,
        'lr': 0.0003,
        'num_epochs': 15,
        'train_batch_size': 32 * 512,  # 16384
        'minibatch_size': 4096,
        'vf_loss_coeff': 0.01,
        'fcnet_hiddens': [64, 64],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,
    }

    if extra_overrides:
        config.update(extra_overrides)

    return config


def get_combination_experiments(baseline_overrides: Dict[str, Any] | None = None):
    """요청된 5개 실험 환경 세트"""
    baseline = get_baseline_config(baseline_overrides)

    def with_common_overrides(**overrides):
        cfg = baseline.copy()
        cfg.update(overrides)
        return cfg

    return [
        {
            'name': 'baseline_default',
            'description': 'Exact baseline hyperparameters (no overrides)',
            'category': 'baseline',
            'rationale': 'Pure reference run using untouched defaults',
            'expected': 'Matches single-parameter baseline performance',
            'params': baseline.copy(),
        },
        {
            'name': 'baseline_conservative',
            'description': 'Baseline (conservative defaults)',
            'category': 'baseline',
            'rationale': 'Reference configuration for comparisons',
            'expected': 'Stable but slower convergence',
            'params': with_common_overrides(
                clip_param=0.2,
                gamma=0.99,
                kl_target=0.01,
                optimizer_config={'sgd_momentum': 0.0},
            )
        },
        {
            'name': 'aggressive_exploration',
            'description': 'Aggressive exploration with scheduled entropy/clip',
            'category': 'speed',
            'rationale': 'Fast initial learning via high clip & entropy',
            'expected': 'Fast early gains, late instability risk',
            'params': with_common_overrides(
                clip_param=0.3,
                gamma=0.90,
                kl_target=0.2,
                entropy_coeff=0.01,
                optimizer_config={'sgd_momentum': 0.9},
            )
        },
        {
            'name': 'stable_conservative',
            'description': 'Stable conservative setting with weight decay',
            'category': 'stability',
            'rationale': 'Maximum stability focus',
            'expected': 'Slowest but most stable convergence',
            'params': with_common_overrides(
                clip_param=0.1,
                gamma=0.99,
                kl_target=0.01,
                entropy_coeff=0.0,
                optimizer_config={'sgd_momentum': 0.99, 'weight_decay': 1e-4},
            )
        },
        {
            'name': 'balanced_high_momentum',
            'description': 'Balanced config with high momentum and more workers',
            'category': 'balanced',
            'rationale': 'Practical balance of speed and stability',
            'expected': 'Fast convergence while remaining stable',
            'params': with_common_overrides(
                clip_param=0.25,
                gamma=0.95,
                kl_target=0.02,
                entropy_coeff=0.001,
                optimizer_config={'sgd_momentum': 0.95},
            )
        },
        {
            'name': 'kl_focused',
            'description': 'KL-focused training with stronger penalties',
            'category': 'stability',
            'rationale': 'Tight KL and VF regularisation',
            'expected': 'Most accurate value estimates',
            'params': with_common_overrides(
                clip_param=0.25,
                gamma=0.95,
                kl_target=0.1,
                kl_coeff=0.1,
                entropy_coeff=0.001,
                optimizer_config={'sgd_momentum': 0.9},
                vf_loss_coeff=0.5,
            )
        },
    ]


DEFAULT_RUNNER_CONFIG = {
    'num_env_runners': 10,
    'num_envs_per_env_runner': 5,
    'num_cpus_per_env_runner': 1,
}


def _scheduled_value(schedule, current_step):
    if not schedule:
        return None
    value = schedule[0][1]
    for step, val in schedule:
        if current_step >= step:
            value = val
        else:
            break
    return value


def _foreach_policy(algo, fn: Callable):
    if hasattr(algo, 'env_runner_group') and algo.env_runner_group is not None:
        def _wrapper(policy, policy_id):
            fn(policy)
        algo.env_runner_group.foreach_policy(_wrapper)
        return

    workers = getattr(algo, 'workers', None)
    if workers is not None and hasattr(workers, 'foreach_policy'):
        workers.foreach_policy(lambda policy, *_: fn(policy))
        return

    default_policy = algo.get_policy() if hasattr(algo, 'get_policy') else None
    if default_policy is not None:
        fn(default_policy)


def _apply_clip_param(algo, new_value):
    if new_value is None:
        return
    algo.config['clip_param'] = new_value

    def _set(policy):
        policy.config['clip_param'] = new_value

    _foreach_policy(algo, _set)


def _build_model_config(config_dict):
    base_model = {
        "fcnet_hiddens": config_dict['fcnet_hiddens'],
        "fcnet_activation": config_dict['fcnet_activation'],
        "vf_share_layers": config_dict['vf_share_layers'],
    }

    initial_log_std = config_dict.get('initial_log_std')
    model_overrides = config_dict.get('model_overrides') or {}
    overrides_copy = dict(model_overrides)

    # Legacy format: {'action_dist_config': {'initial_log_std': x}}
    action_dist_cfg = overrides_copy.pop('action_dist_config', None)
    if isinstance(action_dist_cfg, dict) and 'initial_log_std' in action_dist_cfg:
        initial_log_std = action_dist_cfg['initial_log_std']

    # Allow direct usage via overrides as well.
    if 'initial_log_std' in overrides_copy:
        initial_log_std = overrides_copy.pop('initial_log_std')

    base_model.update(overrides_copy)
    return base_model, initial_log_std


def _apply_initial_log_std(algo, log_std_value):
    if log_std_value is None or torch is None:  # type: ignore
        return

    log_std_value = float(log_std_value)

    def _setter(policy):
        action_space = getattr(policy, 'action_space', None)
        if not action_space or not getattr(action_space, 'shape', None):
            return
        action_dim = int(np.prod(action_space.shape))
        model = policy.model

        # free_log_std uses dedicated parameter tensor.
        if getattr(model, 'free_log_std', False) and hasattr(model, '_append_free_log_std'):
            append_layer = getattr(model, '_append_free_log_std')
            log_std_tensor = getattr(append_layer, 'log_std', None)
            if log_std_tensor is None:
                return
            with torch.no_grad():  # type: ignore
                log_std_tensor.fill_(log_std_value)
            return

        logits = getattr(model, '_logits', None)
        bias = getattr(logits, 'bias', None) if logits else None
        if bias is None or bias.shape[0] < action_dim * 2:
            return

        with torch.no_grad():  # type: ignore
            bias[action_dim:action_dim * 2] = log_std_value

    _foreach_policy(algo, _setter)
    print(f"    ↺ initial log_std set to {log_std_value:.3f}")


def _resolve_checkpoint_path(raw_checkpoint):
    checkpoint_obj = getattr(raw_checkpoint, "checkpoint", None)
    if checkpoint_obj:
        path_attr = getattr(checkpoint_obj, "path", None)
        if path_attr:
            return Path(path_attr)

    if isinstance(raw_checkpoint, Path):
        return raw_checkpoint
    if isinstance(raw_checkpoint, str):
        return Path(raw_checkpoint)

    path_attr = getattr(raw_checkpoint, "path", None)
    if path_attr:
        return Path(path_attr)

    to_uri = getattr(raw_checkpoint, "to_uri", None)
    if callable(to_uri):
        uri = to_uri()
        if isinstance(uri, str) and uri.startswith("file://"):
            return Path(uri[7:])
    return None


def print_system_info():
    """시스템 환경 정보 출력"""
    import platform
    import psutil
    
    print("="*80)
    print("파라미터 조합 실험 - 시스템 환경 정보")
    print("="*80)
    
    # 기본 정보
    print(f"\n[시스템 정보]")
    print(f"  운영체제: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Ray: {ray.__version__}")
    
    # CPU 정보
    print(f"\n[CPU 정보]")
    print(f"  프로세서: {platform.processor()}")
    print(f"  물리 코어: {psutil.cpu_count(logical=False)}개")
    print(f"  논리 코어: {psutil.cpu_count(logical=True)}개")
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"\n[메모리 정보]")
    print(f"  총 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"  사용 가능: {memory.available / (1024**3):.1f}GB")
    
    # GPU 정보
    print(f"\n[GPU 정보]")
    if GPU_AVAILABLE:
        print(f"  CUDA 사용 가능: Yes")
        print(f"  CUDA 버전: {torch.version.cuda}")  # type: ignore[attr-defined]
        print(f"  GPU 개수: {torch.cuda.device_count()}개")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    메모리: {props.total_memory / (1024**3):.1f}GB")
    elif torch is not None:
        print(f"  CUDA 사용 가능: No (PyTorch 설치됨)")
    else:
        print(f"  CUDA 사용 가능: PyTorch 미설치")
    
    # 병렬화 설정
    print(f"\n[병렬화 기본 설정]")
    print(f"  num_env_runners: {DEFAULT_RUNNER_CONFIG['num_env_runners']}")
    print(f"  num_envs_per_env_runner: {DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']}")
    total_envs = DEFAULT_RUNNER_CONFIG['num_env_runners'] * DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']
    print(f"  총 환경 수: {total_envs}")
    learner_desc = "1 (GPU 사용)" if GPU_AVAILABLE else "1 (CPU 모드)"
    print(f"  num_learners: {learner_desc}")
    if not GPU_AVAILABLE:
        print("  ⚠️ GPU 미탐지 → CPU 학습 (느릴 수 있음)")
    print(f"  근거: CleanRL 벤치마크 + PPO 논문 (32-64 envs 권장, 기본 10 runners)")
    print(f"  예상 SPS: 25,000-35,000")
    print(f"  예상 효율: 50-60%")
    
    print("="*80)


def run_single_trial(config_dict, exp_name, trial_num, num_iterations=10, save_checkpoint=False):
    """단일 시행 실행
    
    Args:
        config_dict: 설정 딕셔너리
        exp_name: 실험 이름
        trial_num: 시행 번호
        num_iterations: 반복 횟수
        save_checkpoint: 체크포인트 저장 여부
    """
    runner_cfg = DEFAULT_RUNNER_CONFIG.copy()

    clip_schedule = config_dict.get('clip_param_schedule')
    initial_clip = _scheduled_value(clip_schedule, 0) or _optional_value(config_dict, 'clip_param')

    model_config, initial_log_std = _build_model_config(config_dict)

    optimizer_config = config_dict.get('optimizer_config')

    training_kwargs = dict(
        lambda_=config_dict['lambda_'],
        lr=config_dict['lr'],
        num_epochs=config_dict['num_epochs'],
        train_batch_size=config_dict['train_batch_size'],
        minibatch_size=config_dict['minibatch_size'],
        vf_loss_coeff=config_dict['vf_loss_coeff'],
        clip_param=initial_clip,
        vf_clip_param=_optional_value(config_dict, 'vf_clip_param'),
        entropy_coeff=_optional_value(config_dict, 'entropy_coeff'),
        use_kl_loss=_optional_value(config_dict, 'use_kl_loss'),
        kl_coeff=_optional_value(config_dict, 'kl_coeff'),
        kl_target=_optional_value(config_dict, 'kl_target'),
        grad_clip=_optional_value(config_dict, 'grad_clip'),
        gamma=_optional_value(config_dict, 'gamma'),
        use_gae=_optional_value(config_dict, 'use_gae'),
        use_critic=_optional_value(config_dict, 'use_critic'),
        model=model_config,
    )

    if config_dict.get('entropy_coeff_schedule'):
        training_kwargs['entropy_coeff_schedule'] = config_dict['entropy_coeff_schedule']

    config = (
        PPOConfig()
        .environment("HalfCheetah-v5")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    if optimizer_config:
        config = config.training(**training_kwargs, optimizer=optimizer_config)
    else:
        config = config.training(**training_kwargs)

    config = (
        config
        .learners(num_learners=1, num_gpus_per_learner=NUM_GPUS_PER_LEARNER)
        .debugging(
            seed=20227128 + trial_num,
            log_level="WARN",
            log_sys_usage=True,
        )
        .env_runners(
            num_env_runners=runner_cfg['num_env_runners'],
            num_envs_per_env_runner=runner_cfg['num_envs_per_env_runner'],
            num_cpus_per_env_runner=runner_cfg['num_cpus_per_env_runner'],
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=0,
            evaluation_duration=5
        )
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_logdir = RAY_RESULTS_DIR / f"{exp_name}_trial{trial_num + 1}_{timestamp}"
    trial_logdir.mkdir(parents=True, exist_ok=True)

    def _logger_creator(cfg):
        from ray.tune.logger import UnifiedLogger
        return UnifiedLogger(cfg, str(trial_logdir))

    # 알고리즘 빌드
    algo = config.build(logger_creator=_logger_creator)
    
    results = []
    start_time = time.time()
    checkpoint_path = None

    _apply_clip_param(algo, initial_clip)
    _apply_initial_log_std(algo, initial_log_std)
    total_env_steps = 0
    monitor_samples = []
    
    try:
        for iteration in range(num_iterations):
            scheduled_clip = _scheduled_value(clip_schedule, total_env_steps)
            current_clip = algo.config.get('clip_param', initial_clip)
            if scheduled_clip is not None and abs(scheduled_clip - current_clip) > 1e-6:
                _apply_clip_param(algo, scheduled_clip)
                print(f"    ↺ clip_param schedule applied: {scheduled_clip:.3f} (steps={total_env_steps})")
            
            monitor = ResourceMonitor()
            monitor.start()
            iter_start = time.time()
            result = algo.train()
            iter_time = time.time() - iter_start
            monitor.stop()
            monitor_stats = monitor.get_stats()
            if monitor_stats:
                monitor_samples.append(monitor_stats)
            
            # 메트릭 추출
            env_runners = result.get('env_runners', {})
            episode_reward_mean = env_runners.get('episode_reward_mean', 
                                                  env_runners.get('episode_return_mean', 0))
            
            # 학습 단계 추출 (중요!)
            def _first_positive(*values):
                for val in values:
                    if isinstance(val, (int, float)) and val > 0:
                        return float(val)
                return 0.0

            num_env_steps_trained = _first_positive(
                result.get('num_env_steps_trained'),
                result.get('num_env_steps_trained_this_iter'),
                result.get('num_env_steps_trained_total'),
                result.get('counters', {}).get('num_env_steps_trained'),
                env_runners.get('num_env_steps_trained'),
                env_runners.get('num_env_steps_sampled'),
                result.get('num_env_steps_sampled')
            )
            
            # 첫 iteration에서 학습 검증
            if iteration == 0:
                if num_env_steps_trained == 0:
                    raise RuntimeError(
                        f"❌ 학습이 시작되지 않았습니다! "
                        f"num_env_steps_trained=0\n"
                        f"learner 설정을 확인하세요 (num_learners >= 1 필요)"
                    )
                else:
                    print(f"    ✓ 학습 시작 확인: {num_env_steps_trained} steps trained")
            
            # SPS 계산
            num_env_steps = 0
            for key in ['num_env_steps_sampled', 'num_env_steps_sampled_this_iter']:
                if key in env_runners:
                    num_env_steps = env_runners[key]
                    break
            if num_env_steps == 0:
                num_env_steps = config_dict['train_batch_size']
            
            sps = num_env_steps / iter_time if iter_time > 0 else 0
            
            metrics = {
                'iteration': iteration + 1,
                'episode_reward_mean': float(episode_reward_mean),
                'episode_reward_min': float(env_runners.get('episode_reward_min', 0)),
                'episode_reward_max': float(env_runners.get('episode_reward_max', 0)),
                'episode_len_mean': float(env_runners.get('episode_len_mean', 0)),
                'num_env_steps_sampled': int(num_env_steps),
                'num_env_steps_trained': int(num_env_steps_trained),
                'time_this_iter_s': float(iter_time),
                'sps': float(sps),
                'resource_monitor': monitor_stats,
            }
            
            results.append(metrics)
            total_env_steps = max(total_env_steps, int(result.get('num_env_steps_sampled', total_env_steps)))
            
            print(f"    Iter {iteration + 1}/{num_iterations}: "
                  f"Reward={metrics['episode_reward_mean']:.2f}, "
                  f"Trained={num_env_steps_trained}, "
                  f"Time={iter_time:.2f}s, "
                  f"SPS={sps:.0f}")
            if monitor_stats:
                cpu_avg = monitor_stats.get('cpu_avg')
                cpu_max = monitor_stats.get('cpu_max')
                ram_avg = monitor_stats.get('ram_avg')
                gpu_avg = monitor_stats.get('gpu_avg')
                vram_avg = monitor_stats.get('vram_avg_mb')
                monitor_msg = "      Resource ⇨ "
                if cpu_avg is not None:
                    monitor_msg += f"CPU {cpu_avg:.1f}% avg / {cpu_max:.1f}% max"
                if ram_avg is not None:
                    monitor_msg += f", RAM {ram_avg:.1f}%"
                if gpu_avg is not None:
                    monitor_msg += f", GPU {gpu_avg:.1f}%"
                if vram_avg is not None:
                    monitor_msg += f", VRAM {vram_avg:.0f}MB"
                print(monitor_msg)
    
        
        # 마지막 iteration 체크포인트 저장 (옵션)
        if save_checkpoint and iteration == num_iterations - 1:
            raw_checkpoint = algo.save()
            checkpoint_src = _resolve_checkpoint_path(raw_checkpoint)
            repo_checkpoint_dir = trial_logdir / f"checkpoint_iter{iteration + 1}"
            if checkpoint_src and checkpoint_src.exists():
                if repo_checkpoint_dir.exists():
                    shutil.rmtree(repo_checkpoint_dir)
                shutil.copytree(checkpoint_src, repo_checkpoint_dir)
                checkpoint_path = str(repo_checkpoint_dir)
            else:
                checkpoint_path = str(checkpoint_src or raw_checkpoint)
            print(f"    💾 체크포인트 저장: {checkpoint_path}")
    
    finally:
        algo.stop()
    
    total_time = time.time() - start_time
    resource_summary = aggregate_resource_stats(monitor_samples)
    
    return {
        'trial_num': trial_num,
        'iterations': results,
        'total_time': float(total_time),
        'final_reward': float(results[-1]['episode_reward_mean']) if results else 0.0,
        'checkpoint_path': checkpoint_path,
        'resource_summary': resource_summary,
    }


def run_experiment(exp_config, num_trials=5, num_iterations=10, save_checkpoints=False):
    """단일 실험 실행 (여러 시행)
    
    Args:
        exp_config: 실험 설정
        num_trials: 시행 횟수
        num_iterations: 반복 횟수
        save_checkpoints: 체크포인트 저장 여부 (마지막 trial만)
    """
    exp_name = exp_config['name']
    print(f"\n{'='*80}")
    print(f"실험: {exp_name}")
    print(f"설명: {exp_config['description']}")
    print(f"카테고리: {exp_config['category']}")
    print(f"근거: {exp_config['rationale']}")
    if 'expected' in exp_config:
        print(f"예상 결과: {exp_config['expected']}")
    print(f"{'='*80}")
    
    trials_results = []
    for trial in range(num_trials):
        print(f"\n  Trial {trial + 1}/{num_trials}")
        print(f"  {'='*50}")
        
        try:
            # 마지막 trial만 체크포인트 저장
            save_this_trial = save_checkpoints and (trial == num_trials - 1)
            trial_result = run_single_trial(
                exp_config['params'],
                exp_name,
                trial,
                num_iterations,
                save_checkpoint=save_this_trial
            )
            trials_results.append(trial_result)
            
        except Exception as e:
            print(f"  [ERROR] Trial {trial + 1} failed: {str(e)}")
            continue
    
    if not trials_results:
        print(f"  [WARNING] No successful trials for {exp_name}")
        return None
    
    # 통계 계산
    final_rewards = [t['final_reward'] for t in trials_results]
    all_iterations = [t['iterations'] for t in trials_results]
    
    # 각 iteration별 평균 계산
    mean_rewards_per_iter = []
    if all_iterations:
        num_iters = len(all_iterations[0])
        for i in range(num_iters):
            iter_rewards = [trial[i]['episode_reward_mean'] for trial in all_iterations if len(trial) > i]
            if iter_rewards:
                mean_rewards_per_iter.append(np.mean(iter_rewards))
    
    # SPS 통계
    all_sps = []
    for trial in trials_results:
        for iteration in trial['iterations']:
            if 'sps' in iteration and iteration['sps'] > 0:
                all_sps.append(iteration['sps'])
    
    statistics = {
        'final_reward_mean': float(np.mean(final_rewards)),
        'final_reward_std': float(np.std(final_rewards)),
        'final_reward_min': float(np.min(final_rewards)),
        'final_reward_max': float(np.max(final_rewards)),
        'final_reward_cv': float(np.std(final_rewards) / abs(np.mean(final_rewards))) if np.mean(final_rewards) != 0 else 0.0,
        'mean_rewards_per_iter': [float(x) for x in mean_rewards_per_iter],
        'sps_mean': float(np.mean(all_sps)) if all_sps else 0.0,
        'sps_std': float(np.std(all_sps)) if all_sps else 0.0,
    }

    resource_samples = [
        iteration.get('resource_monitor')
        for trial in trials_results
        for iteration in trial['iterations']
        if iteration.get('resource_monitor')
    ]
    trial_resource_summaries = [trial.get('resource_summary') for trial in trials_results if trial.get('resource_summary')]
    resource_usage = aggregate_resource_stats(resource_samples or trial_resource_summaries)
    if resource_usage:
        statistics['resource_usage'] = resource_usage
    
    print(f"\n  {'='*50}")
    print(f"  최종 통계:")
    print(f"    평균 보상: {statistics['final_reward_mean']:.2f} ± {statistics['final_reward_std']:.2f}")
    print(f"    변동계수(CV): {statistics['final_reward_cv']:.4f}")
    print(f"    범위: [{statistics['final_reward_min']:.2f}, {statistics['final_reward_max']:.2f}]")
    print(f"    평균 SPS: {statistics['sps_mean']:.0f} ± {statistics['sps_std']:.0f}")
    if 'resource_usage' in statistics:
        ru = statistics['resource_usage']
        cpu_msg = f"CPU {ru.get('cpu_avg', 0):.1f}% avg"
        if 'cpu_max' in ru:
            cpu_msg += f" / {ru['cpu_max']:.1f}% max"
        print(f"    리소스: {cpu_msg}")
        if 'gpu_avg' in ru:
            gpu_msg = f"GPU {ru['gpu_avg']:.1f}% avg"
            if 'gpu_max' in ru:
                gpu_msg += f" / {ru['gpu_max']:.1f}% max"
            print(f"            {gpu_msg}, VRAM {ru.get('vram_avg_mb', 0):.0f}MB avg")
    
    return {
        'name': exp_name,
        'description': exp_config['description'],
        'category': exp_config['category'],
        'rationale': exp_config['rationale'],
        'expected': exp_config.get('expected', ''),
        'params': exp_config['params'],
        'trials': trials_results,
        'statistics': statistics,
        'resource_summary': resource_usage,
    }


def main(cli_args=None):
    """메인 실행 함수"""
    if cli_args is None:
        cli_args = _parse_args()

    try:
        baseline_overrides = _load_overrides(getattr(cli_args, 'baseline_overrides', None))
    except Exception as exc:  # pragma: no cover - CLI 유효성 검사용
        print(f"[ERROR] baseline overrides 파싱 실패: {exc}")
        return

    print("\n" + "="*80)
    print("파라미터 조합 실험 시작")
    print("20227128 김지훈")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 시스템 정보 출력
    print_system_info()
    
    # 실험 설정 로드
    experiments = get_combination_experiments(baseline_overrides)
    
    print(f"\n총 {len(experiments)}개의 조합 실험 예정")
    print(f"각 실험당 5회 시행, 시행당 10회 반복")
    total_envs = DEFAULT_RUNNER_CONFIG['num_env_runners'] * DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']
    print(f"병렬화: 기본 {DEFAULT_RUNNER_CONFIG['num_env_runners']} runners × {DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']} envs = {total_envs}개 환경 (균형 실험은 16 runners)")
    learner_desc = "num_learners=1 (GPU 사용)" if GPU_AVAILABLE else "num_learners=1 (CPU 모드)"
    print(f"학습: {learner_desc}")
    print(f"근거: PPO 논문 32-64 envs, CleanRL 8 workers → 기본 10 runners로 확장")
    print(f"예상 소요 시간: 약 20-25분\n")
    
    # 사용자 확인
    if getattr(cli_args, 'auto_yes', False):
        response = 'yes'
        print("실험을 시작하시겠습니까? (yes/no): yes (auto)")
    else:
        response = input("실험을 시작하시겠습니까? (yes/no): ")
    if response.lower() != 'yes':
        print("실험을 취소합니다.")
        return
    
    # Ray 초기화
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 결과 저장 디렉토리
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 실험 실행
    all_results = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'num_experiments': len(experiments),
            'num_trials_per_experiment': 5,
            'num_iterations_per_trial': 10,
            'parallelization': {
                'default_num_env_runners': DEFAULT_RUNNER_CONFIG['num_env_runners'],
                'num_envs_per_env_runner': DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner'],
                'total_envs_default': total_envs,
                'special_cases': {},
                'num_learners': 1,
                'num_gpus_per_learner': NUM_GPUS_PER_LEARNER,
                'rationale': 'PPO paper: 32-64 parallel envs, CleanRL: 8 workers (default expanded to 10 runners)'
            }
        },
        'experiments': []
    }
    
    start_time = time.time()
    
    for i, exp_config in enumerate(experiments):
        print(f"\n{'#'*80}")
        print(f"진행 상황: 실험 {i + 1}/{len(experiments)}")
        print(f"{'#'*80}")
        
        # baseline 실험만 체크포인트 저장
        save_ckpt = (exp_config['name'] == 'baseline_conservative')
        exp_result = run_experiment(exp_config, num_trials=5, num_iterations=10, 
                                   save_checkpoints=save_ckpt)
        
        if exp_result:
            all_results['experiments'].append(exp_result)
        
        # 중간 저장
        with open(results_dir / "combination_experiments_progress.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[진행 저장: {results_dir}/combination_experiments_progress.json]")
    
    total_time = time.time() - start_time
    all_results['metadata']['end_time'] = datetime.now().isoformat()
    all_results['metadata']['total_time_seconds'] = float(total_time)
    all_results['metadata']['total_time_minutes'] = float(total_time / 60)
    
    # 최종 저장
    with open(results_dir / "combination_experiments_final.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("모든 실험 완료!")
    print(f"총 소요 시간: {total_time / 60:.1f}분 ({total_time / 3600:.2f}시간)")
    print(f"결과 저장 위치: {results_dir}")
    print(f"  - 최종 결과: combination_experiments_final.json")
    print(f"  - 진행 기록: combination_experiments_progress.json")
    print("="*80)

    if not all_results['experiments']:
        print("\n[ERROR] 단 한 개의 trial도 성공적으로 완료되지 않았습니다.")
        print("       위에 표시된 오류 메시지를 먼저 해결하세요 (예: MuJoCo 미설치, GPU 리소스 부족 등).")
        print("       문제가 지속되면 --cpu-only 환경을 확인하거나 requirements를 재설치하십시오.")
        ray.shutdown()
        return
    
    # 간단한 요약
    print("\n실험 요약:")
    print(f"{'='*80}")
    print(f"{'실험명':<25} {'최종 보상':<20} {'CV':<10} {'SPS':<10}")
    print(f"{'-'*80}")
    
    for exp in all_results['experiments']:
        stats = exp['statistics']
        print(f"{exp['name']:<25} "
            f"{stats['final_reward_mean']:>7.2f} ± {stats['final_reward_std']:<7.2f} "
            f"{stats['final_reward_cv']:>8.4f} "
            f"{stats['sps_mean']:>9.0f}")
    
    print(f"{'='*80}")
    
    # Ray 종료
    ray.shutdown()


if __name__ == "__main__":
    main(_parse_args())
