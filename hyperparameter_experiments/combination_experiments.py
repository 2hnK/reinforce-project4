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
    - GPU 사용 (학습 가속화)
    - 근거: Schulman et al. 2017, CleanRL MuJoCo benchmark
    - 예상 SPS: 25,000-35,000
    - 예상 효율: 50-60%
"""

import json
import time
from datetime import datetime
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np


def get_baseline_config():
    """베이스라인 설정 (고정 파라미터)"""
    return {
        # 고정 학습 파라미터 (변경 불가)
        'lambda_': 0.95,
        'lr': 0.0003,
        'num_epochs': 15,
        'train_batch_size': 32 * 512,  # 16384
        'minibatch_size': 4096,
        'vf_loss_coeff': 0.01,
        'fcnet_hiddens': [64, 64],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,
        
        # 실험 대상 파라미터 (기본값)
        'clip_param': 0.2,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.0,
        'use_kl_loss': True,
        'kl_coeff': 0.2,
        'kl_target': 0.01,
        'grad_clip': None,
        'gamma': 0.99,
        'use_gae': True,
        'use_critic': True
    }


def get_combination_experiments():
    """5개 실험 환경 세트"""
    baseline = get_baseline_config()

    def with_common_overrides(**overrides):
        cfg = baseline.copy()
        cfg.update(overrides)
        return cfg

    return [
        {
            'name': 'baseline_conservative',
            'description': 'Baseline (conservative defaults)',
            'category': 'baseline',
            'rationale': 'More conservative than baseline_default for variance reduction',
            'expected': 'Similar or slightly lower reward with lower CV',
            'params': with_common_overrides(
                clip_param=0.12,
                gamma=0.99,
                kl_target=0.005,
                kl_coeff=0.3,
                entropy_coeff=0.0,
                optimizer_config={'sgd_momentum': 0.0},
                model_overrides={'action_dist_config': {'initial_log_std': -1.0}},
            ),
            'runner_overrides': {'num_env_runners': 16}
        },
        {
            'name': 'aggressive_exploration',
            'description': 'Aggressive exploration with scheduled entropy/clip',
            'category': 'speed',
            'rationale': 'Fast initial learning via high clip & entropy',
            'expected': 'Fast early gains, late instability risk',
            'params': with_common_overrides(
                clip_param=0.3,
                gamma=0.95,
                kl_target=0.05,
                entropy_coeff=0.01,
            ),
            'runner_overrides': {'num_env_runners': 16}
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
                optimizer_config={'sgd_momentum': 0.99},
                model_overrides={'action_dist_config': {'initial_log_std': -1.2}}
            ),
            'runner_overrides': {'num_env_runners': 16}
        },
        {
            'name': 'balanced_high_momentum',
            'description': 'Balanced config with high momentum and more workers',
            'category': 'balanced',
            'rationale': 'Practical balance of speed and stability',
            'expected': 'Fast convergence while remaining stable',
            'params': with_common_overrides(
                clip_param=0.25,
                gamma=0.99,
                kl_target=0.02,
                entropy_coeff=0.0,
                entropy_coeff_schedule=[
                    [0,    0.01],
                    [200_000, 0.005],
                    [500_000, 0.0],
                ],
                optimizer_config={'sgd_momentum': 0.9},
                model_overrides={'action_dist_config': {'initial_log_std': -0.7}}
            ),
            'runner_overrides': {'num_env_runners': 16}
        },
        {
            'name': 'kl_focused',
            'description': 'KL-focused training with stronger penalties',
            'category': 'stability',
            'rationale': 'Tight KL and VF regularisation',
            'expected': 'Most accurate value estimates',
            'params': with_common_overrides(
                clip_param=0.25,
                gamma=0.99,
                kl_target=0.01,
                kl_coeff=0.2,
                entropy_coeff=0.003,
                optimizer_config={'sgd_momentum': 0.9},
                vf_loss_coeff=0.5,
                model_overrides={'action_dist_config': {'initial_log_std': -0.8}}
            ),
            'runner_overrides': {'num_env_runners': 16}
        },
        {
            'name': 'performance_tuned_v1',
            'description': 'PPO config tuned for HalfCheetah performance',
            'category': 'performance',
            'rationale': 'Based on PPO paper, CleanRL, SB3 MuJoCo defaults',
            'expected': 'Higher final return than baseline_default',
            'params': with_common_overrides(
                clip_param=0.27,
                gamma=0.99,
                entropy_coeff=0.0,
                entropy_coeff_schedule=[
                    [0,      0.02],
                    [200_000, 0.01],
                    [500_000, 0.0],
                ],
                use_kl_loss=True,
                kl_target=0.02,
                kl_coeff=0.15,
                optimizer_config={'sgd_momentum': 0.9},
                model_overrides={'action_dist_config': {'initial_log_std': -0.7}},
            ),
            'runner_overrides': {'num_env_runners': 16}
        },
    ]


DEFAULT_RUNNER_CONFIG = {
    'num_env_runners': 16,
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


def _apply_clip_param(algo, new_value):
    if new_value is None:
        return
    algo.config['clip_param'] = new_value

    def _set(policy, *_):
        policy.config['clip_param'] = new_value

    algo.workers.foreach_policy(_set)


def print_system_info():
    """시스템 환경 정보 출력"""
    import platform
    import psutil

    try:
        import torch  # type: ignore
        torch_available = True
    except ImportError:
        torch_available = False
        torch = None  # type: ignore
    
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
    if torch_available and torch.cuda.is_available():
        print(f"  CUDA 사용 가능: Yes")
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 개수: {torch.cuda.device_count()}개")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    메모리: {props.total_memory / (1024**3):.1f}GB")
    elif torch_available:
        print(f"  CUDA 사용 가능: No (PyTorch 설치됨)")
    else:
        print(f"  CUDA 사용 가능: PyTorch 미설치")
    
    # 병렬화 설정
    print(f"\n[병렬화 기본 설정]")
    print(f"  num_env_runners: {DEFAULT_RUNNER_CONFIG['num_env_runners']}")
    print(f"  num_envs_per_env_runner: {DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']}")
    total_envs = DEFAULT_RUNNER_CONFIG['num_env_runners'] * DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']
    print(f"  총 환경 수: {total_envs}")
    print(f"  num_learners: 1 (GPU 사용)")
        print(f"  근거: CleanRL 벤치마크 + PPO 논문 (32-64 envs 권장, 기본 10 runners)")
    print(f"  예상 SPS: 25,000-35,000")
    print(f"  예상 효율: 50-60%")
    
    print("="*80)


def run_single_trial(config_dict, exp_name, trial_num, num_iterations=10, save_checkpoint=False, runner_overrides=None):
    """단일 시행 실행
    
    Args:
        config_dict: 설정 딕셔너리
        exp_name: 실험 이름
        trial_num: 시행 번호
        num_iterations: 반복 횟수
        save_checkpoint: 체크포인트 저장 여부
    """
    runner_cfg = DEFAULT_RUNNER_CONFIG.copy()
    if runner_overrides:
        runner_cfg.update(runner_overrides)

    clip_schedule = config_dict.get('clip_param_schedule')
    initial_clip = _scheduled_value(clip_schedule, 0) or config_dict['clip_param']

    model_config = {
        "fcnet_hiddens": config_dict['fcnet_hiddens'],
        "fcnet_activation": config_dict['fcnet_activation'],
        "vf_share_layers": config_dict['vf_share_layers'],
    }
    if config_dict.get('model_overrides'):
        model_config.update(config_dict['model_overrides'])

    optimizer_config = config_dict.get('optimizer_config')

    training_kwargs = dict(
        lambda_=config_dict['lambda_'],
        lr=config_dict['lr'],
        num_epochs=config_dict['num_epochs'],
        train_batch_size=config_dict['train_batch_size'],
        minibatch_size=config_dict['minibatch_size'],
        vf_loss_coeff=config_dict['vf_loss_coeff'],
        clip_param=initial_clip,
        vf_clip_param=config_dict['vf_clip_param'],
        entropy_coeff=config_dict['entropy_coeff'],
        use_kl_loss=config_dict['use_kl_loss'],
        kl_coeff=config_dict['kl_coeff'],
        kl_target=config_dict['kl_target'],
        grad_clip=config_dict['grad_clip'],
        gamma=config_dict['gamma'],
        use_gae=config_dict['use_gae'],
        use_critic=config_dict['use_critic'],
        model=model_config,
    )

    if config_dict.get('entropy_coeff_schedule'):
        training_kwargs['entropy_coeff_schedule'] = config_dict['entropy_coeff_schedule']
    if optimizer_config:
        training_kwargs['optimizer_config'] = optimizer_config

    config = (
        PPOConfig()
        .environment("HalfCheetah-v5")
        .training(**training_kwargs)
        .learners(num_learners=1, num_gpus_per_learner=1)  # GPU 사용으로 학습 가속화
        .debugging(seed=20227128 + trial_num)
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
    
    # 알고리즘 빌드
    algo = config.build()
    
    results = []
    start_time = time.time()
    checkpoint_path = None

    _apply_clip_param(algo, initial_clip)
    total_env_steps = 0
    
    try:
        for iteration in range(num_iterations):
            scheduled_clip = _scheduled_value(clip_schedule, total_env_steps)
            current_clip = algo.config.get('clip_param', initial_clip)
            if scheduled_clip is not None and abs(scheduled_clip - current_clip) > 1e-6:
                _apply_clip_param(algo, scheduled_clip)
                print(f"    ↺ clip_param schedule applied: {scheduled_clip:.3f} (steps={total_env_steps})")
            
            iter_start = time.time()
            result = algo.train()
            iter_time = time.time() - iter_start
            
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
            }
            
            results.append(metrics)
            total_env_steps = max(total_env_steps, int(result.get('num_env_steps_sampled', total_env_steps)))
            
            print(f"    Iter {iteration + 1}/{num_iterations}: "
                  f"Reward={metrics['episode_reward_mean']:.2f}, "
                  f"Trained={num_env_steps_trained}, "
                  f"Time={iter_time:.2f}s, "
                  f"SPS={sps:.0f}")
    
        
        # 마지막 iteration 체크포인트 저장 (옵션)
        if save_checkpoint and iteration == num_iterations - 1:
            raw_checkpoint = algo.save()
            if isinstance(raw_checkpoint, (str, Path)):
                checkpoint_path = str(raw_checkpoint)
            elif hasattr(raw_checkpoint, "to_uri"):
                checkpoint_path = raw_checkpoint.to_uri()
            elif hasattr(raw_checkpoint, "path"):
                checkpoint_path = str(raw_checkpoint.path)
            else:
                checkpoint_path = str(raw_checkpoint)
            print(f"    💾 체크포인트 저장: {checkpoint_path}")
    
    finally:
        algo.stop()
    
    total_time = time.time() - start_time
    
    return {
        'trial_num': trial_num,
        'iterations': results,
        'total_time': float(total_time),
        'final_reward': float(results[-1]['episode_reward_mean']) if results else 0.0,
        'checkpoint_path': checkpoint_path
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
    runner_overrides = exp_config.get('runner_overrides')
    
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
                save_checkpoint=save_this_trial,
                runner_overrides=runner_overrides
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
    
    print(f"\n  {'='*50}")
    print(f"  최종 통계:")
    print(f"    평균 보상: {statistics['final_reward_mean']:.2f} ± {statistics['final_reward_std']:.2f}")
    print(f"    변동계수(CV): {statistics['final_reward_cv']:.4f}")
    print(f"    범위: [{statistics['final_reward_min']:.2f}, {statistics['final_reward_max']:.2f}]")
    print(f"    평균 SPS: {statistics['sps_mean']:.0f} ± {statistics['sps_std']:.0f}")
    
    return {
        'name': exp_name,
        'description': exp_config['description'],
        'category': exp_config['category'],
        'rationale': exp_config['rationale'],
        'expected': exp_config.get('expected', ''),
        'params': exp_config['params'],
        'runner_overrides': exp_config.get('runner_overrides'),
        'trials': trials_results,
        'statistics': statistics
    }


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("파라미터 조합 실험 시작")
    print("20227128 김지훈")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 시스템 정보 출력
    print_system_info()
    
    # 실험 설정 로드
    experiments = get_combination_experiments()
    
    print(f"\n총 {len(experiments)}개의 조합 실험 예정")
    print(f"각 실험당 5회 시행, 시행당 10회 반복")
    total_envs = DEFAULT_RUNNER_CONFIG['num_env_runners'] * DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']
    print(f"병렬화: 기본 {DEFAULT_RUNNER_CONFIG['num_env_runners']} runners × {DEFAULT_RUNNER_CONFIG['num_envs_per_env_runner']} envs = {total_envs}개 환경 (균형 실험은 16 runners)")
    print(f"학습: num_learners=1 (GPU 사용)")
    print(f"근거: PPO 논문 32-64 envs, CleanRL 8 workers → 기본 10 runners로 확장")
    print(f"예상 소요 시간: 약 20-25분\n")
    
    # 사용자 확인
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
                'special_cases': {'balanced_high_momentum': 16},
                'num_learners': 1,
                'num_gpus_per_learner': 1,
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
    main()
