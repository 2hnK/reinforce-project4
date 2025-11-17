"""
20227128 김지훈

하이퍼파라미터 안정성 및 성능 비교 실험

목적:
    PPO 하이퍼파라미터 변경에 따른 학습 성능 및 안정성 분석
    각 설정을 5회 반복 실행하여 평균 성능과 분산(안정성) 측정

베이스라인 설정:
    lambda_=0.95,
    lr=0.0003,
    num_epochs=15,
    train_batch_size=32 * 512 (16384),
    minibatch_size=4096,
    vf_loss_coeff=0.01,
    fcnet_hiddens=[64, 64],
    fcnet_activation="tanh"

실험 파라미터:
    1. Learning Rate (lr) - 학습 속도
       - 0.0001 (낮음), 0.0003 (베이스라인), 0.001 (높음)
    
    2. GAE Lambda (lambda_) - 어드밴티지 추정
       - 0.9 (낮음), 0.95 (베이스라인), 0.99 (높음)
    
    3. Epochs (num_epochs) - 업데이트 반복 횟수
       - 10 (낮음), 15 (베이스라인), 20 (높음)
    
    4. Batch Size (train_batch_size) - 샘플 수
       - 8192 (작음), 16384 (베이스라인), 32768 (큼)
    
    5. Network Architecture (fcnet_hiddens) - 신경망 구조
       - [32, 32] (작음), [64, 64] (베이스라인), [128, 128] (큼), [256, 256] (매우 큼)
    
    6. Value Function Coefficient (vf_loss_coeff) - 가치 함수 손실 가중치
       - 0.005 (낮음), 0.01 (베이스라인), 0.05 (높음), 0.1 (매우 높음)
    
    7. Clip Parameter (clip_param) - PPO 클리핑
       - 0.1 (낮음), 0.2 (베이스라인), 0.3 (높음)
    
    8. Entropy Coefficient (entropy_coeff) - 탐험 장려
       - 0.0 (없음), 0.001 (낮음), 0.01 (중간), 0.1 (높음)

측정 지표:
    - episode_reward_mean: 평균 보상 (성능)
    - episode_reward_std: 보상 표준편차 (안정성)
    - SPS: 처리량
    - time_this_iter_s: 반복당 시간
    - convergence_speed: 수렴 속도 (목표 보상 도달 시간)
"""

from ray.rllib.algorithms.ppo import PPOConfig
import json
import time
import numpy as np
from datetime import datetime
import os


def get_baseline_config():
    """베이스라인 설정 반환
    
    고정 파라미터 (12-22 line, 변경 불가):
        lambda_, lr, num_epochs, train_batch_size, minibatch_size,
        vf_loss_coeff, fcnet_hiddens, fcnet_activation, vf_share_layers
    
    실험 가능한 파라미터 (추가/변경 가능):
        clip_param, vf_clip_param, entropy_coeff, use_kl_loss,
        kl_coeff, kl_target, grad_clip, gamma, use_gae, use_critic
    """
    return {
        # ===== 고정 파라미터 (변경 불가) =====
        'lambda_': 0.95,
        'lr': 0.0003,
        'num_epochs': 15,
        'train_batch_size': 32 * 512,  # 16384
        'minibatch_size': 4096,
        'vf_loss_coeff': 0.01,
        'fcnet_hiddens': [64, 64],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,
        
        # ===== 실험 가능한 파라미터 =====
        'clip_param': 0.2,          # PPO 클리핑
        'vf_clip_param': 10.0,      # 가치 함수 클리핑
        'entropy_coeff': 0.0,       # 엔트로피 계수
        'use_kl_loss': True,        # KL divergence 사용
        'kl_coeff': 0.2,            # KL 계수
        'kl_target': 0.01,          # KL 목표
        'grad_clip': None,          # 그래디언트 클리핑
        'gamma': 0.99,              # 할인율
        'use_gae': True,            # GAE 사용
        'use_critic': True,         # Critic 사용
    }


def get_experiment_configs():
    """실험할 하이퍼파라미터 설정들
    
    실험 설계 원칙:
    - 단일 파라미터 변화 실험 (Single-variable experiments)
    - 각 파라미터의 독립적 효과 측정
    - 실증 연구에서 검증된 범위 사용
    
    총 18개 실험 (베이스라인 1개 + 단일 변화 17개)
    """
    baseline = get_baseline_config()
    
    experiments = []
    
    # ===== 0. Baseline =====
    experiments.append({
        'name': 'baseline',
        'description': 'Baseline configuration (all defaults)',
        'params': baseline.copy(),
        'group': 'baseline',
        'rationale': 'PPO paper default values (Schulman et al., 2017)'
    })
    
    # ===== Group 1: Clipping 메커니즘 (2개) =====
    # Schulman et al. (2017): ε=0.2가 최적 균형
    config = baseline.copy()
    config['clip_param'] = 0.1
    experiments.append({
        'name': 'clip_conservative',
        'description': 'Conservative clipping: clip_param=0.1',
        'params': config,
        'group': 'clipping',
        'rationale': 'Lower clip → stable but slower convergence (PPO paper)'
    })
    
    config = baseline.copy()
    config['clip_param'] = 0.3
    experiments.append({
        'name': 'clip_aggressive',
        'description': 'Aggressive clipping: clip_param=0.3',
        'params': config,
        'group': 'clipping',
        'rationale': 'Higher clip → faster convergence but unstable (PPO paper)'
    })
    
    # ===== Group 2: Entropy Regularization (3개) =====
    # HalfCheetah는 연속 제어로 자연스럽게 탐험됨
    config = baseline.copy()
    config['entropy_coeff'] = 0.001
    experiments.append({
        'name': 'entropy_minimal',
        'description': 'Minimal entropy: entropy_coeff=0.001',
        'params': config,
        'group': 'entropy',
        'rationale': 'Slight exploration for continuous control (Haarnoja et al., 2018)'
    })
    
    config = baseline.copy()
    config['entropy_coeff'] = 0.01
    experiments.append({
        'name': 'entropy_medium',
        'description': 'Medium entropy: entropy_coeff=0.01',
        'params': config,
        'group': 'entropy',
        'rationale': 'Balanced exploration-exploitation (RLlib best practices)'
    })
    
    config = baseline.copy()
    config['entropy_coeff'] = 0.05
    experiments.append({
        'name': 'entropy_high',
        'description': 'High entropy: entropy_coeff=0.05',
        'params': config,
        'group': 'entropy',
        'rationale': 'Strong exploration for diverse solutions (SAC paper)'
    })
    
    # ===== Group 3: Discount Factor (2개) =====
    # γ는 가장 중요한 하이퍼파라미터
    config = baseline.copy()
    config['gamma'] = 0.95
    experiments.append({
        'name': 'gamma_short',
        'description': 'Short-term focus: gamma=0.95',
        'params': config,
        'group': 'gamma',
        'rationale': 'Short horizon for immediate rewards (HalfCheetah distance)'
    })
    
    config = baseline.copy()
    config['gamma'] = 0.995
    experiments.append({
        'name': 'gamma_long',
        'description': 'Long-term focus: gamma=0.995',
        'params': config,
        'group': 'gamma',
        'rationale': 'Long horizon for strategic planning'
    })
    
    # ===== Group 4: Gradient Clipping (2개) =====
    # 안정성 확보 기법
    config = baseline.copy()
    config['grad_clip'] = 0.5
    experiments.append({
        'name': 'grad_clip_tight',
        'description': 'Tight gradient clipping: grad_clip=0.5',
        'params': config,
        'group': 'gradient',
        'rationale': 'Strong stabilization, prevent divergence (Engstrom et al., 2020)'
    })
    
    config = baseline.copy()
    config['grad_clip'] = 1.0
    experiments.append({
        'name': 'grad_clip_loose',
        'description': 'Loose gradient clipping: grad_clip=1.0',
        'params': config,
        'group': 'gradient',
        'rationale': 'Moderate stabilization, faster learning'
    })
    
    # ===== Group 5: Value Function Clipping (2개) =====
    config = baseline.copy()
    config['vf_clip_param'] = 1.0
    experiments.append({
        'name': 'vf_clip_tight',
        'description': 'Tight VF clipping: vf_clip_param=1.0',
        'params': config,
        'group': 'value_function',
        'rationale': 'Conservative value updates for stability'
    })
    
    config = baseline.copy()
    config['vf_clip_param'] = 100.0
    experiments.append({
        'name': 'vf_clip_loose',
        'description': 'Loose VF clipping: vf_clip_param=100.0',
        'params': config,
        'group': 'value_function',
        'rationale': 'Free value learning for faster adaptation'
    })
    
    # ===== Group 6: KL Divergence (3개) =====
    # PPO 논문: KL penalty < clipping, 하지만 추가 안정성 제공
    config = baseline.copy()
    config['use_kl_loss'] = False
    experiments.append({
        'name': 'kl_disabled',
        'description': 'KL loss disabled (PPO-Clip only)',
        'params': config,
        'group': 'kl_divergence',
        'rationale': 'Simpler PPO-Clip only (Schulman et al., 2017)'
    })
    
    config = baseline.copy()
    config['use_kl_loss'] = True
    config['kl_coeff'] = 0.1
    experiments.append({
        'name': 'kl_weak',
        'description': 'Weak KL constraint: kl_coeff=0.1',
        'params': config,
        'group': 'kl_divergence',
        'rationale': 'Light policy constraint for additional stability'
    })
    
    config = baseline.copy()
    config['use_kl_loss'] = True
    config['kl_coeff'] = 0.5
    experiments.append({
        'name': 'kl_strong',
        'description': 'Strong KL constraint: kl_coeff=0.5',
        'params': config,
        'group': 'kl_divergence',
        'rationale': 'Strong policy constraint for maximum stability'
    })
    
    # ===== Group 7: GAE (1개) =====
    config = baseline.copy()
    config['use_gae'] = False
    experiments.append({
        'name': 'no_gae',
        'description': 'GAE disabled (simple advantage)',
        'params': config,
        'group': 'advantage_estimation',
        'rationale': 'Simple advantage without bias-variance tradeoff'
    })
    
    return experiments


def run_single_trial(config_params, trial_num, total_iterations=10):
    """단일 실험 실행"""
    print(f"\n  Trial {trial_num}/5")
    print(f"  {'='*50}")
    
    # PPO Config 생성
    config = (
        PPOConfig()
        .environment("HalfCheetah-v5")
        .training(
            # ===== 고정 파라미터 (12-22 line) =====
            lambda_=config_params['lambda_'],
            lr=config_params['lr'],
            num_epochs=config_params['num_epochs'],
            train_batch_size=config_params['train_batch_size'],
            minibatch_size=config_params['minibatch_size'],
            vf_loss_coeff=config_params['vf_loss_coeff'],
            model={
                "fcnet_hiddens": config_params['fcnet_hiddens'],
                "fcnet_activation": config_params['fcnet_activation'],
                "vf_share_layers": config_params['vf_share_layers'],
            },
            
            # ===== 실험 파라미터 =====
            clip_param=config_params['clip_param'],
            vf_clip_param=config_params['vf_clip_param'],
            entropy_coeff=config_params['entropy_coeff'],
            use_kl_loss=config_params['use_kl_loss'],
            kl_coeff=config_params['kl_coeff'],
            kl_target=config_params['kl_target'],
            grad_clip=config_params['grad_clip'],
            gamma=config_params['gamma'],
            use_gae=config_params['use_gae'],
            use_critic=config_params['use_critic'],
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=1
        )
        .debugging(seed=20227128 + trial_num)  # 각 trial마다 다른 시드
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=0,
            evaluation_duration=5
        )
    )
    
    # 알고리즘 생성
    algo = config.build_algo()
    
    trial_results = {
        'trial': trial_num,
        'seed': 20227128 + trial_num,
        'iterations': [],
        'total_time': 0,
    }
    
    start_time = time.time()
    
    try:
        for i in range(total_iterations):
            iter_start = time.time()
            res = algo.train()
            iter_time = time.time() - iter_start
            
            # 메트릭 추출
            env_runners = res.get('env_runners', {})
            episode_reward_mean = env_runners.get('episode_reward_mean', 
                                                  env_runners.get('episode_return_mean', 0))
            time_this_iter_s = res.get('time_this_iter_s', iter_time)
            
            # SPS 계산
            num_env_steps = 0
            for key in ['num_env_steps_sampled', 'num_env_steps_sampled_this_iter']:
                if key in env_runners:
                    num_env_steps = env_runners[key]
                    break
            if num_env_steps == 0:
                num_env_steps = config_params['train_batch_size']
            
            sps = num_env_steps / time_this_iter_s if time_this_iter_s > 0 else 0
            
            iter_result = {
                'iteration': i + 1,
                'episode_reward_mean': episode_reward_mean,
                'episode_len_mean': env_runners.get('episode_len_mean', 0),
                'time_this_iter_s': time_this_iter_s,
                'sps': sps,
            }
            
            trial_results['iterations'].append(iter_result)
            
            # 간단한 출력
            print(f"    Iter {i+1}/{total_iterations}: Reward={episode_reward_mean:.2f}, "
                  f"Time={time_this_iter_s:.2f}s, SPS={sps:.2f}")
    
    finally:
        algo.stop()
    
    trial_results['total_time'] = time.time() - start_time
    
    return trial_results


def run_experiment(experiment_config, num_trials=5, total_iterations=10):
    """하나의 하이퍼파라미터 설정에 대해 여러 번 실행"""
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_config['name']}")
    print(f"Description: {experiment_config['description']}")
    print(f"{'='*70}")
    
    # 변경된 파라미터 출력
    baseline = get_baseline_config()
    changed_params = {}
    for key, value in experiment_config['params'].items():
        if baseline[key] != value:
            changed_params[key] = {'baseline': baseline[key], 'new': value}
    
    if changed_params:
        print("\nChanged parameters:")
        for key, values in changed_params.items():
            print(f"  • {key}: {values['baseline']} → {values['new']}")
    else:
        print("\nNo changes (baseline configuration)")
    
    experiment_results = {
        'name': experiment_config['name'],
        'description': experiment_config['description'],
        'params': experiment_config['params'],
        'changed_params': changed_params,
        'trials': [],
        'num_trials': num_trials,
        'num_iterations': total_iterations,
    }
    
    # 여러 번 실행
    for trial in range(1, num_trials + 1):
        trial_result = run_single_trial(
            experiment_config['params'], 
            trial, 
            total_iterations
        )
        experiment_results['trials'].append(trial_result)
    
    # 통계 계산
    all_final_rewards = []
    all_mean_rewards = []
    all_sps = []
    all_times = []
    
    for trial in experiment_results['trials']:
        final_reward = trial['iterations'][-1]['episode_reward_mean']
        mean_reward = np.mean([it['episode_reward_mean'] for it in trial['iterations']])
        mean_sps = np.mean([it['sps'] for it in trial['iterations']])
        total_time = trial['total_time']
        
        all_final_rewards.append(final_reward)
        all_mean_rewards.append(mean_reward)
        all_sps.append(mean_sps)
        all_times.append(total_time)
    
    experiment_results['statistics'] = {
        'final_reward_mean': float(np.mean(all_final_rewards)),
        'final_reward_std': float(np.std(all_final_rewards)),
        'final_reward_min': float(np.min(all_final_rewards)),
        'final_reward_max': float(np.max(all_final_rewards)),
        
        'mean_reward_mean': float(np.mean(all_mean_rewards)),
        'mean_reward_std': float(np.std(all_mean_rewards)),
        
        'sps_mean': float(np.mean(all_sps)),
        'sps_std': float(np.std(all_sps)),
        
        'time_mean': float(np.mean(all_times)),
        'time_std': float(np.std(all_times)),
        
        # 안정성 지표: 변동계수 (CV = std/mean)
        'final_reward_cv': float(np.std(all_final_rewards) / np.mean(all_final_rewards)) if np.mean(all_final_rewards) != 0 else 0,
    }
    
    # 결과 출력
    stats = experiment_results['statistics']
    print(f"\n{'='*70}")
    print(f"Results Summary for: {experiment_config['name']}")
    print(f"{'='*70}")
    print(f"Performance (5 trials average):")
    print(f"  Final Reward: {stats['final_reward_mean']:.2f} ± {stats['final_reward_std']:.2f}")
    print(f"  Range: [{stats['final_reward_min']:.2f}, {stats['final_reward_max']:.2f}]")
    print(f"  Mean Reward (across iterations): {stats['mean_reward_mean']:.2f} ± {stats['mean_reward_std']:.2f}")
    print(f"\nStability:")
    print(f"  Coefficient of Variation (CV): {stats['final_reward_cv']:.4f}")
    print(f"  Lower CV = More Stable")
    print(f"\nEfficiency:")
    print(f"  SPS: {stats['sps_mean']:.2f} ± {stats['sps_std']:.2f}")
    print(f"  Time per experiment: {stats['time_mean']:.2f}s ± {stats['time_std']:.2f}s")
    print(f"{'='*70}\n")
    
    return experiment_results


def main():
    """메인 실험 실행"""
    print("="*70)
    print("Hyperparameter Stability and Performance Comparison Experiment")
    print("Student ID: 20227128")
    print("="*70)
    
    # 실험 환경 정보 출력
    print("\n" + "="*70)
    print("실험 환경 정보")
    print("="*70)
    
    # CPU 정보
    try:
        import subprocess
        cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        cpu_count = subprocess.check_output("nproc", shell=True).decode().strip()
        print(f"CPU: {cpu_info.split(':')[1].strip()}")
        print(f"CPU Cores: {cpu_count}")
    except:
        import os as os_mod
        print(f"CPU Cores: {os_mod.cpu_count()}")
    
    # Memory 정보
    try:
        mem_info = subprocess.check_output("free -h | grep Mem", shell=True).decode().split()
        print(f"Memory: {mem_info[1]} (Available: {mem_info[6]})")
    except:
        print("Memory: 정보 확인 불가")
    
    # GPU 정보
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True).decode().strip()
        print(f"GPU: {gpu_info}")
        
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    except:
        print("GPU: NVIDIA GPU 없음 (CPU 모드)")
    
    # Ray 버전
    import ray
    print(f"Ray: {ray.__version__}")
    
    print("="*70 + "\n")
    
    # 결과 디렉토리 생성
    results_dir = 'hyperparameter_experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 실험 설정 로드
    experiments = get_experiment_configs()
    
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Trials per experiment: 5")
    print(f"Iterations per trial: 10")
    print(f"Total training runs: {len(experiments) * 5} ({len(experiments)} × 5)")
    
    # 사용자 확인
    print("\nExperiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    
    response = input("\nDo you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Experiment cancelled.")
        return
    
    # 모든 실험 결과
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'student_id': '20227128',
        'baseline_config': get_baseline_config(),
        'experiments': []
    }
    
    # 실험 실행
    start_time = time.time()
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'#'*70}")
        
        try:
            exp_result = run_experiment(exp_config, num_trials=5, total_iterations=10)
            all_results['experiments'].append(exp_result)
            
            # 중간 저장
            temp_file = f'{results_dir}/hyperparameter_experiments_progress.json'
            with open(temp_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n[Progress saved to {temp_file}]")
            
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 최종 결과 저장
    all_results['total_time'] = time.time() - start_time
    final_file = f'{results_dir}/hyperparameter_experiments_final.json'
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*70}")
    print(f"Total time: {all_results['total_time']:.2f}s ({all_results['total_time']/60:.2f} minutes)")
    print(f"Results saved to: {final_file}")
    print(f"{'='*70}")
    
    # 요약 출력
    print("\n" + "="*70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*70)
    print(f"{'Experiment':<30} {'Final Reward':<20} {'Stability (CV)':<15}")
    print("-"*70)
    
    for exp in all_results['experiments']:
        stats = exp['statistics']
        name = exp['name']
        reward = f"{stats['final_reward_mean']:.2f} ± {stats['final_reward_std']:.2f}"
        cv = f"{stats['final_reward_cv']:.4f}"
        print(f"{name:<30} {reward:<20} {cv:<15}")
    
    print("="*70)


if __name__ == "__main__":
    main()
