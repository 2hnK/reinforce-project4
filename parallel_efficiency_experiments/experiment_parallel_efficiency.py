"""
병렬화 효율성 분석 실험 자동화 스크립트

이 스크립트는 다양한 병렬화 설정으로 PPO 학습을 실행하고
성능 메트릭을 수집하여 분석합니다.

실험 변수:
- num_env_runners: 환경 러너 수
- num_envs_per_env_runner: 러너당 환경 수

측정 항목:
- time_this_iter_s: 각 iteration 소요 시간
- SPS (Steps Per Second): 처리량
- 시스템 리소스 사용률 (CPU, GPU, RAM, VRAM)
"""

from ray.rllib.algorithms.ppo import PPOConfig
import json
import time
import psutil
import os
from datetime import datetime
import traceback

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU metrics will not be collected.")
    print("Install with: pip install gputil")


def get_system_metrics():
    """시스템 리소스 사용률 측정"""
    import os
    
    # 현재 프로세스와 모든 자식 프로세스의 CPU 사용률 합산
    current_process = psutil.Process(os.getpid())
    cpu_percent = current_process.cpu_percent(interval=0.1)
    
    # 자식 프로세스들 (Ray workers 포함)
    try:
        children = current_process.children(recursive=True)
        for child in children:
            try:
                cpu_percent += child.cpu_percent(interval=0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    # 전체 시스템 CPU도 측정 (비교용)
    system_cpu_percent = psutil.cpu_percent(interval=0.1)
    
    metrics = {
        'cpu_percent': cpu_percent,  # 이 프로세스 + 자식들
        'system_cpu_percent': system_cpu_percent,  # 전체 시스템
        'cpu_count': psutil.cpu_count(),
        'ram_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / (1024**3),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
    }
    
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 첫 번째 GPU 사용
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['vram_used_mb'] = gpu.memoryUsed
                metrics['vram_total_mb'] = gpu.memoryTotal
                metrics['vram_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                metrics['gpu_temp'] = gpu.temperature
            else:
                metrics['gpu_available'] = False
        except Exception as e:
            print(f"GPU metrics collection error: {e}")
            metrics['gpu_available'] = False
    else:
        metrics['gpu_available'] = False
    
    return metrics


def run_experiment(num_env_runners, num_envs_per_env_runner, num_iterations=3):
    """
    특정 병렬화 설정으로 학습 실험 실행
    
    Args:
        num_env_runners: 환경 러너 수
        num_envs_per_env_runner: 러너당 환경 수
        num_iterations: 학습 반복 횟수
    
    Returns:
        실험 결과 딕셔너리
    """
    print(f"\n{'='*80}")
    print(f"Experiment: num_env_runners={num_env_runners}, "
          f"num_envs_per_env_runner={num_envs_per_env_runner}")
    print(f"{'='*80}")
    
    # 설정 생성
    config = (
        PPOConfig()
        .environment("HalfCheetah-v5")
        .training(
            lambda_=0.95,
            lr=0.0003,
            num_epochs=5,
            train_batch_size=32 * 512,
            minibatch_size=4096,
            vf_loss_coeff=0.01,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            },
        )
        .learners(
            num_learners=0,  # 로컬 learner 사용
            num_gpus_per_learner=1
        )
        .debugging(seed=0)
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=1,
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=0,
            evaluation_duration=5
        )
    )
    
    experiment_results = {
        'config': {
            'num_env_runners': num_env_runners,
            'num_envs_per_env_runner': num_envs_per_env_runner,
            'total_envs': num_env_runners * num_envs_per_env_runner,
            'num_iterations': num_iterations,
            'num_epochs': 3,
        },
        'iterations': [],
        'summary': {},
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        },
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # 알고리즘 빌드 전 초기 메트릭
        initial_metrics = get_system_metrics()
        experiment_results['initial_system_metrics'] = initial_metrics
        
        print("Building algorithm...")
        build_start = time.time()
        algo = config.build()
        build_time = time.time() - build_start
        experiment_results['build_time_s'] = build_time
        print(f"✓ Algorithm built in {build_time:.2f}s")
        
        # 학습 반복
        total_training_time = 0
        
        for i in range(num_iterations):
            print(f"\nIteration {i+1}/{num_iterations}")
            
            # 학습 전 시스템 메트릭
            pre_metrics = get_system_metrics()
            
            # 학습 실행
            iter_start = time.time()
            result = algo.train()
            iter_time = time.time() - iter_start
            total_training_time += iter_time
            
            # 학습 후 시스템 메트릭
            post_metrics = get_system_metrics()
            
            # 결과에서 주요 메트릭 추출
            time_this_iter = result.get('time_this_iter_s', iter_time)
            
            # SPS 계산 (Steps Per Second) - 올바른 메트릭 경로 사용
            env_runners_info = result.get('env_runners', {})
            num_env_steps = env_runners_info.get('num_env_steps_sampled', 0)  # 수정: 올바른 키
            sps = num_env_steps / time_this_iter if time_this_iter > 0 and num_env_steps > 0 else 0
            
            # Iteration 결과 저장
            iter_result = {
                'iteration': i + 1,
                'time_this_iter_s': time_this_iter,
                'wall_clock_time': iter_time,
                'num_env_steps': num_env_steps,
                'sps': sps,
                'episode_reward_mean': env_runners_info.get('episode_reward_mean', 0),
                'episode_len_mean': env_runners_info.get('episode_len_mean', 0),
                'pre_system_metrics': pre_metrics,
                'post_system_metrics': post_metrics,
            }
            
            experiment_results['iterations'].append(iter_result)
            
            # 진행 상황 출력
            print(f"  Time: {time_this_iter:.2f}s")
            print(f"  SPS: {sps:.2f} steps/s")
            print(f"  Steps: {num_env_steps}")
            print(f"  Reward: {iter_result['episode_reward_mean']:.2f}")
            print(f"  Process CPU: {post_metrics['cpu_percent']:.1f}%")
            print(f"  System CPU: {post_metrics['system_cpu_percent']:.1f}%")
            print(f"  RAM: {post_metrics['ram_percent']:.1f}%")
            if post_metrics.get('gpu_utilization') is not None:
                print(f"  GPU Util: {post_metrics['gpu_utilization']:.1f}%")
                print(f"  VRAM: {post_metrics['vram_used_mb']:.0f}MB / {post_metrics['vram_total_mb']:.0f}MB ({post_metrics['vram_percent']:.1f}%)")
                if post_metrics.get('gpu_temp'):
                    print(f"  GPU Temp: {post_metrics['gpu_temp']:.1f}°C")
        
        # 요약 통계 계산
        times = [it['time_this_iter_s'] for it in experiment_results['iterations']]
        sps_values = [it['sps'] for it in experiment_results['iterations']]
        rewards = [it['episode_reward_mean'] for it in experiment_results['iterations']]
        
        experiment_results['summary'] = {
            'avg_time_per_iter_s': sum(times) / len(times),
            'min_time_per_iter_s': min(times),
            'max_time_per_iter_s': max(times),
            'avg_sps': sum(sps_values) / len(sps_values),
            'max_sps': max(sps_values),
            'total_training_time_s': total_training_time,
            'avg_reward': sum(rewards) / len(rewards),
            'final_reward': rewards[-1],
        }
        
        # GPU 요약 통계 추가
        gpu_utils = [it['post_system_metrics'].get('gpu_utilization', 0) 
                     for it in experiment_results['iterations'] 
                     if it['post_system_metrics'].get('gpu_utilization') is not None]
        vram_usages = [it['post_system_metrics'].get('vram_used_mb', 0) 
                       for it in experiment_results['iterations'] 
                       if it['post_system_metrics'].get('vram_used_mb') is not None]
        
        if gpu_utils:
            experiment_results['summary']['gpu'] = {
                'avg_gpu_utilization': sum(gpu_utils) / len(gpu_utils),
                'max_gpu_utilization': max(gpu_utils),
                'min_gpu_utilization': min(gpu_utils),
                'avg_vram_used_mb': sum(vram_usages) / len(vram_usages) if vram_usages else 0,
                'max_vram_used_mb': max(vram_usages) if vram_usages else 0,
            }
        
        print(f"\n{'='*80}")
        print(f"Experiment Summary:")
        print(f"  Average Time/Iter: {experiment_results['summary']['avg_time_per_iter_s']:.2f}s")
        print(f"  Average SPS: {experiment_results['summary']['avg_sps']:.2f}")
        print(f"  Max SPS: {experiment_results['summary']['max_sps']:.2f}")
        print(f"  Total Time: {total_training_time:.2f}s")
        print(f"  Final Reward: {experiment_results['summary']['final_reward']:.2f}")
        
        # GPU 요약 출력
        if 'gpu' in experiment_results['summary']:
            print(f"\n  GPU Summary:")
            print(f"    Avg GPU Util: {experiment_results['summary']['gpu']['avg_gpu_utilization']:.1f}%")
            print(f"    Max GPU Util: {experiment_results['summary']['gpu']['max_gpu_utilization']:.1f}%")
            print(f"    Avg VRAM: {experiment_results['summary']['gpu']['avg_vram_used_mb']:.0f}MB")
            print(f"    Max VRAM: {experiment_results['summary']['gpu']['max_vram_used_mb']:.0f}MB")
        
        print(f"{'='*80}")
        
        # 알고리즘 정리
        algo.stop()
        
        experiment_results['status'] = 'success'
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print(traceback.format_exc())
        experiment_results['status'] = 'failed'
        experiment_results['error'] = str(e)
        experiment_results['traceback'] = traceback.format_exc()
    
    return experiment_results


def run_all_experiments():
    """모든 병렬화 설정 조합에 대해 실험 실행"""
    
    # 실험 설정 정의
    # 시스템 사양: i7-12700 (16 logical cores), 32GB RAM, RTX 3070
    experiments_config = [
        # Baseline: 단일 러너, 단일 환경
        {'num_env_runners': 1, 'num_envs_per_env_runner': 1},
        
        # 러너 수 증가 (환경당 1개 유지) - CPU 병렬화 테스트
        {'num_env_runners': 4, 'num_envs_per_env_runner': 1},
        {'num_env_runners': 8, 'num_envs_per_env_runner': 1},
        {'num_env_runners': 16, 'num_envs_per_env_runner': 1},
        
        # 러너당 환경 수 증가 (러너 1개 유지) - 환경 병렬화 테스트
        {'num_env_runners': 1, 'num_envs_per_env_runner': 4},
        {'num_env_runners': 1, 'num_envs_per_env_runner': 8},
        {'num_env_runners': 1, 'num_envs_per_env_runner': 16},
        
        # 조합 테스트 - 최적 균형점 찾기
        {'num_env_runners': 4, 'num_envs_per_env_runner': 2},
        {'num_env_runners': 4, 'num_envs_per_env_runner': 4},
        {'num_env_runners': 8, 'num_envs_per_env_runner': 2},
        {'num_env_runners': 8, 'num_envs_per_env_runner': 4},
    ]
    
    all_results = {
        'experiment_info': {
            'total_experiments': len(experiments_config),
            'iterations_per_experiment': 3,
            'start_time': datetime.now().isoformat(),
        },
        'experiments': []
    }
    
    print("=" * 80)
    print("PARALLEL EFFICIENCY EXPERIMENT SUITE")
    print("=" * 80)
    print(f"Total experiments: {len(experiments_config)}")
    print(f"Iterations per experiment: 3")
    print("=" * 80)
    
    # 각 설정에 대해 실험 실행
    for idx, exp_config in enumerate(experiments_config, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Experiment {idx}/{len(experiments_config)}")
        print(f"{'#'*80}")
        
        result = run_experiment(
            num_env_runners=exp_config['num_env_runners'],
            num_envs_per_env_runner=exp_config['num_envs_per_env_runner'],
            num_iterations=3
        )
        
        all_results['experiments'].append(result)
        
        # 중간 결과 저장 (실험 실패 시 데이터 보존)
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        progress_file = os.path.join(results_dir, 'parallel_experiments_progress.json')
        with open(progress_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Experiment {idx} completed and saved")
        
        # 다음 실험 전 대기 (시스템 안정화)
        if idx < len(experiments_config):
            print("\nWaiting 10s before next experiment...")
            time.sleep(10)
    
    # 최종 결과 저장
    all_results['experiment_info']['end_time'] = datetime.now().isoformat()
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    final_file = os.path.join(results_dir, 'parallel_experiments_final.json')
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {progress_file} (incremental)")
    print(f"  - {final_file} (final)")
    print(f"\nRun 'python analyze_parallel_efficiency_simple.py' to generate analysis and visualizations")
    
    return all_results


if __name__ == "__main__":
    # 시스템 정보 출력
    print("\nSystem Information:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"  GPU: {gpu.name}")
            print(f"  VRAM: {gpu.memoryTotal} MB")
        else:
            print(f"  GPU: Not detected")
    else:
        print(f"  GPU monitoring: Not available (install gputil)")
    
    print("\nStarting experiments...\n")
    
    results = run_all_experiments()
