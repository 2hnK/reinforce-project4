"""
20227128 김지훈

병렬화 효율성 분석 실험 자동화 스크립트

목적:
    RLlib PPO 알고리즘의 병렬화 설정에 따른 성능 및 효율성 분석
    다양한 num_env_runners와 num_envs_per_env_runner 조합 실험

주요 기능:
    1. 백그라운드 리소스 모니터링 (CPU, GPU, RAM)
       - ResourceMonitor 클래스: 학습 중 0.5초마다 시스템 리소스 샘플링
       - 평균, 최대, 최소 사용률 통계 계산
    
    2. 정확한 성능 메트릭 수집
       - SPS (Steps Per Second): 초당 환경 스텝 수
       - 반복당 시간 (Time per Iteration)
       - 에피소드 보상 및 길이
    
    3. 다양한 병렬화 설정 실험
       - Baseline: 1 runner × 1 env
       - Runner 증가: 2/4/8 runners × 1 env (CPU 병렬화)
       - 환경 증가: 1 runner × 2/4/8 envs (환경 병렬화)
       - 조합: 2/4/8 runners × 2 envs (하이브리드)
    
    4. 실험 결과 저장
       - JSON 형식으로 상세 메트릭 저장
       - 중간 결과 자동 저장 (실험 중단 시 복구 가능)

실험 구성:
    - 환경: HalfCheetah-v5 (MuJoCo)
    - 알고리즘: PPO (Proximal Policy Optimization)
    - 총 실험 수: 10개 설정
    - 각 설정당 반복: 5회
    - 최대 러너 수: 8 (리소스 안정성)

출력 파일:
    - results/parallel_experiments_FIXED_progress.json (진행 중 저장)
    - results/parallel_experiments_FIXED_final.json (최종 결과)
"""

from ray.rllib.algorithms.ppo import PPOConfig
import json
import time
import psutil
import os
from datetime import datetime
import traceback
import threading

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU metrics will not be collected.")
    print("Install with: pip install gputil")


class ResourceMonitor:
    """백그라운드에서 시스템 리소스 모니터링"""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
    
    def start(self):
        """모니터링 시작"""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """모니터링 루프 (백그라운드 스레드)"""
        while self.monitoring:
            sample = self._get_sample()
            self.samples.append(sample)
            time.sleep(0.5)  # 0.5초마다 샘플링
    
    def _get_sample(self):
        """단일 샘플 수집"""
        sample = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=None),  # 비블로킹
            'cpu_per_core': psutil.cpu_percent(interval=None, percpu=True),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
        }
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    sample['gpu_utilization'] = gpu.load * 100
                    sample['vram_used_mb'] = gpu.memoryUsed
                    sample['vram_total_mb'] = gpu.memoryTotal
                    sample['gpu_temp'] = gpu.temperature
            except Exception:
                pass
        
        return sample
    
    def get_stats(self):
        """수집된 샘플의 통계"""
        if not self.samples:
            return {}
        
        import numpy as np
        
        cpu_percents = [s['cpu_percent'] for s in self.samples]
        ram_percents = [s['ram_percent'] for s in self.samples]
        
        stats = {
            'cpu_avg': np.mean(cpu_percents),
            'cpu_max': np.max(cpu_percents),
            'cpu_min': np.min(cpu_percents),
            'ram_avg': np.mean(ram_percents),
            'ram_max': np.max(ram_percents),
            'sample_count': len(self.samples),
        }
        
        # GPU 통계
        gpu_utils = [s.get('gpu_utilization', 0) for s in self.samples if 'gpu_utilization' in s]
        if gpu_utils:
            stats['gpu_avg'] = np.mean(gpu_utils)
            stats['gpu_max'] = np.max(gpu_utils)
            stats['gpu_min'] = np.min(gpu_utils)
            
            vram_usages = [s.get('vram_used_mb', 0) for s in self.samples if 'vram_used_mb' in s]
            stats['vram_avg_mb'] = np.mean(vram_usages)
            stats['vram_max_mb'] = np.max(vram_usages)
        
        # 코어별 CPU 평균
        if self.samples[0].get('cpu_per_core'):
            cpu_per_core_samples = [s['cpu_per_core'] for s in self.samples]
            stats['cpu_per_core_avg'] = np.mean(cpu_per_core_samples, axis=0).tolist()
            stats['cpu_per_core_max'] = np.max(cpu_per_core_samples, axis=0).tolist()
        
        return stats


def get_snapshot_metrics():
    """단일 시점 메트릭 (빠른 스냅샷)"""
    metrics = {
        'cpu_percent': psutil.cpu_percent(interval=None),
        'cpu_count': psutil.cpu_count(),
        'ram_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / (1024**3),
        'ram_total_gb': psutil.virtual_memory().total / (1024**3),
    }
    
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['vram_used_mb'] = gpu.memoryUsed
                metrics['vram_total_mb'] = gpu.memoryTotal
                metrics['vram_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except Exception as e:
            metrics['gpu_available'] = False
    else:
        metrics['gpu_available'] = False
    
    return metrics


def run_experiment(num_env_runners, num_envs_per_env_runner, num_iterations=5):
    """
    특정 병렬화 설정으로 학습 실험 실행
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
            num_learners=0,
            num_gpus_per_learner=1
        )
        .debugging(seed=20227128)
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
            'num_epochs': 5,
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
            
            # 백그라운드 모니터링 시작
            monitor = ResourceMonitor()
            monitor.start()
            
            # 학습 실행
            iter_start = time.time()
            result = algo.train()
            iter_time = time.time() - iter_start
            total_training_time += iter_time
            
            # 모니터링 중지
            monitor.stop()
            
            # 모니터링 통계 수집
            monitor_stats = monitor.get_stats()
            
            # 결과에서 주요 메트릭 추출
            time_this_iter = result.get('time_this_iter_s', iter_time)
            
            # SPS 계산 - 여러 가능한 키 시도
            num_env_steps = 0
            env_runners_info = result.get('env_runners', {})
            
            # 가능한 키들 시도
            for key in ['num_env_steps_sampled', 'num_env_steps_sampled_this_iter', 
                       'num_agent_steps_sampled', 'timesteps_this_iter']:
                if key in env_runners_info:
                    num_env_steps = env_runners_info[key]
                    break
            
            # 혹시 최상위에 있을 수도
            if num_env_steps == 0:
                for key in ['timesteps_this_iter', 'timesteps_total']:
                    if key in result:
                        num_env_steps = result[key]
                        break
            
            # train_batch_size로 추정 (최후의 수단)
            if num_env_steps == 0:
                num_env_steps = 32 * 512  # train_batch_size
            
            sps = num_env_steps / time_this_iter if time_this_iter > 0 else 0
            
            # Iteration 결과 저장
            iter_result = {
                'iteration': i + 1,
                'time_this_iter_s': time_this_iter,
                'wall_clock_time': iter_time,
                'num_env_steps': num_env_steps,
                'sps': sps,
                'episode_reward_mean': env_runners_info.get('episode_reward_mean', 
                                                             env_runners_info.get('episode_return_mean', 0)),
                'episode_len_mean': env_runners_info.get('episode_len_mean', 0),
                'resource_monitor': monitor_stats,  # 백그라운드 모니터링 결과
            }
            
            experiment_results['iterations'].append(iter_result)
            
            # 진행 상황 출력
            print(f"  Time: {time_this_iter:.2f}s")
            print(f"  SPS: {sps:.2f} steps/s")
            print(f"  Steps: {num_env_steps}")
            print(f"  Reward: {iter_result['episode_reward_mean']:.2f}")
            
            if monitor_stats:
                print(f"  CPU (학습 중 평균): {monitor_stats.get('cpu_avg', 0):.1f}%")
                print(f"  CPU (최대): {monitor_stats.get('cpu_max', 0):.1f}%")
                print(f"  RAM (평균): {monitor_stats.get('ram_avg', 0):.1f}%")
                if 'gpu_avg' in monitor_stats:
                    print(f"  GPU (평균): {monitor_stats['gpu_avg']:.1f}%")
                    print(f"  GPU (최대): {monitor_stats['gpu_max']:.1f}%")
                    print(f"  VRAM (평균): {monitor_stats['vram_avg_mb']:.0f}MB")
                print(f"  샘플 수: {monitor_stats['sample_count']}")
        
        # 요약 통계 계산
        times = [it['time_this_iter_s'] for it in experiment_results['iterations']]
        sps_values = [it['sps'] for it in experiment_results['iterations']]
        rewards = [it['episode_reward_mean'] for it in experiment_results['iterations']]
        
        # 모니터링 통계 평균
        cpu_avgs = [it['resource_monitor'].get('cpu_avg', 0) 
                   for it in experiment_results['iterations'] 
                   if it.get('resource_monitor')]
        
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
        
        # 리소스 요약
        if cpu_avgs:
            experiment_results['summary']['resource_usage'] = {
                'cpu_avg': sum(cpu_avgs) / len(cpu_avgs),
                'cpu_max': max([it['resource_monitor'].get('cpu_max', 0) 
                               for it in experiment_results['iterations']]),
            }
            
            # GPU 요약
            gpu_avgs = [it['resource_monitor'].get('gpu_avg', 0) 
                       for it in experiment_results['iterations'] 
                       if it['resource_monitor'].get('gpu_avg')]
            
            if gpu_avgs:
                experiment_results['summary']['resource_usage']['gpu_avg'] = sum(gpu_avgs) / len(gpu_avgs)
                experiment_results['summary']['resource_usage']['gpu_max'] = max(
                    [it['resource_monitor'].get('gpu_max', 0) 
                     for it in experiment_results['iterations']])
        
        print(f"\n{'='*80}")
        print(f"Experiment Summary:")
        print(f"  Average Time/Iter: {experiment_results['summary']['avg_time_per_iter_s']:.2f}s")
        print(f"  Average SPS: {experiment_results['summary']['avg_sps']:.2f}")
        print(f"  Total Time: {total_training_time:.2f}s")
        
        if 'resource_usage' in experiment_results['summary']:
            print(f"\n  Resource Usage (학습 중):")
            ru = experiment_results['summary']['resource_usage']
            print(f"    CPU 평균: {ru['cpu_avg']:.1f}%")
            print(f"    CPU 최대: {ru['cpu_max']:.1f}%")
            if 'gpu_avg' in ru:
                print(f"    GPU 평균: {ru['gpu_avg']:.1f}%")
                print(f"    GPU 최대: {ru['gpu_max']:.1f}%")
        
        print(f"{'='*80}")
        
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
    
    experiments_config = [
        # Baseline: 단일 러너, 단일 환경
        {'num_env_runners': 1, 'num_envs_per_env_runner': 1},
        
        # 러너 수 증가 (환경당 1개 유지) - CPU 병렬화 테스트
        {'num_env_runners': 2, 'num_envs_per_env_runner': 1},
        {'num_env_runners': 4, 'num_envs_per_env_runner': 1},
        {'num_env_runners': 8, 'num_envs_per_env_runner': 1},
        
        # 러너당 환경 수 증가 (러너 1개 유지) - 환경 병렬화 테스트
        {'num_env_runners': 1, 'num_envs_per_env_runner': 2},
        {'num_env_runners': 1, 'num_envs_per_env_runner': 4},
        {'num_env_runners': 1, 'num_envs_per_env_runner': 8},
        
        # 조합 테스트 - 최적 균형점 찾기
        {'num_env_runners': 2, 'num_envs_per_env_runner': 2},
        {'num_env_runners': 4, 'num_envs_per_env_runner': 2},
        {'num_env_runners': 8, 'num_envs_per_env_runner': 2},
    ]
    
    all_results = {
        'experiment_info': {
            'total_experiments': len(experiments_config),
            'iterations_per_experiment': 5,
            'start_time': datetime.now().isoformat(),
            'note': 'Parallel efficiency experiment with 5 iterations per config',
            'max_envs': 16,
            'reason': 'Limited to 8 runners max to avoid CPU resource exhaustion'
        },
        'experiments': []
    }
    
    print("=" * 80)
    print("PARALLEL EFFICIENCY EXPERIMENT - FIXED VERSION")
    print("=" * 80)
    print(f"Total experiments: {len(experiments_config)}")
    print(f"Max runners: 8 (to avoid Ray autoscaler resource issues)")
    print(f"Max total envs: 16 (8r×2e)")
    print("\nChanges:")
    print("  ✓ Background monitoring during training")
    print("  ✓ CPU/GPU measured while training is active")
    print("  ✓ Fixed SPS calculation")
    print("  ✓ Resource-safe configurations (max 8 runners)")
    print("=" * 80)
    
    # 각 설정에 대해 실험 실행
    for idx, exp_config in enumerate(experiments_config, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Experiment {idx}/{len(experiments_config)}")
        print(f"{'#'*80}")
        
        result = run_experiment(
            num_env_runners=exp_config['num_env_runners'],
            num_envs_per_env_runner=exp_config['num_envs_per_env_runner'],
            num_iterations=5
        )
        
        all_results['experiments'].append(result)
        
        # 중간 결과 저장
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        progress_file = os.path.join(results_dir, 'parallel_experiments_FIXED_progress.json')
        with open(progress_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Experiment {idx} completed and saved")
        
        if idx < len(experiments_config):
            print("\nWaiting 10s before next experiment...")
            time.sleep(10)
    
    # 최종 결과 저장
    all_results['experiment_info']['end_time'] = datetime.now().isoformat()
    
    final_file = os.path.join(results_dir, 'parallel_experiments_FIXED_final.json')
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {progress_file}")
    print(f"  - {final_file}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    print("\nSystem Information:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    if GPU_AVAILABLE:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"  GPU: {gpu.name}")
            print(f"  VRAM: {gpu.memoryTotal} MB")
    
    print("\n⚠️  This is the FIXED version with proper monitoring!")
    print("Changes:")
    print("  1. CPU/GPU monitored in background thread DURING training")
    print("  2. Sampling every 0.5s while training is active")
    print("  3. Fixed SPS calculation with multiple key attempts")
    print()
    
    # numpy 필요
    try:
        import numpy
    except ImportError:
        print("Installing numpy...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
    print("Starting experiments...\n")
    results = run_all_experiments()
