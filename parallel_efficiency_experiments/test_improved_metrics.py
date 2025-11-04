"""
개선된 메트릭 수집 테스트 스크립트
- 올바른 SPS 계산
- 개선된 CPU 측정
"""

from ray.rllib.algorithms.ppo import PPOConfig
import psutil
import os
import time

def get_system_metrics():
    """시스템 리소스 사용률 측정"""
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
    
    # 전체 시스템 CPU도 측정
    system_cpu_percent = psutil.cpu_percent(interval=0.1)
    
    return {
        'process_cpu_percent': cpu_percent,
        'system_cpu_percent': system_cpu_percent,
        'cpu_count': psutil.cpu_count(),
        'ram_percent': psutil.virtual_memory().percent,
    }

print("="*80)
print("메트릭 수집 테스트")
print("="*80)

# 2개 구성 테스트
configs = [
    {'num_env_runners': 1, 'num_envs_per_env_runner': 1},
    {'num_env_runners': 4, 'num_envs_per_env_runner': 1},
]

for cfg in configs:
    print(f"\n{'#'*80}")
    print(f"# Config: {cfg['num_env_runners']}r × {cfg['num_envs_per_env_runner']}e")
    print(f"{'#'*80}")
    
    config = (
        PPOConfig()
        .environment("HalfCheetah-v5")
        .training(
            lambda_=0.95,
            lr=0.0003,
            num_epochs=3,
            train_batch_size=16384,
            minibatch_size=4096,
        )
        .learners(num_learners=0, num_gpus_per_learner=1)
        .env_runners(
            num_env_runners=cfg['num_env_runners'],
            num_envs_per_env_runner=cfg['num_envs_per_env_runner'],
        )
    )
    
    print("\nBuilding...")
    algo = config.build()
    
    # 2번 iteration
    for i in range(2):
        print(f"\nIteration {i+1}/2")
        
        pre_metrics = get_system_metrics()
        iter_start = time.time()
        result = algo.train()
        iter_time = time.time() - iter_start
        post_metrics = get_system_metrics()
        
        # 메트릭 추출
        time_this_iter = result.get('time_this_iter_s', iter_time)
        env_runners_info = result.get('env_runners', {})
        
        # SPS 계산 - 올바른 키 사용
        num_env_steps = env_runners_info.get('num_env_steps_sampled', 0)
        sps = num_env_steps / time_this_iter if time_this_iter > 0 and num_env_steps > 0 else 0
        
        reward = env_runners_info.get('episode_return_mean', 0)
        
        print(f"  Time: {time_this_iter:.2f}s")
        print(f"  Steps: {num_env_steps}")
        print(f"  SPS: {sps:.2f} steps/s")
        print(f"  Reward: {reward:.2f}")
        print(f"  Process CPU: {post_metrics['process_cpu_percent']:.1f}%")
        print(f"  System CPU: {post_metrics['system_cpu_percent']:.1f}%")
        print(f"  RAM: {post_metrics['ram_percent']:.1f}%")
        
        # SPS 검증
        if num_env_steps > 0:
            print(f"  ✓ SPS 계산 성공!")
        else:
            print(f"  ✗ WARNING: num_env_steps = 0")
    
    algo.stop()
    print("\nWaiting 5s...")
    time.sleep(5)

print("\n" + "="*80)
print("테스트 완료!")
print("="*80)
