"""
20227128 김지훈

병렬화 효율성 실험 결과 분석 및 시각화 스크립트

목적:
    experiment_parallel_efficiency.py로 수집한 병렬화 실험 결과를 
    분석하고 시각화하여 최적의 병렬화 설정 도출

주요 기능:
    1. 실험 결과 로드 및 파싱
       - JSON 형식의 실험 결과 파일 로드
       - 각 설정별 성능 메트릭 추출
    
    2. 성능 분석
       - Speedup 계산: Baseline 대비 속도 향상 배율
       - Efficiency 계산: 이상적 속도 향상 대비 실제 효율성 (%)
       - GPU 병목 분석: GPU 사용률 기반 병목 현상 감지
    
    3. 시각화 대시보드 생성 (5개 차트)
       - Strong Scaling: 환경 수 대비 Speedup (이상적 선형 비교)
       - Parallel Efficiency: 각 설정의 효율성 (%)
       - Time per Iteration: 반복당 소요 시간
       - GPU Utilization: GPU 사용률 (GPU 사용 시)
       - VRAM Usage: GPU 메모리 사용량 (GPU 사용 시)
    
    4. 성능 순위 및 추천
       - Best Speedup: 가장 빠른 설정
       - Best Efficiency: 가장 효율적인 설정
       - Fastest Time: 절대 시간 기준 최고 성능
       - GPU 병목 분석: High/Low/Balanced 분류

분석 메트릭:
    - Speedup = T_baseline / T_config
    - Efficiency = (Speedup / Ideal_Speedup) × 100%
    - Ideal_Speedup = num_runners × num_envs_per_runner

입력 파일:
    - results/parallel_experiments_final.json (실험 결과 데이터)

출력 파일:
    - results/parallel_efficiency_dashboard.png (시각화 대시보드)
    - results/parallel_efficiency_report.txt (텍스트 리포트)
"""

import json
import os
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np

def load_results(filename='results/parallel_experiments_FIXED_final.json'):
    """실험 결과 로드"""
    import os
    if not os.path.exists(filename):
        print(f"Error: Results file '{filename}' not found!")
        print("Please run experiment_parallel_efficiency.py first.")
        exit(1)
    
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_results(data):
    """실험 결과 분석"""
    experiments = data['experiments']
    
    # Baseline 찾기
    baseline = next((exp for exp in experiments 
                    if exp['config']['num_env_runners'] == 1 
                    and exp['config']['num_envs_per_env_runner'] == 1), None)
    
    if not baseline:
        print("⚠️  No baseline found!")
        return None
    
    baseline_time = baseline['summary']['avg_time_per_iter_s']
    
    # GPU 가용 여부 확인
    has_gpu = any(exp['summary'].get('gpu') is not None for exp in experiments)
    
    # 분석 결과 생성
    results = []
    for exp in experiments:
        config = exp['config']
        total_envs = config['num_env_runners'] * config['num_envs_per_env_runner']
        avg_time = exp['summary']['avg_time_per_iter_s']
        
        # Speedup 계산 (시간 기준)
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        
        # Efficiency 계산
        ideal_speedup = total_envs
        efficiency = (speedup / ideal_speedup) * 100 if ideal_speedup > 0 else 0
        
        result = {
            'config_str': f"{config['num_env_runners']}r×{config['num_envs_per_env_runner']}e",
            'num_runners': config['num_env_runners'],
            'envs_per_runner': config['num_envs_per_env_runner'],
            'total_envs': total_envs,
            'avg_time': avg_time,
            'speedup': speedup,
            'efficiency': efficiency,
        }
        
        # 리소스 사용량 추가 (CPU, GPU, RAM)
        resource_usage = exp['summary'].get('resource_usage', {})
        result['cpu_avg'] = resource_usage.get('cpu_avg', 0)
        result['cpu_max'] = resource_usage.get('cpu_max', 0)
        result['gpu_avg'] = resource_usage.get('gpu_avg', None)
        result['gpu_max'] = resource_usage.get('gpu_max', None)
        
        # GPU 메트릭 추가 (이전 버전 호환)
        if exp['summary'].get('gpu'):
            gpu = exp['summary']['gpu']
            result['avg_gpu_util'] = gpu.get('avg_gpu_utilization', 0)
            result['max_gpu_util'] = gpu.get('max_gpu_utilization', 0)
            result['avg_vram_mb'] = gpu.get('avg_vram_used_mb', 0)
            result['max_vram_mb'] = gpu.get('max_vram_used_mb', 0)
        else:
            result['avg_gpu_util'] = result['gpu_avg']
            result['max_gpu_util'] = result['gpu_max']
            result['avg_vram_mb'] = None
            result['max_vram_mb'] = None
        
        results.append(result)
    
    # Total envs로 정렬
    results.sort(key=lambda x: (x['total_envs'], x['num_runners']))
    
    return results, baseline_time, has_gpu

def print_analysis(results, baseline_time, has_gpu):
    """분석 결과 출력"""
    print("\n" + "="*100)
    print("PARALLEL EFFICIENCY ANALYSIS - PPO on HalfCheetah-v5")
    print("="*100)
    print(f"\nBaseline (1r×1e): {baseline_time:.2f}s per iteration\n")
    
    if has_gpu:
        print(f"{'Configuration':<15} {'Total Envs':<12} {'Avg Time/Iter':<16} {'Speedup':<10} {'Efficiency':<12} {'GPU Util':<12} {'VRAM (MB)':<12}")
        print("-"*100)
    else:
        print(f"{'Configuration':<15} {'Total Envs':<12} {'Avg Time/Iter':<16} {'Speedup':<10} {'Efficiency':<12}")
        print("-"*80)
    
    for r in results:
        if has_gpu and r['avg_gpu_util'] is not None:
            print(f"{r['config_str']:<15} {r['total_envs']:<12} "
                  f"{r['avg_time']:<16.2f} {r['speedup']:<10.2f} {r['efficiency']:<11.1f}% "
                  f"{r['avg_gpu_util']:<11.1f}% {r['avg_vram_mb']:<11.0f}")
        else:
            print(f"{r['config_str']:<15} {r['total_envs']:<12} "
                  f"{r['avg_time']:<16.2f} {r['speedup']:<10.2f} {r['efficiency']:<11.1f}%")
    
    # 최고 성능 설정 찾기
    best_speedup = max(results, key=lambda x: x['speedup'])
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    best_time = min(results, key=lambda x: x['avg_time'])
    
    print("\n" + "="*100)
    print("BEST CONFIGURATIONS")
    print("="*100)
    print(f"\n🚀 Best Speedup: {best_speedup['config_str']}")
    print(f"   Speedup: {best_speedup['speedup']:.2f}×")
    print(f"   Time: {best_speedup['avg_time']:.2f}s")
    print(f"   Efficiency: {best_speedup['efficiency']:.1f}%")
    if has_gpu and best_speedup['avg_gpu_util'] is not None:
        print(f"   GPU Util: {best_speedup['avg_gpu_util']:.1f}%")
        print(f"   VRAM: {best_speedup['avg_vram_mb']:.0f}MB")
    
    print(f"\n⚡ Best Efficiency: {best_efficiency['config_str']}")
    print(f"   Efficiency: {best_efficiency['efficiency']:.1f}%")
    print(f"   Speedup: {best_efficiency['speedup']:.2f}×")
    print(f"   Time: {best_efficiency['avg_time']:.2f}s")
    if has_gpu and best_efficiency['avg_gpu_util'] is not None:
        print(f"   GPU Util: {best_efficiency['avg_gpu_util']:.1f}%")
    
    print(f"\n⏱️  Fastest Time: {best_time['config_str']}")
    print(f"   Time: {best_time['avg_time']:.2f}s")
    print(f"   Speedup: {best_time['speedup']:.2f}×")
    print(f"   Efficiency: {best_time['efficiency']:.1f}%")
    if has_gpu and best_time['avg_gpu_util'] is not None:
        print(f"   GPU Util: {best_time['avg_gpu_util']:.1f}%")
        print(f"   VRAM: {best_time['avg_vram_mb']:.0f}MB")
    
    # GPU 병목 분석
    if has_gpu:
        print(f"\n" + "="*100)
        print("GPU BOTTLENECK ANALYSIS")
        print("="*100)
        
        high_gpu = [r for r in results if r['avg_gpu_util'] is not None and r['avg_gpu_util'] > 80]
        low_gpu = [r for r in results if r['avg_gpu_util'] is not None and r['avg_gpu_util'] < 20]
        balanced_gpu = [r for r in results if r['avg_gpu_util'] is not None and 20 <= r['avg_gpu_util'] <= 80]
        
        if high_gpu:
            print(f"\n🔴 HIGH GPU Utilization (>80%) - GPU Bottleneck:")
            for r in high_gpu:
                print(f"   {r['config_str']}: GPU {r['avg_gpu_util']:.1f}%, VRAM {r['avg_vram_mb']:.0f}MB")
        
        if low_gpu:
            print(f"\n🟢 LOW GPU Utilization (<20%) - GPU Underutilized:")
            for r in low_gpu:
                print(f"   {r['config_str']}: GPU {r['avg_gpu_util']:.1f}%")
        
        if balanced_gpu:
            print(f"\n✅ BALANCED GPU Utilization (20-80%):")
            for r in balanced_gpu:
                print(f"   {r['config_str']}: GPU {r['avg_gpu_util']:.1f}%")
    
    print()

def plot_results(results, has_gpu):
    """결과 시각화"""
    print("\nGenerating visualizations...")
    
    # 데이터 추출
    configs = [r['config_str'] for r in results]
    total_envs = [r['total_envs'] for r in results]
    speedups = [r['speedup'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    
    # 리소스 사용량 데이터
    cpu_avgs = [r['cpu_avg'] for r in results]
    cpu_maxs = [r['cpu_max'] for r in results]
    
    # GPU 데이터
    has_gpu_data = any(r['gpu_avg'] is not None for r in results)
    if has_gpu_data:
        gpu_avgs = [r['gpu_avg'] if r['gpu_avg'] is not None else 0 for r in results]
        gpu_maxs = [r['gpu_max'] if r['gpu_max'] is not None else 0 for r in results]
    
    # 차트 개수 결정: 6개 (Speedup, Efficiency, Time, CPU, GPU, RAM)
    num_charts = 6
    fig_width = 30
    
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, 10))
    fig.suptitle('Parallel Efficiency Analysis - PPO on HalfCheetah-v5', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    axes = axes.flatten()  # 2D 배열을 1D로 변환
    
    # 1. Speedup vs Total Environments
    ax1 = axes[0]
    max_envs = max(total_envs)
    ideal_line = list(range(1, max_envs + 1))
    ax1.plot(ideal_line, ideal_line, 'k--', label='Ideal Speedup', linewidth=2, alpha=0.5)
    ax1.scatter(total_envs, speedups, s=200, c=total_envs, 
               cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    for i, config in enumerate(configs):
        ax1.annotate(config, (total_envs[i], speedups[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Total Environments', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Speedup (vs Baseline)', fontsize=11, fontweight='bold')
    ax1.set_title('Strong Scaling: Speedup', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency
    ax2 = axes[1]
    bars = ax2.bar(range(len(results)), efficiencies, color='#FF6B35', alpha=0.7)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax2.axhline(y=50, color='red', linestyle='--', label='50% threshold', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Time per Iteration
    ax3 = axes[2]
    bars = ax3.bar(range(len(results)), avg_times, color='#A23B72', alpha=0.7)
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Time per Iteration', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. CPU Usage (Average)
    ax4 = axes[3]
    bars = ax4.bar(range(len(results)), cpu_avgs, color='#FF6B6B', alpha=0.7, label='Average')
    ax4.scatter(range(len(results)), cpu_maxs, color='#C92A2A', s=100, 
                marker='_', linewidths=3, label='Max', zorder=3)
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('CPU Utilization', fontsize=12, fontweight='bold')
    ax4.axhline(y=100, color='orange', linestyle='--', alpha=0.3, label='1 Core (100%)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (cpu_avg, cpu_max) in enumerate(zip(cpu_avgs, cpu_maxs)):
        ax4.text(i, cpu_avg, f'{cpu_avg:.0f}%', ha='center', va='bottom', fontsize=7)
    
    # 5. GPU Usage (if available)
    ax5 = axes[4]
    if has_gpu_data:
        bars = ax5.bar(range(len(results)), gpu_avgs, color='#50C878', alpha=0.7, label='Average')
        ax5.scatter(range(len(results)), gpu_maxs, color='#2F855A', s=100,
                    marker='_', linewidths=3, label='Max', zorder=3)
        ax5.set_xticks(range(len(results)))
        ax5.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax5.set_ylabel('GPU Utilization (%)', fontsize=11, fontweight='bold')
        ax5.set_title('GPU Utilization', fontsize=12, fontweight='bold')
        ax5.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='High (80%)')
        ax5.axhline(y=20, color='green', linestyle='--', alpha=0.3, label='Low (20%)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        for i, (gpu_avg, gpu_max) in enumerate(zip(gpu_avgs, gpu_maxs)):
            if gpu_avg > 0:
                ax5.text(i, gpu_avg, f'{gpu_avg:.1f}%', ha='center', va='bottom', fontsize=7)
    else:
        ax5.text(0.5, 0.5, 'No GPU Data Available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=14, color='gray')
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title('GPU Utilization', fontsize=12, fontweight='bold')
    
    # 6. SPS (Steps Per Second)
    ax6 = axes[5]
    sps_values = [speedup * (16384 / 11.69) for speedup in speedups]  # 대략적인 SPS 계산
    bars = ax6.bar(range(len(results)), sps_values, color='#4ECDC4', alpha=0.7)
    ax6.set_xticks(range(len(results)))
    ax6.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Steps Per Second', fontsize=11, fontweight='bold')
    ax6.set_title('Throughput (SPS)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, sps in zip(bars, sps_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{sps:.0f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, 'parallel_efficiency_dashboard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Dashboard saved to: {output_file}")
    
    plt.close()

def write_report(results, baseline_time):
    """텍스트 리포트 생성"""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, 'parallel_efficiency_report.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PARALLEL EFFICIENCY EXPERIMENT REPORT\n")
        f.write("PPO on MuJoCo HalfCheetah-v5\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Baseline (1r×1e): {baseline_time:.2f}s per iteration\n\n")
        
        f.write("CONFIGURATION RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Config':<12} {'Total Envs':<12} {'Time/Iter':<15} {'Speedup':<10} {'Efficiency':<12}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            f.write(f"{r['config_str']:<12} {r['total_envs']:<12} "
                   f"{r['avg_time']:<15.2f} {r['speedup']:<10.2f} {r['efficiency']:<11.1f}%\n")
        
        # 최고 성능
        best_speedup = max(results, key=lambda x: x['speedup'])
        best_efficiency = max(results, key=lambda x: x['efficiency'])
        best_time = min(results, key=lambda x: x['avg_time'])
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP PERFORMERS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Best Speedup: {best_speedup['config_str']}\n")
        f.write(f"  Speedup: {best_speedup['speedup']:.2f}×\n")
        f.write(f"  Time: {best_speedup['avg_time']:.2f}s\n")
        f.write(f"  Efficiency: {best_speedup['efficiency']:.1f}%\n\n")
        
        f.write(f"Best Efficiency: {best_efficiency['config_str']}\n")
        f.write(f"  Efficiency: {best_efficiency['efficiency']:.1f}%\n")
        f.write(f"  Speedup: {best_efficiency['speedup']:.2f}×\n\n")
        
        f.write(f"Fastest Time: {best_time['config_str']}\n")
        f.write(f"  Time: {best_time['avg_time']:.2f}s\n")
        f.write(f"  Speedup: {best_time['speedup']:.2f}×\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        if best_speedup['efficiency'] > 50:
            f.write(f"✅ Best configuration ({best_speedup['config_str']}) achieves good efficiency (>{50}%)\n")
            f.write(f"   Recommended for production use.\n\n")
        else:
            f.write(f"⚠️  Best speedup configuration has low efficiency (<50%)\n")
            f.write(f"   Consider using {best_efficiency['config_str']} for better resource utilization.\n\n")
        
        f.write(f"💡 For your hardware (16 cores):\n")
        f.write(f"   - Best absolute performance: {best_speedup['config_str']}\n")
        f.write(f"   - Best resource efficiency: {best_efficiency['config_str']}\n")
        f.write(f"   - Balanced choice: Look for configs with 50%+ efficiency and 3-5× speedup\n")
    
    print(f"✓ Report saved to: {output_file}")

def main():
    print("="*80)
    print("PARALLEL EFFICIENCY ANALYSIS - SIMPLIFIED VERSION")
    print("="*80)
    
    # 결과 로드
    data = load_results()
    print(f"\nLoaded {len(data['experiments'])} experiments")
    
    # 분석
    results, baseline_time, has_gpu = analyze_results(data)
    
    if results:
        # 출력
        print_analysis(results, baseline_time, has_gpu)
        
        # 시각화
        plot_results(results, has_gpu)
        
        # 리포트 작성
        write_report(results, baseline_time)
        
        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("="*80)

if __name__ == '__main__':
    main()
