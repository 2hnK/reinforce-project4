"""
20227128 김지훈

하이퍼파라미터 실험 결과 분석 및 시각화

분석 내용:
    1. 성능 비교 (평균 보상)
    2. 안정성 비교 (표준편차, CV)
    3. 효율성 비교 (SPS, 학습 시간)
    4. 파라미터별 영향 분석
    5. 베이스라인 대비 개선율
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 9


def load_results(file_path):
    """결과 파일 로드"""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_results(results):
    """결과 분석"""
    
    experiments = results['experiments']
    baseline = None
    
    # 베이스라인 찾기
    for exp in experiments:
        if exp['name'] == 'baseline':
            baseline = exp
            break
    
    if not baseline:
        print("Warning: Baseline experiment not found!")
        return
    
    baseline_reward = baseline['statistics']['final_reward_mean']
    baseline_cv = baseline['statistics']['final_reward_cv']
    baseline_sps = baseline['statistics']['sps_mean']
    
    print("="*80)
    print("HYPERPARAMETER EXPERIMENT ANALYSIS")
    print("="*80)
    print(f"\nBaseline Performance:")
    print(f"  Final Reward: {baseline_reward:.2f} ± {baseline['statistics']['final_reward_std']:.2f}")
    print(f"  Stability (CV): {baseline_cv:.4f}")
    print(f"  SPS: {baseline_sps:.2f}")
    
    # 분석 데이터 수집
    analysis_data = []
    
    for exp in experiments:
        if exp['name'] == 'baseline':
            continue
        
        stats = exp['statistics']
        
        # 베이스라인 대비 개선율
        reward_improvement = ((stats['final_reward_mean'] - baseline_reward) / abs(baseline_reward)) * 100
        stability_improvement = ((baseline_cv - stats['final_reward_cv']) / baseline_cv) * 100  # CV가 낮을수록 좋음
        sps_improvement = ((stats['sps_mean'] - baseline_sps) / baseline_sps) * 100
        
        analysis_data.append({
            'name': exp['name'],
            'description': exp['description'],
            'reward_mean': stats['final_reward_mean'],
            'reward_std': stats['final_reward_std'],
            'reward_cv': stats['final_reward_cv'],
            'sps_mean': stats['sps_mean'],
            'time_mean': stats['time_mean'],
            'reward_improvement': reward_improvement,
            'stability_improvement': stability_improvement,
            'sps_improvement': sps_improvement,
        })
    
    # 정렬
    analysis_data_by_reward = sorted(analysis_data, key=lambda x: x['reward_mean'], reverse=True)
    analysis_data_by_stability = sorted(analysis_data, key=lambda x: x['reward_cv'])
    
    # 상위 실험 출력
    print("\n" + "="*80)
    print("TOP 5 BY PERFORMANCE (Final Reward)")
    print("="*80)
    print(f"{'Rank':<6} {'Experiment':<30} {'Reward':<20} {'vs Baseline':<15}")
    print("-"*80)
    for i, exp in enumerate(analysis_data_by_reward[:5], 1):
        reward_str = f"{exp['reward_mean']:.2f} ± {exp['reward_std']:.2f}"
        improvement = f"{exp['reward_improvement']:+.2f}%"
        print(f"{i:<6} {exp['name']:<30} {reward_str:<20} {improvement:<15}")
    
    print("\n" + "="*80)
    print("TOP 5 BY STABILITY (Lowest CV)")
    print("="*80)
    print(f"{'Rank':<6} {'Experiment':<30} {'CV':<15} {'vs Baseline':<15}")
    print("-"*80)
    for i, exp in enumerate(analysis_data_by_stability[:5], 1):
        cv_str = f"{exp['reward_cv']:.4f}"
        improvement = f"{exp['stability_improvement']:+.2f}%"
        print(f"{i:<6} {exp['name']:<30} {cv_str:<15} {improvement:<15}")
    
    # 파라미터 그룹별 분석
    print("\n" + "="*80)
    print("PARAMETER GROUP ANALYSIS")
    print("="*80)
    
    param_groups = {
        'Learning Rate': ['lr_0.0001', 'lr_0.001'],
        'GAE Lambda': ['lambda_0.9', 'lambda_0.99'],
        'Epochs': ['epochs_10', 'epochs_20'],
        'Batch Size': ['batch_8192', 'batch_32768'],
        'Network Size': ['network_small', 'network_large', 'network_xlarge'],
        'VF Coefficient': ['vf_coeff_0.005', 'vf_coeff_0.05', 'vf_coeff_0.1'],
        'Clip Parameter': ['clip_0.1', 'clip_0.3'],
        'Entropy': ['entropy_0.001', 'entropy_0.01', 'entropy_0.1'],
        'Combined': ['fast_convergence', 'stable_learning', 'large_capacity'],
    }
    
    for group_name, exp_names in param_groups.items():
        print(f"\n{group_name}:")
        for exp_name in exp_names:
            exp_data = next((x for x in analysis_data if x['name'] == exp_name), None)
            if exp_data:
                print(f"  • {exp_name:<25} Reward: {exp_data['reward_mean']:>7.2f} "
                      f"({exp_data['reward_improvement']:+6.2f}%), "
                      f"CV: {exp_data['reward_cv']:.4f} "
                      f"({exp_data['stability_improvement']:+6.2f}%)")
    
    return analysis_data, baseline


def create_visualizations(results, analysis_data, baseline, output_dir):
    """시각화 생성"""
    
    # 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 성능 vs 안정성 산점도
    fig, ax = plt.subplots(figsize=(12, 8))
    
    rewards = [exp['reward_mean'] for exp in analysis_data]
    cvs = [exp['reward_cv'] for exp in analysis_data]
    names = [exp['name'] for exp in analysis_data]
    
    # 베이스라인
    baseline_stats = baseline['statistics']
    ax.scatter(baseline_stats['final_reward_mean'], baseline_stats['final_reward_cv'],
               s=200, c='red', marker='*', label='Baseline', zorder=3)
    
    # 나머지 실험
    scatter = ax.scatter(rewards, cvs, s=100, alpha=0.6, c=rewards, cmap='viridis')
    
    # 라벨 추가 (상위/하위만)
    top_5_by_reward = sorted(analysis_data, key=lambda x: x['reward_mean'], reverse=True)[:5]
    for exp in top_5_by_reward:
        ax.annotate(exp['name'], (exp['reward_mean'], exp['reward_cv']),
                   fontsize=7, alpha=0.7, ha='left')
    
    ax.set_xlabel('Final Reward Mean (Performance) →', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (Instability) →', fontsize=11, fontweight='bold')
    ax.set_title('Performance vs Stability Trade-off', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Reward Mean')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_vs_stability.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/performance_vs_stability.png")
    plt.close()
    
    # 2. 베이스라인 대비 개선율 비교
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 정렬
    sorted_by_reward = sorted(analysis_data, key=lambda x: x['reward_improvement'], reverse=True)
    names = [exp['name'] for exp in sorted_by_reward]
    reward_improvements = [exp['reward_improvement'] for exp in sorted_by_reward]
    
    # 상위 15개만
    names = names[:15]
    reward_improvements = reward_improvements[:15]
    
    bars = ax1.barh(range(len(names)), reward_improvements)
    for i, bar in enumerate(bars):
        if reward_improvements[i] > 0:
            bar.set_color('#2ECC71')
        else:
            bar.set_color('#E74C3C')
    
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Performance Improvement vs Baseline (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Top 15: Performance Improvement', fontsize=11, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 안정성 개선
    sorted_by_stability = sorted(analysis_data, key=lambda x: x['stability_improvement'], reverse=True)
    names_s = [exp['name'] for exp in sorted_by_stability][:15]
    stability_improvements = [exp['stability_improvement'] for exp in sorted_by_stability][:15]
    
    bars = ax2.barh(range(len(names_s)), stability_improvements)
    for i, bar in enumerate(bars):
        if stability_improvements[i] > 0:
            bar.set_color('#3498DB')
        else:
            bar.set_color('#E67E22')
    
    ax2.set_yticks(range(len(names_s)))
    ax2.set_yticklabels(names_s, fontsize=8)
    ax2.set_xlabel('Stability Improvement vs Baseline (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Top 15: Stability Improvement (Lower CV)', fontsize=11, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvements_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/improvements_comparison.png")
    plt.close()
    
    # 3. 파라미터 그룹별 효과
    param_groups = {
        'Learning Rate': ['lr_0.0001', 'baseline', 'lr_0.001'],
        'GAE Lambda': ['lambda_0.9', 'baseline', 'lambda_0.99'],
        'Epochs': ['epochs_10', 'baseline', 'epochs_20'],
        'Batch Size': ['batch_8192', 'baseline', 'batch_32768'],
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (group_name, exp_names) in enumerate(param_groups.items()):
        ax = axes[idx]
        
        group_rewards = []
        group_stds = []
        labels = []
        
        for exp_name in exp_names:
            if exp_name == 'baseline':
                stats = baseline['statistics']
                group_rewards.append(stats['final_reward_mean'])
                group_stds.append(stats['final_reward_std'])
                labels.append('baseline')
            else:
                exp_data = next((x for x in analysis_data if x['name'] == exp_name), None)
                if exp_data:
                    group_rewards.append(exp_data['reward_mean'])
                    group_stds.append(exp_data['reward_std'])
                    labels.append(exp_name.replace(group_name.lower().replace(' ', '_') + '_', ''))
        
        x = range(len(labels))
        bars = ax.bar(x, group_rewards, yerr=group_stds, capsize=5, alpha=0.7)
        
        # 베이스라인 강조
        for i, label in enumerate(labels):
            if label == 'baseline':
                bars[i].set_color('#E74C3C')
            else:
                bars[i].set_color('#3498DB')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Final Reward', fontsize=9)
        ax.set_title(group_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_group_effects.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/parameter_group_effects.png")
    plt.close()
    
    # 4. 학습 곡선 비교 (베이스라인 + 상위 3개)
    top_3 = sorted(analysis_data, key=lambda x: x['reward_mean'], reverse=True)[:3]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 베이스라인
    for trial in baseline['trials']:
        rewards = [it['episode_reward_mean'] for it in trial['iterations']]
        iterations = list(range(1, len(rewards) + 1))
        ax.plot(iterations, rewards, color='red', alpha=0.3, linewidth=1)
    
    # 평균
    all_baseline_rewards = []
    for i in range(len(baseline['trials'][0]['iterations'])):
        iter_rewards = [trial['iterations'][i]['episode_reward_mean'] 
                       for trial in baseline['trials']]
        all_baseline_rewards.append(np.mean(iter_rewards))
    ax.plot(range(1, len(all_baseline_rewards) + 1), all_baseline_rewards, 
           color='red', linewidth=2.5, label='Baseline (avg)', marker='o')
    
    # 상위 3개
    colors = ['#2ECC71', '#3498DB', '#9B59B6']
    for i, exp_data in enumerate(top_3):
        exp_name = exp_data['name']
        exp_full = next(x for x in results['experiments'] if x['name'] == exp_name)
        
        # 모든 trial
        for trial in exp_full['trials']:
            rewards = [it['episode_reward_mean'] for it in trial['iterations']]
            iterations = list(range(1, len(rewards) + 1))
            ax.plot(iterations, rewards, color=colors[i], alpha=0.2, linewidth=1)
        
        # 평균
        all_rewards = []
        for j in range(len(exp_full['trials'][0]['iterations'])):
            iter_rewards = [trial['iterations'][j]['episode_reward_mean'] 
                           for trial in exp_full['trials']]
            all_rewards.append(np.mean(iter_rewards))
        ax.plot(range(1, len(all_rewards) + 1), all_rewards, 
               color=colors[i], linewidth=2.5, label=f'{exp_name} (avg)', marker='s')
    
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Reward Mean', fontsize=11, fontweight='bold')
    ax.set_title('Learning Curves: Baseline vs Top 3 Experiments', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/learning_curves_comparison.png")
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def generate_report(results, analysis_data, baseline, output_file):
    """마크다운 보고서 생성"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Hyperparameter Stability and Performance Analysis Report\n\n")
        f.write("**Student ID:** 20227128 김지훈\n\n")
        f.write(f"**Date:** {results['timestamp']}\n\n")
        f.write("---\n\n")
        
        # 실험 개요
        f.write("## 1. Experiment Overview\n\n")
        f.write(f"- **Total Experiments:** {len(results['experiments'])}\n")
        f.write(f"- **Trials per Experiment:** 5\n")
        f.write(f"- **Iterations per Trial:** 10\n")
        f.write(f"- **Environment:** HalfCheetah-v5 (MuJoCo)\n")
        f.write(f"- **Algorithm:** PPO (Proximal Policy Optimization)\n\n")
        
        # 베이스라인
        f.write("## 2. Baseline Configuration\n\n")
        f.write("```python\n")
        for key, value in results['baseline_config'].items():
            f.write(f"{key} = {value}\n")
        f.write("```\n\n")
        
        baseline_stats = baseline['statistics']
        f.write("**Baseline Performance:**\n")
        f.write(f"- Final Reward: {baseline_stats['final_reward_mean']:.2f} ± {baseline_stats['final_reward_std']:.2f}\n")
        f.write(f"- Coefficient of Variation: {baseline_stats['final_reward_cv']:.4f}\n")
        f.write(f"- SPS: {baseline_stats['sps_mean']:.2f}\n\n")
        
        # 상위 실험
        f.write("## 3. Top Performing Experiments\n\n")
        f.write("### 3.1 By Performance (Final Reward)\n\n")
        f.write("| Rank | Experiment | Final Reward | vs Baseline | CV |\n")
        f.write("|------|------------|--------------|-------------|----|\n")
        
        sorted_by_reward = sorted(analysis_data, key=lambda x: x['reward_mean'], reverse=True)
        for i, exp in enumerate(sorted_by_reward[:10], 1):
            f.write(f"| {i} | {exp['name']} | {exp['reward_mean']:.2f} ± {exp['reward_std']:.2f} | "
                   f"{exp['reward_improvement']:+.2f}% | {exp['reward_cv']:.4f} |\n")
        
        f.write("\n### 3.2 By Stability (Lowest CV)\n\n")
        f.write("| Rank | Experiment | CV | vs Baseline | Final Reward |\n")
        f.write("|------|------------|-------|-------------|---------------|\n")
        
        sorted_by_stability = sorted(analysis_data, key=lambda x: x['reward_cv'])
        for i, exp in enumerate(sorted_by_stability[:10], 1):
            f.write(f"| {i} | {exp['name']} | {exp['reward_cv']:.4f} | "
                   f"{exp['stability_improvement']:+.2f}% | {exp['reward_mean']:.2f} ± {exp['reward_std']:.2f} |\n")
        
        # 파라미터 분석
        f.write("\n## 4. Parameter Analysis\n\n")
        
        param_groups = {
            'Learning Rate': ['lr_0.0001', 'lr_0.001'],
            'GAE Lambda': ['lambda_0.9', 'lambda_0.99'],
            'Epochs': ['epochs_10', 'epochs_20'],
            'Batch Size': ['batch_8192', 'batch_32768'],
            'Network Size': ['network_small', 'network_large', 'network_xlarge'],
            'VF Coefficient': ['vf_coeff_0.005', 'vf_coeff_0.05', 'vf_coeff_0.1'],
            'Clip Parameter': ['clip_0.1', 'clip_0.3'],
            'Entropy': ['entropy_0.001', 'entropy_0.01', 'entropy_0.1'],
        }
        
        for group_name, exp_names in param_groups.items():
            f.write(f"\n### 4.{list(param_groups.keys()).index(group_name) + 1} {group_name}\n\n")
            f.write("| Experiment | Final Reward | vs Baseline | CV | vs Baseline |\n")
            f.write("|------------|--------------|-------------|----|--------------|\n")
            
            for exp_name in exp_names:
                exp_data = next((x for x in analysis_data if x['name'] == exp_name), None)
                if exp_data:
                    f.write(f"| {exp_name} | {exp_data['reward_mean']:.2f} ± {exp_data['reward_std']:.2f} | "
                           f"{exp_data['reward_improvement']:+.2f}% | {exp_data['reward_cv']:.4f} | "
                           f"{exp_data['stability_improvement']:+.2f}% |\n")
        
        # 조합 실험
        f.write("\n## 5. Combined Configuration Experiments\n\n")
        combined_exps = ['fast_convergence', 'stable_learning', 'large_capacity']
        f.write("| Experiment | Description | Final Reward | CV | SPS |\n")
        f.write("|------------|-------------|--------------|----|\n")
        
        for exp_name in combined_exps:
            exp_data = next((x for x in analysis_data if x['name'] == exp_name), None)
            if exp_data:
                exp_full = next(x for x in results['experiments'] if x['name'] == exp_name)
                f.write(f"| {exp_name} | {exp_data['description']} | "
                       f"{exp_data['reward_mean']:.2f} ± {exp_data['reward_std']:.2f} | "
                       f"{exp_data['reward_cv']:.4f} | {exp_data['sps_mean']:.2f} |\n")
        
        # 결론
        f.write("\n## 6. Key Findings\n\n")
        
        best_perf = sorted_by_reward[0]
        best_stable = sorted_by_stability[0]
        
        f.write(f"### Best Performance\n")
        f.write(f"- **Experiment:** {best_perf['name']}\n")
        f.write(f"- **Final Reward:** {best_perf['reward_mean']:.2f} ± {best_perf['reward_std']:.2f}\n")
        f.write(f"- **Improvement:** {best_perf['reward_improvement']:+.2f}%\n\n")
        
        f.write(f"### Most Stable\n")
        f.write(f"- **Experiment:** {best_stable['name']}\n")
        f.write(f"- **CV:** {best_stable['reward_cv']:.4f}\n")
        f.write(f"- **Stability Improvement:** {best_stable['stability_improvement']:+.2f}%\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("Based on the experiments:\n\n")
        f.write("1. **For Maximum Performance:** Use the configuration that achieved the highest reward\n")
        f.write("2. **For Stable Learning:** Use the configuration with the lowest CV\n")
        f.write("3. **Parameter Insights:**\n")
        f.write("   - Review parameter group analysis section for specific parameter effects\n")
        f.write("   - Consider trade-offs between performance and stability\n")
        f.write("   - Evaluate computational efficiency (SPS) for production use\n\n")
        
        f.write("---\n\n")
        f.write("## 7. Visualizations\n\n")
        f.write("See the following generated plots:\n")
        f.write("- `performance_vs_stability.png`: Scatter plot of all experiments\n")
        f.write("- `improvements_comparison.png`: Bar charts of improvements vs baseline\n")
        f.write("- `parameter_group_effects.png`: Parameter-specific effect analysis\n")
        f.write("- `learning_curves_comparison.png`: Learning curves over iterations\n\n")
    
    print(f"\nReport saved to: {output_file}")


def main():
    """메인 분석 함수"""
    
    # 결과 파일 찾기
    results_dir = 'hyperparameter_experiments/results'
    final_file = f'{results_dir}/hyperparameter_experiments_final.json'
    progress_file = f'{results_dir}/hyperparameter_experiments_progress.json'
    
    # 파일 존재 확인
    if Path(final_file).exists():
        results_file = final_file
        print(f"Loading final results from: {results_file}")
    elif Path(progress_file).exists():
        results_file = progress_file
        print(f"Loading progress results from: {results_file}")
        print("Warning: Using progress file (experiments may be incomplete)")
    else:
        print(f"Error: No results file found in {results_dir}/")
        print("Please run hyperparameter_stability_experiment.py first.")
        return
    
    # 결과 로드
    results = load_results(results_file)
    
    # 분석 수행
    analysis_data, baseline = analyze_results(results)
    
    if not analysis_data or not baseline:
        print("Error: Analysis failed")
        return
    
    # 시각화 생성
    output_dir = f'{results_dir}/visualizations'
    create_visualizations(results, analysis_data, baseline, output_dir)
    
    # 보고서 생성
    report_file = f'{results_dir}/HYPERPARAMETER_ANALYSIS_REPORT.md'
    generate_report(results, analysis_data, baseline, report_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results directory: {results_dir}/")
    print(f"Visualizations: {output_dir}/")
    print(f"Report: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()
