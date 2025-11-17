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

# 한글 폰트 설정 (NanumGothic 없으면 DejaVu Sans로 폴백)
try:
    matplotlib.rcParams['font.family'] = 'NanumGothic'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


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
    print("하이퍼파라미터 실험 결과 분석")
    print("="*80)
    print(f"\n베이스라인 성능:")
    print(f"  최종 보상: {baseline_reward:.2f} ± {baseline['statistics']['final_reward_std']:.2f}")
    print(f"  안정성 (CV): {baseline_cv:.4f}")
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
    print("성능 기준 상위 5개 실험 (최종 보상)")
    print("="*80)
    print(f"{'순위':<6} {'실험명':<30} {'보상':<20} {'개선율':<15}")
    print("-"*80)
    for i, exp in enumerate(analysis_data_by_reward[:5], 1):
        reward_str = f"{exp['reward_mean']:.2f} ± {exp['reward_std']:.2f}"
        improvement = f"{exp['reward_improvement']:+.2f}%"
        print(f"{i:<6} {exp['name']:<30} {reward_str:<20} {improvement:<15}")
    
    print("\n" + "="*80)
    print("안정성 기준 상위 5개 실험 (낮은 CV)")
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

    experiments = results['experiments']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 성능 vs 안정성 산점도
    fig, ax = plt.subplots(figsize=(12, 8))

    rewards = [exp['reward_mean'] for exp in analysis_data]
    cvs = [exp['reward_cv'] for exp in analysis_data]

    baseline_stats = baseline['statistics']
    ax.scatter(
        baseline_stats['final_reward_mean'],
        baseline_stats['final_reward_cv'],
        s=200,
        c='red',
        marker='*',
        label='Baseline',
        zorder=3,
    )

    scatter = ax.scatter(rewards, cvs, s=100, alpha=0.6, c=rewards, cmap='viridis')

    top_5 = sorted(analysis_data, key=lambda x: x['reward_mean'], reverse=True)[:5]
    for exp in top_5:
        ax.annotate(
            exp['name'],
            (exp['reward_mean'], exp['reward_cv']),
            fontsize=7,
            alpha=0.7,
            ha='left',
        )

    ax.set_xlabel('Final Reward Mean (Performance) →', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (Instability) →', fontsize=11, fontweight='bold')
    ax.set_title('성능 vs 안정성 트레이드오프', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Reward Mean')
    plt.tight_layout()
    perf_path = output_dir / 'performance_vs_stability.png'
    plt.savefig(perf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {perf_path}")

    # 2. 베이스라인 대비 개선율 비교
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    sorted_by_reward = sorted(analysis_data, key=lambda x: x['reward_improvement'], reverse=True)
    names = [exp['name'] for exp in sorted_by_reward[:15]]
    reward_improvements = [exp['reward_improvement'] for exp in sorted_by_reward[:15]]

    bars = ax1.barh(range(len(names)), reward_improvements)
    for idx, bar in enumerate(bars):
        bar.set_color('#2ECC71' if reward_improvements[idx] > 0 else '#E74C3C')

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Performance Improvement vs Baseline (%)', fontsize=10, fontweight='bold')
    ax1.set_title('상위 15개: 성능 개선율', fontsize=11, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='x')

    sorted_by_stability = sorted(
        analysis_data,
        key=lambda x: x['stability_improvement'],
        reverse=True,
    )
    names_s = [exp['name'] for exp in sorted_by_stability[:15]]
    stability_improvements = [exp['stability_improvement'] for exp in sorted_by_stability[:15]]

    bars = ax2.barh(range(len(names_s)), stability_improvements)
    for idx, bar in enumerate(bars):
        bar.set_color('#3498DB' if stability_improvements[idx] > 0 else '#E67E22')

    ax2.set_yticks(range(len(names_s)))
    ax2.set_yticklabels(names_s, fontsize=8)
    ax2.set_xlabel('Stability Improvement vs Baseline (%)', fontsize=10, fontweight='bold')
    ax2.set_title('상위 15개: 안정성 개선율 (낮은 CV)', fontsize=11, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    improvement_path = output_dir / 'improvements_comparison.png'
    plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {improvement_path}")

    # 3. 파라미터 그룹별 효과 (통합 박스플롯)
    param_groups = {
        'Clipping': ['clip_conservative', 'clip_aggressive'],
        'Entropy': ['entropy_minimal', 'entropy_medium', 'entropy_high'],
        'Gamma': ['gamma_short', 'gamma_long'],
        'Grad Clip': ['grad_clip_tight', 'grad_clip_loose'],
        'VF Clip': ['vf_clip_tight', 'vf_clip_loose'],
        'KL Loss': ['kl_disabled', 'kl_weak', 'kl_strong'],
        'GAE': ['no_gae'],
    }

    group_colors = {
        'Clipping': '#3498DB',
        'Entropy': '#2ECC71',
        'Gamma': '#9B59B6',
        'Grad Clip': '#F39C12',
        'VF Clip': '#1ABC9C',
        'KL Loss': '#E67E22',
        'GAE': '#95A5A6',
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    all_box_data = []
    all_labels = []
    all_colors = []

    baseline_rewards = [trial['iterations'][-1]['episode_reward_mean'] for trial in baseline['trials']]
    all_box_data.append(baseline_rewards)
    all_labels.append('Baseline')
    all_colors.append('#E74C3C')

    for group_name, exp_names in param_groups.items():
        for exp_name in exp_names:
            exp = next((e for e in experiments if e['name'] == exp_name), None)
            if not exp:
                continue
            trial_rewards = [trial['iterations'][-1]['episode_reward_mean'] for trial in exp['trials']]
            all_box_data.append(trial_rewards)
            all_labels.append(exp_name.replace('_', ' ').title())
            all_colors.append(group_colors[group_name])

    bp = ax.boxplot(
        all_box_data,
        tick_labels=all_labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=5),
    )

    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Reward', fontsize=12, fontweight='bold')
    ax.set_title('하이퍼파라미터 그룹별 성능 분포 비교', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(
        y=baseline['statistics']['final_reward_mean'],
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.5,
        label='Baseline Mean',
    )
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.legend(loc='lower right')
    plt.tight_layout()

    group_path = output_dir / 'parameter_group_effects.png'
    plt.savefig(group_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {group_path}")

    # 4. 학습 곡선 비교 (베이스라인 + 상위 3개)
    top_3 = sorted(analysis_data, key=lambda x: x['reward_mean'], reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(12, 7))

    for trial in baseline['trials']:
        rewards = [it['episode_reward_mean'] for it in trial['iterations']]
        iterations = list(range(1, len(rewards) + 1))
        ax.plot(iterations, rewards, color='red', alpha=0.3, linewidth=1)

    baseline_mean = []
    for idx in range(len(baseline['trials'][0]['iterations'])):
        iter_rewards = [trial['iterations'][idx]['episode_reward_mean'] for trial in baseline['trials']]
        baseline_mean.append(np.mean(iter_rewards))
    ax.plot(range(1, len(baseline_mean) + 1), baseline_mean, color='red', linewidth=2.5, label='Baseline (avg)', marker='o')

    colors = ['#2ECC71', '#3498DB', '#9B59B6']
    for color, exp_data in zip(colors, top_3):
        exp_full = next(x for x in experiments if x['name'] == exp_data['name'])
        for trial in exp_full['trials']:
            rewards = [it['episode_reward_mean'] for it in trial['iterations']]
            iterations = list(range(1, len(rewards) + 1))
            ax.plot(iterations, rewards, color=color, alpha=0.2, linewidth=1)

        avg_rewards = []
        for idx in range(len(exp_full['trials'][0]['iterations'])):
            iter_rewards = [trial['iterations'][idx]['episode_reward_mean'] for trial in exp_full['trials']]
            avg_rewards.append(np.mean(iter_rewards))
        ax.plot(range(1, len(avg_rewards) + 1), avg_rewards, color=color, linewidth=2.5, label=f"{exp_data['name']} (avg)", marker='s')

    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Reward Mean', fontsize=11, fontweight='bold')
    ax.set_title('학습 곡선: 베이스라인 vs 상위 3개 실험', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    curve_path = output_dir / 'learning_curves_comparison.png'
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {curve_path}")

    print(f"\nAll visualizations saved to: {output_dir}")


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
            f.write(f"{key}: {value}\n")
        f.write("```\n\n")
        
        baseline_stats = baseline['statistics']
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
    results_dir = RESULTS_DIR
    final_file = results_dir / 'hyperparameter_experiments_final.json'
    progress_file = results_dir / 'hyperparameter_experiments_progress.json'
    
    # 파일 존재 확인
    if final_file.exists():
        results_file = final_file
        print(f"Loading final results from: {results_file}")
    elif progress_file.exists():
        results_file = progress_file
        print(f"Loading progress results from: {results_file}")
        print("Warning: Using progress file (experiments may be incomplete)")
    else:
        print(f"Error: No results file found in {results_dir}")
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
    output_dir = results_dir / 'visualizations'
    create_visualizations(results, analysis_data, baseline, output_dir)
    
    # 보고서 생성
    report_file = results_dir / 'HYPERPARAMETER_ANALYSIS_REPORT.md'
    generate_report(results, analysis_data, baseline, report_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Visualizations: {output_dir}")
    print(f"Report: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()
