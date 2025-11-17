# PPO 실험 표 & 그래프 개요

- 데이터 출처: `hyperparameter_experiments_final.json`, `combination_experiments_final.json`
- 그래프/테이블 생성 스크립트: `hyperparameter_experiments/analyze_results.py`
- 실행 방법:

```bash
cd /home/kimjihun/reinforce-project4
python hyperparameter_experiments/analyze_results.py
```

스크립트를 실행하면 아래와 동일한 표가 `SUMMARY_TABLES.md`로 저장되고, 단일·조합 실험 그래프가 각각 `hyperparameter_experiments/stability/results/visualizations/summary/`와 `hyperparameter_experiments/combination/results/visualizations/`에 생성된다.

---

## 단일 하이퍼파라미터 실험 (5회 반복 평균)

### Clip Parameter Sweep

설정 | clip_param | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
clip=0.10 (conservative) | 0.1 | -335.14 | 24.26 | 0.072
clip=0.20 (baseline) | 0.2 | -307.88 | 23.92 | 0.078
clip=0.30 (aggressive) | 0.3 | -290.94 | 22.29 | 0.077

### Entropy Coefficient Sweep

설정 | entropy_coeff | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
entropy=0.000 (baseline) | 0.0 | -307.88 | 23.92 | 0.078
entropy=0.001 | 0.001 | -313.11 | 16.05 | 0.051
entropy=0.010 | 0.01 | -323.24 | 21.07 | 0.065
entropy=0.050 | 0.05 | -362.69 | 10.60 | 0.029

### Discount Factor Sweep

설정 | gamma | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
gamma=0.95 (short) | 0.95 | -302.64 | 8.69 | 0.029
gamma=0.99 (baseline) | 0.99 | -307.88 | 23.92 | 0.078
gamma=0.995 (long) | 0.995 | -316.59 | 22.38 | 0.071

### Gradient Clipping Sweep

설정 | grad_clip | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
no clip (baseline) | None | -307.88 | 23.92 | 0.078
grad_clip=0.5 | 0.5 | -302.04 | 11.19 | 0.037
grad_clip=1.0 | 1.0 | -317.17 | 25.14 | 0.079

### Value Function Clip Sweep

설정 | vf_clip_param | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
vf_clip=1.0 | 1.0 | -324.25 | 12.84 | 0.040
vf_clip=10.0 (baseline) | 10.0 | -307.88 | 23.92 | 0.078
vf_clip=100.0 | 100.0 | -309.52 | 14.73 | 0.048

### KL Constraint Sweep

설정 | kl_coeff | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
KL disabled | off | -299.25 | 21.55 | 0.072
kl_coeff=0.2 (baseline) | 0.2 | -307.88 | 23.92 | 0.078
kl_coeff=0.1 | 0.1 | -300.53 | 20.66 | 0.069
kl_coeff=0.5 | 0.5 | -317.73 | 14.95 | 0.047

---

## 조합 실험 (5회 반복 평균)

실험 | 설명 | 최종 보상 평균 | 표준편차
--- | --- | --- | ---
aggressive | Aggressive: high clip_param=0.3 | -250.98 | 38.33
fast_convergence | Fast convergence: standard PPO-Clip without KL | -294.14 | 27.03
balanced | Balanced: standard PPO + minimal entropy | -294.44 | 36.77
adaptive_kl | Adaptive KL: PPO-Lagrange style | -296.18 | 21.70
baseline | Baseline configuration (all defaults) | -305.33 | 16.33
ultra_stable | Maximum stability: conservative clip + gradient clip + KL | -328.91 | 9.91
exploration_focused | High exploration: entropy_coeff=0.05 | -377.93 | 26.13

---

## 그래프 산출물 위치

- 단일 실험 그래프: `hyperparameter_experiments/stability/results/visualizations/summary/*.png`
- 조합 실험 그래프: `hyperparameter_experiments/combination/results/visualizations/combination_final_reward.png`

필요 시 위 그림을 바로 보고서에 삽입하거나, `analyze_results.py`를 수정해 추가 지표(예: SPS, episode_len 등)를 확장할 수 있다.
