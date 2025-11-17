# 단일 하이퍼파라미터 실험 테이블

## Clip Parameter Sweep

설정 | clip_param | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
clip=0.10 (conservative) | 0.1 | -335.14 | 24.26 | 0.072
clip=0.20 (baseline) | 0.2 | -307.88 | 23.92 | 0.078
clip=0.30 (aggressive) | 0.3 | -290.94 | 22.29 | 0.077

## Entropy Coefficient Sweep

설정 | entropy_coeff | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
entropy=0.000 (baseline) | 0.0 | -307.88 | 23.92 | 0.078
entropy=0.001 | 0.001 | -313.11 | 16.05 | 0.051
entropy=0.010 | 0.01 | -323.24 | 21.07 | 0.065
entropy=0.050 | 0.05 | -362.69 | 10.60 | 0.029

## Discount Factor Sweep

설정 | gamma | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
gamma=0.95 (short) | 0.95 | -302.64 | 8.69 | 0.029
gamma=0.99 (baseline) | 0.99 | -307.88 | 23.92 | 0.078
gamma=0.995 (long) | 0.995 | -316.59 | 22.38 | 0.071

## Gradient Clipping Sweep

설정 | grad_clip | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
no clip (baseline) | None | -307.88 | 23.92 | 0.078
grad_clip=0.5 | 0.5 | -302.04 | 11.19 | 0.037
grad_clip=1.0 | 1.0 | -317.17 | 25.14 | 0.079

## Value Function Clip Sweep

설정 | vf_clip_param | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
vf_clip=1.0 | 1.0 | -324.25 | 12.84 | 0.040
vf_clip=10.0 (baseline) | 10.0 | -307.88 | 23.92 | 0.078
vf_clip=100.0 | 100.0 | -309.52 | 14.73 | 0.048

## KL Constraint Sweep

설정 | kl_coeff | 최종 보상 평균 | 표준편차 | CV
--- | --- | --- | --- | ---
KL disabled | off | -299.25 | 21.55 | 0.072
kl_coeff=0.2 (baseline) | 0.2 | -307.88 | 23.92 | 0.078
kl_coeff=0.1 | 0.1 | -300.53 | 20.66 | 0.069
kl_coeff=0.5 | 0.5 | -317.73 | 14.95 | 0.047

# 조합 실험 테이블
실험 | 설명 | 최종 보상 평균 | 표준편차
--- | --- | --- | ---
aggressive_exploration | Aggressive exploration with scheduled entropy/clip | -297.34 | 4.06
balanced_high_momentum | Balanced config with high momentum and more workers | -297.66 | 9.82
kl_focused | KL-focused training with stronger penalties | -297.91 | 8.47
baseline_default | Exact baseline hyperparameters (no overrides) | -301.48 | 3.95
baseline_conservative | Baseline (conservative defaults) | -304.29 | 2.67
stable_conservative | Stable conservative setting with weight decay | -331.45 | 8.75