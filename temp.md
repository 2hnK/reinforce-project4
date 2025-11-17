실증 근거 기반으로 **단일 파라미터 변화 실험**을 제안드립니다. 조합보다는 각 파라미터의 독립적 효과를 먼저 파악하는 것이 과학적으로 타당합니다.

## 📋 **실험 설계 원칙**

1. **한 번에 하나씩 변경** (Single-variable experiments)
2. **실증 연구에서 검증된 범위** 사용
3. **총 실험 수 최소화** (시간 효율성)

---

## 🎯 **제안 실험 그룹**

### **Group 1: Clipping 메커니즘 (3개 실험)**

PPO 논문에서 ε=0.2가 최적 균형을 제공하지만, 더 작은 값(안정성)과 큰 값(빠른 학습)의 트레이드오프가 존재

```python
# Experiment 1-1: 보수적 클리핑
{
    'clip_param': 0.1,  # 기본값(0.2)의 절반
}

# Experiment 1-2: Baseline
{
    'clip_param': 0.2,  # PPO 논문 기본값
}

# Experiment 1-3: 공격적 클리핑
{
    'clip_param': 0.3,  # 더 큰 업데이트 허용
}
```

**예상 결과**:
- 0.1: 높은 안정성(낮은 분산), 느린 수렴
- 0.3: 빠른 초기 학습, 높은 분산

---

### **Group 2: Entropy Regularization (4개 실험)**

HalfCheetah에서 entropy 효과는 Hopper/Walker보다 덜 명확하지만, 다양한 표준편차 실험이 수행됨

```python
# Experiment 2-1: Entropy 없음
{
    'entropy_coeff': 0.0,  # 탐험 최소화
}

# Experiment 2-2: 최소 Entropy
{
    'entropy_coeff': 0.001,  # 약간의 탐험
}

# Experiment 2-3: 중간 Entropy (Baseline)
{
    'entropy_coeff': 0.01,  # RLlib 일반적 기본값
}

# Experiment 2-4: 높은 Entropy
{
    'entropy_coeff': 0.05,  # 강한 탐험 유도
}
```

**예상 결과**:
- 0.0: 초기 빠른 수렴, 지역 최적해 위험
- 0.05: 느린 수렴, 더 나은 최종 성능 가능성

---

### **Group 3: Discount Factor (3개 실험)**

Discount factor γ는 가장 중요한 하이퍼파라미터이며 환경별 튜닝이 필요함

```python
# Experiment 3-1: 단기 보상 중심
{
    'gamma': 0.95,  # 더 짧은 시간 지평
}

# Experiment 3-2: Baseline
{
    'gamma': 0.99,  # 표준값
}

# Experiment 3-3: 장기 보상 중심
{
    'gamma': 0.995,  # 더 긴 시간 지평
}
```

**예상 결과**:
- 0.95: HalfCheetah는 단기 보상(거리)이 명확하므로 효과적일 수 있음
- 0.995: 과도하게 먼 미래 고려로 학습 불안정 가능

---

### **Group 4: Gradient Clipping (3개 실험)**

안정성 확보를 위한 기법

```python
# Experiment 4-1: Gradient Clipping 없음
{
    'grad_clip': None,  # 제약 없음
}

# Experiment 4-2: 적당한 Clipping
{
    'grad_clip': 0.5,  # 일반적 권장값
}

# Experiment 4-3: 강한 Clipping
{
    'grad_clip': 1.0,  # 더 넓은 허용 범위
}
```

**예상 결과**:
- None: 빠른 학습, 발산 위험
- 0.5: 안정적 학습, 약간 느린 수렴

---

### **Group 5: Value Function Clipping (3개 실험)**

```python
# Experiment 5-1: 강한 VF 제약
{
    'vf_clip_param': 1.0,  # 작은 값 = 강한 제약
}

# Experiment 5-2: Baseline
{
    'vf_clip_param': 10.0,  # RLlib 기본값
}

# Experiment 5-3: VF 제약 거의 없음
{
    'vf_clip_param': 100.0,  # 큰 값 = 약한 제약
}
```

**예상 결과**:
- 1.0: 가치 함수 업데이트 보수적, 안정성 증가
- 100.0: 빠른 가치 학습, 불안정 가능

---

### **Group 6: KL Divergence Constraint (2개 실험)**

PPO 논문에서 KL penalty가 clipping보다 성능이 낮았지만, 추가 안정성 제공 가능

```python
# Experiment 6-1: KL Loss 비활성화 (Baseline)
{
    'use_kl_loss': False,  # PPO-Clip만 사용
}

# Experiment 6-2: KL Loss 활성화
{
    'use_kl_loss': True,
    'kl_coeff': 0.2,  # 초기 계수
    'kl_target': 0.01,  # 목표 KL divergence
}
```

**예상 결과**:
- KL Loss 활성화: 더 안정적이지만 느린 학습

---

## 📊 **최종 실험 구성 요약**

| Group | 파라미터 | 실험 수 | 총 실행 (×5회) |
|-------|----------|---------|----------------|
| 1 | clip_param | 3 | 15 |
| 2 | entropy_coeff | 4 | 20 |
| 3 | gamma | 3 | 15 |
| 4 | grad_clip | 3 | 15 |
| 5 | vf_clip_param | 3 | 15 |
| 6 | use_kl_loss | 2 | 10 |
| **합계** | - | **18** | **90** |

---

## 🎯 **우선순위가 높은 실험 (시간 제약 시)**

가장 영향력이 큰 파라미터만 선택:

### **Minimal Set (10개 실험, 50회 실행)**

```python
experiments = [
    # 1. Gamma (가장 중요)
    {'gamma': 0.95},
    {'gamma': 0.99},  # Baseline
    {'gamma': 0.995},
    
    # 2. Clip param (PPO 핵심)
    {'clip_param': 0.1},
    {'clip_param': 0.2},  # Baseline
    {'clip_param': 0.3},
    
    # 3. Entropy (탐험-활용 균형)
    {'entropy_coeff': 0.0},
    {'entropy_coeff': 0.01},  # Baseline
    {'entropy_coeff': 0.05},
    
    # 4. Gradient Clipping (안정성)
    {'grad_clip': 0.5},
]
```

---

## 💡 **실험 코드 예시**

```python
# experiments.py
EXPERIMENTS = {
    # Group 1: Clipping
    "clip_conservative": {'clip_param': 0.1},
    "clip_baseline": {'clip_param': 0.2},
    "clip_aggressive": {'clip_param': 0.3},
    
    # Group 2: Entropy
    "entropy_none": {'entropy_coeff': 0.0},
    "entropy_low": {'entropy_coeff': 0.001},
    "entropy_medium": {'entropy_coeff': 0.01},
    "entropy_high": {'entropy_coeff': 0.05},
    
    # Group 3: Gamma
    "gamma_short": {'gamma': 0.95},
    "gamma_standard": {'gamma': 0.99},
    "gamma_long": {'gamma': 0.995},
    
    # Group 4: Gradient Clipping
    "grad_clip_none": {'grad_clip': None},
    "grad_clip_tight": {'grad_clip': 0.5},
    "grad_clip_loose": {'grad_clip': 1.0},
    
    # Group 5: VF Clipping
    "vf_clip_tight": {'vf_clip_param': 1.0},
    "vf_clip_standard": {'vf_clip_param': 10.0},
    "vf_clip_loose": {'vf_clip_param': 100.0},
    
    # Group 6: KL Loss
    "kl_disabled": {'use_kl_loss': False},
    "kl_enabled": {
        'use_kl_loss': True,
        'kl_coeff': 0.2,
        'kl_target': 0.01
    },
}
```

---

## 📈 **분석 방법**

각 실험 후:
1. **성능**: 5회 평균 최종 reward
2. **안정성**: 5회 표준편차 또는 변동계수(CV)
3. **수렴 속도**: 목표 성능(예: 2000 reward) 도달 시간

**통계적 유의성 검정**:
- Baseline 대비 t-test (p<0.05)
- 효과 크기(Cohen's d) 계산

이렇게 하면 **실증적 근거**를 확보하면서도 **실험 수를 관리 가능한 수준**으로 유지할 수 있습니다!