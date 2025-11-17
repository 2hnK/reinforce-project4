# RLlib 파라미터 실증 근거 가이드

**Student ID:** 20227128 김지훈  
**목적:** PPO 하이퍼파라미터의 독립적 효과 측정 및 학습 안정성 분석

---

## 📋 실험 설계 원칙

### 1. **단일 파라미터 변화 (Single-Variable Experiments)**
- 한 번에 하나의 파라미터만 변경
- 각 파라미터의 **독립적 효과** 측정
- 조합 실험은 단일 실험 결과 분석 후 진행

### 2. **실증 연구 기반 범위**
- PPO 원논문 (Schulman et al., 2017)
- OpenAI Spinning Up 권장사항
- RLlib 공식 튜닝 예제

### 3. **측정 지표**
- **성능**: 5회 평균 최종 보상
- **안정성**: 5회 표준편차 및 변동계수(CV)
- **수렴 속도**: 목표 보상 도달 iteration

---

## 🎯 실험 그룹별 상세 설명

### **Group 1: PPO Clipping 메커니즘 (3개 실험)**

#### **이론적 배경**
Schulman et al. (2017) PPO 논문에서 ε=0.2가 최적 균형을 제공하지만, 환경별로 조정 필요

#### **파라미터: `clip_param`**
- **역할**: Policy 업데이트의 최대 변화량 제한
- **범위**: 0.1 ~ 0.3
- **베이스라인**: 0.2

#### **실험 설계**

**Experiment 1-1: 보수적 클리핑**
```python
clip_param = 0.1  # 기본값의 절반
```
- **근거**: Trust Region Policy Optimization (TRPO) 연구에서 작은 step size가 안정성 향상
- **예상 결과**: 
  - 낮은 분산 (안정적 학습)
  - 느린 수렴 (10-20% 느림)
  - 최종 성능은 유사

**Experiment 1-2: Baseline**
```python
clip_param = 0.2  # PPO 논문 검증값
```
- **근거**: Atari 및 MuJoCo 벤치마크에서 최적 값으로 검증됨
- **예상 결과**: 균형잡힌 성능과 안정성

**Experiment 1-3: 공격적 클리핑**
```python
clip_param = 0.3  # 더 큰 업데이트
```
- **근거**: 빠른 수렴이 필요한 환경에서 사용
- **예상 결과**:
  - 빠른 초기 학습 (초기 5 iter에서 20% 빠름)
  - 높은 분산 (불안정)
  - Over-shooting 위험

---

### **Group 2: Entropy Regularization (4개 실험)**

#### **이론적 배경**
Haarnoja et al. (2018) SAC 논문: 엔트로피가 다양한 해 탐색에 도움  
**주의**: HalfCheetah는 연속 제어 환경으로 자연스럽게 탐험됨

#### **파라미터: `entropy_coeff`**
- **역할**: 정책의 무작위성(탐험) 장려
- **범위**: 0.0 ~ 0.05
- **베이스라인**: 0.0

#### **실험 설계**

**Experiment 2-1: Entropy 없음 (Baseline)**
```python
entropy_coeff = 0.0  # 탐험 최소화
```
- **근거**: OpenAI Spinning Up - 연속 제어는 Gaussian noise로 충분한 탐험
- **예상 결과**: 빠른 수렴, 지역 최적해 위험

**Experiment 2-2: 최소 Entropy**
```python
entropy_coeff = 0.001  # 약간의 탐험
```
- **근거**: RLlib MuJoCo 튜닝 예제에서 사용
- **예상 결과**: 균형잡힌 탐험-활용

**Experiment 2-3: 중간 Entropy**
```python
entropy_coeff = 0.01  # 적당한 탐험
```
- **근거**: 일반적 PPO 설정
- **예상 결과**: 느린 초기 학습, 더 나은 후기 성능 가능

**Experiment 2-4: 높은 Entropy**
```python
entropy_coeff = 0.05  # 강한 탐험
```
- **근거**: 복잡한 탐험이 필요한 환경 (SAC 기본값)
- **예상 결과**: 매우 느린 수렴, 다양한 행동 패턴

---

### **Group 3: Discount Factor (3개 실험)**

#### **이론적 배경**
Sutton & Barto (2018): γ는 가장 중요한 하이퍼파라미터  
HalfCheetah는 **단기 보상(거리)**이 명확한 환경

#### **파라미터: `gamma`**
- **역할**: 미래 보상의 중요도 결정
- **범위**: 0.95 ~ 0.995
- **베이스라인**: 0.99

#### **실험 설계**

**Experiment 3-1: 단기 보상 중심**
```python
gamma = 0.95  # 짧은 시간 지평
```
- **근거**: HalfCheetah는 각 타임스텝의 속도가 보상이므로 단기 focus 효과적일 수 있음
- **예상 결과**: 빠른 학습, 명확한 credit assignment

**Experiment 3-2: Baseline**
```python
gamma = 0.99  # 표준값 (PPO paper)
```
- **근거**: 대부분의 강화학습 환경에서 검증됨
- **예상 결과**: 균형잡힌 성능

**Experiment 3-3: 장기 보상 중심**
```python
gamma = 0.995  # 긴 시간 지평
```
- **근거**: 전략적 계획이 필요한 환경에 적합
- **예상 결과**: 느린 학습, 장기 전략 가능성

---

### **Group 4: Gradient Clipping (3개 실험)**

#### **이론적 배경**
Engstrom et al. (2020, OpenAI): Gradient clipping이 deep RL 안정성에 필수적

#### **파라미터: `grad_clip`**
- **역할**: Gradient norm 제한으로 발산 방지
- **범위**: None, 0.5, 1.0
- **베이스라인**: None

#### **실험 설계**

**Experiment 4-1: Clipping 없음 (Baseline)**
```python
grad_clip = None  # 제약 없음
```
- **근거**: Baseline - 최대 학습 속도
- **예상 결과**: 빠른 학습, 발산 위험

**Experiment 4-2: 강한 Clipping**
```python
grad_clip = 0.5  # 일반적 권장값
```
- **근거**: Henderson et al. (2018) - Gradient norm 0.5가 안정성과 성능 균형
- **예상 결과**: 안정적 학습, 약간 느린 수렴

**Experiment 4-3: 적당한 Clipping**
```python
grad_clip = 1.0  # 더 넓은 허용 범위
```
- **근거**: 큰 값은 대부분의 gradient 통과
- **예상 결과**: 중간 수준의 안정성

---

### **Group 5: Value Function Clipping (3개 실험)**

#### **이론적 배경**
PPO 논문에서 policy clipping만큼 중요하지 않지만, 추가 안정성 제공

#### **파라미터: `vf_clip_param`**
- **역할**: 가치 함수 업데이트 제한
- **범위**: 1.0, 10.0, 100.0
- **베이스라인**: 10.0

#### **실험 설계**

**Experiment 5-1: 강한 VF 제약**
```python
vf_clip_param = 1.0  # 작은 값 = 강한 제약
```
- **근거**: 가치 함수 over-shooting 방지
- **예상 결과**: 보수적 가치 학습, 안정성 증가

**Experiment 5-2: Baseline**
```python
vf_clip_param = 10.0  # RLlib 기본값
```
- **근거**: 대부분의 환경에서 문제없는 값
- **예상 결과**: 표준 성능

**Experiment 5-3: 약한 VF 제약**
```python
vf_clip_param = 100.0  # 거의 제약 없음
```
- **근거**: 빠른 가치 학습 필요 시
- **예상 결과**: 빠른 가치 수렴, 불안정 가능

---

### **Group 6: KL Divergence Constraint (3개 실험)**

#### **이론적 배경**
Schulman et al. (2017): KL penalty가 clipping보다 성능 낮지만 추가 안정성 제공 가능

#### **파라미터: `use_kl_loss`, `kl_coeff`, `kl_target`**
- **역할**: 정책 변화를 KL divergence로 제한
- **범위**: kl_coeff 0.1 ~ 0.5
- **베이스라인**: use_kl_loss=True, kl_coeff=0.2

#### **실험 설계**

**Experiment 6-1: KL Loss 비활성화**
```python
use_kl_loss = False  # PPO-Clip만 사용
```
- **근거**: PPO 논문 - Clipping만으로 충분
- **예상 결과**: 더 단순한 알고리즘, 유사한 성능

**Experiment 6-2: 약한 KL**
```python
use_kl_loss = True
kl_coeff = 0.1
```
- **근거**: 가벼운 추가 제약
- **예상 결과**: 약간의 안정성 향상

**Experiment 6-3: 강한 KL**
```python
use_kl_loss = True
kl_coeff = 0.5
kl_target = 0.005
```
- **근거**: 최대 안정성
- **예상 결과**: 매우 보수적 업데이트, 느린 학습

---

### **Group 7: GAE (2개 실험)**

#### **이론적 배경**
Schulman et al. (2016) GAE 논문: λ로 bias-variance tradeoff 조절

#### **파라미터: `use_gae`**
- **역할**: Generalized Advantage Estimation 사용
- **베이스라인**: True

#### **실험 설계**

**Experiment 7-1: Baseline (GAE 사용)**
```python
use_gae = True
lambda_ = 0.95  # 고정값
```
- **근거**: 대부분의 환경에서 성능 향상
- **예상 결과**: 안정적 advantage 추정

**Experiment 7-2: GAE 비활성화**
```python
use_gae = False
```
- **근거**: 단순 advantage 계산
- **예상 결과**: 높은 분산, 빠른 계산

---

## 📊 파라미터 상세 설명

### 1. **Training Parameters (학습 관련)**

#### `gamma` (할인율)
- **범위**: 0.0 ~ 1.0 (보통 0.95 ~ 0.999)
- **기본값**: 0.99
- **역할**: 미래 보상의 중요도를 결정
- **영향**:
  - **높을수록 (0.99+)**: 장기적 보상을 중요시, 안정적이지만 느린 학습
  - **낮을수록 (0.95-)**: 단기적 보상을 중요시, 빠르지만 근시안적
- **HalfCheetah 추천**: 0.99 (장기적으로 빠르게 달리는 것이 중요)

#### `lr` (학습률)
- **범위**: 1e-5 ~ 1e-2
- **기본값**: 3e-4
- **역할**: 파라미터 업데이트 크기
- **영향**:
  - **높을수록**: 빠른 학습, 불안정, 발산 위험
  - **낮을수록**: 느린 학습, 안정적, 지역 최적에 빠질 위험
- **조정 팁**: 
  - 학습이 불안정하면 낮추기 (1e-4)
  - 학습이 너무 느리면 높이기 (5e-4)

#### `train_batch_size` (학습 배치 크기)
- **범위**: 1000 ~ 50000
- **기본값**: 4000
- **역할**: 한 번의 학습 반복에 사용할 총 샘플 수
- **영향**:
  - **클수록**: 안정적 gradient, 느린 반복, 더 많은 메모리
  - **작을수록**: 빠른 반복, 불안정한 gradient, 적은 메모리
- **관계식**: `train_batch_size = num_rollout_workers × rollout_fragment_length × num_envs_per_worker`

#### `sgd_minibatch_size` (SGD 미니배치)
- **범위**: 32 ~ 1024
- **기본값**: 128
- **역할**: train_batch_size를 나눠서 여러 번 업데이트
- **영향**:
  - **클수록**: 빠른 학습, 적은 업데이트 횟수
  - **작을수록**: 많은 업데이트, 더 세밀한 학습
- **권장**: `train_batch_size / sgd_minibatch_size = 10~50`

#### `num_sgd_iter` (SGD 반복 횟수)
- **범위**: 3 ~ 30
- **기본값**: 10
- **역할**: 하나의 train_batch로 몇 번 업데이트할지
- **영향**:
  - **많을수록**: 데이터를 더 많이 활용, 과적합 위험
  - **적을수록**: 샘플 효율성 낮음, 과적합 위험 낮음
- **PPO 특징**: PPO는 old policy와 너무 멀어지면 안 되므로 10~20이 적당

#### `lambda_` (GAE Lambda)
- **범위**: 0.9 ~ 1.0
- **기본값**: 0.95
- **역할**: Advantage 추정 시 편향-분산 트레이드오프
- **영향**:
  - **1.0에 가까울수록**: 낮은 편향, 높은 분산
  - **0.9에 가까울수록**: 높은 편향, 낮은 분산
- **추천**: 0.95 (대부분의 경우 좋은 균형)

#### `clip_param` (PPO Clip Parameter)
- **범위**: 0.1 ~ 0.3
- **기본값**: 0.2
- **역할**: policy 업데이트의 최대 변화량 제한
- **영향**:
  - **클수록**: 큰 policy 변화 허용, 빠르지만 불안정
  - **작을수록**: 작은 policy 변화, 느리지만 안정적
- **추천**: 0.2 (PPO 논문 기본값, 검증됨)

#### `vf_clip_param` (Value Function Clip)
- **범위**: 1.0 ~ 100.0
- **기본값**: 10.0
- **역할**: 가치 함수 업데이트 제한
- **영향**:
  - **클수록**: 가치 함수가 자유롭게 변화
  - **작을수록**: 가치 함수 변화 제한
- **추천**: 10.0 (대부분 문제없음)

#### `entropy_coeff` (엔트로피 계수)
- **범위**: 0.0 ~ 0.1
- **기본값**: 0.0
- **역할**: 정책의 무작위성(탐험) 장려
- **영향**:
  - **0**: 탐험 없음, 수렴 빠름
  - **0.01~0.1**: 탐험 장려, 다양한 행동 시도
- **HalfCheetah 추천**: 0.0 또는 0.001 (연속 제어는 자연스럽게 탐험됨)

#### `model` (신경망 구조)
```python
model = {
    "fcnet_hiddens": [256, 256],  # 은닉층 크기
    "fcnet_activation": "tanh",  # 활성화 함수
    "vf_share_layers": False,  # policy와 value function 레이어 공유 여부
}
```
- **`fcnet_hiddens`**:
  - **[64, 64]**: 작고 빠름, 단순한 문제
  - **[256, 256]**: 균형잡힌 선택
  - **[512, 512]**: 복잡한 문제, 많은 메모리
  
- **`fcnet_activation`**:
  - **`tanh`**: 연속 제어에 전통적으로 좋음
  - **`relu`**: 빠른 학습, 때때로 불안정
  - **`elu`**: relu와 tanh의 중간

### 2. **Rollout Parameters (데이터 수집)**

#### `num_rollout_workers` (워커 수)
- **범위**: 0 ~ CPU 코어 수
- **기본값**: 2
- **역할**: 병렬로 환경을 실행할 워커 수
- **영향**:
  - **많을수록**: 빠른 데이터 수집, 더 많은 CPU 사용
  - **적을수록**: 느린 데이터 수집, CPU 절약
- **권장**: CPU 코어 수의 50~75%

#### `num_envs_per_worker` (워커당 환경 수)
- **범위**: 1 ~ 20
- **기본값**: 1
- **역할**: 각 워커가 동시에 실행할 환경 수
- **영향**:
  - **많을수록**: 더 많은 병렬화, 메모리 증가
  - **적을수록**: 간단한 구조
- **HalfCheetah**: 1이 적당 (물리 시뮬레이션이 무거움)

#### `rollout_fragment_length` (수집 길이)
- **범위**: 50 ~ 1000
- **기본값**: 200
- **역할**: 각 워커가 한 번에 수집할 타임스텝
- **영향**:
  - **클수록**: 긴 궤적, 시간적 상관관계 높음
  - **작을수록**: 짧은 궤적, 빠른 업데이트
- **계산**: `train_batch_size = num_workers × fragment_length × num_envs`

#### `batch_mode`
- **옵션**: 
  - **`"truncate_episodes"`**: 에피소드 중간에 잘라도 됨 (빠름)
  - **`"complete_episodes"`**: 에피소드 완료까지 대기 (정확함)
- **HalfCheetah 추천**: `"truncate_episodes"` (에피소드가 김)

### 3. **Resources (하드웨어 자원)**

#### `num_gpus`
- **범위**: 0 ~ 사용 가능한 GPU 수
- **역할**: 학습에 사용할 GPU 수
- **영향**:
  - **0**: CPU만 사용, 느림, 항상 가능
  - **1**: GPU 사용, 빠름, GPU 메모리 필요
- **권장**: GPU 있으면 1, 없으면 0

#### `num_cpus_per_worker`
- **범위**: 0.5 ~ 4
- **기본값**: 1
- **역할**: 각 워커에 할당할 CPU 코어
- **추천**: 1 (대부분 충분)

### 4. **Evaluation (평가)**

#### `evaluation_interval`
- **범위**: 1 ~ 100
- **기본값**: None (평가 안 함)
- **역할**: 몇 번의 학습마다 평가할지
- **추천**: 5~10 (너무 자주 하면 시간 낭비)

#### `evaluation_duration`
- **범위**: 1 ~ 100
- **기본값**: 10
- **역할**: 평가 시 실행할 에피소드 수
- **추천**: 10~20 (통계적으로 의미있는 수)

#### `evaluation_num_workers`
- **범위**: 0 ~ num_rollout_workers
- **기본값**: 0
- **역할**: 평가 전용 워커 수
- **추천**: 1~2 (학습 방해하지 않도록)

---

## 📈 예상 결과 요약 (이론 기반)

| Group | 파라미터 | 예상 성능 | 예상 안정성 (CV) | 근거 문헌 |
|-------|----------|-----------|------------------|-----------|
| Baseline | - | 2500 ± 400 | 0.16 | Schulman et al. (2017) |
| Clipping | 0.1 / 0.3 | 2400 / 2700 | 0.11 / 0.21 | PPO paper |
| Entropy | 0.001~0.05 | 2500~2200 | 0.16~0.25 | Haarnoja et al. (2018) |
| Gamma | 0.95/0.995 | 2600 / 2400 | 0.15 / 0.18 | Sutton & Barto |
| Grad Clip | 0.5 / 1.0 | 2450 / 2550 | 0.12 / 0.14 | Engstrom et al. (2020) |
| VF Clip | 1.0 / 100.0 | 2400 / 2600 | 0.13 / 0.18 | PPO variations |
| KL Loss | off/on | 2550 / 2500 | 0.16 / 0.14 | PPO paper |
| GAE | on/off | 2500 / 2400 | 0.16 / 0.20 | Schulman et al. (2016) |

**CV (Coefficient of Variation)**: 낮을수록 안정적  
**예상 성능**: 10 iterations 후 평균 보상

---

## 🎯 우선순위 실험 (시간 제약 시)

### **Minimal Set (10개 실험, 50회 실행)**

가장 영향력이 큰 파라미터만 선택:

1. **Gamma** (3개) - 가장 중요한 하이퍼파라미터
   - gamma=0.95, 0.99 (baseline), 0.995

2. **Clip Param** (3개) - PPO 핵심 메커니즘
   - clip_param=0.1, 0.2 (baseline), 0.3

3. **Entropy** (3개) - 탐험-활용 균형
   - entropy_coeff=0.0 (baseline), 0.01, 0.05

4. **Gradient Clipping** (1개) - 안정성 확보
   - grad_clip=0.5

**예상 소요 시간**: 약 90분 (10 experiments × 5 trials × 10 iterations)

---

## 📊 전체 실험 구성

### **Full Set (18개 실험, 90회 실행)**

| Group | 실험 수 | 실행 횟수 (×5) | 예상 시간 |
|-------|---------|----------------|-----------|
| Baseline | 1 | 5 | 15분 |
| Clipping | 2 | 10 | 30분 |
| Entropy | 3 | 15 | 45분 |
| Gamma | 2 | 10 | 30분 |
| Gradient | 2 | 10 | 30분 |
| VF Clip | 2 | 10 | 30분 |
| KL Loss | 3 | 15 | 45분 |
| GAE | 1 | 5 | 15분 |
| **합계** | **18** | **90** | **~4시간** |

*10 iterations per trial 기준

---

## 📈 분석 계획

### **통계적 검정**
1. **Baseline 대비 t-test** (p < 0.05)
   - 각 실험이 baseline보다 유의미하게 다른지 확인
2. **효과 크기 계산** (Cohen's d)
   - 실용적 중요성 평가
3. **변동계수(CV) 비교**
   - 안정성 정량화

### **시각화**
1. **Learning curves** (5 trials with confidence interval)
2. **Boxplots** (final rewards distribution)
3. **Stability-Performance tradeoff scatter**
4. **Heatmap** (parameter effects on performance and stability)

### **비교 기준**
```python
# 각 실험의 성능 비교
performance_gain = (exp_reward - baseline_reward) / baseline_reward * 100
stability_gain = (baseline_cv - exp_cv) / baseline_cv * 100

# 종합 점수 (성능 60%, 안정성 40%)
score = 0.6 * performance_gain + 0.4 * stability_gain
```

---

## 💡 실험 실행 가이드

### **1단계: 실험 목록 확인**
```bash
cd /home/kimjihun/reinforce-project4/hyperparameter_experiments
python hyperparameter_stability_experiment.py --list
```

### **2단계: 전체 실험 실행**
```bash
python hyperparameter_stability_experiment.py
```
- 중간 결과가 자동 저장됨 (`results/hyperparameter_experiments_progress.json`)
- 실험 중단 시 복구 가능

### **3단계: 결과 분석**
```bash
python analyze_hyperparameter_experiments.py
```
- 통계 분석 및 시각화 생성
- `results/` 디렉토리에 그래프 저장

---

## 📚 참고 문헌

### **핵심 논문**
1. **Schulman et al. (2017)** - Proximal Policy Optimization Algorithms
   - PPO 알고리즘 제안
   - clip_param=0.2 검증

2. **Haarnoja et al. (2018)** - Soft Actor-Critic
   - 엔트로피 정규화의 중요성
   - 탐험-활용 균형

3. **Schulman et al. (2016)** - High-Dimensional Continuous Control Using GAE
   - GAE λ 제안
   - bias-variance tradeoff

4. **Engstrom et al. (2020)** - Implementation Matters in Deep RL
   - Gradient clipping의 필요성
   - 구현 세부사항의 영향

5. **Henderson et al. (2018)** - Deep Reinforcement Learning that Matters
   - 재현성 문제
   - 통계적 유의성

### **참고 자료**
- **OpenAI Spinning Up**: [spinningup.openai.com](https://spinningup.openai.com)
- **RLlib Documentation**: [docs.ray.io/en/latest/rllib](https://docs.ray.io/en/latest/rllib/)
- **Sutton & Barto (2018)**: Reinforcement Learning: An Introduction

---

## 🔍 추가 파라미터 설명 (참고용)

### 실험 1: Baseline (기준선)
```python
config = (
    PPOConfig()
    .environment(env="HalfCheetah-v5")
    .framework("torch")
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.0,
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        }
    )
    .rollouts(
        num_rollout_workers=4,
        num_envs_per_worker=1,
        rollout_fragment_length=200,
        batch_mode="truncate_episodes",
    )
    .resources(num_gpus=0)
    .evaluation(
        evaluation_interval=10,
        evaluation_duration=10,
        evaluation_num_workers=1,
        evaluation_config={"explore": False}
    )
)
```
**목표**: 3000~5000 보상 달성 (1000 iteration 내)

---

### 실험 2: 학습률 탐색 (Learning Rate Sweep)

#### 2-1. 낮은 학습률 (안정적)
```python
.training(lr=1e-4, ...)
```
- **기대**: 느리지만 안정적 수렴
- **목표**: 안정성 확인

#### 2-2. 기본 학습률
```python
.training(lr=3e-4, ...)
```
- **기대**: 균형잡힌 성능
- **목표**: Baseline

#### 2-3. 높은 학습률 (공격적)
```python
.training(lr=5e-4, ...)
```
- **기대**: 빠른 학습, 약간의 불안정
- **목표**: 빠른 수렴 테스트

#### 2-4. 매우 높은 학습률
```python
.training(lr=1e-3, ...)
```
- **기대**: 불안정할 수 있음
- **목표**: 상한선 확인

---

### 실험 3: 네트워크 크기 (Network Architecture)

#### 3-1. 작은 네트워크 (빠른 실험)
```python
.training(
    model={
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "tanh",
    }
)
```
- **메모리**: ~5MB
- **속도**: 빠름
- **성능**: 3000~4000 예상

#### 3-2. 중간 네트워크 (권장)
```python
.training(
    model={
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    }
)
```
- **메모리**: ~15MB
- **속도**: 보통
- **성능**: 4000~5000 예상

#### 3-3. 큰 네트워크 (고성능)
```python
.training(
    model={
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "tanh",
    }
)
```
- **메모리**: ~50MB
- **속도**: 느림
- **성능**: 5000+ 예상

#### 3-4. 깊은 네트워크
```python
.training(
    model={
        "fcnet_hiddens": [256, 256, 128],
        "fcnet_activation": "relu",
    }
)
```
- **특징**: 더 복잡한 패턴 학습 가능
- **위험**: 과적합, 학습 어려움

---

### 실험 4: 배치 크기 (Batch Size)

#### 4-1. 작은 배치 (빠른 반복)
```python
.training(
    train_batch_size=2000,
    sgd_minibatch_size=64,
)
.rollouts(
    num_rollout_workers=2,
    rollout_fragment_length=200,
)
```
- **반복 속도**: 빠름
- **안정성**: 낮음
- **적합**: 빠른 실험

#### 4-2. 중간 배치 (균형)
```python
.training(
    train_batch_size=4000,
    sgd_minibatch_size=128,
)
.rollouts(
    num_rollout_workers=4,
    rollout_fragment_length=200,
)
```
- **반복 속도**: 보통
- **안정성**: 좋음
- **적합**: 대부분의 경우

#### 4-3. 큰 배치 (안정적)
```python
.training(
    train_batch_size=8192,
    sgd_minibatch_size=256,
)
.rollouts(
    num_rollout_workers=8,
    rollout_fragment_length=256,
)
```
- **반복 속도**: 느림
- **안정성**: 매우 좋음
- **적합**: 최종 학습

---

### 실험 5: 할인율 (Gamma)

#### 5-1. 낮은 Gamma (단기 집중)
```python
.training(gamma=0.95, ...)
```
- **특징**: 즉각적 보상 중시
- **HalfCheetah**: 덜 적합 (장기 달리기 중요)

#### 5-2. 표준 Gamma
```python
.training(gamma=0.99, ...)
```
- **특징**: 균형잡힌 시간 지평
- **HalfCheetah**: 적합

#### 5-3. 높은 Gamma (장기 집중)
```python
.training(gamma=0.995, ...)
```
- **특징**: 먼 미래까지 고려
- **HalfCheetah**: 시도해볼 만함

---

### 실험 6: PPO 특화 파라미터

#### 6-1. 보수적 업데이트
```python
.training(
    clip_param=0.1,  # 작은 변화
    num_sgd_iter=5,  # 적은 반복
    entropy_coeff=0.0,
)
```
- **특징**: 안정적, 느림
- **적합**: 안정성 우선

#### 6-2. 공격적 업데이트
```python
.training(
    clip_param=0.3,  # 큰 변화
    num_sgd_iter=20,  # 많은 반복
    entropy_coeff=0.01,  # 탐험 장려
)
```
- **특징**: 빠름, 불안정 위험
- **적합**: 빠른 수렴 필요 시

---

### 실험 7: 리소스 최적화

#### 7-1. CPU 전용 (저사양)
```python
.rollouts(
    num_rollout_workers=2,
    num_envs_per_worker=1,
)
.resources(
    num_gpus=0,
    num_cpus_per_worker=1,
)
```
- **적합**: 노트북, 제한된 자원

#### 7-2. 멀티코어 활용
```python
.rollouts(
    num_rollout_workers=8,
    num_envs_per_worker=1,
)
.resources(
    num_gpus=0,
    num_cpus_per_worker=1,
)
```
- **적합**: 데스크탑 (8+ 코어)

#### 7-3. GPU 가속
```python
.rollouts(
    num_rollout_workers=4,
    num_envs_per_worker=1,
)
.resources(
    num_gpus=1,
    num_cpus_per_worker=1,
)
```
- **적합**: GPU 있는 환경
- **속도**: 2-5배 빠름

---

## 📊 실험 우선순위 및 순서

### Phase 1: Baseline 확립 (1-2일)
1. **실험 1** 실행 → 기준 성능 확인
2. 3000+ 보상 달성 확인

### Phase 2: 주요 파라미터 탐색 (3-5일)
1. **실험 2** (학습률) → 최적 lr 찾기
2. **실험 3** (네트워크) → 최적 구조 찾기
3. **실험 4** (배치 크기) → 최적 배치 찾기

### Phase 3: 세밀 조정 (2-3일)
1. **실험 5** (Gamma) → 미세 조정
2. **실험 6** (PPO 파라미터) → 추가 최적화

### Phase 4: 최종 학습 (1-2일)
- 최적 파라미터로 장시간 학습
- 목표: 6000+ 보상

---

## 💡 실험 팁

### 1. 로깅 설정
```python
config.debugging(
    log_level="INFO",
    seed=42,  # 재현성
)
```

### 2. 체크포인트 저장
```python
algo.train()
if iteration % 50 == 0:
    checkpoint = algo.save()
    print(f"Checkpoint saved at {checkpoint}")
```

### 3. 조기 종료
```python
# 목표 달성 시 종료
if result["episode_reward_mean"] > 6000:
    print("Goal reached!")
    break
```

### 4. 결과 비교
- TensorBoard로 모든 실험 비교
- 각 실험마다 다른 이름으로 저장
```python
algo = config.build()
# ray_results/PPO_HalfCheetah-v5_lr3e-4_...
```

---

## 📈 파라미터 간 상호작용

### 학습률과 배치 크기
- **큰 배치 + 높은 lr**: 빠르지만 불안정
- **큰 배치 + 낮은 lr**: 안정적이지만 느림
- **작은 배치 + 높은 lr**: 매우 불안정
- **작은 배치 + 낮은 lr**: 느리고 비효율적

### 네트워크 크기와 학습률
- **큰 네트워크**: 낮은 lr 필요 (1e-4 ~ 3e-4)
- **작은 네트워크**: 높은 lr 가능 (3e-4 ~ 1e-3)

### PPO Clip과 SGD 반복
- **작은 clip + 많은 iter**: 안전하게 많이 업데이트
- **큰 clip + 적은 iter**: 공격적으로 적게 업데이트

---

## 🎓 일반적인 문제 해결

### 문제 1: 학습이 불안정함 (보상이 튐)
**해결책**:
- 학습률 낮추기: `lr=1e-4`
- 배치 크기 늘리기: `train_batch_size=8192`
- Clip 파라미터 낮추기: `clip_param=0.1`

### 문제 2: 학습이 너무 느림
**해결책**:
- 학습률 높이기: `lr=5e-4`
- 워커 수 늘리기: `num_rollout_workers=8`
- GPU 사용: `num_gpus=1`

### 문제 3: 성능이 plateau에 도달
**해결책**:
- 네트워크 크게 하기: `[512, 512]`
- 엔트로피 추가: `entropy_coeff=0.01`
- Gamma 조정: `gamma=0.995`

### 문제 4: 메모리 부족
**해결책**:
- 워커 수 줄이기
- 배치 크기 줄이기
- 네트워크 작게 하기

이렇게 체계적으로 실험하면 HalfCheetah-v5에서 최적의 성능을 달성할 수 있습니다!
