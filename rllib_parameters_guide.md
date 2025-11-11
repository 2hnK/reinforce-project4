# RLlib 파라미터 상세 설명 및 실험 설정 가이드

## 📋 파라미터 상세 설명

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

## 🎯 추천 실험 설정 정리

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
