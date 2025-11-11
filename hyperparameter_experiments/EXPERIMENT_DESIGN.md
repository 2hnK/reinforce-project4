# PPO 하이퍼파라미터 실험 설계

**Student ID:** 20227128 김지훈  
**Date:** 2025-11-11

---

## 🔒 고정 파라미터 (변경 불가)

```python
lambda_ = 0.95                 # GAE Lambda
lr = 0.0003                    # Learning Rate
num_epochs = 15                # Training Epochs
train_batch_size = 32 * 512    # 16384
minibatch_size = 4096          # Minibatch Size
vf_loss_coeff = 0.01           # Value Function Loss Coefficient
fcnet_hiddens = [64, 64]       # Network Architecture
fcnet_activation = "tanh"      # Activation Function
vf_share_layers = False        # Separate Value/Policy Networks
```

---

## ✅ 실험 가능한 하이퍼파라미터

### 1. **PPO Clip Parameter** (`clip_param`)
**기본값:** 0.2  
**공식 문서:** PPO의 핵심 파라미터. 정책 업데이트의 클리핑 범위

**제안 실험값:**
- **0.1** (보수적): 안정적이지만 느린 수렴
- **0.2** (기본값)
- **0.3** (공격적): 빠른 수렴, 불안정 가능

**예상 효과:**
- 낮은 값 → 정책 변화 제한, 안정적 학습
- 높은 값 → 큰 정책 업데이트 허용, 빠른 학습

**코드:**
```python
config.training(clip_param=0.1)  # or 0.3
```

---

### 2. **Value Function Clip Parameter** (`vf_clip_param`)
**기본값:** 10.0 (또는 None)  
**공식 문서:** 가치 함수 손실에 대한 클리핑

**제안 실험값:**
- **None** (클리핑 없음)
- **1.0** (작은 클리핑)
- **10.0** (기본값)
- **100.0** (큰 클리핑)

**예상 효과:**
- None → 제한 없는 가치 함수 업데이트
- 작은 값 → 가치 함수 변화 제한
- 큰 값 → 더 자유로운 가치 추정

**코드:**
```python
config.training(vf_clip_param=1.0)
```

---

### 3. **Entropy Coefficient** (`entropy_coeff`)
**기본값:** 0.0  
**공식 문서:** 탐험을 장려하는 엔트로피 정규화

**제안 실험값:**
- **0.0** (기본값, 탐험 없음)
- **0.001** (약간의 탐험)
- **0.01** (적당한 탐험)
- **0.05** (많은 탐험)

**예상 효과:**
- 0.0 → 빠른 수렴, 지역 최적화 위험
- 높은 값 → 더 많은 탐험, 느린 수렴, 더 나은 최종 성능 가능

**코드:**
```python
config.training(entropy_coeff=0.01)
```

**스케줄 옵션:**
```python
config.training(
    entropy_coeff=[[0, 0.01], [100000, 0.001], [500000, 0.0]]
)  # 시간에 따라 감소
```

---

### 4. **KL Divergence 설정** (`use_kl_loss`, `kl_coeff`, `kl_target`)
**기본값:** `use_kl_loss=True`, `kl_coeff=0.2`, `kl_target=0.01`  
**공식 문서:** 정책 변화를 제한하는 KL divergence 페널티

**제안 실험값:**

**A. KL Loss 사용 여부:**
- **True** (기본값): KL 페널티 사용
- **False**: KL 페널티 미사용 (PPO 클리핑만)

**B. KL Coefficient:**
- **0.1** (낮음)
- **0.2** (기본값)
- **0.5** (높음)

**C. KL Target:**
- **0.005** (엄격)
- **0.01** (기본값)
- **0.02** (느슨)

**예상 효과:**
- use_kl_loss=False → 클리핑만 사용, 단순화
- 높은 kl_coeff → 정책 변화 강하게 제한
- 낮은 kl_target → 보수적 업데이트

**코드:**
```python
config.training(
    use_kl_loss=True,
    kl_coeff=0.3,
    kl_target=0.01
)
```

---

### 5. **Gradient Clipping** (`grad_clip`)
**기본값:** None  
**공식 문서:** 그래디언트 노름 클리핑

**제안 실험값:**
- **None** (클리핑 없음)
- **0.5** (강한 클리핑)
- **1.0** (중간 클리핑)
- **5.0** (약한 클리핑)

**예상 효과:**
- None → 큰 그래디언트 허용, 불안정 가능
- 작은 값 → 안정적 학습, 느린 수렴
- 큰 값 → 대부분의 그래디언트 통과

**코드:**
```python
config.training(grad_clip=0.5)
```

---

### 6. **GAE 사용 여부** (`use_gae`, `use_critic`)
**기본값:** `use_gae=True`, `use_critic=True`  
**공식 문서:** Generalized Advantage Estimation 사용

**제안 실험값:**
- **use_gae=True** (기본값)
- **use_gae=False**: 단순 advantage 계산

**예상 효과:**
- True → bias-variance 트레이드오프 조절
- False → 단순하지만 높은 분산

**코드:**
```python
config.training(use_gae=False)  # 실험적
```

---

### 7. **Discount Factor** (`gamma`)
**기본값:** 0.99  
**공식 문서:** 미래 보상 할인율

**제안 실험값:**
- **0.95** (단기 보상 중시)
- **0.99** (기본값)
- **0.995** (장기 보상 중시)

**예상 효과:**
- 낮은 값 → 즉각적 보상 선호
- 높은 값 → 장기적 전략 선호

**코드:**
```python
config.training(gamma=0.995)
```

---

### 8. **Learning Rate Schedule** (`lr_schedule`)
**기본값:** None (고정 LR)  
**공식 문서:** 학습률 스케줄링

**제안 실험값:**
```python
# 선형 감소
lr_schedule = [
    [0, 0.0003],
    [500000, 0.00001]
]

# 단계적 감소
lr_schedule = [
    [0, 0.0003],
    [100000, 0.0001],
    [300000, 0.00003]
]
```

**예상 효과:**
- 초기 높은 LR → 빠른 학습
- 후기 낮은 LR → 안정적 수렴

**코드:**
```python
config.training(lr_schedule=[[0, 0.0003], [500000, 0.00001]])
```

**주의:** 고정 파라미터(`lr=0.0003`)와 충돌 가능성 확인 필요

---

### 9. **SGD Minibatch 크기** (`sgd_minibatch_size`)
**기본값:** `minibatch_size`와 동일  
**공식 문서:** SGD 업데이트 시 미니배치 크기

**참고:** `minibatch_size`는 고정이지만, 다른 배치 관련 설정 확인

---

### 10. **Rollout Fragment Length** (`rollout_fragment_length`)
**기본값:** "auto"  
**공식 문서:** EnvRunner가 수집하는 타임스텝 수

**제안 실험값:**
- **200** (짧음)
- **400** (중간)
- **"auto"** (기본값)

**예상 효과:**
- 짧은 길이 → 빈번한 업데이트
- 긴 길이 → 효율적 수집

**코드:**
```python
config.env_runners(rollout_fragment_length=200)
```

---

### 11. **Exploration 설정**
**기본값:** 없음  
**공식 문서:** 탐험 전략 추가

**제안 실험:**
```python
config.exploration(
    explore=True,
    exploration_config={
        "type": "StochasticSampling",  # 기본값
    }
)
```

---

### 12. **Optimizer 설정** (`_optimizer_config`)
**기본값:** Adam optimizer  
**공식 문서:** 옵티마이저 관련 설정

**제안 실험:**
```python
# Adam epsilon 조정
config.training(
    _optimizer_config={
        "adam_epsilon": 1e-5  # 기본값 1e-8
    }
)
```

---

## 🧪 추천 실험 조합

### 실험 1: 클리핑 파라미터 영향
```python
experiments = [
    {"clip_param": 0.1, "vf_clip_param": 1.0},    # 보수적
    {"clip_param": 0.2, "vf_clip_param": 10.0},   # 기본
    {"clip_param": 0.3, "vf_clip_param": 100.0},  # 공격적
]
```

### 실험 2: 탐험 vs 활용
```python
experiments = [
    {"entropy_coeff": 0.0},     # 활용 중심
    {"entropy_coeff": 0.01},    # 균형
    {"entropy_coeff": 0.05},    # 탐험 중심
]
```

### 실험 3: 안정성 강화
```python
experiments = [
    {"grad_clip": None},                           # 제한 없음
    {"grad_clip": 0.5, "clip_param": 0.1},        # 강한 안정화
    {"grad_clip": 1.0, "use_kl_loss": True},      # 중간 안정화
]
```

### 실험 4: KL Divergence 효과
```python
experiments = [
    {"use_kl_loss": False},                                    # KL 미사용
    {"use_kl_loss": True, "kl_coeff": 0.1, "kl_target": 0.01}, # 약한 KL
    {"use_kl_loss": True, "kl_coeff": 0.5, "kl_target": 0.005}, # 강한 KL
]
```

### 실험 5: 종합 최적화
```python
experiments = [
    # 빠른 수렴
    {
        "clip_param": 0.3,
        "entropy_coeff": 0.0,
        "grad_clip": None,
        "gamma": 0.95
    },
    # 안정적 학습
    {
        "clip_param": 0.1,
        "entropy_coeff": 0.001,
        "grad_clip": 0.5,
        "use_kl_loss": True,
        "kl_coeff": 0.3
    },
    # 탐험 중심
    {
        "clip_param": 0.2,
        "entropy_coeff": 0.05,
        "grad_clip": 1.0,
        "gamma": 0.995
    }
]
```

---

## 📊 측정 지표 (동일)

### 성능
- `episode_reward_mean`: 평균 보상
- `episode_reward_std`: 보상 표준편차 (5회 trial)

### 안정성
- `reward_cv`: 변동계수 (std/mean)
- `min/max reward`: 범위

### 효율성
- `SPS`: Steps Per Second
- `time_per_experiment`: 소요 시간

---

## 🎯 실험 목표

1. **클리핑 효과**: clip_param과 vf_clip_param이 성능/안정성에 미치는 영향
2. **탐험 효과**: entropy_coeff가 최종 성능에 미치는 영향
3. **안정화 기법**: grad_clip, KL loss가 학습 안정성에 미치는 영향
4. **할인율 영향**: gamma가 장기/단기 전략에 미치는 영향
5. **조합 효과**: 여러 파라미터의 상호작용

---

## 💡 구현 팁

### 1. 베이스라인 유지
```python
def get_baseline_config():
    return {
        # 고정 파라미터 (12-22 line)
        'lambda_': 0.95,
        'lr': 0.0003,
        'num_epochs': 15,
        'train_batch_size': 16384,
        'minibatch_size': 4096,
        'vf_loss_coeff': 0.01,
        'fcnet_hiddens': [64, 64],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,
        
        # 변경 가능한 파라미터 (기본값)
        'clip_param': 0.2,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.0,
        'use_kl_loss': True,
        'kl_coeff': 0.2,
        'kl_target': 0.01,
        'grad_clip': None,
        'gamma': 0.99,
        'use_gae': True,
        'use_critic': True,
    }
```

### 2. Config 적용
```python
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .training(
        # 고정 파라미터
        lambda_=params['lambda_'],
        lr=params['lr'],
        num_epochs=params['num_epochs'],
        train_batch_size=params['train_batch_size'],
        minibatch_size=params['minibatch_size'],
        vf_loss_coeff=params['vf_loss_coeff'],
        model={
            "fcnet_hiddens": params['fcnet_hiddens'],
            "fcnet_activation": params['fcnet_activation'],
            "vf_share_layers": params['vf_share_layers'],
        },
        
        # 실험 파라미터
        clip_param=params['clip_param'],
        vf_clip_param=params['vf_clip_param'],
        entropy_coeff=params['entropy_coeff'],
        use_kl_loss=params['use_kl_loss'],
        kl_coeff=params['kl_coeff'],
        kl_target=params['kl_target'],
        grad_clip=params['grad_clip'],
        gamma=params['gamma'],
        use_gae=params['use_gae'],
        use_critic=params['use_critic'],
    )
)
```

---

## 📚 참고 문서

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [RLlib PPO Config](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#proximal-policy-optimization-ppo)
- [RLlib Training API](https://docs.ray.io/en/latest/rllib/rllib-training.html)

---

## ✅ 최종 권장 실험 리스트

총 **25개** 실험 (베이스라인 + 24개 변형)

1. **Baseline** (기본값)
2-4. **Clip Parameter** (0.1, 0.3, 조합)
5-7. **VF Clip Parameter** (1.0, 100.0, None)
8-11. **Entropy Coefficient** (0.001, 0.01, 0.05, 스케줄)
12-14. **KL Loss** (미사용, 약함, 강함)
15-17. **Gradient Clipping** (0.5, 1.0, 5.0)
18-19. **Gamma** (0.95, 0.995)
20-21. **GAE** (False, True with different lambda)
22-25. **조합 실험** (빠른수렴, 안정학습, 탐험중심, 균형)

각 실험 **5회 반복** → 총 **125회** 학습 실행
예상 소요 시간: **3-4시간** (환경에 따라 다름)

---

**Good Luck! 🚀**
