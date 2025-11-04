# Reinforcement Learning Project 4: RLlib with MuJoCo
하이퍼 파라미터 참고: https://docs.ray.io/en/latest/rllib/getting-started.html
학습 중 리소스 모니터링 추가와 더 정확한 CPU/GPU 측정 적용해줘
## 프로젝트 개요

이 프로젝트는 Ray RLlib를 사용하여 MuJoCo 환경(HalfCheetah-v5)에서 PPO(Proximal Policy Optimization) 알고리즘을 학습하고 평가하는 과제입니다.

## 환경 설정

### 필수 요구사항

- Python 3.8 이상
- Ray[rllib]
- Gymnasium
- MuJoCo
- NumPy

### 설치 방법

```bash
# Ray RLlib 설치
pip install ray[rllib]

# Gymnasium과 MuJoCo 설치
pip install gymnasium
pip install gymnasium[mujoco]

# 추가 의존성
pip install numpy
```

## 프로젝트 구조

```
reinforce-project4/
├── README.md                      # 프로젝트 문서
├── pdf-docs.pdf                   # 과제 문서
├── rllib_mujoco.py               # PPO 학습 스크립트
└── rllib_mujoco_compute_action.py # 학습된 모델 평가 스크립트
```

## 파일 설명

### 1. rllib_mujoco.py

PPO 알고리즘을 사용하여 HalfCheetah-v5 환경에서 에이전트를 학습합니다.

**주요 구성:**
- **환경**: HalfCheetah-v5 (MuJoCo)
- **알고리즘**: PPO (Proximal Policy Optimization)
- **학습 파라미터**:
  - Lambda (GAE): 0.95
  - Learning rate: 0.0003
  - Training batch size: 16,384 (32 * 512)
  - Minibatch size: 4,096
  - Number of epochs: 15
  - Value function loss coefficient: 0.01
- **네트워크 구조**:
  - Hidden layers: [64, 64]
  - Activation: tanh
  - Separate value function layers

**실행 방법:**
```bash
python rllib_mujoco.py
```

### 2. rllib_mujoco_compute_action.py

학습된 모델을 로드하여 평가하는 스크립트입니다.

**주요 기능:**
- 체크포인트에서 모델 로드
- 10개 에피소드에 대한 평가
- 평균 및 표준편차 리턴 계산

**실행 방법:**
```bash
python rllib_mujoco_compute_action.py
```

## 과제 수행 단계

### Step 1: 학습 환경 설정

1. `rllib_mujoco.py`에서 PPO 설정 확인
2. 필요에 따라 하이퍼파라미터 조정
3. 학습 반복 횟수 설정 (현재: 5회)

### Step 2: 모델 학습

```bash
python rllib_mujoco.py
```

학습 중 출력되는 정보:
- Episode reward mean
- Training loss
- Learning rate
- Policy loss, Value loss
- 기타 학습 메트릭

### Step 3: 체크포인트 저장

학습된 모델은 자동으로 `~/ray_results/` 디렉토리에 저장됩니다.

체크포인트 경로 확인:
```bash
ls ~/ray_results/PPO_*/checkpoint_*
```

### Step 4: 모델 평가

1. `rllib_mujoco_compute_action.py`에서 체크포인트 경로 설정
2. `compute_action()` 함수 구현
3. 평가 실행

```bash
python rllib_mujoco_compute_action.py
```

## 구현 가이드

### compute_action() 함수 구현

```python
def compute_action(obs):
    """
    학습된 정책을 사용하여 행동 선택
    
    Args:
        obs: 환경의 관측값
    
    Returns:
        action: 선택된 행동
    """
    # 방법 1: 학습된 모델 사용
    action = algo_eval.compute_single_action(obs)
    
    # 방법 2: 랜덤 행동 (baseline)
    # action = env.action_space.sample()
    
    return action
```

### 체크포인트 로딩

```python
from ray.rllib.algorithms.algorithm import Algorithm

# 체크포인트 경로 설정
ckpt_path = "path/to/checkpoint"

# 알고리즘 복원
algo_eval = Algorithm.from_checkpoint(ckpt_path)
```

## 평가 메트릭

### 학습 중 모니터링
- **episode_reward_mean**: 평균 에피소드 리워드
- **episode_len_mean**: 평균 에피소드 길이
- **policy_loss**: 정책 손실
- **vf_loss**: 가치 함수 손실
- **entropy**: 정책 엔트로피

### 평가 메트릭
- **Mean Return**: 10개 에피소드의 평균 리턴
- **Std Return**: 리턴의 표준편차
- **Success Rate**: 특정 임계값 이상의 성공률

## 하이퍼파라미터 튜닝 가이드

### 학습률 (Learning Rate)
- 현재 값: 0.0003
- 너무 높으면: 학습 불안정
- 너무 낮으면: 학습 속도 느림

### 배치 크기 (Batch Size)
- train_batch_size: 전체 배치 크기
- minibatch_size: SGD 업데이트용 미니배치
- 더 큰 배치: 안정적이지만 메모리 소비 많음

### GAE Lambda
- 현재 값: 0.95
- 높을수록: 장기 리워드 강조
- 낮을수록: 단기 리워드 강조

### 네트워크 크기
```python
model={
    "fcnet_hiddens": [64, 64],  # 레이어 크기 조정 가능
    "fcnet_activation": "tanh",  # relu, elu 등으로 변경 가능
}
```

## 트러블슈팅

### 1. MuJoCo 설치 오류
```bash
# MuJoCo 라이센스 확인 (최신 버전은 무료)
pip install mujoco
```

### 2. GPU 메모리 부족
```python
# learners 설정에서 GPU 사용 조정
.learners(
    num_learners=0,  # CPU 사용
    num_gpus_per_learner=0
)
```

### 3. 학습이 수렴하지 않는 경우
- 학습률 감소
- 배치 크기 증가
- 네트워크 크기 증가
- 더 많은 학습 반복

## 결과 분석

### 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

# 학습 중 기록된 리워드 플롯
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
```

### TensorBoard 사용

```bash
# Ray는 자동으로 TensorBoard 로그 생성
tensorboard --logdir ~/ray_results
```

## 참고 자료

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 과제 제출 요구사항

1. **코드 파일**
   - `rllib_mujoco.py`: 완성된 학습 스크립트
   - `rllib_mujoco_compute_action.py`: 완성된 평가 스크립트

2. **실행 결과**
   - 학습 로그 (최소 5회 이상)
   - 평가 결과 (10 에피소드)
   - 학습 곡선 그래프

3. **체크포인트**
   - 최종 학습된 모델 체크포인트

4. **보고서**
   - 사용한 하이퍼파라미터 설명
   - 학습 과정 및 결과 분석
   - 성능 개선 시도 및 결과

## 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 연락처

과제 관련 문의사항은 담당 조교에게 연락하시기 바랍니다.
