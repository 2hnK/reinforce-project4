# 강화학습 프로젝트 4 - 과제 수행 플랜

## 📋 프로젝트 목표

Ray RLlib를 사용하여 MuJoCo 환경(HalfCheetah-v5)에서 PPO 알고리즘을 학습하고, 학습된 모델을 평가하는 시스템 구축

---

## 🎯 Phase 1: 환경 설정 및 검증 (예상 소요: 30분)

### Task 1.1: 개발 환경 확인
- [ ] Python 버전 확인 (3.8 이상)
  ```bash
  python --version
  ```

- [ ] 필수 패키지 설치 확인
  ```bash
  pip list | grep ray
  pip list | grep gymnasium
  pip list | grep mujoco
  ```

### Task 1.2: 패키지 설치 (필요시)
- [ ] Ray RLlib 설치
  ```bash
  pip install "ray[rllib]"
  ```

- [ ] Gymnasium 및 MuJoCo 설치
  ```bash
  pip install gymnasium
  pip install gymnasium[mujoco]
  pip install mujoco
  ```

- [ ] 추가 의존성 설치
  ```bash
  pip install numpy matplotlib tensorboard
  ```

### Task 1.3: MuJoCo 환경 테스트
- [ ] 간단한 테스트 스크립트 작성
  ```python
  import gymnasium as gym
  env = gym.make("HalfCheetah-v5")
  obs, info = env.reset()
  print(f"Observation shape: {obs.shape}")
  print(f"Action space: {env.action_space}")
  env.close()
  ```

- [ ] 테스트 실행 및 정상 작동 확인

**완료 기준**: 모든 패키지가 설치되고 MuJoCo 환경이 정상적으로 로드됨

---

## 🎯 Phase 2: PPO 학습 준비 (예상 소요: 1시간)

### Task 2.1: rllib_mujoco.py 코드 이해
- [ ] PPOConfig 파라미터 분석
  - Lambda (GAE): 0.95
  - Learning rate: 0.0003
  - Batch sizes: train_batch_size=16384, minibatch_size=4096
  - Network architecture: [64, 64] with tanh activation

- [ ] 학습 설정 검토
  - num_learners=0 (로컬 학습)
  - num_env_runners=1
  - evaluation 설정

### Task 2.2: 체크포인트 저장 설정 추가
- [ ] 코드에 체크포인트 저장 로직 추가
  ```python
  # 학습 루프 수정
  for i in range(5):
      res = algo.train()
      print(f"Iteration {i+1}")
      print(f"Episode reward mean: {res['env_runners']['episode_reward_mean']}")
      
      # 체크포인트 저장
      if (i + 1) % 1 == 0:  # 매 iteration마다 저장
          checkpoint_dir = algo.save()
          print(f"Checkpoint saved at: {checkpoint_dir}")
  ```

### Task 2.3: 로깅 개선
- [ ] 학습 메트릭 추출 및 저장
  ```python
  import json
  
  training_history = []
  for i in range(5):
      res = algo.train()
      
      # 중요 메트릭 저장
      metrics = {
          "iteration": i + 1,
          "episode_reward_mean": res['env_runners']['episode_reward_mean'],
          "episode_len_mean": res['env_runners']['episode_len_mean'],
          "policy_loss": res.get('info', {}).get('learner', {}).get('default_policy', {}).get('policy_loss', 0),
          "vf_loss": res.get('info', {}).get('learner', {}).get('default_policy', {}).get('vf_loss', 0),
      }
      training_history.append(metrics)
      
  # 결과 저장
  with open('training_history.json', 'w') as f:
      json.dump(training_history, f, indent=2)
  ```

**완료 기준**: 학습 코드가 체크포인트를 저장하고 메트릭을 기록하도록 수정됨

---

## 🎯 Phase 3: 모델 학습 실행 (예상 소요: 2-4시간)

### Task 3.1: 초기 학습 실행
- [ ] 기본 설정으로 학습 시작
  ```bash
  python rllib_mujoco.py
  ```

- [ ] 학습 진행 모니터링
  - Episode reward mean 추이 관찰
  - 학습 안정성 확인
  - 메모리 사용량 모니터링

### Task 3.2: 체크포인트 확인
- [ ] 저장된 체크포인트 위치 확인
  ```bash
  ls -la ~/ray_results/PPO_*/checkpoint_*
  ```

- [ ] 최신 체크포인트 경로 기록
  ```bash
  # 예시
  # ~/ray_results/PPO_HalfCheetah-v5_2025-11-04_10-30-45/checkpoint_000005
  ```

### Task 3.3: TensorBoard 모니터링 (선택사항)
- [ ] TensorBoard 실행
  ```bash
  tensorboard --logdir ~/ray_results
  ```

- [ ] 브라우저에서 http://localhost:6006 접속
- [ ] 학습 곡선 실시간 확인

**완료 기준**: 학습이 완료되고 최소 1개 이상의 체크포인트가 저장됨

---

## 🎯 Phase 4: 평가 스크립트 구현 (예상 소요: 1시간)

### Task 4.1: compute_action() 함수 구현
- [ ] rllib_mujoco_compute_action.py 수정
  ```python
  from ray.rllib.algorithms.algorithm import Algorithm
  
  # 체크포인트 경로 설정
  ckpt_path = "~/ray_results/PPO_HalfCheetah-v5_XXXXX/checkpoint_000005"
  algo_eval = Algorithm.from_checkpoint(ckpt_path)
  
  def compute_action(obs):
      """학습된 정책으로 행동 선택"""
      action = algo_eval.compute_single_action(obs, explore=False)
      return action
  ```

### Task 4.2: 평가 로직 개선
- [ ] 더 상세한 평가 정보 추가
  ```python
  returns = []
  episode_lengths = []
  
  for ep in range(NUM_EVAL_EPISODES):
      obs, info = env.reset()
      done = False
      ep_ret = 0.0
      ep_len = 0
      
      while not done:
          action = compute_action(obs)
          obs, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          ep_ret += float(reward)
          ep_len += 1
      
      returns.append(ep_ret)
      episode_lengths.append(ep_len)
      print(f"[EVAL] Episode {ep+1}/{NUM_EVAL_EPISODES}: return={ep_ret:.3f}, length={ep_len}")
  ```

### Task 4.3: 결과 저장
- [ ] 평가 결과를 JSON 파일로 저장
  ```python
  import json
  
  eval_results = {
      "num_episodes": NUM_EVAL_EPISODES,
      "mean_return": mean_ret,
      "std_return": std_ret,
      "mean_episode_length": float(np.mean(episode_lengths)),
      "returns": returns,
      "episode_lengths": episode_lengths
  }
  
  with open('evaluation_results.json', 'w') as f:
      json.dump(eval_results, f, indent=2)
  ```

**완료 기준**: 평가 스크립트가 체크포인트를 로드하고 10개 에피소드를 평가함

---

## 🎯 Phase 5: 평가 실행 및 검증 (예상 소요: 30분)

### Task 5.1: 평가 실행
- [ ] 평가 스크립트 실행
  ```bash
  python rllib_mujoco_compute_action.py
  ```

- [ ] 출력 결과 확인
  - 각 에피소드의 리턴 값
  - 평균 및 표준편차
  - 에피소드 길이

### Task 5.2: 베이스라인 비교
- [ ] 랜덤 정책과 비교
  ```python
  # compute_action()을 랜덤으로 변경
  def compute_action(obs):
      return env.action_space.sample()
  ```

- [ ] 학습된 모델과 랜덤 정책의 성능 차이 비교

**완료 기준**: 평가가 성공적으로 완료되고 학습된 모델이 랜덤 정책보다 우수한 성능을 보임

---

## 🎯 Phase 6: 결과 분석 및 시각화 (예상 소요: 1시간)

### Task 6.1: 학습 곡선 시각화
- [ ] 시각화 스크립트 작성 (visualize_results.py)
  ```python
  import json
  import matplotlib.pyplot as plt
  
  # 학습 데이터 로드
  with open('training_history.json', 'r') as f:
      history = json.load(f)
  
  # 학습 곡선 그리기
  iterations = [h['iteration'] for h in history]
  rewards = [h['episode_reward_mean'] for h in history]
  
  plt.figure(figsize=(10, 6))
  plt.plot(iterations, rewards, marker='o')
  plt.xlabel('Iteration')
  plt.ylabel('Episode Reward Mean')
  plt.title('PPO Training Progress on HalfCheetah-v5')
  plt.grid(True)
  plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
  plt.show()
  ```

### Task 6.2: 평가 결과 시각화
- [ ] 평가 결과 히스토그램 생성
  ```python
  with open('evaluation_results.json', 'r') as f:
      eval_data = json.load(f)
  
  plt.figure(figsize=(10, 6))
  plt.hist(eval_data['returns'], bins=10, edgecolor='black')
  plt.axvline(eval_data['mean_return'], color='r', linestyle='--', 
              label=f"Mean: {eval_data['mean_return']:.2f}")
  plt.xlabel('Episode Return')
  plt.ylabel('Frequency')
  plt.title('Distribution of Evaluation Returns')
  plt.legend()
  plt.savefig('evaluation_distribution.png', dpi=300, bbox_inches='tight')
  plt.show()
  ```

### Task 6.3: 비교 분석
- [ ] 학습된 모델 vs 랜덤 정책 비교 표 작성
- [ ] 하이퍼파라미터의 영향 분석 (선택사항)

**완료 기준**: 학습 및 평가 결과가 시각화되고 분석됨

---

## 🎯 Phase 6B: 병렬화 효율성 실험 (예상 소요: 3-4시간)

### Task 6B.1: 실험 설계
- [ ] 테스트할 파라미터 조합 정의
  ```python
  # 병렬화 변수
  - num_env_runners: [1, 2, 4, 8]
  - num_envs_per_env_runner: [1, 2, 4, 8]
  
  # 측정 항목
  - time_this_iter_s: 반복당 소요 시간
  - SPS (Steps Per Second): 처리량
  - CPU utilization: CPU 사용률
  - GPU utilization: GPU 사용률
  - RAM utilization: 메모리 사용률
  - VRAM utilization: GPU 메모리 사용률
  ```

- [ ] 실험 조합 선정
  ```python
  experiments_config = [
      # Baseline
      {'num_env_runners': 1, 'num_envs_per_env_runner': 1},
      
      # 러너 수 증가
      {'num_env_runners': 2, 'num_envs_per_env_runner': 1},
      {'num_env_runners': 4, 'num_envs_per_env_runner': 1},
      
      # 러너당 환경 수 증가
      {'num_env_runners': 1, 'num_envs_per_env_runner': 2},
      {'num_env_runners': 1, 'num_envs_per_env_runner': 4},
      
      # 조합
      {'num_env_runners': 2, 'num_envs_per_env_runner': 2},
      {'num_env_runners': 2, 'num_envs_per_env_runner': 4},
      {'num_env_runners': 4, 'num_envs_per_env_runner': 2},
  ]
  ```

### Task 6B.2: 자동화 스크립트 작성 ✓
- [x] parallel_efficiency_experiment.py 작성
  - 각 설정별 자동 학습 실행
  - 시스템 리소스 모니터링 (psutil, GPUtil)
  - 메트릭 수집 및 JSON 저장
  - 실패 시 복구 및 중간 저장

- [x] analyze_parallel_efficiency.py 작성
  - 실험 결과 로드 및 분석
  - 확장성(Scalability) 분석
  - 병렬 효율성 계산
  - 종합 시각화

### Task 6B.3: 필수 패키지 설치
- [ ] 리소스 모니터링 패키지 설치
  ```bash
  pip install psutil
  pip install gputil  # GPU 모니터링용 (선택)
  ```

### Task 6B.4: 실험 실행
- [ ] 병렬화 실험 실행
  ```bash
  python parallel_efficiency_experiment.py
  ```
  
  예상 소요 시간:
  - 8개 설정 × 3 iterations × 약 5분 = 약 2시간
  - 실제 시간은 시스템 사양에 따라 달라질 수 있음

- [ ] 실시간 진행 상황 모니터링
  - 콘솔 출력 확인
  - parallel_experiments_progress.json 주기적 확인

### Task 6B.5: 결과 분석 및 시각화
- [ ] 분석 스크립트 실행
  ```bash
  python analyze_parallel_efficiency.py
  ```

- [ ] 생성되는 분석 자료
  - **parallel_efficiency_analysis.png**
    - Throughput vs Parallelism
    - Speedup Analysis
    - Parallel Efficiency
    - Training Time Comparison
    - Learning Performance
    - Performance Summary
  
  - **resource_utilization_analysis.png**
    - CPU Utilization
    - RAM Utilization
    - GPU Utilization
    - VRAM Utilization
  
  - **parallel_efficiency_report.txt**
    - 상세 실험 결과 요약
    - 확장성 분석
    - 병목 현상 분석

### Task 6B.6: 분석 포인트

#### 1. 처리량(Throughput) 분석
- [ ] SPS (Steps Per Second) 추이 확인
- [ ] 병렬화 수준에 따른 SPS 증가율
- [ ] 이상적인 선형 확장과 실제 성능 비교

#### 2. 확장성(Scalability) 분석
- [ ] Speedup 계산: `실제 SPS / 베이스라인 SPS`
- [ ] 병렬 효율성 계산: `Speedup / 병렬화 수준 × 100%`
- [ ] 확장성 한계점 파악

#### 3. 자원 병목(Resource Bottleneck) 분석
- [ ] CPU 사용률이 병목인가?
  - 높은 병렬화에서 CPU 100% 도달 시
- [ ] 메모리가 병목인가?
  - RAM 사용률이 90% 이상일 때
- [ ] GPU가 병목인가?
  - GPU 사용률이 낮으면 데이터 전송 병목
- [ ] I/O가 병목인가?
  - 디스크 읽기/쓰기 대기 시간

#### 4. 최적 설정 도출
- [ ] 최고 처리량 설정 식별
- [ ] 최고 효율성 설정 식별
- [ ] 비용 대비 성능 최적점 찾기

### Task 6B.7: 실험 결과 해석
- [ ] 왜 선형 확장이 되지 않는가?
  - 통신 오버헤드
  - 동기화 비용
  - 공유 자원 경쟁
  - 직렬화 구간 존재

- [ ] 어느 시점부터 효율이 떨어지는가?
  - 임계 병렬화 수준
  - 성능 포화점

- [ ] 자원별 최적 활용 방안
  - CPU 코어 수에 맞는 러너 수
  - 메모리 용량에 맞는 환경 수

**완료 기준**: 
- 8개 병렬화 설정 실험 완료
- 분석 그래프 및 보고서 생성
- 병목 현상 및 최적 설정 도출

---

## 🎯 Phase 7: 보고서 작성 (예상 소요: 2시간)

### Task 7.1: 실험 방법 문서화
- [ ] 사용한 환경 및 알고리즘 설명
- [ ] 하이퍼파라미터 선택 근거
- [ ] 학습 과정 설명

### Task 7.2: 결과 정리
- [ ] 학습 결과 요약
  - 최종 평균 리워드
  - 학습 소요 시간
  - 수렴 여부

- [ ] 평가 결과 요약
  - 10개 에피소드 평균 리턴
  - 표준편차
  - 랜덤 정책 대비 성능 향상

### Task 7.3: 분석 및 고찰
- [ ] 학습 과정에서 관찰된 현상
- [ ] 성능 개선을 위한 시도 (있다면)
- [ ] 한계점 및 개선 방향

### Task 7.4: 보고서 구조
```markdown
# 강화학습 프로젝트 4 보고서

## 1. 실험 개요
- 목적
- 환경 및 알고리즘

## 2. 실험 설정
- 하이퍼파라미터
- 네트워크 구조
- 학습 설정

## 3. 실험 결과
- 학습 곡선
- 평가 결과
- 시각화 자료

## 4. 분석 및 고찰
- 관찰된 현상
- 성능 분석
- 개선 시도

## 5. 결론
- 요약
- 한계점
- 향후 연구 방향
```

**완료 기준**: 상세한 보고서가 작성됨

---

## 🎯 Phase 8: 제출 준비 (예상 소요: 30분)

### Task 8.1: 파일 정리
- [ ] 제출 파일 목록 확인
  ```
  reinforce-project4/
  ├── README.md
  ├── PROJECT_PLAN.md (이 파일)
  ├── rllib_mujoco.py (수정됨)
  ├── rllib_mujoco_compute_action.py (완성됨)
  ├── training_history.json
  ├── evaluation_results.json
  ├── training_curve.png
  ├── evaluation_distribution.png
  ├── visualize_results.py
  └── REPORT.md
  ```

### Task 8.2: 체크포인트 압축
- [ ] 최종 체크포인트 폴더 압축
  ```bash
  cd ~/ray_results
  tar -czf checkpoint_final.tar.gz PPO_HalfCheetah-v5_*/checkpoint_000005
  ```

### Task 8.3: 코드 실행 가능성 검증
- [ ] 새로운 터미널에서 전체 파이프라인 재실행
  ```bash
  # 1. 학습
  python rllib_mujoco.py
  
  # 2. 평가
  python rllib_mujoco_compute_action.py
  
  # 3. 시각화
  python visualize_results.py
  ```

### Task 8.4: 최종 점검
- [ ] README.md 업데이트 (실제 결과 반영)
- [ ] 모든 그래프와 수치 재확인
- [ ] 보고서 맞춤법 검사
- [ ] 제출 요구사항 재확인

**완료 기준**: 모든 파일이 정리되고 제출 준비 완료

---

## 📊 체크리스트 요약

### 필수 항목
- [ ] 학습 코드 실행 완료
- [ ] 체크포인트 저장 완료
- [ ] 평가 코드 구현 완료
- [ ] 평가 실행 완료 (10 에피소드)
- [ ] 학습 곡선 그래프 생성
- [ ] 보고서 작성 완료

### 선택 항목
- [ ] TensorBoard 모니터링
- [ ] 하이퍼파라미터 튜닝 실험
- [ ] 추가 분석 (policy entropy, value loss 등)
- [ ] 여러 시드로 실험 반복

---

## 🚨 주의사항

1. **메모리 관리**: 학습 중 메모리 사용량 모니터링
2. **체크포인트 백업**: 학습 중간에 체크포인트 복사본 저장
3. **경로 확인**: 체크포인트 경로가 정확한지 확인
4. **환경 일관성**: 학습과 평가 시 동일한 환경 사용
5. **시간 관리**: Phase별 예상 시간 고려하여 계획적으로 진행

---

## 📝 진행 상황 기록

### 작업 로그
```
[YYYY-MM-DD HH:MM] Phase 1 시작
[YYYY-MM-DD HH:MM] Phase 1 완료
[YYYY-MM-DD HH:MM] Phase 2 시작
...
```

### 이슈 및 해결
```
Issue 1: MuJoCo 설치 오류
Solution: pip install mujoco 대신 conda install mujoco 사용

Issue 2: GPU 메모리 부족
Solution: num_gpus_per_learner=0으로 변경하여 CPU 사용
```

---

## 🎓 학습 목표 달성도

- [ ] Ray RLlib 사용법 이해
- [ ] PPO 알고리즘 이해
- [ ] MuJoCo 환경 사용법 이해
- [ ] 체크포인트 저장/로드 방법 이해
- [ ] 강화학습 평가 방법 이해
- [ ] 결과 분석 및 시각화 능력 향상

---

## 📚 추가 학습 자료

- [PPO 논문 읽기](https://arxiv.org/abs/1707.06347)
- [Ray RLlib 튜토리얼](https://docs.ray.io/en/latest/rllib/rllib-training.html)
- [MuJoCo 환경 설명](https://gymnasium.farama.org/environments/mujoco/)
- [강화학습 평가 베스트 프랙티스](https://spinningup.openai.com/en/latest/spinningup/bench.html)

---

## 💡 성공을 위한 팁

1. **단계별 진행**: 각 Phase를 순차적으로 완료
2. **자주 저장**: 중간 결과를 자주 저장
3. **문서화**: 실험 과정을 상세히 기록
4. **검증**: 각 단계 완료 후 결과 검증
5. **질문**: 막힐 때는 조교나 동료에게 질문

---

**작성일**: 2025-11-04  
**최종 수정일**: 2025-11-04  
**예상 총 소요 시간**: 8-12시간
