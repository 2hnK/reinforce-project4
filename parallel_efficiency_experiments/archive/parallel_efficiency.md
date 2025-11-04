# Phase 6B: 병렬화 효율성 실험 실행 가이드

## 📋 개요

이 가이드는 Ray RLlib의 병렬화 설정에 따른 성능 분석을 수행하는 방법을 설명합니다.

## 🎯 실험 목적

다양한 병렬화 설정(`num_env_runners`, `num_envs_per_env_runner`)이 학습 성능에 미치는 영향을 정량적으로 분석하고, 최적의 설정을 도출합니다.

---

## 📦 1단계: 필수 패키지 설치

### 기본 패키지
```bash
pip install ray[rllib]
pip install gymnasium
pip install gymnasium[mujoco]
pip install numpy matplotlib
```

### 리소스 모니터링 패키지
```bash
# 필수: CPU, RAM 모니터링
pip install psutil

# 선택: GPU 모니터링 (GPU가 있는 경우)
pip install gputil
```

### 설치 확인
```bash
python -c "import psutil; print('psutil OK')"
python -c "import GPUtil; print('GPUtil OK')"  # GPU 있는 경우
```

---

## 🧪 2단계: 실험 실행

### 2.1 실험 스크립트 확인

파일 구조:
```
reinforce-project4/
├── parallel_efficiency_experiment.py    # 실험 실행
├── analyze_parallel_efficiency.py       # 결과 분석
├── rllib_mujoco.py                      # 기본 학습 스크립트
└── PROJECT_PLAN.md                      # 전체 플랜
```

### 2.2 실험 실행

```bash
# 실험 시작 (예상 소요: 2-3시간)
python parallel_efficiency_experiment.py
```

**실행 중 확인사항:**
- 각 설정별로 3 iterations 학습
- 총 8개 설정 조합 테스트
- 중간 결과가 `parallel_experiments_progress.json`에 저장됨
- 실험 실패 시 자동으로 다음 설정으로 진행

**진행 상황 모니터링:**
```bash
# 다른 터미널에서
watch -n 5 'tail -20 parallel_experiments_progress.json'

# 또는 실시간 파일 크기 확인
watch -n 5 'ls -lh parallel_experiments_*.json'
```

### 2.3 실험 중단 및 재개

실험이 중단된 경우:
- `parallel_experiments_progress.json` 파일이 이미 저장되어 있음
- 완료된 실험은 건너뛰고 이어서 실행하려면 스크립트 수정 필요
- 처음부터 다시 시작하려면 JSON 파일 삭제 후 재실행

---

## 📊 3단계: 결과 분석

### 3.1 분석 실행

실험이 완료되면:
```bash
python analyze_parallel_efficiency.py
```

### 3.2 생성되는 파일

#### 1. parallel_efficiency_analysis.png
6개 차트로 구성:
- **Throughput vs Parallelism**: 병렬화 수준에 따른 처리량
- **Speedup Analysis**: 베이스라인 대비 속도 향상
- **Parallel Efficiency**: 병렬 효율성 (%)
- **Training Time Comparison**: 학습 시간 비교
- **Learning Performance**: 학습 성능 (보상)
- **Performance Summary**: 요약 통계

#### 2. resource_utilization_analysis.png
시스템 리소스 사용률:
- CPU Utilization
- RAM Utilization
- GPU Utilization (GPU 있는 경우)
- VRAM Utilization (GPU 있는 경우)

#### 3. parallel_efficiency_report.txt
텍스트 보고서:
- 실험 요약
- 각 설정별 상세 결과
- 확장성 분석
- 최적 설정 추천

### 3.3 결과 해석

```bash
# 보고서 확인
cat parallel_efficiency_report.txt

# 또는
less parallel_efficiency_report.txt
```

---

## 📈 4단계: 결과 해석 가이드

### 4.1 핵심 메트릭

#### SPS (Steps Per Second)
```
높을수록 좋음
= 단위 시간당 처리하는 환경 스텝 수
= 학습 속도의 직접적 지표
```

#### Speedup
```
Speedup = (현재 설정의 SPS) / (베이스라인 SPS)
이상적: Speedup = 병렬화 수준 (N개 병렬 → N배 빠름)
실제: 통신 오버헤드로 인해 이상적보다 낮음
```

#### Parallel Efficiency
```
Efficiency = (Speedup / 병렬화 수준) × 100%
100%: 완벽한 선형 확장
80% 이상: 우수한 확장성
60-80%: 양호한 확장성
60% 미만: 병목 현상 존재
```

### 4.2 병목 현상 진단

#### CPU 병목
```
증상: CPU 사용률 90% 이상
원인: CPU 코어 수 < 총 작업 수
해결: num_env_runners 감소 또는 CPU 업그레이드
```

#### 메모리 병목
```
증상: RAM 사용률 90% 이상
원인: 너무 많은 환경 동시 실행
해결: num_envs_per_env_runner 감소
```

#### GPU 병목
```
증상: GPU 사용률 100%, 하지만 SPS 낮음
원인: 데이터 전송 병목 또는 배치 크기 부족
해결: 배치 크기 조정, 데이터 파이프라인 최적화
```

#### 통신 오버헤드
```
증상: 병렬화 증가해도 Speedup 증가 미미
원인: Ray의 프로세스 간 통신 비용
해결: 러너당 환경 수를 증가 (통신 빈도 감소)
```

### 4.3 최적 설정 선택

**목표별 추천:**

1. **최대 처리량 (빠른 학습)**
   - 가장 높은 SPS를 달성한 설정
   - 학습 시간 최소화
   - 자원 사용률이 높아도 OK

2. **최고 효율성 (비용 효율)**
   - 가장 높은 Parallel Efficiency
   - 자원 대비 성능 최적
   - 제한된 자원 환경에 적합

3. **균형잡힌 설정 (추천)**
   - Efficiency 80% 이상
   - 자원 사용률 80% 이하
   - 안정적인 학습

---

## 🔍 5단계: 심화 분석

### 5.1 추가 실험 아이디어

#### 실험 1: 더 많은 병렬화
```python
# parallel_efficiency_experiment.py 수정
experiments_config = [
    {'num_env_runners': 8, 'num_envs_per_env_runner': 1},
    {'num_env_runners': 16, 'num_envs_per_env_runner': 1},
    # ...
]
```

#### 실험 2: 긴 학습으로 수렴성 확인
```python
# num_iterations를 3에서 10으로 증가
run_experiment(
    num_env_runners=4,
    num_envs_per_env_runner=2,
    num_iterations=10  # 더 긴 학습
)
```

#### 실험 3: 다른 환경에서 테스트
```python
config = (
    PPOConfig()
    .environment("Ant-v5")  # 더 복잡한 환경
    # ...
)
```

### 5.2 결과 비교

여러 실험 결과를 비교하려면:
```python
# analyze_parallel_efficiency.py 수정
import glob
import pandas as pd

# 모든 실험 파일 로드
files = glob.glob('parallel_experiments_*.json')
results = []

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        results.append(data)

# 비교 분석...
```

---

## 📝 6단계: 보고서 작성

### 분석 항목

#### 1. 실험 설계
```markdown
- 테스트한 병렬화 설정 나열
- 각 설정의 의도 설명
- 측정 메트릭 정의
```

#### 2. 실험 결과
```markdown
- 각 설정별 SPS, Speedup, Efficiency 표
- 시각화 그래프 첨부
- 자원 사용률 분석
```

#### 3. 병목 현상 분석
```markdown
- 확장성 한계점 식별
- 병목 원인 분석 (CPU/RAM/GPU/통신)
- 임계점 도출
```

#### 4. 최적 설정 제안
```markdown
- 목적별 최적 설정 추천
- Trade-off 분석
- 실무 적용 가이드
```

#### 5. 결론 및 향후 연구
```markdown
- 주요 발견사항 요약
- 한계점
- 개선 방향
```

---

## ⚠️ 주의사항

### 실험 시간
- 전체 실험: 2-4시간 소요
- 시스템 사양에 따라 달라질 수 있음
- 백그라운드 실행 권장: `nohup python parallel_efficiency_experiment.py > experiment.log 2>&1 &`

### 시스템 부하
- CPU/RAM 사용률 높음
- 다른 작업 동시 수행 자제
- 시스템 온도 모니터링

### 데이터 저장
- JSON 파일 크기: 수 MB
- 충분한 디스크 공간 확보
- 정기적으로 백업

### GPU 사용
- `num_gpus_per_learner=1` 설정 시 GPU 필수
- GPU 없으면 `num_gpus_per_learner=0`으로 변경
- GPU 메모리 부족 시 배치 크기 감소

---

## 🐛 문제 해결

### 문제 1: Out of Memory
```bash
증상: RAM 부족으로 종료
해결: 
  - num_envs_per_env_runner 감소
  - 배치 크기 감소
  - 시스템 재시작 후 재실행
```

### 문제 2: Ray 초기화 실패
```bash
증상: "Ray failed to initialize"
해결:
  ray stop  # 기존 Ray 프로세스 종료
  python parallel_efficiency_experiment.py
```

### 문제 3: GPU 감지 안 됨
```bash
증상: "GPU data not available"
해결:
  pip install gputil
  nvidia-smi  # GPU 작동 확인
```

### 문제 4: 실험 중단
```bash
증상: 스크립트가 중간에 멈춤
해결:
  - progress.json 파일 확인
  - 완료된 실험은 results에서 제외
  - 남은 실험만 재실행
```

---

## 📚 참고 자료

- [Ray RLlib Scaling Guide](https://docs.ray.io/en/latest/rllib/rllib-scaling-guide.html)
- [Parallel and Distributed RL](https://docs.ray.io/en/latest/rllib/rllib-concepts.html)
- [Performance Tuning](https://docs.ray.io/en/latest/ray-core/performance-tips.html)

---

**작성일**: 2025-11-04  
**Phase**: 6B - Parallel Efficiency Experiment  
**예상 소요 시간**: 3-4시간
