# 병렬화 효율성 실험 (Parallel Efficiency Experiments)

Ray RLlib PPO의 병렬화 설정이 학습 성능에 미치는 영향을 체계적으로 분석합니다.

## 📁 디렉토리 구조

```
parallel_efficiency_experiments/
├── README.md                              # 이 파일
├── EXPERIMENT_ANALYSIS.md                 # 상세한 실험 분석 및 재실험 제안
├── GPU_MEASUREMENT_GUIDE.md               # GPU 측정 가이드
│
├── parallel_efficiency_experiment.py      # 원본 실험 스크립트 (12개 구성)
├── parallel_experiment_runners_only.py    # 재실험 스크립트 (러너만 변경)
│
├── analyze_parallel_efficiency_simple.py  # 원본 분석 스크립트
├── analyze_runners_only.py                # 러너 전용 분석 스크립트
│
├── results/                               # 실험 결과 파일들
│   ├── parallel_experiments_final.json    # 원본 실험 결과
│   ├── parallel_experiments_progress.json
│   ├── parallel_efficiency_dashboard.png  # 시각화
│   ├── parallel_efficiency_report.txt     # 텍스트 리포트
│   ├── runners_only_final.json            # 재실험 결과 (생성 예정)
│   └── runners_only_dashboard.png         # 재실험 시각화 (생성 예정)
│
└── archive/                               # 사용하지 않는 파일들
    ├── experiment_utils.py
    ├── analyze_gpu_efficiency.py
    └── parallel_efficiency.md
```

## 🎯 실험 목적

Ray RLlib의 주요 병렬화 파라미터가 학습 속도에 미치는 영향 분석:
- `num_env_runners`: 환경 샘플링을 수행하는 워커 수
- `num_envs_per_env_runner`: 각 워커가 실행하는 환경 인스턴스 수

## 📊 주요 발견 (원본 실험)

### 실험 1: 12개 구성 테스트

**실행 날짜**: 2025-11-04  
**환경**: HalfCheetah-v5 (MuJoCo)  
**시스템**: i7-12700 (16 logical cores), 32GB RAM, RTX 3070 8GB

#### 결과 요약

| Config | Total Envs | Speedup | Efficiency | 추천 |
|--------|-----------|---------|-----------|------|
| 2r×1e  | 2         | 1.72×   | 86.2%     | ⭐⭐⭐ 최고 효율 |
| 4r×1e  | 4         | 2.68×   | 67.0%     | ⭐⭐ 균형잡힌 선택 |
| 8r×1e  | 8         | 4.12×   | 51.5%     | ⭐ 여전히 양호 |
| 8r×2e  | 16        | 5.76×   | 36.0%     | ❌ 비효율적 |

#### 핵심 인사이트

1. **러너 증가 > 환경 증가**
   - `2r×1e` (86.2%) > `1r×2e` (78.4%)
   - 러너 병렬화가 더 효율적

2. **확장성 한계**
   - 8 envs까지는 50%+ 효율성 유지
   - 16 envs에서는 35-36%로 급격히 감소

3. **GPU 활용 불가**
   - MuJoCo는 CPU 시뮬레이션
   - GPU 사용률 0-5% (거의 미사용)

### 문제점 및 한계

❌ **현재 실험의 한계**:
- env_runners vs envs_per_runner 비교가 불명확
- GPU 측정이 의미 없음 (MuJoCo는 CPU 기반)
- 확장성 한계의 원인 불명확
- 변수가 2개라서 인과관계 파악 어려움

✅ **개선 방안** → `EXPERIMENT_ANALYSIS.md` 참조

## 🚀 빠른 시작

### 1. 원본 실험 결과 분석

```bash
cd parallel_efficiency_experiments
python analyze_parallel_efficiency_simple.py
```

**출력**:
- `results/parallel_efficiency_dashboard.png`: 3-5개 차트 (GPU 포함 여부에 따라 다름)
- `results/parallel_efficiency_report.txt`: 텍스트 리포트

### 2. 재실험 실행 (권장)

**목적**: 러너 수만 변경하여 순수 병렬화 효과 측정

```bash
# 기본 실행 (1,2,3,4,6,8,12,16 러너)
python parallel_experiment_runners_only.py

# 커스텀 러너 수
python parallel_experiment_runners_only.py --runners "1,2,4,8,16"

# 반복 횟수 조정
python parallel_experiment_runners_only.py --iterations 10
```

**결과 분석**:
```bash
python analyze_runners_only.py
```

## 📈 주요 메트릭

### 측정 항목

1. **time_this_iter_s**: 각 iteration 소요 시간
2. **SPS** (Steps Per Second): 샘플 처리량
3. **Speedup**: 기준선 대비 속도 향상 (baseline_time / current_time)
4. **Efficiency**: 병렬 효율성 (speedup / total_envs × 100%)

### 평가 기준

- **Excellent**: Efficiency > 70%
- **Good**: Efficiency 60-70%
- **Acceptable**: Efficiency 50-60%
- **Poor**: Efficiency < 50%

## 🔧 실험 구성

### 하드웨어 사양

- **CPU**: Intel i7-12700 (8P+4E cores, 16 logical)
- **RAM**: 32GB (컨테이너에서 15.54GB 사용 가능)
- **GPU**: NVIDIA RTX 3070 8GB VRAM

### PPO 설정

```python
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .training(
        lambda_=0.95,
        lr=0.0003,
        num_epochs=3,
        train_batch_size=16384,
        minibatch_size=4096,
    )
    .env_runners(
        num_env_runners=N,  # 변경되는 파라미터
        num_envs_per_env_runner=M,  # 변경되는 파라미터
    )
)
```

## 📝 재실험 제안

상세한 분석은 `EXPERIMENT_ANALYSIS.md` 참조

### 옵션 A: 러너만 측정 (⭐⭐ 최추천)

**목적**: 변수 1개만 변경하여 명확한 인과관계 파악

```bash
python parallel_experiment_runners_only.py
```

**장점**:
- 결과 해석 간단
- 실험 시간 단축
- 실용적 가치 높음

### 옵션 B: GPU 활용 극대화

**목적**: GPU를 실제로 사용하는 구성 테스트

**변경 사항**:
- `num_epochs`: 3 → 20
- `train_batch_size`: 16384 → 65536
- 신경망 크기 증가
- `num_learners`: 0 → 1

### 옵션 C: 세밀한 측정

**목적**: 물리 코어 vs 논리 코어 차이 확인

**테스트**: 1, 2, 3, 4, 6, 8, 12, 16 러너

## 🎓 학습 자료

### 관련 문서

- `EXPERIMENT_ANALYSIS.md`: 상세 분석 + 재실험 제안
- `GPU_MEASUREMENT_GUIDE.md`: GPU 메트릭 수집 방법

### 핵심 개념

**Strong Scaling**: 작업량 고정, 프로세서 증가  
→ 이상적으로는 N배 프로세서 = N배 속도

**Parallel Efficiency**: 실제 speedup / 이론적 speedup  
→ 오버헤드가 적을수록 100%에 가까움

**Amdahl's Law**: 병렬화 가능한 부분만 가속  
→ 직렬 부분(통신, 동기화)이 bottleneck

## 🔍 트러블슈팅

### GPU 사용률이 낮음

**원인**: MuJoCo는 CPU 시뮬레이션  
**해결**: 정상 - GPU는 신경망 학습에만 사용됨

### 효율성이 낮음 (< 50%)

**원인**: 
- 통신 오버헤드
- 동기화 대기 시간
- 캐시 경합

**해결**:
- 러너 수 줄이기
- 배치 크기 늘리기
- 더 긴 에피소드 사용

### 실험이 실패함

**디버깅**:
```bash
# 진행 상황 확인
tail -f results/parallel_experiments_progress.json

# 시스템 리소스 확인
htop  # CPU/RAM
nvidia-smi  # GPU
```

## 📚 참고 자료

- [Ray RLlib 공식 문서](https://docs.ray.io/en/latest/rllib/)
- [Parallel Training Guide](https://docs.ray.io/en/latest/rllib/rllib-training.html#scaling-guide)
- [PPO Algorithm](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)

## 🤝 기여

실험 결과나 개선 사항이 있다면 이슈를 열어주세요!
