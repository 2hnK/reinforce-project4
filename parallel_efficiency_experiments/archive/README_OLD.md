# 🔬 Phase 6B: Parallel Efficiency Experiments

병렬화 효율성 분석 실험 및 결과

## 📁 폴더 구조

```
parallel_efficiency_experiments/
├── README.md                              # 이 파일
│
├── 📊 실험 데이터
├── parallel_experiments_final.json       # 최종 실험 결과 (53KB)
├── parallel_experiments_progress.json    # 증분 실험 결과 (53KB)
│
├── 📈 분석 결과
├── parallel_efficiency_dashboard.png     # 3개 차트 시각화 (400KB)
├── parallel_efficiency_report.txt        # 텍스트 리포트
│
├── 🔧 실험 스크립트
├── parallel_efficiency_experiment.py     # 기본 병렬화 실험 (CPU/RAM)
├── parallel_experiment_with_gpu.py       # GPU 측정 포함 실험
│
├── 📊 분석 스크립트
├── analyze_parallel_efficiency.py        # 원본 분석 (복잡)
├── analyze_parallel_efficiency_simple.py # 간소화 분석 (권장)
├── analyze_gpu_efficiency.py             # GPU 분석
│
└── 📖 문서
    ├── PHASE6B_RESULTS_SUMMARY.md        # 실험 결과 요약 ⭐
    ├── HARDWARE_OPTIMIZED_CONFIG.md      # 하드웨어 최적화 가이드
    ├── GPU_MEASUREMENT_GUIDE.md          # GPU 측정 가이드
    └── parallel_efficiency.md            # 초기 계획 문서
```

## 🎯 실험 개요

### 목적
Ray RLlib PPO의 병렬화 설정별 성능 측정 및 최적 설정 도출

### 시스템 사양
- **CPU**: Intel i7-12700 (16 logical cores, 8 physical cores)
- **RAM**: 32GB (컨테이너: 15.54GB)
- **GPU**: NVIDIA RTX 3070 (8GB VRAM)

### 실험 설정
- **알고리즘**: PPO (Proximal Policy Optimization)
- **환경**: MuJoCo HalfCheetah-v5
- **설정 조합**: 12가지
- **반복**: 각 3 iterations

## 🚀 빠른 시작

### 1. 결과 확인 (이미 완료된 실험)
```bash
cd /home/com/reinforce-project4/parallel_efficiency_experiments

# 요약 보기
cat PHASE6B_RESULTS_SUMMARY.md

# 텍스트 리포트
cat parallel_efficiency_report.txt

# 시각화
xdg-open parallel_efficiency_dashboard.png  # 또는 이미지 뷰어로 열기
```

### 2. 재분석 (새로운 차트 생성)
```bash
python analyze_parallel_efficiency_simple.py
```

### 3. GPU 측정 실험 실행 (30-60분 소요)
```bash
# 필수 패키지 설치
pip install gputil nvidia-ml-py3

# GPU 측정 포함 실험
python parallel_experiment_with_gpu.py

# GPU 분석
python analyze_gpu_efficiency.py
```

## 📊 주요 결과

### 🏆 Top 3 설정

#### 1위: 최고 속도 🚀
```python
config.env_runners(
    num_env_runners=8,
    num_envs_per_env_runner=2
)
```
- **Time**: 2.03초/iteration
- **Speedup**: 5.84×
- **Efficiency**: 36.5%
- **용도**: 빠른 프로토타이핑

#### 2위: 권장 설정 ✅ (프로덕션)
```python
config.env_runners(
    num_env_runners=4,
    num_envs_per_env_runner=2
)
```
- **Time**: 2.87초/iteration
- **Speedup**: 4.14×
- **Efficiency**: 51.8% ✅
- **용도**: 일반 학습, 안정적 운영

#### 3위: 최고 효율 💎
```python
config.env_runners(
    num_env_runners=8,
    num_envs_per_env_runner=1
)
```
- **Time**: 2.74초/iteration
- **Speedup**: 4.33×
- **Efficiency**: 54.1% ✅
- **용도**: 물리 코어 활용, 안정성

### 📈 성능 지표 이해

#### Speedup (속도 향상)
```
Speedup = Baseline 시간 / 현재 설정 시간

예: 11.88s / 2.03s = 5.84×
→ "5.84배 빠르다"
```

#### Efficiency (효율성)
```
Efficiency = (Speedup / 병렬 수) × 100%

예: 5.84 / 16 = 36.5%
→ "이상적 성능의 36.5% 달성"
```

**해석:**
- **70-100%**: 매우 우수 💎
- **50-70%**: 양호 ✅
- **30-50%**: 보통
- **<30%**: 비효율

## 🔬 실험 방법론

### Phase 1: 기본 병렬화 측정 (완료 ✅)
```bash
python parallel_efficiency_experiment.py
```
**측정 항목:**
- CPU 사용률
- RAM 사용량
- 학습 시간
- Speedup & Efficiency

### Phase 2: GPU 측정 (선택사항)
```bash
python parallel_experiment_with_gpu.py
```
**추가 측정:**
- GPU 활용률
- VRAM 사용량
- GPU 온도
- CPU vs GPU 병목 분석

## 📖 문서 가이드

### 🌟 핵심 문서 (필독)
1. **PHASE6B_RESULTS_SUMMARY.md**
   - 전체 실험 결과 요약
   - Top 설정 및 권장 사항
   - 성능 비교 표

2. **parallel_efficiency_report.txt**
   - 간단한 텍스트 리포트
   - 빠른 참조용

### 📚 참고 문서
3. **HARDWARE_OPTIMIZED_CONFIG.md**
   - 하드웨어별 최적 설정
   - 시스템 사양 분석
   - 실험 설계 근거

4. **GPU_MEASUREMENT_GUIDE.md**
   - GPU 측정 방법
   - 결과 해석
   - 트러블슈팅

## 🎯 시나리오별 사용법

### 시나리오 A: 결과만 확인하고 싶음
```bash
# 요약 문서 읽기
cat PHASE6B_RESULTS_SUMMARY.md

# 시각화 확인
xdg-open parallel_efficiency_dashboard.png
```

### 시나리오 B: 다른 설정으로 재실험
```bash
# parallel_efficiency_experiment.py 수정
# experiments_config 리스트에 원하는 설정 추가

python parallel_efficiency_experiment.py
python analyze_parallel_efficiency_simple.py
```

### 시나리오 C: GPU 활용 확인 필요
```bash
pip install gputil nvidia-ml-py3
python parallel_experiment_with_gpu.py
python analyze_gpu_efficiency.py
```

### 시나리오 D: 자신의 환경에서 실험
```bash
# 1. 시스템 사양 확인
python -c "import psutil; print(f'Cores: {psutil.cpu_count()}')"

# 2. 설정 수정 (parallel_efficiency_experiment.py)
experiments_config = [
    {'num_env_runners': 1, 'num_envs_per_env_runner': 1},
    # ... 자신의 코어 수에 맞게 조정
]

# 3. 실행
python parallel_efficiency_experiment.py
python analyze_parallel_efficiency_simple.py
```

## 🔍 주요 발견 사항

### 1. 선형 확장 불가
- 16배 병렬화 → 5.84배 속도 향상
- 이유: 통신 오버헤드, Amdahl's Law

### 2. 러너 분산 > 환경 벡터화
```
8r×1e (54.1%) > 1r×8e (42.5%)
→ 프로세스 분산이 더 효과적
```

### 3. Sweet Spot: 8 total envs
- 4-8개 러너 사용 시 효율 50% 이상 유지
- 속도와 효율의 균형점

### 4. CPU 미활용 (1-3%)
- 짧은 iteration 시간
- GPU 측정 필요

## 🐛 트러블슈팅

### 문제 1: ModuleNotFoundError: 'matplotlib'
```bash
pip install matplotlib
```

### 문제 2: GPU 측정 오류
```bash
pip install gputil nvidia-ml-py3
# 또는
pip install pynvml
```

### 문제 3: Out of Memory
```python
# parallel_efficiency_experiment.py에서
NUM_ITERATIONS = 3  # 줄이기
```

### 문제 4: Ray 오류
```bash
pip uninstall -y ray
pip install --no-cache-dir "ray[rllib]"
```

## 📊 데이터 형식

### parallel_experiments_final.json
```json
{
  "experiment_info": {
    "total_experiments": 12,
    "timestamp": "2025-11-04T..."
  },
  "experiments": [
    {
      "config": {
        "num_env_runners": 4,
        "num_envs_per_env_runner": 2,
        "total_envs": 8
      },
      "summary": {
        "avg_time_per_iter_s": 2.87,
        "avg_sps": 0.0,
        "total_training_time_s": 8.63
      },
      "iterations": [...]
    }
  ]
}
```

## 🎓 다음 단계

### 즉시 적용
```python
# rllib_mujoco.py에 권장 설정 적용
config.env_runners(
    num_env_runners=4,
    num_envs_per_env_runner=2
)
```

### 추가 최적화
1. GPU 활용 측정 및 분석
2. 더 긴 학습 설정 (`num_sgd_iter=10`)
3. Rollout fragment length 조정
4. Multi-GPU 실험

### Phase 7: 최종 리포트
- 전체 실험 통합
- 학습 곡선 시각화
- 성능 벤치마크

## 📞 참고 자료

- [Ray RLlib 공식 문서](https://docs.ray.io/en/latest/rllib/)
- [PPO 알고리즘](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [MuJoCo 환경](https://gymnasium.farama.org/environments/mujoco/)

---

**작성일**: 2025년 11월 4일  
**프로젝트**: reinforce-project4  
**Phase**: 6B - Parallel Efficiency Analysis
