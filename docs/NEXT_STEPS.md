# Next Steps

최신 기획안(역할 ①~⑤) 기준으로, 지금 상태에서 무엇부터 시작할지 정리한 문서입니다.
(2026-11-09 작성. 비교 대상: 팀 저장소 `261029_TetherRescue`, 연습 프로젝트 `261106_TetherRescue2`)

---

## 결론

**① 물리 코어**와 **② 공통 환경 API·데이터 규격**을 먼저 확정합니다.

- ③ PanicSim, ④ AirwayTow, ⑤ TetherSense는 모두 ①이 만든 바다·로봇 위에서, ②의 API로 돌아갑니다.
- 이 둘이 없으면 나머지 세 파트가 각자 따로 만든 뒤 나중에 다시 맞춰야 합니다.
- 연습 프로젝트에 ①·②의 상당 부분이 이미 있으므로, 이를 출발점으로 쓰면 시간을 크게 줄일 수 있습니다.

---

## 현재 상태

### 팀 저장소

- 아직 이전 Unity 기획 구조입니다. `unity_env/` 폴더와 `feature/unity` 브랜치가 남아 있습니다.
- 새 기획안의 `sim/`, `ros2_ws/` 폴더와 `feature/sim`, `feature/infra` 브랜치는 아직 없습니다.

### 연습 프로젝트 ↔ 새 기획안 대응

| 새 기획안 | 연습 프로젝트에 이미 있는 것 | 비어 있는 것 |
|---|---|---|
| ① `sim/water.py` | [`waves.py`](../tether_rescue/ocean/waves.py), [`hydro.py`](../tether_rescue/ocean/hydro.py) (복셀 부력·저항), [`added_mass.py`](../tether_rescue/ocean/added_mass.py) | — |
| ① `robot.py`, `world.py` | [`scene.py`](../tether_rescue/scene.py) (선체·추진기·조향 선외기), 전복 판정, 물리 테스트, [`REALISM_GAPS.md`](REALISM_GAPS.md) (사실성 점검) | 지금은 XML 문자열로 조립함. 기획안은 **MjSpec** 방식 |
| ② `sim/base_env.py` | [`tow_env.py`](../tether_rescue/tow_env.py)의 `reset`/`step`/관측 dict | Gymnasium 형태의 공통 API |
| ② `telemetry.py` | [`run_log.py`](../tether_rescue/ui/run_log.py) (판별 결과 기록만) | 50 Hz 시계열 CSV 기록 |
| ② `ros2_ws/` | `bridge/` (UDP → WSL → rviz2, 마커·목표 좌표) | `.msg` 정의, launch 파일 |
| ③ `rescuee_fsm.py` | [`boarding.py`](../tether_rescue/mission/boarding.py) (평온 → 배를 봄 → 달려듦 → 매달림), [`handoff.py`](../tether_rescue/handoff.py) | 체력·의식 등 패닉 상태 전체, 보상 함수 |
| ④ `tether.py`, `pid_controller.py`, `tow_env.py` | [`tether.py`](../tether_rescue/tow/tether.py) (k·c·끊김), [`pid.py`](../tether_rescue/tow/pid.py), [`sweep_speed.py`](../scripts/sweep_speed.py) | Composite Cost, 파레토, TowAgent |
| ⑤ `domain_randomizer.py` | 설정값 구조([`config.py`](../tether_rescue/config.py)), 노이즈가 섞인 IMU | 데이터셋, 1D CNN, 실시간 추론 |

---

## 추천 순서

### 1단계 — 팀이 함께 정할 것 (이번 주)

1. **연습 프로젝트를 물리 코어의 출발점으로 쓸지** ① 담당자와 결정합니다.
   쓰기로 하면 `feature/sim` 브랜치로 옮기고 MjSpec 방식으로 바꿉니다.
2. **공통 규격을 먼저 확정합니다.** 나중에 바꾸면 다섯 명이 모두 고쳐야 하기 때문입니다.
   - `base_env` API: `reset`/`step` 인자와 반환값, 관측 키 이름, 시간 간격 (예: 물리 250 Hz, 제어 50 Hz)
   - TelemetryLogger CSV 열: 기존 `TetherData`(장력, 가속도 x/y/z, 추진 명령, 조향 명령)를 기준으로
   - `.msg` 초안
3. **개발 환경을 통일합니다.** 팀 문서는 Python 3.10.12, 연습 프로젝트는 Python 3.12 + MuJoCo 3.17입니다.
4. **저장소를 정리합니다.** Unity 구조를 걷어 내고 새 기획안의 폴더와 브랜치를 만듭니다.

### 2단계 — 병렬로 시작

- **①** 연습 프로젝트의 바다·로봇 코드를 `sim/`으로 옮기고 MjSpec으로 바꿉니다. 물리 테스트도 함께 옮깁니다.
- **②** `base_env.py`와 `telemetry.py`를 만들고, 연습 프로젝트의 `bridge/`를 ROS2 토픽 브리지로 키웁니다.
- **③④⑤**는 ①이 끝나기를 기다리지 않고 시작합니다.
  - ③ 사람 상태(FSM)는 물리와 따로 만들 수 있습니다. `boarding.py`를 참고합니다.
  - ④ 줄과 PID가 이미 있으므로 Composite Cost부터 시작합니다.
  - ⑤ 연습 시뮬레이터로 장력 CSV를 미리 뽑아 데이터셋 파이프라인을 먼저 만듭니다.

### 3단계 — 처음부터 끝까지 한 번 이어 보기 (PoC)

- 접근 → 접촉 → 견인 → 텔레메트리 CSV → 규칙 기반 놓침 감지 → (가능하면) 되먹임까지, 엉성하게라도 한 번 연결합니다.
- 연습 프로젝트는 이미 좌표 이동부터 견인 완료까지 한 번에 돌아가므로 이 단계에 빨리 도달할 수 있습니다.
- 이전 README의 M0 PoC 마감은 11/22였습니다. 새 기획에서 일정이 바뀌었는지 확인이 필요합니다.

---

## ① 물리·통합 작업 분담안 (5개 작업 단위)

① 물리·통합의 일을 서로 겹치지 않는 5개 작업 단위로 나누고, 파트별 관련성에 맞춰 팀원에게 하나씩 맡기는 안입니다.
물리 코어는 모든 파트의 기반이므로, 각자 자기 파트에 필요한 물리를 직접 챙길 수 있습니다.

| # | 작업 단위 | 하는 일 | 파일 | 연습 프로젝트의 출발점 | 맡기 좋은 사람 |
|---|---|---|---|---|---|
| 1 | **물** | 파도, 복셀 부력, 물 저항, 부가질량 | `sim/water.py` | [`waves.py`](../tether_rescue/ocean/waves.py), [`hydro.py`](../tether_rescue/ocean/hydro.py), [`added_mass.py`](../tether_rescue/ocean/added_mass.py) | ④ AirwayTow — 견인 결과(얼굴 잠김, 줄 힘)가 물 모델에 가장 크게 좌우됨 |
| 2 | **로봇** | 선체, 추진기(추력·효율), 조향, 전복 판정 | `sim/robot.py` | [`scene.py`](../tether_rescue/scene.py)의 로봇 부분, [`tow_env.py`](../tether_rescue/tow_env.py)의 추진 계산 | ③ PanicSim — 접근 단계 핵심 지표가 전복률 |
| 3 | **세계 조립** | MjSpec으로 바다·로봇·사람·줄을 하나의 모델로 조립 (사람 몸 모델 포함) | `sim/world.py` | [`scene.py`](../tether_rescue/scene.py) 전체 (XML → MjSpec 전환) | ① 물리·통합 담당 — 모든 조각이 모이는 곳이라 책임자가 맡음 |
| 4 | **검증** | 물리 테스트, 사실성 점검, 파라미터 근거 정리, 속도 측정(초당 스텝 수) | `tests/`, `docs/` | [`test_physics.py`](../tests/test_physics.py), [`REALISM_GAPS.md`](REALISM_GAPS.md) | ⑤ TetherSense — 문서 검토·무작위화(DR) 담당이라 "어떤 값이 얼마나 흔들려도 되는지"와 바로 이어짐 |
| 5 | **단계 연결** | PhaseManager(접근 → 접촉 → 견인), 되먹임 연결, InferenceWatchdog, 통합 테스트 | `sim/phase_manager.py` 등 | [`mission_sim.py`](../tether_rescue/mission/mission_sim.py)의 단계 전환, [`guidance.py`](../tether_rescue/mission/guidance.py) | ② 인프라·ROS2 — 공통 환경 API와 통합 테스트·CI를 이미 맡음 |

### 순서와 의존 관계

```
1 물 ─┐
      ├─▶ 3 세계 조립 ─▶ 5 단계 연결
2 로봇 ┘        │
                ▼
             4 검증 (1·2·3이 바뀔 때마다 계속)
```

- 1·2는 바로 동시에 시작할 수 있습니다.
- 3은 1·2가 무엇을 주고받는지 정해져야 조립할 수 있습니다.
- 5는 ②의 `base_env` API가 정해지면 가짜 데이터(mock)로 먼저 만들 수 있습니다.

### 주의할 점

- **최종 책임자는 한 명으로 둡니다.** 작업은 나눠도 3(세계 조립)과 5(단계 연결)의 결정권은 ① 담당자에게 있어야 합니다. 그렇지 않으면 각자 다르게 이어 붙여 통합 단계에서 충돌이 납니다.
- **작업 단위 사이의 약속을 먼저 정합니다.** 예: 물이 로봇에 주는 힘을 어떤 함수로, 어떤 좌표계로 넘기는지. 연습 프로젝트에서 부가질량 버그가 생긴 곳도 이런 경계였습니다.
- **각자의 원래 일이 늘어납니다.** ③④⑤는 자기 강화학습·모델 작업이 따로 있으므로, 물리 작업은 초기 이식만 나눠 맡고 이후 유지보수는 ① 담당자에게 모으는 방식이 현실적입니다.
- 기획안 표에서 ①에 들어 있는 "RescuerAgent 환경 설계·PPO 튜닝"은 물리가 아니라 ③ PanicSim의 학습 일이라 위 5개 단위에서 뺐습니다.

---

## 확인이 필요한 것

- 새 기획안에서 역할 ①~⑤의 담당자와 일정
- MjSpec 전환에 드는 시간. 연습 프로젝트의 XML 생성 코드는 분량이 커서 단순히 옮기는 일보다 오래 걸릴 수 있습니다.
