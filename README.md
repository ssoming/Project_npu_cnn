# NPU for Wafer Varification

> CNN 추론을 FPGA로 직접 구현해, 서버·GPU 없이 저전력 보드 한 장으로 웨이퍼 결함을 자동 분류하는 시스템

[![YOUTUBE](https://img.shields.io/badge/YOUTUBE-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%EC%98%81%EC%83%81%20%EB%B3%B4%EA%B8%B0-555555?style=for-the-badge&logo=youtube&logoColor=white&labelColor=FF0000)](https://www.youtube.com/watch?v=iz0Gmx1RPGs)

---

## 1. Overview

| 항목 | 내용 |
|------|------|
| 플랫폼 | Zybo Z7-20 (Zynq XC7Z020) |
| 언어 | Python, Verilog, C |
| 도구 | Vivado 2024.2, Vitis 2024.2 |
| 통신 | UART (115200 bps), AXI4-Lite |
| 개발 기간 | 2026.06.23 - 07.03 |
| 팀 구성 | 3인 팀 프로젝트 |

---

## 2. 주요 기능

- **웨이퍼 결함 분류**: WM-811K로 학습한 CNN을 양자화해 FPGA(PL)에서 직접 추론, 9종 결함 분류
- **SoC 통합**: CNN 가속기 IP를 AXI4-Lite로 패키징해 Zynq PS에 통합
- **PC 연동 GUI**: PyQt5로 단일/배치 추론 및 결과 CSV·PDF 저장·비교

---

## 3. 담당 역할

**CNN 모델 생성 및 CNN IP 생성**

- CNN 모델 학습 및 Dense → GAP 구조 재설계 (가중치 수를 크게 축소)
- 가중치 양자화 및 export (레이어별 소수점 위치 최적화)
- CNN 가속기 RTL 설계 (conv/pool/GAP/fc 레이어, FSM, AXI-Lite 인터페이스)

---

## 4. 시스템 아키텍처 및 핵심 구현

![block_diagram](docs/npu_wafer_block_diagram.png)

PC가 UART로 이미지를 보내면 보드의 PS(ARM)가 이를 받아 AXI-Lite로 PL(CNN 가속기)에 전달하고, 추론 결과를 다시 PC로 돌려준다.

**핵심 구현**

- **GAP 기반 경량화**: 마지막 레이어를 Dense 대신 GAP(전역 평균 풀링)으로 바꿔 가중치를 576개 수준으로 축소
- **cnn_core_fsm**: conv-pool 3단 → GAP → fc 순서로 각 레이어를 제어하는 상태 기계, 레이어별로 다른 소수점 위치(FRAC_BITS) 적용
- **레지스터 인터페이스** (기준주소 `0x43C00000`): `CTRL`(추론 시작) · `STATUS`(busy/done) · `CLASS_RESULT`(결과) · `IMG_DATA`/`IMG_ADDR`(이미지 입력)

---

## 5. 트러블슈팅

| 발생 문제 | 발생 원인 | 해결 방안 | 결과 |
|-----------|-----------|-----------|------|
| 레이어별 TB는 모두 통과했는데, 실제 Vitis+GUI로 추론하면 예측이 특정 클래스로만 쏠림 | 활성화 값 표현 범위(Q1.7, 0~0.992)가 실측 활성값 최대치(약 16)를 감당 못해 다수 값이 saturate | 활성화 프랙션 비트를 7→2로 낮춰 표현 범위 확장(0~31.75). 비트폭은 그대로 두고 스케일만 조정 | 정확도 89.1%로 회복, 쏠림 현상 해소 |

---

## 6. 디렉토리 구조

| 경로 | 역할 |
|------|------|
| `01_python/` | CNN 모델 학습·평가·가중치 양자화 (Python) |
| `02_vivado/ip_repo_cnn_npu/src/` | CNN 가속기 RTL 및 가중치 .mem (Verilog) |
| `03_vitis/wafer_npu_app/src/` | UART 통신 및 추론 실행 펌웨어 (C) |
| `04_python_ui/` | PC용 PyQt5 GUI |
