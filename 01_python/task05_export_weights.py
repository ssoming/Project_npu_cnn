# -*- coding: utf-8 -*-
"""
학습된 GAP 모델(wafer_cnn_gap_*.h5)의 가중치/바이어스를
RTL $readmemh 로 읽을 수 있는 .mem(hex) 파일로 변환.

고정소수점 컨벤션 (v3 - 활성화 range 재조정)
  - 입력/활성화 : DATA_W=8bit 고정, frac_bits=ACT_FRAC_BITS (아래 상수). conv/pool 체인
                  전체에서 활성값은 항상 이 스케일이어야 다음 레이어가 그대로 읽을 수 있음
                  (파이프라인 호환성) - 이 값 하나가 시스템 전체에서 공유된다.
                  ** v2까지는 ACT_FRAC_BITS=7 (Q1.7, 표현범위 0~0.992)을 썼는데, 이는
                  입력 이미지(전처리 후 [0,1])만 보고 정한 값이었다. 실제로 conv+ReLU를 거친
                  중간 활성값(conv1/2/3 출력, GAP 출력)은 최대 16 근처까지 올라가는 것으로
                  확인됐다(model.predict로 실측, N=2000 샘플: conv1 max=2.91, conv2
                  max=9.54, conv3 max=16.01, GAP max=12.19). Q1.7로는 1.0 이상을 아예
                  표현할 수 없어서 전체 활성값의 12~18%가 모든 레이어에서 saturate 되고
                  있었고, 이게 클래스 예측이 2~3개 클래스로 쏠리는 버그의 실제 원인이었다
                  (fc weight clipping이 원인일 거라 처음 의심했지만, 그건 고쳐도 쏠림이
                  해소되지 않아 활성화 쪽을 ablation으로 다시 진단해서 찾음).
                  DATA_W=8bit는 그대로 두고 ACT_FRAC_BITS만 7→2로 낮춰서(표현범위
                  0~31.75, 실측 최대 16.01 대비 약 2배 여유) 문제를 해결했다 - bit폭 확장
                  없이 정수 비트를 늘리는 쪽으로 스케일만 바꾼 것이므로 RTL 포트 폭은
                  전혀 바뀌지 않는다(아래 가중치/바이어스 재계산만 반영하면 됨).
  - 가중치      : 레이어마다 실제 float 가중치의 |max| 값을 보고 clipping이 없는 선에서
                  가장 세밀한 frac_bits를 자동으로 고른다 (아래 choose_weight_frac_bits).
                  활성화 range 조정과는 독립적 - 가중치 자체의 분포만 보고 정하므로 변경 없음.
                  (참고로 v1에서는 전 레이어에 Q1.7을 강제해서 fc 가중치의 25%가 clipping
                  되는 별개의 문제가 있었는데, 그건 이미 레이어별 자동 스케일로 고쳐져 있다.)
  - 바이어스    : frac = 활성화 frac(ACT_FRAC_BITS) + 그 레이어의 가중치 frac_bits.
                  MAC 누산값의 스케일이 정확히 이 값이 되므로(활성 Q(*.actfrac) × 가중치
                  Q(*.wfrac)), 누산기에 바이어스를 그대로 더하려면 바이어스도 같은 frac이어야
                  한다. RTL에서는: acc += bias 후, 그 레이어의 가중치 frac_bits만큼
                  right-shift 하여 다시 활성화 스케일로 되돌린다(그 다음 ReLU/clip, fc는
                  마지막 레이어라 ReLU 없음). conv_layer.v/fc.v의 FRAC_BITS 파라미터가 바로
                  이 shift량이며, activation의 실제 frac_bits 값과 무관하게 "가중치
                  frac_bits만큼 shift"라는 관계식 자체는 안 바뀌므로 RTL은 수정이 필요 없다
                  (cnn_core_fsm.v에서 conv1/2/3/fc 인스턴스마다 다른 FRAC_BITS를 넘겨주는
                  구조는 이미 되어 있고, 그 값(6/7/7/3)도 그대로 유지된다 - 바뀌는 건 바이어스
                  .mem 파일의 실제 수치뿐).
"""

import os
import glob
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dense

# ───────────────────────────────────────────────
# 0. 설정
# ───────────────────────────────────────────────
MODEL_PATH = None              # None이면 MODEL_DIR 안의 wafer_cnn_gap_*.h5 중 정확도가 가장 높은 파일 자동 선택
MODEL_DIR  = '/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/models'   # task03_train_gap.py 의 저장 경로와 동일하게 맞춤
OUT_DIR    = '/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem'

ACT_BITS, ACT_FRAC_BITS = 8, 2   # 활성화 포맷: DATA_W=8 고정, frac=2 (표현범위 0~31.75)
                                  # - 실측 활성값 최대치(~16.01) 대비 여유 있게 잡음. 7이었다가
                                  #   쏠림 버그 진단 후 2로 조정됨 (위 docstring 참고)
W_BITS = 8                       # 가중치 폭은 8bit 고정, frac_bits는 레이어별로 자동 결정
B_BITS = 16                      # 바이어스 폭 고정, frac_bits = ACT_FRAC_BITS + 그 레이어의 weight_frac_bits

MIN_W_FRAC_BITS = 0
MAX_W_FRAC_BITS = 7              # W_BITS=8 (1 sign bit) 이므로 frac_bits는 최대 7


# ───────────────────────────────────────────────
# 1. 양자화 / hex 변환 유틸
# ───────────────────────────────────────────────
def choose_weight_frac_bits(max_abs):
    """clipping 없이 표현 가능한 가장 큰 frac_bits를 고른다.
    (qmax=127 기준: round(max_abs * 2**frac_bits) <= 127 을 만족하는 최대 frac_bits)
    """
    qmax = 2 ** (W_BITS - 1) - 1  # 127
    if max_abs <= 0:
        return MAX_W_FRAC_BITS
    frac_bits = int(np.floor(np.log2(qmax / max_abs)))
    return int(np.clip(frac_bits, MIN_W_FRAC_BITS, MAX_W_FRAC_BITS))


def quantize(arr, frac_bits, bits, label):
    scale = 2 ** frac_bits
    q = np.round(np.asarray(arr, dtype=np.float64) * scale)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    n_clip = int(np.sum((q < qmin) | (q > qmax)))
    if n_clip > 0:
        print(f"  ! 경고 [{label}]: {n_clip}개 값이 [{qmin}, {qmax}] 범위를 벗어나 clipping 됨 "
              f"→ frac_bits/bit폭 재검토 필요")
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int64)


def to_hex_lines(values, bits):
    mask = (1 << bits) - 1
    hex_digits = bits // 4
    return [format(int(v) & mask, f'0{hex_digits}x') for v in values]


def flatten_conv_weight(kernel):
    # Keras kernel shape: (kh, kw, in_ch, out_ch)
    # → RTL 순서: out_ch 우선, 그 안에서 in_ch, 그 안에서 kh, kw (행 우선)
    return np.transpose(kernel, (3, 2, 0, 1)).flatten()


def flatten_dense_weight(kernel):
    # Keras kernel shape: (in_features, out_features)
    # → RTL 순서: out_features 우선, 그 안에서 in_features
    return np.transpose(kernel, (1, 0)).flatten()


# ───────────────────────────────────────────────
# 2. 모델 로드
# ───────────────────────────────────────────────
if MODEL_PATH is None:
    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, 'wafer_cnn_gap_*.h5')))
    if not candidates:
        raise FileNotFoundError(
            f"{MODEL_DIR} 안에 wafer_cnn_gap_*.h5 파일이 없습니다. "
            f"먼저 task03_train_gap.py를 실행하거나 MODEL_PATH를 직접 지정하세요."
        )
    MODEL_PATH = candidates[-1]   # 파일명에 정확도가 포함되어 있어 정렬 시 마지막 = 가장 높은 정확도

print(f"[1] 모델 로드: {MODEL_PATH}")
model = load_model(MODEL_PATH)
model.summary()

# ───────────────────────────────────────────────
# 3. 레이어별 양자화 및 .mem 저장
# ───────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
weight_layers = [l for l in model.layers if isinstance(l, (Conv2D, Dense))]

print("\n[2] 레이어별 양자화 및 .mem 파일 저장")
meta_layers = []
conv_idx = 0

for layer in weight_layers:
    kernel, bias = layer.get_weights()

    if isinstance(layer, Conv2D):
        conv_idx += 1
        name = f'conv{conv_idx}'
        w_flat = flatten_conv_weight(kernel)
        kh, kw, in_ch, out_ch = kernel.shape
        shape_info = {'layer_type': 'conv2d', 'kh': int(kh), 'kw': int(kw),
                      'in_ch': int(in_ch), 'out_ch': int(out_ch)}
        weight_order = 'out_ch, in_ch, kh, kw (row-major)'
    else:
        name = 'fc'
        w_flat = flatten_dense_weight(kernel)
        in_f, out_f = kernel.shape
        shape_info = {'layer_type': 'dense', 'in_features': int(in_f), 'out_features': int(out_f)}
        weight_order = 'out_features, in_features'

    w_max_abs = float(np.max(np.abs(w_flat))) if w_flat.size else 0.0
    w_frac_bits = choose_weight_frac_bits(w_max_abs)
    b_frac_bits = ACT_FRAC_BITS + w_frac_bits

    w_q = quantize(w_flat, w_frac_bits, W_BITS, f'{name}.weight')
    b_q = quantize(bias,   b_frac_bits, B_BITS, f'{name}.bias')

    w_hex = to_hex_lines(w_q, W_BITS)
    b_hex = to_hex_lines(b_q, B_BITS)

    w_path = os.path.join(OUT_DIR, f'{name}_weight.mem')
    b_path = os.path.join(OUT_DIR, f'{name}_bias.mem')
    with open(w_path, 'w') as f:
        f.write('\n'.join(w_hex) + '\n')
    with open(b_path, 'w') as f:
        f.write('\n'.join(b_hex) + '\n')

    print(f"  {name:6s}: |w|max={w_max_abs:.4f} → weight_frac_bits={w_frac_bits} (bias_frac_bits={b_frac_bits})  "
          f"weight {len(w_hex):5d}개 → {w_path}  |  bias {len(b_hex):3d}개 → {b_path}")

    meta_layers.append({
        'name': name,
        **shape_info,
        'weight_count': len(w_hex),
        'bias_count': len(b_hex),
        'weight_max_abs_float': w_max_abs,
        'weight_bits': W_BITS, 'weight_frac_bits': w_frac_bits,
        'bias_bits': B_BITS, 'bias_frac_bits': b_frac_bits,
        'weight_order': weight_order,
        'weight_file': os.path.basename(w_path),
        'bias_file': os.path.basename(b_path),
    })

# ───────────────────────────────────────────────
# 4. 메타데이터 저장 (다음 단계 RTL 설계 시 메모리 깊이/주소/포맷 참고용)
# ───────────────────────────────────────────────
meta_path = os.path.join(OUT_DIR, 'quant_meta.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump({
        'source_model': MODEL_PATH,
        'activation_format': {
            'bits': ACT_BITS, 'frac_bits': ACT_FRAC_BITS,
            'note': ('활성화는 전 레이어 공통 Q1.7 고정. 가중치는 레이어별로 다른 frac_bits를 쓸 수 있음 '
                     '(아래 layers[].weight_frac_bits 참고) - conv/fc 출력은 acc(bias 포함) 후 '
                     '그 레이어의 weight_frac_bits만큼 right-shift 하면 다시 Q1.7로 돌아온다.'),
        },
        'layers': meta_layers,
    }, f, indent=2, ensure_ascii=False)

print(f"\n[3] 메타데이터 저장: {meta_path}")
print("\n=== 완료 ===")
