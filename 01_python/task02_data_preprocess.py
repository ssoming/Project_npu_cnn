"""
WM-811K 전처리 및 오버샘플링 스크립트
저장 결과는 CSV 대신 numpy .npy 형식으로 저장 - 이미지 배열을 CSV에 넣으면 너무 커지고 느림
학습 코드에서 np.load('./data/X.npy')로 바로 불러올 수 있음.

x.npy: 웨이퍼 맵 이미지 배열; 모델 입력
y.npy: 클래스 레이블; 정답
"""

import pickle
import numpy as np
import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
from skimage.transform import resize
from sklearn.model_selection import train_test_split


PKL_PATH     = '../data/LSWMD.pkl'
IMG_SIZE     = (48, 48)
SAMPLE_RATIO = 0.5      # 0.0~1.0, 1.0이면 전체 사용

# ───────────────────────────────────────────────
# 1. pkl 로드
# ───────────────────────────────────────────────
print("[1] pkl 로드 중...")
with open(PKL_PATH, 'rb') as f:
    df = pickle.load(f, encoding='latin1')

def unpack(val):
    try:
        return val[0][0]
    except:
        return ''

df['failureType'] = df['failureType'].apply(unpack)
df = df[df['failureType'] != ''].reset_index(drop=True)
print(f"  레이블 있는 샘플: {len(df):,}")

# ───────────────────────────────────────────────
# 2. 전처리 함수
# ───────────────────────────────────────────────
def preprocess(wmap):
    wmap = np.array(wmap, dtype=np.float32)
    wmap = resize(wmap, IMG_SIZE, anti_aliasing=False, preserve_range=True)
    wmap = wmap / 2.0
    return wmap

# ───────────────────────────────────────────────
# 3. 오버샘플링 함수
# ───────────────────────────────────────────────
def augment(wmap):
    return [
        np.fliplr(wmap),
        np.flipud(wmap),
        np.rot90(wmap, k=1),
        np.rot90(wmap, k=2),
        np.rot90(wmap, k=3),
    ]

# ───────────────────────────────────────────────
# 4. 전처리 + 오버샘플링
# ───────────────────────────────────────────────
print("[2] 전처리 및 오버샘플링 중...")
X, y = [], []

for _, row in df.iterrows():
    wmap  = preprocess(row['waferMap'])
    label = row['failureType']

    X.append(wmap)
    y.append(label)

    if label != 'none':
        for aug_wmap in augment(wmap):
            X.append(aug_wmap)
            y.append(label)

X = np.array(X)
y = np.array(y)
print(f"  오버샘플링 후: {len(X):,} 샘플")

# ───────────────────────────────────────────────
# 5. 비율 유지하며 샘플링
# ───────────────────────────────────────────────
if SAMPLE_RATIO < 1.0:
    print(f"[3] {SAMPLE_RATIO*100:.0f}% 샘플링 중 (클래스 비율 유지)...")
    _, X, _, y = train_test_split(
        X, y,
        test_size=SAMPLE_RATIO,
        random_state=42,
        stratify=y
    )
    print(f"  샘플링 후: {len(X):,} 샘플")

# ───────────────────────────────────────────────
# 6. 클래스별 분포 출력
# ───────────────────────────────────────────────
print("\n클래스별 샘플 수:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls:12s}: {cnt:,}")

# ───────────────────────────────────────────────
# 7. 저장
# ───────────────────────────────────────────────
print("\n[4] 저장 중...")
np.save('../data/X.npy', X)
np.save('../data/y.npy', y)
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print("\n=== 완료 ===")