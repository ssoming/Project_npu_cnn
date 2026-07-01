# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping


# ───────────────────────────────────────────────
# 1. 데이터 로드
# ───────────────────────────────────────────────
X = np.load('../data/X.npy')
y = np.load('../data/y.npy')

print(X.shape)
print(y.shape)
# ───────────────────────────────────────────────
# 2. 레이블 인코딩
# ───────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot  = to_categorical(y_encoded)
num_classes = len(le.classes_)

print(y[0])
print(y_onehot[0])
# ───────────────────────────────────────────────
# 3. 전처리 (채널 추가)
# ───────────────────────────────────────────────
X = X.reshape(-1, 48, 48, 1)

print(X.shape)
print(y_onehot.shape)

# ───────────────────────────────────────────────
# 4. train / test 분할 (90:10)
# ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.1, random_state=42, stratify=y_encoded
)

print(X_train.shape)
print(X_test.shape)

# ───────────────────────────────────────────────
# 5. 모델 정의 (Flatten+Dense64 → GlobalAveragePooling2D)
#    - FC 가중치(기존 73,792개, 전체의 92%)를 제거해 RTL 가중치
#      어드레싱을 단순화. 대신 conv3 채널을 32→64로 늘려 GAP
#      출력 특징 벡터 차원을 확보(정확도 손실 완화 목적).
# ───────────────────────────────────────────────
L2 = regularizers.l2(0.001)   # 가중치 크기 억제 → int8 범위 내 유지

model = Sequential()
model.add(Conv2D(8,  kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1),
                 activation='relu', kernel_regularizer=L2))
model.add(MaxPooling2D(pool_size=(2, 2)))   # → 24×24×8

model.add(Conv2D(16, kernel_size=(3, 3), padding='same',
                 activation='relu', kernel_regularizer=L2))
model.add(MaxPooling2D(pool_size=(2, 2)))   # → 12×12×16

model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                 activation='relu', kernel_regularizer=L2))
model.add(MaxPooling2D(pool_size=(2, 2)))   # → 6×6×64

model.add(GlobalAveragePooling2D())         # → 64
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# ───────────────────────────────────────────────
# 6. 학습

early_stop = EarlyStopping(
    monitor='val_accuracy',   # 검증 정확도 기준
    patience=5,               # 5 epoch 동안 개선 없으면 중단
    restore_best_weights=True # 중단 시 가장 좋았던 가중치로 복원
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.2,
          callbacks=[early_stop],
          verbose=1)

# ───────────────────────────────────────────────
# 7. 평가 및 저장
# ───────────────────────────────────────────────
accuracy_score = model.evaluate(X_test, y_test, verbose=0)[1]
print('Final test set accuracy', accuracy_score)
model.save('python/models/wafer_cnn_gap_{:.3f}.h5'.format(accuracy_score))
