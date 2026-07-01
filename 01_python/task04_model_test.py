# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ───────────────────────────────────────────────
# 1. 데이터 로드
# ───────────────────────────────────────────────
X = np.load('../data/X.npy')
y = np.load('../data/y.npy')

# ───────────────────────────────────────────────
# 2. 레이블 인코딩
# ───────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot  = to_categorical(y_encoded)
class_names = le.classes_

# ───────────────────────────────────────────────
# 3. 전처리 (채널 추가)
# ───────────────────────────────────────────────
X = X.reshape(-1, 48, 48, 1)

# ───────────────────────────────────────────────
# 4. train/test 분할 (학습과 동일한 설정)
# ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.1, random_state=42, stratify=y_encoded
)
y_test_class = np.argmax(y_test, axis=1)

# ───────────────────────────────────────────────
# 5. 모델 로드
# ───────────────────────────────────────────────
model = load_model('/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/models/wafer_cnn_gap_0.891.h5')
model.summary()

# ───────────────────────────────────────────────
# 6. 평가
# ───────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# ───────────────────────────────────────────────
# 7. 예측
# ───────────────────────────────────────────────
y_pred       = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# ───────────────────────────────────────────────
# 8. Classification Report
# ───────────────────────────────────────────────
print("\n=== Classification Report ===")
print(classification_report(y_test_class, y_pred_class, target_names=class_names, digits=4))

# ───────────────────────────────────────────────
# 9. Confusion Matrix
# ───────────────────────────────────────────────
cm = confusion_matrix(y_test_class, y_pred_class)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────
# 10. 샘플 이미지 출력 (정답 / 오답)
# ───────────────────────────────────────────────
wrong_indices = np.where(y_test_class != y_pred_class)[0]
print(f"\n전체 오분류 개수: {len(wrong_indices)}")

# 오분류 샘플 최대 12개 출력
sample_count = min(12, len(wrong_indices))
plt.figure(figsize=(15, 10))
for i, idx in enumerate(wrong_indices[:sample_count]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title(f"T: {class_names[y_test_class[idx]]}\nP: {class_names[y_pred_class[idx]]}", fontsize=8)
    plt.axis('off')

plt.suptitle('Wrong Predictions')
plt.tight_layout()
plt.show()