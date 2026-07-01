"""
Wafer NPU Inference GUI - PC-side client for the FPGA wafer-defect-map NPU board.

UART protocol (must match the board firmware, main.c):
    PC -> board : 0xAA (start byte) + 2304 image bytes (48x48 grayscale, row-major)
    board -> PC : 1 result byte (class 0-8) + 0x55 (end byte)

Preprocessing mirrors task02_data_preprocess.py / npu_validate.py:
    raw WM-811K categorical map {0,1,2} --resize(48x48)--> /2.0 --> [0,1] --> *255 --> uint8
This GUI additionally accepts ordinary rendered grayscale images and already-
normalized [0,1] float arrays (e.g. a single X.npy sample), auto-detecting which
domain a loaded array is in (see preprocess_to_bytes).
"""

import os
import sys
import time

import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize

import serial
import serial.tools.list_ports

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QSizePolicy,
)

IMG_W = 48
IMG_H = 48
IMG_PIXELS = IMG_W * IMG_H
BAUD_RATE = 115200
START_BYTE = 0xAA
END_BYTE = 0x55
DEFAULT_TIMEOUT_SEC = 5.0

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none",
]

FILE_FILTER = (
    "지원 파일 (*.png *.jpg *.jpeg *.bmp *.npy);;"
    "이미지 (*.png *.jpg *.jpeg *.bmp);;"
    "NumPy 배열 (*.npy);;"
    "모든 파일 (*)"
)

STYLE_SHEET = """
QWidget {
    background-color: #f5f6f8;
    color: #1f2430;
    font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dcdfe4;
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #3b4252;
}
QPushButton {
    background-color: #2f6fed;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover { background-color: #2a5fd0; }
QPushButton:pressed { background-color: #234ea8; }
QPushButton:disabled { background-color: #c2c8d1; color: #eef0f3; }
QPushButton#secondary {
    background-color: #ffffff;
    color: #2f6fed;
    border: 1px solid #2f6fed;
}
QPushButton#secondary:hover { background-color: #eef3ff; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #dcdfe4;
    border-radius: 4px;
    padding: 4px;
}
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #dcdfe4;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}
QLabel#previewLabel, QLabel#resultCard {
    background-color: #ffffff;
    border: 1px solid #dcdfe4;
    border-radius: 6px;
}
QLabel#sectionHint {
    color: #6b7280;
    font-size: 12px;
}
"""


def load_image_any(path):
    """
    Load a single 2D array from either a raster image or a .npy file.
    Returns (array2d_float32, stack_or_None, note). When the .npy file holds
    a stack (N, H, W), stack is the full array (index 0 is used initially)
    and the caller is expected to offer index selection.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 2:
            return arr.astype(np.float32), None, ""
        elif arr.ndim == 3:
            note = f"스택 {arr.shape[0]}개 샘플 감지 (인덱스로 선택 가능)"
            return arr[0].astype(np.float32), arr.astype(np.float32), note
        else:
            raise ValueError(f".npy 배열 shape이 지원되지 않습니다: {arr.shape} (2D 또는 3D 필요)")
    else:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32), None, ""


def preprocess_to_bytes(arr2d):
    """
    Convert a 2D array (of unknown value domain) into the 48x48 uint8 byte
    sequence sent over UART, auto-detecting the domain by peak value.

    The input image is read directly as conv1's activation with no separate
    quantization step, so it must use the SAME scale as every other
    activation in the pipeline: DATA_W=8 bits, frac_bits=ACT_FRAC_BITS=2
    (see quant_meta.json's activation_format.frac_bits, and the header
    comments in conv_layer.v/fc.v). That means raw byte value = real_value *
    2^ACT_FRAC_BITS = real_value * 4 - NOT *128 and NOT *255.

    (test03_picture.py's INPUT_SCALE=128 comment - and an earlier version of
    this function that copied it - predates the ACT_FRAC_BITS 7->2 fix and
    was never updated; confirmed by re-running the actual deployed
    conv1/conv2/conv3/fc .mem weights layer-by-layer in Python: x128 and x255
    both collapse predictions to one class (11-14% accuracy) while x4 gives
    a healthy, diverse 62.5% accuracy matching the float model's ballpark.)

      > 2.0  : ordinary rendered/raster grayscale image (0-255) - normalized
               to [0,1] first, since it has no direct categorical meaning
      > 1.0  : raw WM-811K categorical map {0,1,2}, same as task02 (/2.0 step)
      <= 1.0 : already normalized to [0,1] (e.g. one X.npy sample, Keras-style)
    Returns (byte_array_2d_uint8, domain_description).
    """
    INPUT_SCALE = 4.0  # = 2^ACT_FRAC_BITS (ACT_FRAC_BITS=2, see quant_meta.json)

    a = np.asarray(arr2d, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"2D 이미지가 필요합니다 (현재 shape={a.shape})")
    if a.size == 0:
        raise ValueError("빈 배열입니다.")

    vmax = float(a.max())

    if vmax > 2.0:
        domain = "grayscale image (0-255), normalized before FPGA quantization"
        resized = sk_resize(a, (IMG_H, IMG_W), anti_aliasing=False, preserve_range=True)
        byte_vals = (resized / 255.0) * INPUT_SCALE
    elif vmax > 1.0:
        domain = "raw categorical map (0/1/2)"
        resized = sk_resize(a, (IMG_H, IMG_W), anti_aliasing=False, preserve_range=True)
        byte_vals = (resized / 2.0) * INPUT_SCALE
    else:
        domain = "normalized [0,1] (Keras-style)"
        if a.shape != (IMG_H, IMG_W):
            a = sk_resize(a, (IMG_H, IMG_W), anti_aliasing=False, preserve_range=True)
        byte_vals = a * INPUT_SCALE

    # clip to [0,127], not 255: the RTL activation is a signed 8-bit value
    # (conv_layer.v's SAT_MAX = 2^(DATA_W-1)-1 = 127) - a byte >127 would be
    # read back as negative fixed-point by the hardware.
    byte_vals = np.clip(np.round(byte_vals), 0, 127).astype(np.uint8)
    return byte_vals, domain


def make_display_image(byte_vals_2d):
    """
    Min-max normalize byte_vals (the 0-4-ish, INPUT_SCALE=4.0 hardware-domain
    bytes) up to a full 0-255 range for on-screen preview ONLY. This is a
    purely cosmetic rescaling - it never touches image_bytes, which is what
    actually gets sent over UART, so it has no effect on inference.
    """
    a = byte_vals_2d.astype(np.float32)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax > vmin:
        norm = (a - vmin) / (vmax - vmin) * 255.0
    else:
        norm = np.zeros_like(a)
    return np.clip(np.round(norm), 0, 255).astype(np.uint8)


class InferenceWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(int, float)
    failed = pyqtSignal(str)

    def __init__(self, ser, image_bytes, timeout_sec):
        super().__init__()
        self.ser = ser
        self.image_bytes = image_bytes
        self.timeout_sec = timeout_sec

    def run(self):
        try:
            self.progress.emit("이미지 전송 중...")
            t0 = time.time()

            self.ser.timeout = self.timeout_sec
            self.ser.reset_input_buffer()
            payload = bytes([START_BYTE]) + self.image_bytes
            self.ser.write(payload)
            self.ser.flush()

            self.progress.emit("보드 응답 대기 중...")
            resp = self.ser.read(2)
            elapsed = time.time() - t0

            if len(resp) < 2:
                self.failed.emit(
                    f"응답 타임아웃: {self.timeout_sec:.1f}초 내에 {len(resp)}/2 바이트만 수신했습니다."
                )
                return

            class_byte, end_byte = resp[0], resp[1]
            if end_byte != END_BYTE:
                self.failed.emit(
                    f"프로토콜 오류: 종료 바이트가 0x{END_BYTE:02X}가 아니라 0x{end_byte:02X}입니다."
                )
                return
            if not (0 <= class_byte <= 8):
                self.failed.emit(f"프로토콜 오류: 잘못된 클래스 값 {class_byte} (0~8 범위를 벗어남)")
                return

            self.finished_ok.emit(class_byte, elapsed)
        except serial.SerialException as e:
            self.failed.emit(f"시리얼 통신 오류: {e}")
        except Exception as e:
            self.failed.emit(f"알 수 없는 오류: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer NPU Inference - 반도체 웨이퍼 결함 분류")
        self.resize(860, 640)

        self.ser = None
        self.worker = None
        self.npy_stack = None
        self.image_bytes = None

        self._build_ui()
        self.refresh_ports()
        self.update_infer_button_state()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # ---- connection group ----
        conn_group = QGroupBox("시리얼 연결")
        conn_layout = QHBoxLayout(conn_group)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(260)

        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.setObjectName("secondary")
        self.refresh_btn.clicked.connect(self.refresh_ports)

        self.connect_btn = QPushButton("연결")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.status_label = QLabel("● 연결 안 됨")
        self.status_label.setStyleSheet("color: #888; font-weight: 600;")

        timeout_label = QLabel("응답 타임아웃(초):")
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.5, 30.0)
        self.timeout_spin.setSingleStep(0.5)
        self.timeout_spin.setValue(DEFAULT_TIMEOUT_SEC)

        conn_layout.addWidget(self.port_combo)
        conn_layout.addWidget(self.refresh_btn)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.status_label)
        conn_layout.addStretch(1)
        conn_layout.addWidget(timeout_label)
        conn_layout.addWidget(self.timeout_spin)

        root.addWidget(conn_group)

        # ---- image load group ----
        load_group = QGroupBox("이미지 불러오기")
        load_layout = QHBoxLayout(load_group)

        self.load_btn = QPushButton("이미지 불러오기...")
        self.load_btn.clicked.connect(self.load_image)

        self.path_label = QLabel("불러온 파일 없음")
        self.path_label.setObjectName("sectionHint")

        self.index_label = QLabel("샘플 인덱스:")
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 0)
        self.index_spin.valueChanged.connect(self.on_index_changed)
        self.index_label.setVisible(False)
        self.index_spin.setVisible(False)

        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.path_label, 1)
        load_layout.addWidget(self.index_label)
        load_layout.addWidget(self.index_spin)

        root.addWidget(load_group)

        # ---- preview / result side by side ----
        mid_layout = QHBoxLayout()

        preview_group = QGroupBox("원본 이미지 미리보기")
        preview_v = QVBoxLayout(preview_group)
        self.preview_label = QLabel("이미지를 불러오세요")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(260, 260)
        self.domain_label = QLabel("")
        self.domain_label.setObjectName("sectionHint")
        preview_v.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        preview_v.addWidget(self.domain_label, alignment=Qt.AlignCenter)

        result_group = QGroupBox("예측 결과")
        result_v = QVBoxLayout(result_group)
        self.result_label = QLabel("-")
        self.result_label.setObjectName("resultCard")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(260, 260)
        self.result_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.result_sub_label = QLabel("")
        self.result_sub_label.setObjectName("sectionHint")
        self.result_sub_label.setAlignment(Qt.AlignCenter)
        result_v.addWidget(self.result_label, alignment=Qt.AlignCenter)
        result_v.addWidget(self.result_sub_label, alignment=Qt.AlignCenter)

        mid_layout.addWidget(preview_group)
        mid_layout.addWidget(result_group)
        root.addLayout(mid_layout)

        # ---- infer button + progress ----
        infer_layout = QVBoxLayout()
        self.infer_btn = QPushButton("추론 시작")
        self.infer_btn.setMinimumHeight(40)
        self.infer_btn.clicked.connect(self.start_inference)
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setObjectName("sectionHint")
        infer_layout.addWidget(self.infer_btn)
        infer_layout.addWidget(self.progress_label)
        root.addLayout(infer_layout)

        # ---- log ----
        log_group = QGroupBox("로그")
        log_v = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        log_v.addWidget(self.log_box)
        root.addWidget(log_group)

        self.setStyleSheet(STYLE_SHEET)

    # ------------------------------------------------------------ helpers --
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    def update_infer_button_state(self):
        busy = self.worker is not None and self.worker.isRunning()
        ready = (
            self.ser is not None and self.ser.is_open
            and self.image_bytes is not None
            and not busy
        )
        self.infer_btn.setEnabled(ready)

    # --------------------------------------------------------- connection --
    def refresh_ports(self):
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self.port_combo.addItem("(사용 가능한 포트 없음)", userData=None)
            return
        for p in ports:
            self.port_combo.addItem(f"{p.device}  -  {p.description}", userData=p.device)
        for i in range(self.port_combo.count()):
            text = self.port_combo.itemText(i).lower()
            if "usb" in text or "uart" in text:
                self.port_combo.setCurrentIndex(i)
                break
        self.log("시리얼 포트 목록을 새로고침했습니다.")

    def toggle_connection(self):
        if self.ser is not None and self.ser.is_open:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        port_name = self.port_combo.currentData()
        if not port_name:
            QMessageBox.warning(self, "포트 선택 오류", "사용 가능한 시리얼 포트를 선택하세요.")
            return
        try:
            self.ser = serial.Serial(port_name, BAUD_RATE, timeout=self.timeout_spin.value())
            self.connect_btn.setText("연결 해제")
            self.status_label.setText("● 연결됨")
            self.status_label.setStyleSheet("color: #1a7f37; font-weight: 600;")
            self.log(f"{port_name} @ {BAUD_RATE}bps 연결 성공")
        except serial.SerialException as e:
            self.ser = None
            QMessageBox.critical(self, "연결 실패", f"시리얼 포트 연결에 실패했습니다:\n{e}")
            self.log(f"[오류] 연결 실패: {e}")
        self.update_infer_button_state()

    def disconnect_serial(self):
        try:
            if self.ser:
                self.ser.close()
        finally:
            self.ser = None
            self.connect_btn.setText("연결")
            self.status_label.setText("● 연결 안 됨")
            self.status_label.setStyleSheet("color: #888; font-weight: 600;")
            self.log("포트 연결을 해제했습니다.")
        self.update_infer_button_state()

    # --------------------------------------------------------- image load --
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "이미지 불러오기", "", FILE_FILTER)
        if not path:
            return
        try:
            arr2d, stack, note = load_image_any(path)
        except Exception as e:
            QMessageBox.critical(self, "이미지 로드 실패", f"이미지를 불러오지 못했습니다:\n{e}")
            self.log(f"[오류] 이미지 로드 실패: {e}")
            return

        self.npy_stack = stack
        self.path_label.setText(os.path.basename(path))

        if stack is not None:
            self.index_spin.blockSignals(True)
            self.index_spin.setRange(0, stack.shape[0] - 1)
            self.index_spin.setValue(0)
            self.index_spin.blockSignals(False)
            self.index_label.setVisible(True)
            self.index_spin.setVisible(True)
        else:
            self.index_label.setVisible(False)
            self.index_spin.setVisible(False)

        self.log(f"'{os.path.basename(path)}' 로드 완료" + (f" - {note}" if note else ""))
        self.apply_image(arr2d)

    def on_index_changed(self, idx):
        if self.npy_stack is not None:
            self.apply_image(self.npy_stack[idx])

    def apply_image(self, arr2d):
        try:
            byte_vals, domain = preprocess_to_bytes(arr2d)
        except Exception as e:
            QMessageBox.critical(self, "전처리 실패", f"이미지 전처리에 실패했습니다:\n{e}")
            self.log(f"[오류] 전처리 실패: {e}")
            return

        self.image_bytes = byte_vals.tobytes()
        self.domain_label.setText(f"감지된 입력 형식: {domain}")
        self.show_preview(make_display_image(byte_vals))
        self.result_label.setText("-")
        self.result_sub_label.setText("")
        self.update_infer_button_state()

    def show_preview(self, byte_vals_2d):
        h, w = byte_vals_2d.shape
        qimg = QImage(bytes(byte_vals_2d.tobytes()), w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation,
        )
        self.preview_label.setPixmap(pix)

    # --------------------------------------------------------- inference --
    def start_inference(self):
        if self.ser is None or not self.ser.is_open:
            QMessageBox.warning(self, "연결 필요", "먼저 시리얼 포트에 연결하세요.")
            return
        if self.image_bytes is None or len(self.image_bytes) != IMG_PIXELS:
            QMessageBox.warning(self, "이미지 필요", "먼저 이미지를 불러오세요.")
            return

        self.infer_btn.setEnabled(False)
        self.result_label.setText("추론 중")
        self.result_sub_label.setText("")
        self.progress_label.setText("전송 준비 중...")
        self.log("추론 시작 - 0xAA + 2304 바이트 전송")

        self.worker = InferenceWorker(self.ser, self.image_bytes, self.timeout_spin.value())
        self.worker.progress.connect(self.on_infer_progress)
        self.worker.finished_ok.connect(self.on_infer_success)
        self.worker.failed.connect(self.on_infer_failed)
        self.worker.start()

    def on_infer_progress(self, msg):
        self.progress_label.setText(msg)
        self.log(msg)

    def on_infer_success(self, class_idx, elapsed):
        name = CLASS_NAMES[class_idx]
        self.result_label.setText(name)
        self.result_sub_label.setText(f"class {class_idx}  ·  {elapsed * 1000:.0f} ms")
        self.progress_label.setText("완료")
        self.log(f"추론 완료 - 결과={class_idx} ({name}), 소요시간={elapsed * 1000:.0f}ms")
        self.update_infer_button_state()

    def on_infer_failed(self, msg):
        self.result_label.setText("오류")
        self.result_sub_label.setText("")
        self.progress_label.setText("실패")
        QMessageBox.critical(self, "추론 실패", msg)
        self.log(f"[오류] {msg}")
        self.update_infer_button_state()

    # ------------------------------------------------------------- close --
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
