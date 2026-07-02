"""
wafer_cnn_gui_compare.py — Wafer NPU 배치 이력 비교/트렌드 화면 (독립 실행)

wafer_cnn_gui_csv.py에서 생성한 배치 결과 CSV를 대상으로 한다. 배치를 돌릴 때마다
저장되는 result_*.csv (추론 결과 테이블 — 파일명/예측 클래스/지연시간 등) 파일들을
직접 읽어서 그 자리에서 수율/불량률/평균 지연시간 통계를 계산한다.
(이전 버전은 별도의 batch_summary_*.csv 요약 파일을 읽었지만, 이제 요약 CSV를
따로 저장하지 않으므로 result_*.csv에서 직접 집계한다.)

화면 색 구성(#252525/#1a1a1a/#1abc9c 등)과 클래스 정의(CLASS_NAMES 등)는
wafer_cnn_gui_csv.py와 통일해서 같은 느낌을 유지한다.

result_*.csv 컬럼(wafer_cnn_gui_csv.py의 _write_result_csv 기준):
  파일명, 예측 클래스, 예측 클래스 인덱스, 관련 공정 단계, 권장 조치 방안,
  실행 날짜, 실행 시각, 추론 지연시간(ms)
  (한 행 = 웨이퍼 이미지 한 장. 예측 클래스 인덱스가 -1이면 오류로 처리된 항목.)

실행: python wafer_cnn_gui_compare.py
"""

import os
import sys
import csv
import glob

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox,
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor

import matplotlib
from matplotlib import font_manager as fm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from wafer_cnn_gui_csv import CLASS_NAMES, NORMAL_IDX  # 클래스명/정상클래스 정의 재사용

NORMAL_NAME = CLASS_NAMES[NORMAL_IDX]
RESULT_CSV_GLOB = "result_*.csv"

# ── 한글 폰트 설정 (그래프 축/범례에 "수율(%)" 등 한글 표시용) ────────────
_KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _font_path in _KOREAN_FONT_CANDIDATES:
    if os.path.isfile(_font_path):
        fm.fontManager.addfont(_font_path)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False
        break


# ── 배치 결과 CSV 로드 + 집계 ─────────────────────────────────────────
def load_batch_result(csv_path):
    """
    result_*.csv(웨이퍼 1장당 1행)를 읽어서, 그 배치 전체의 수율/불량률/
    평균 지연시간 등을 그 자리에서 계산해 요약 dict로 돌려준다.

    형식이 맞지 않는 파일은 조용히 None을 반환한다(비교 목록에서 제외 -
    이 폴더에 다른 CSV가 섞여 있어도 안전하게 걸러내기 위함).
    """
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None

        total = len(rows)
        error_rows = [r for r in rows if r.get("예측 클래스 인덱스", "") == "-1"]
        valid_rows = [r for r in rows if r.get("예측 클래스 인덱스", "") != "-1"]
        classified = len(valid_rows)

        normal_count = sum(1 for r in valid_rows if r.get("예측 클래스") == NORMAL_NAME)
        yield_pct = (normal_count / classified * 100.0) if classified else 0.0
        defect_pct = 100.0 - yield_pct if classified else 0.0

        if valid_rows:
            avg_ms = sum(float(r["추론 지연시간(ms)"]) for r in valid_rows) / len(valid_rows)
        else:
            avg_ms = 0.0

        first = rows[0]
        return {
            "path": csv_path,
            "file": os.path.basename(csv_path),
            "date": first.get("실행 날짜", ""),
            "time": first.get("실행 시각", ""),
            "total": total,
            "error_count": len(error_rows),
            "yield_pct": yield_pct,
            "defect_pct": defect_pct,
            "avg_ms": avg_ms,
        }
    except Exception:
        return None


def find_batch_results(folder):
    """폴더 내 result_*.csv를 모두 찾아 배치 날짜/시각순으로 정렬해서 돌려준다."""
    entries = []
    for path in sorted(glob.glob(os.path.join(folder, RESULT_CSV_GLOB))):
        entry = load_batch_result(path)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: (e["date"], e["time"]))
    return entries


# ── 비교 윈도우 ────────────────────────────────────────────────────────
class CompareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer NPU 배치 이력 비교 (독립 실행)")
        self.setMinimumSize(1100, 800)
        self.entries = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 상단: 폴더 선택 ────────────────────────────────────────
        folder_grp = QGroupBox("폴더 선택")
        fl = QHBoxLayout(folder_grp)
        self.folder_label = QLabel("폴더를 선택하세요 (result_*.csv가 있는 폴더)")
        self.folder_label.setStyleSheet("color:#aaa; font-size:12px;")
        fl.addWidget(self.folder_label, 1)
        btn_browse = QPushButton("폴더 선택")
        btn_browse.setFixedHeight(32)
        btn_browse.clicked.connect(self._choose_folder)
        fl.addWidget(btn_browse)
        root.addWidget(folder_grp)

        # ── 배치 목록(체크박스로 여러 개 선택) ──────────────────────
        list_grp = QGroupBox("배치 목록 (비교할 항목 체크)")
        ll = QVBoxLayout(list_grp)
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget{background:#1a1a1a;color:#ddd;border:1px solid #444;"
            "font-size:13px;}"
            "QListWidget::item{padding:8px 6px; min-height:22px;}"
            "QListWidget::item:selected{background:#2a2a2a;}"
            # 체크박스 인디케이터 크게 + 체크됐을 때 눈에 띄게 초록색
            "QListWidget::indicator{width:20px; height:20px;"
            "border:2px solid #bbbbbb; border-radius:4px; background:#2a2a2a;}"
            "QListWidget::indicator:hover{border:2px solid #1abc9c;}"
            "QListWidget::indicator:checked{background:#1abc9c; border:2px solid #1abc9c;}"
        )
        # 리스트위젯 항목의 체크박스 클릭 판정 영역이 좁아서, 행(텍스트 부분) 어디를
        # 클릭해도 체크/해제가 토글되도록 뷰포트에 이벤트 필터를 건다. Qt 기본
        # 동작은 인디케이터 정확히 클릭해야만 토글되므로, 이 필터가 없으면
        # 사용자가 텍스트를 클릭했을 때 아무 반응이 없어 혼란스러울 수 있다.
        self.list_widget.viewport().installEventFilter(self)
        # 체크 상태 변경 시 배경색(선택된 항목은 초록빛, 해제되면 원래
        # 배경색) 갱신용 콜백 - QSS의 `::item:checked` 지원이 위젯
        # 버전마다 들쭉날쭉해서(체크 상태 반영 안 되는 버전) 직접 처리.
        self.list_widget.itemChanged.connect(self._on_item_check_changed)
        ll.addWidget(self.list_widget)

        self.empty_label = QLabel("해당 폴더에 결과 파일이 없습니다.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color:#888; font-size:13px; padding:24px;")
        self.empty_label.setVisible(False)
        ll.addWidget(self.empty_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_compare = QPushButton("📊 선택 항목 비교")
        self.btn_compare.setFixedHeight(36)
        self.btn_compare.setEnabled(False)
        self.btn_compare.setStyleSheet(
            "QPushButton{background:#1abc9c;color:white;font-weight:bold;"
            "border-radius:6px;padding:0 16px;}"
            "QPushButton:disabled{background:#3a3a3a;color:#666;}"
            "QPushButton:hover{background:#16a085;}"
        )
        self.btn_compare.clicked.connect(self._compare_selected)
        btn_row.addWidget(self.btn_compare)
        ll.addLayout(btn_row)
        root.addWidget(list_grp, 2)

        # ── 최근 2건 증감 배너 ─────────────────────────────────────
        self.delta_banner = QLabel("")
        self.delta_banner.setAlignment(Qt.AlignCenter)
        self.delta_banner.setWordWrap(True)
        self.delta_banner.setVisible(False)
        root.addWidget(self.delta_banner)

        # ── 수율/불량률 비교 표 ────────────────────────────────────
        table_grp = QGroupBox("수율 / 불량률 비교")
        tl = QVBoxLayout(table_grp)
        self.compare_table = QTableWidget(0, 4)
        self.compare_table.setHorizontalHeaderLabels(["날짜/시각", "전체 건수", "수율(%)", "불량률(%)"])
        self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compare_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.compare_table.verticalHeader().setVisible(False)
        hdr = self.compare_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        self.compare_table.setStyleSheet(
            "QTableWidget{background:#1a1a1a;color:#ddd;gridline-color:#3a3a3a;"
            "border:1px solid #444;}"
            "QHeaderView::section{background:#333;color:#eee;border:none;"
            "padding:5px;font-weight:bold;}"
        )
        tl.addWidget(self.compare_table)
        root.addWidget(table_grp, 2)

        # ── 수율 추이 그래프 (2개 이상 선택 시 표시) ─────────────────
        self.chart_grp = QGroupBox("수율 추이 (2개 이상 선택 시 표시)")
        cl = QVBoxLayout(self.chart_grp)
        self.figure = Figure(figsize=(7, 3.2), dpi=100)
        self.figure.patch.set_facecolor("#252525")
        self.canvas = FigureCanvas(self.figure)
        cl.addWidget(self.canvas)
        self.chart_grp.setVisible(False)
        root.addWidget(self.chart_grp, 3)

        self.setStyleSheet("""
            QMainWindow,QWidget{background:#252525;color:#ddd;}
            QGroupBox{border:1px solid #444;border-radius:6px;
                      margin-top:8px;padding-top:8px;color:#eee;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QPushButton{background:#3a3a3a;color:#ddd;
                        border:1px solid #555;border-radius:4px;padding:4px 8px;}
            QPushButton:hover{background:#4a4a4a;}
        """)

    # ── 이벤트필터: 리스트 항목 아무데나 클릭해도 체크 토글 ─────────────
    def eventFilter(self, obj, event):
        if (obj is self.list_widget.viewport()
                and event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton):
            item = self.list_widget.itemAt(event.pos())
            if item is not None:
                item.setCheckState(
                    Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                )
                return True  # Qt 기본 토글까지 겹쳐서 두 번 반전되는 것 방지
        return super().eventFilter(obj, event)

    def _on_item_check_changed(self, item):
        if item.checkState() == Qt.Checked:
            item.setBackground(QColor("#1e824c"))
        else:
            item.setBackground(QColor("#1a1a1a"))

    # ── 폴더 선택 / 목록 채우기 ───────────────────────────────────────
    def _choose_folder(self):
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        folder = QFileDialog.getExistingDirectory(self, "결과 폴더 선택", default_dir)
        if not folder:
            return
        self.folder_label.setText(folder)
        self.entries = find_batch_results(folder)
        self._populate_list()
        self._reset_compare_views()

    def _populate_list(self):
        self.list_widget.clear()
        has_entries = len(self.entries) > 0
        self.list_widget.setVisible(has_entries)
        self.empty_label.setVisible(not has_entries)
        self.btn_compare.setEnabled(has_entries)
        for e in self.entries:
            text = (
                f"{e['date']} {e['time']}  |  {e['total']}장  |  "
                f"수율 {e['yield_pct']:.1f}%  |  불량률 {e['defect_pct']:.1f}%  "
                f"({e['file']})"
            )
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, e)
            self.list_widget.addItem(item)

    def _reset_compare_views(self):
        self.compare_table.setRowCount(0)
        self.delta_banner.setVisible(False)
        self.chart_grp.setVisible(False)

    # ── 비교 실행 ──────────────────────────────────────────────────────
    def _compare_selected(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))

        if not selected:
            QMessageBox.information(self, "안내", "비교할 배치를 하나 이상 체크하세요.")
            return

        selected.sort(key=lambda e: (e["date"], e["time"]))
        self._fill_compare_table(selected)
        self._update_delta_banner(selected)
        self._update_trend_chart(selected)

    def _fill_compare_table(self, selected):
        self.compare_table.setRowCount(len(selected))
        for r, e in enumerate(selected):
            self.compare_table.setItem(r, 0, QTableWidgetItem(f"{e['date']} {e['time']}"))
            self.compare_table.setItem(r, 1, QTableWidgetItem(str(e["total"])))
            self.compare_table.setItem(r, 2, QTableWidgetItem(f"{e['yield_pct']:.1f}"))
            self.compare_table.setItem(r, 3, QTableWidgetItem(f"{e['defect_pct']:.1f}"))

    def _update_delta_banner(self, selected):
        if len(selected) < 2:
            self.delta_banner.setVisible(False)
            return

        prev, latest = selected[-2], selected[-1]
        delta = latest["defect_pct"] - prev["defect_pct"]
        tag = f"(최근 2건 비교: {prev['date']} {prev['time']} → {latest['date']} {latest['time']})"

        if delta > 0:
            text = f"⚠ 불량률 {delta:.1f}%p 증가 {tag}"
            color = "#c0392b"
        elif delta < 0:
            text = f"✓ 불량률 {abs(delta):.1f}%p 감소 {tag}"
            color = "#1e824c"
        else:
            text = f"— 불량률 변동 없음 {tag}"
            color = "#555555"

        self.delta_banner.setText(text)
        self.delta_banner.setStyleSheet(
            f"background:{color}; color:white; font-weight:bold; font-size:13px;"
            "padding:8px; border-radius:6px;"
        )
        self.delta_banner.setVisible(True)

    def _update_trend_chart(self, selected):
        # 2개 이상 선택되면 추이 그래프를 표시한다(기존엔 3개 이상이었음).
        if len(selected) < 2:
            self.chart_grp.setVisible(False)
            return

        self.chart_grp.setVisible(True)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1a1a1a")

        labels = [f"{e['date']}\n{e['time']}" for e in selected]
        yields = [e["yield_pct"] for e in selected]
        x = range(len(selected))

        ax.plot(x, yields, marker="o", color="#1abc9c", linewidth=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8, color="#ddd")
        ax.set_ylabel("수율(%)", color="#ddd")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="#ddd")
        ax.grid(True, alpha=0.25, color="#888")
        for spine in ax.spines.values():
            spine.set_color("#555")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CompareWindow()
    win.show()
    sys.exit(app.exec_())