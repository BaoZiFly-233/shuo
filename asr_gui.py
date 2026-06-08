import sys, tempfile, wave, shutil
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QHBoxLayout, QDialog, QCheckBox,
    QSystemTrayIcon, QMenu, QScrollArea, QSizePolicy, QGridLayout,
    QFileDialog, QSlider, QLineEdit, QGraphicsOpacityEffect)
from PySide6.QtCore import (QThread, Signal, QTimer, Qt, QSize, QRect, QRectF,
    QPointF)
from PySide6.QtGui import QGuiApplication, QColor, QPainter, QFont, QAction, QPen, QFontMetrics, QPixmap
from pynput import mouse, keyboard as kb
import sounddevice as sd
import numpy as np
import ctypes
import qtawesome as qta

_VK_CODE = {"V": 0x56, "LCONTROL": 0xA2}
user32 = ctypes.windll.user32

def send_paste():
    import time
    try:
        user32.keybd_event(_VK_CODE["LCONTROL"], 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(_VK_CODE["V"], 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(_VK_CODE["V"], 0, 2, 0)
        user32.keybd_event(_VK_CODE["LCONTROL"], 0, 2, 0)
    except Exception:
        pass

import i18n
import global_hotkey as gh

from theme import Theme, _ThemeWatcher, _apply_title_bar
from config import (USER_DIR, CONFIG_PATH, HISTORY_PATH, LOG_PATH,
                    logger, DEFAULT_CONFIG, Config, History)

model_root = Path(__file__).parent / "Qwen3-ASR-0.6B-ONNX-CPU"
from onnx_inference import OnnxAsrPipeline

SAMPLE_RATE = 16000
CHUNK = 1024
AUDIO_DTYPE = "int16"
SAMPLE_WIDTH = 2  # bytes per sample for int16
CHANNELS = 1
DEBOUNCE_MS = 300

ASR_LANGUAGES = [
    ("auto", "asr_lang.auto"),
    ("zh", "asr_lang.zh"),
    ("yue", "asr_lang.yue"),
    ("en", "asr_lang.en"),
    ("ja", "asr_lang.ja"),
    ("de", "asr_lang.de"),
    ("ko", "asr_lang.ko"),
    ("ru", "asr_lang.ru"),
    ("fr", "asr_lang.fr"),
    ("pt", "asr_lang.pt"),
    ("ar", "asr_lang.ar"),
    ("it", "asr_lang.it"),
    ("es", "asr_lang.es"),
    ("hi", "asr_lang.hi"),
    ("id", "asr_lang.id"),
    ("th", "asr_lang.th"),
    ("tr", "asr_lang.tr"),
    ("uk", "asr_lang.uk"),
    ("vi", "asr_lang.vi"),
    ("cs", "asr_lang.cs"),
    ("da", "asr_lang.da"),
    ("fil", "asr_lang.fil"),
    ("fi", "asr_lang.fi"),
    ("is", "asr_lang.is"),
    ("ms", "asr_lang.ms"),
    ("no", "asr_lang.no"),
    ("pl", "asr_lang.pl"),
    ("sv", "asr_lang.sv"),
]


class Loader(QThread):
    done = Signal(object)
    error = Signal(str)

    def run(self):
        try:
            mdir = model_root / "onnx_models"
            tok_src = model_root / "tokenizer.json"
            tok_dst = mdir / "tokenizer.json"
            if tok_src.exists() and not tok_dst.exists():
                shutil.copy2(str(tok_src), str(tok_dst))
            logger.info("Loading model...")
            pipeline = OnnxAsrPipeline(onnx_dir=str(mdir), num_threads=6)
            logger.info("Model loaded")
            self.done.emit(pipeline)
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.error.emit(str(e))


class Recorder(QThread):
    finished = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        self._running = True
        logger.info("Recording started")
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype=AUDIO_DTYPE, blocksize=CHUNK)
        stream.start()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wf = wave.open(tmp.name, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        frame_count = 0
        while self._running:
            data, _ = stream.read(CHUNK)
            wf.writeframes(data.tobytes())
            frame_count += 1
        wf.close()
        stream.stop()
        stream.close()
        duration = frame_count * CHUNK / SAMPLE_RATE
        logger.info(f"Recording finished, duration {duration:.1f}s")
        self.finished.emit(tmp.name)

    def stop(self):
        self._running = False


class InferWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, pipeline, audio_path, asr_lang=None):
        super().__init__()
        self.pipeline = pipeline
        self.audio_path = audio_path
        self.asr_lang = asr_lang

    def run(self):
        try:
            logger.info("Transcribing...")
            kwargs = {}
            if self.asr_lang and self.asr_lang != "auto":
                kwargs["language"] = self.asr_lang
            result = self.pipeline.transcribe(self.audio_path, **kwargs)
            logger.info("Transcription done")
            self.done.emit(result["text"])
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.error.emit(str(e))


class HotkeyDialog(QDialog):
    _capture_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(i18n.tr("settings.hotkey"))
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.setFixedSize(300, 120)
        layout = QVBoxLayout(self)
        self.label = QLabel(i18n.tr("settings.press_key"))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.cancel_btn = QPushButton(i18n.tr("btn.cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
        self._captured = None
        self._mods = set()
        self._capture_signal.connect(self._on_captured)

        def on_click(x, y, btn, pressed):
            if pressed:
                m = {mouse.Button.x1: "xbutton1", mouse.Button.x2: "xbutton2"}.get(btn)
                if m:
                    self._capture_signal.emit(m)
            return False

        def on_press(key):
            if isinstance(key, kb.Key):
                k = key.name.lower()
            else:
                try: k = key.char.lower()
                except: return True
            if k in ("ctrl_l", "ctrl_r", "ctrl"): self._mods.add("ctrl"); return True
            if k in ("shift_l", "shift_r", "shift"): self._mods.add("shift"); return True
            if k in ("alt_l", "alt_r", "alt"): self._mods.add("alt"); return True
            parts = list(self._mods) + [k]
            self._capture_signal.emit("+".join(parts))
            return False

        self._ml = mouse.Listener(on_click=on_click)
        self._kl = kb.Listener(on_press=on_press)
        self._ml.start()
        self._kl.start()
        # safety timer: auto-close after 30s to prevent stuck listeners
        self._safety = QTimer(self)
        self._safety.setSingleShot(True)
        self._safety.timeout.connect(self.reject)
        self._safety.start(30000)

    def _on_captured(self, hotkey):
        self._captured = hotkey
        self.accept()

    def get_hotkey(self):
        return self._captured

    def closeEvent(self, e):
        self._ml.stop()
        self._kl.stop()
        super().closeEvent(e)


class SettingsDialog(QDialog):
    """Settings dialog: hotkey, background image, opacity"""
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._temp_hotkey = config.get("hotkey", "f2")
        t = Theme.current()
        self.setWindowTitle(i18n.tr("settings.title"))
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.setMinimumWidth(440)
        self.setStyleSheet(f"SettingsDialog {{ background:{t['bg']}; }}")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 16, 20, 16)
        # hotkey row
        hk_label = QLabel(i18n.tr("settings.hotkey"))
        hk_label.setStyleSheet(f"font-weight:bold; color:{t['text']};")
        layout.addWidget(hk_label)
        hk_row = QHBoxLayout()
        self.hk_edit = QLineEdit(self._temp_hotkey.upper())
        self.hk_edit.setReadOnly(True); self.hk_edit.setFixedWidth(140)
        self.hk_edit.setStyleSheet(f"color:{t['text']}; background:{t['surface']}; border:1px solid {t['border']}; padding:4px 8px;")
        hk_row.addWidget(self.hk_edit)
        change_btn = QPushButton(i18n.tr("settings.change_hotkey")); change_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        change_btn.clicked.connect(self._change_hotkey)
        hk_row.addWidget(change_btn); hk_row.addStretch()
        layout.addLayout(hk_row)
        sep1 = QWidget(); sep1.setFixedHeight(1); sep1.setStyleSheet(f"background:{t['border']};")
        layout.addWidget(sep1)
        # background image
        bg_label = QLabel(i18n.tr("settings.bg_image"))
        bg_label.setStyleSheet(f"font-weight:bold; color:{t['text']};")
        layout.addWidget(bg_label)
        bg_row = QHBoxLayout()
        self.bg_edit = QLineEdit(config.get("bg_image", "")); self.bg_edit.setReadOnly(True)
        self.bg_edit.setPlaceholderText(i18n.tr("settings.bg_none"))
        self.bg_edit.setStyleSheet(f"color:{t['text']}; background:{t['surface']}; border:1px solid {t['border']}; padding:4px 8px;")
        bg_row.addWidget(self.bg_edit)
        browse_btn = QPushButton("..."); browse_btn.setFixedWidth(36); browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.clicked.connect(self._browse_bg)
        bg_row.addWidget(browse_btn)
        clear_btn = QPushButton(_icon("fa5s.times"), ""); clear_btn.setFixedWidth(36); clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_bg)
        bg_row.addWidget(clear_btn)
        layout.addLayout(bg_row)
        fit_label = QLabel(i18n.tr("settings.fit_mode")); fit_label.setStyleSheet(f"font-weight:bold; color:{t['text']};")
        layout.addWidget(fit_label)
        self.fit_box = QComboBox(); self.fit_box.addItems(["Cover", "Contain", "Tile", "Center"])
        cur_fit = config.get("bg_fit", "cover").capitalize()
        for i in range(self.fit_box.count()):
            if self.fit_box.itemText(i).lower() == cur_fit.lower():
                self.fit_box.setCurrentIndex(i); break
        self.fit_box.setStyleSheet(f"color:{t['text']}; background:{t['surface']}; border:1px solid {t['border']}; padding:4px;")
        layout.addWidget(self.fit_box)
        sep2 = QWidget(); sep2.setFixedHeight(1); sep2.setStyleSheet(f"background:{t['border']};")
        layout.addWidget(sep2)
        opacity_label = QLabel(i18n.tr("settings.opacity")); opacity_label.setStyleSheet(f"font-weight:bold; color:{t['text']};")
        layout.addWidget(opacity_label)
        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal); self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(config.get("opacity", 100))
        self.opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow); self.opacity_slider.setTickInterval(10)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider)
        self.opacity_pct = QLabel(f"{self.opacity_slider.value()}%"); self.opacity_pct.setFixedWidth(40)
        self.opacity_pct.setStyleSheet(f"color:{t['text']};"); self.opacity_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        opacity_row.addWidget(self.opacity_pct)
        layout.addLayout(opacity_row)

        # ── 分隔线 ──
        sep3 = QWidget(); sep3.setFixedHeight(1); sep3.setStyleSheet(f"background:{t['border']};")
        layout.addWidget(sep3)

        # ── 桌面快捷方式 ──
        shortcut_btn = QPushButton("Create Desktop Shortcut")
        shortcut_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        shortcut_btn.clicked.connect(self._create_shortcut)
        layout.addWidget(shortcut_btn)

        # ── 开机自启 ──
        self.autostart_cb = QCheckBox("Start with Windows")
        self.autostart_cb.setChecked(self._get_autostart())
        self.autostart_cb.stateChanged.connect(self._on_autostart_changed)
        layout.addWidget(self.autostart_cb)

        layout.addStretch()
        btn_row = QHBoxLayout(); btn_row.addStretch()
        cancel_btn = QPushButton(i18n.tr("btn.cancel")); cancel_btn.setFixedWidth(90)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        save_btn = QPushButton(i18n.tr("settings.save")); save_btn.setFixedWidth(90)
        save_btn.setStyleSheet(f"QPushButton {{ background:{Theme.accent().name()}; color:#ffffff; border:none; border-radius:4px; padding:6px 16px; font-weight:bold; }} QPushButton:hover {{ background:{Theme.accent().lighter(115).name()}; }}")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _change_hotkey(self):
        dlg = HotkeyDialog()
        if dlg.exec() == QDialog.DialogCode.Accepted:
            hk = dlg.get_hotkey()
            if hk: self._temp_hotkey = hk; self.hk_edit.setText(hk.upper())
    def _browse_bg(self):
        path, _ = QFileDialog.getOpenFileName(self, i18n.tr("settings.choose_bg"), "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if path: self.bg_edit.setText(str(Path(path)))
    def _clear_bg(self): self.bg_edit.clear()
    def _on_opacity_changed(self, val): self.opacity_pct.setText(f"{val}%")

    @staticmethod
    def _get_exe_path():
        """Return path to the executable (frozen or source)."""
        if getattr(sys, 'frozen', False):
            return Path(sys.executable)
        return Path(sys.argv[0]).resolve()

    @staticmethod
    def _create_shortcut():
        """Create desktop shortcut to the app via PowerShell."""
        import subprocess
        exe = SettingsDialog._get_exe_path()
        desktop = Path.home() / "Desktop"
        lnk = desktop / "Shuo.lnk"
        try:
            ps = f'''
                $WshShell = New-Object -ComObject WScript.Shell
                $Shortcut = $WshShell.CreateShortcut("{lnk}")
                $Shortcut.TargetPath = "{exe}"
                $Shortcut.WorkingDirectory = "{exe.parent}"
                $Shortcut.IconLocation = "{exe}"
                $Shortcut.Save()
            '''
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, check=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(None, "Shuo", f"Desktop shortcut created:\n{lnk}")
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "Shuo", f"Failed to create shortcut:\n{e}")

    @staticmethod
    def _get_autostart():
        """Check if app is configured to start with Windows."""
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_READ)
            val, _ = winreg.QueryValueEx(key, "Shuo")
            winreg.CloseKey(key)
            return True
        except Exception:
            return False

    @staticmethod
    def _on_autostart_changed(state):
        """Enable/disable auto-start with Windows."""
        import winreg
        exe = SettingsDialog._get_exe_path()
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE)
            if state == Qt.CheckState.Checked.value:
                winreg.SetValueEx(key, "Shuo", 0, winreg.REG_SZ, str(exe))
            else:
                try:
                    winreg.DeleteValue(key, "Shuo")
                except Exception:
                    pass
            winreg.CloseKey(key)
        except Exception as e:
            logger.warning(f"Failed to set auto-start: {e}")
    def _save(self):
        self.config["hotkey"] = self._temp_hotkey
        self.config["bg_image"] = self.bg_edit.text()
        self.config["bg_fit"] = self.fit_box.currentText().lower()
        self.config["opacity"] = self.opacity_slider.value()
        self.accept()


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = Theme.current()
        self.setWindowTitle(i18n.tr("about.title"))
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.setMinimumWidth(520)
        self.setStyleSheet(f"AboutDialog {{ background:{t['bg']}; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(14)
        cl.setContentsMargins(32, 24, 32, 24)

        title = QLabel("Shuo")
        title_font = title.font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{t['text']};")
        cl.addWidget(title)

        subtitle = QLabel(i18n.tr("about.subtitle"))
        sub_font = subtitle.font()
        sub_font.setPointSize(10)
        subtitle.setFont(sub_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color:{t['sub_text']};")
        cl.addWidget(subtitle)

        cl.addSpacing(8)

        accent_hex = Theme.accent().name()
        license_html = i18n.tr("about.license").format(accent=accent_hex)
        info = QLabel(
            f'<p style="line-height:1.6; color:{t["text"]};">'
            f'{license_html}</p>'
        )
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        cl.addWidget(info)

        cl.addSpacing(4)

        deps = [
            ("qtawesome", ">=0.4.2", "MIT", i18n.tr("about.dep_qtawesome")),
            ("pynput", ">=1.8.0", "LGPL-3.0", i18n.tr("about.dep_pynput")),
            ("sounddevice", ">=0.5.0", "MIT", i18n.tr("about.dep_sounddevice")),
            ("onnxruntime", ">=1.26.0", "MIT", i18n.tr("about.dep_onnx")),
            ("numpy", ">=1.4.0", "BSD-3", i18n.tr("about.dep_numpy")),
            ("librosa", ">=0.11.0", "ISC", i18n.tr("about.dep_librosa")),
            ("tokenizers", ">=0.23.0", "Apache-2.0", i18n.tr("about.dep_tokenizers")),
        ]

        grid = QWidget()
        grid_layout = QGridLayout(grid)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        headers = [
            i18n.tr("about.col_lib"),
            i18n.tr("about.col_ver"),
            i18n.tr("about.col_lic"),
            i18n.tr("about.col_use"),
        ]
        header_font = QFont(self.font())
        header_font.setBold(True)

        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setFont(header_font)
            lbl.setContentsMargins(8, 6, 8, 6)
            lbl.setStyleSheet(f"color:{t['text']};")
            grid_layout.addWidget(lbl, 0, col)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{t['border']};")
        grid_layout.addWidget(sep, 1, 0, 1, 4)

        for row, (name, ver, lic, desc) in enumerate(deps):
            r = row + 2
            row_bg = t["surface"] if row % 2 == 1 else t["bg"]
            for col, val in enumerate([name, ver, lic, desc]):
                lbl = QLabel(val)
                lbl.setContentsMargins(8, 5, 8, 5)
                lbl.setStyleSheet(f"color:{t['text']}; background:{row_bg};")
                grid_layout.addWidget(lbl, r, col)

        cl.addWidget(grid)

        layout.addWidget(content, 1)

        btn_bar = QWidget()
        btn_bar.setStyleSheet(f"background:{t['surface']}; border-top:1px solid {t['border']};")
        bl = QHBoxLayout(btn_bar)
        bl.setContentsMargins(32, 10, 32, 10)
        bl.addStretch()
        close_btn = QPushButton(i18n.tr("btn.cancel"))
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        bl.addWidget(close_btn)
        layout.addWidget(btn_bar, 0)


class LoadingOverlay(QWidget):
    """Semi-transparent overlay with spinner arc animation"""

    _ARC_SPAN = 270 * 16
    _TIMER_MS = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._opacity = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._TIMER_MS)
        self.resize(parent.size() if parent else self.size())

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        if self._opacity < 1.0:
            self._opacity = min(1.0, self._opacity + 0.05)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        t = Theme.current()

        bg_alpha = int(140 * self._opacity) if Theme.is_dark() else int(60 * self._opacity)
        painter.fillRect(self.rect(), QColor(0, 0, 0, bg_alpha))

        cx = self.width() // 2
        cy = self.height() // 2
        spinner_r = 20

        accent = Theme.accent()
        pen = QPen(accent, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        start_angle = (self._angle * 16) % (360 * 16)
        painter.drawArc(QRectF(cx - spinner_r, cy - spinner_r,
                               spinner_r * 2, spinner_r * 2),
                        start_angle, self._ARC_SPAN)

        text = i18n.tr("loading.model")
        text_font = QFont(self.font().family(), 11)
        painter.setFont(text_font)
        painter.setPen(QColor(t["sub_text"]))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        painter.drawText(cx - tw // 2, cy + spinner_r + 28, text)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent():
            self.resize(self.parent().size())


class ResultItem(QWidget):
    """Card-style result: accent bar + text + copy button"""

    _RADIUS = 12
    _ACCENT_W = 4
    _PADDING_H = 14
    _PADDING_V = 12
    _MAX_LINES = 3

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._text = text
        self._hover = False
        self._copy_hover = False
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        fm = QFontMetrics(self.font())
        self._card_h = fm.lineSpacing() * self._MAX_LINES + self._PADDING_V * 2

        self._copy_w = 32
        self._copy_rect = None

    def sizeHint(self):
        return QSize(self.width(), self._card_h)

    def minimumHeight(self):
        return self._card_h

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        t = Theme.current()
        accent = Theme.accent()
        card = QRectF(self.rect()).adjusted(0.5, 1.5, -0.5, -1.5)

        if self._hover:
            bg = QColor(t["surface"]).lighter(112)
            border = accent
        else:
            bg = QColor(t["surface"])
            border = QColor(t["border"])

        painter.setPen(QPen(border, 1.0, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(bg)
        painter.drawRoundedRect(card, self._RADIUS, self._RADIUS)

        if self._hover:
            bar = QRectF(card.left() + 2, card.top() + 8,
                         self._ACCENT_W, card.height() - 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            painter.drawRoundedRect(bar, 2, 2)

        bar_w = self._ACCENT_W + 4
        text_left = card.left() + self._PADDING_H + bar_w
        text_w = card.width() - self._PADDING_H * 2 - bar_w

        if self._hover:
            copy_area_right = card.right() - 8
            copy_area_left = copy_area_right - self._copy_w
            self._copy_rect = QRectF(copy_area_left, card.top() + (card.height() - self._copy_w) / 2,
                                     self._copy_w, self._copy_w)
            text_w = copy_area_left - text_left - 8

            if self._copy_hover:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(accent.lighter(160))
                painter.drawRoundedRect(self._copy_rect.adjusted(1, 1, -1, -1), 6, 6)

            copy_icon = _icon("fa5s.copy", Theme.accent().name())
            icon_sz = 16
            cx = self._copy_rect.x() + (self._copy_w - icon_sz) / 2
            cy = self._copy_rect.y() + (self._copy_w - icon_sz) / 2
            copy_icon.paint(painter, int(cx), int(cy), icon_sz, icon_sz)
        else:
            self._copy_rect = None

        text_rect = QRectF(text_left, card.top() + self._PADDING_V,
                           text_w, card.height() - self._PADDING_V * 2)
        painter.setPen(QColor(t["text"]))
        fm = painter.fontMetrics()
        elided = fm.elidedText(self._text, Qt.TextElideMode.ElideRight,
                               text_rect.width() * 3)
        painter.drawText(text_rect, Qt.TextFlag.TextWordWrap, elided)

        painter.end()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        old_hover = self._hover
        old_copy = self._copy_hover
        self._hover = self.rect().contains(pos)
        self._copy_hover = bool(self._copy_rect and self._copy_rect.contains(
            QPointF(pos)))
        if self._hover != old_hover or self._copy_hover != old_copy:
            self.setCursor(Qt.CursorShape.PointingHandCursor
                           if self._copy_hover else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, event):
        if self._hover or self._copy_hover:
            self._hover = False
            self._copy_hover = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

    def mousePressEvent(self, event):
        if self._copy_hover:
            QGuiApplication.clipboard().setText(self._text)


def _icon(name, color=None):
    return qta.icon(name, color=color or Theme.current()["text"])


class BgLabel(QLabel):
    """Background image label that respects fit mode (cover, contain, tile, center)."""
    def __init__(self, pixmap, fit_mode="cover", parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._fit_mode = fit_mode
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_fit_mode(self, mode):
        self._fit_mode = mode
        self.update()

    def paintEvent(self, event):
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or ww <= 0 or wh <= 0:
            return

        if self._fit_mode == "cover":
            # Scale to cover entire widget, crop
            scale = max(ww / pw, wh / ph)
            sw, sh = int(pw * scale), int(ph * scale)
            x, y = (ww - sw) // 2, (wh - sh) // 2
            painter.drawPixmap(x, y, sw, sh, self._pixmap)
        elif self._fit_mode == "contain":
            # Scale to fit within widget, letterbox
            scale = min(ww / pw, wh / ph)
            sw, sh = int(pw * scale), int(ph * scale)
            x, y = (ww - sw) // 2, (wh - sh) // 2
            painter.drawPixmap(x, y, sw, sh, self._pixmap)
        elif self._fit_mode == "tile":
            # Repeat the image at original size
            for tx in range(0, ww, pw):
                for ty in range(0, wh, ph):
                    painter.drawPixmap(tx, ty, self._pixmap)
        elif self._fit_mode == "center":
            # Center without scaling
            x, y = (ww - pw) // 2, (wh - ph) // 2
            painter.drawPixmap(x, y, self._pixmap)


class MainWindow(QMainWindow):
    hotkey_pressed = Signal()
    hotkey_released = Signal()

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.recorder = None
        self.worker = None
        self._temp_wav = None
        self._pending_wavs = []
        self.config = Config.load()
        i18n.load(self.config.get("language", "en"))
        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.on_debounce_end)
        self.hotkey_pressed.connect(self.on_press)
        self.hotkey_released.connect(self.on_release)

        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.setMinimumSize(500, 350)
        self.resize(700, 500)
        self.setWindowIcon(_icon("fa5s.microphone"))
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move((geo.width() - self.width()) // 2,
                      (geo.height() - self.height()) // 2)

        cw = QWidget()
        cw.setObjectName("centralWidget")
        self.setCentralWidget(cw)
        main_layout = QVBoxLayout(cw)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.settings_btn = QPushButton(_icon("fa5s.cog"), f"  {i18n.tr('btn.settings')}")
        self.settings_btn.clicked.connect(self.open_settings)
        toolbar.addWidget(self.settings_btn)

        self.hint = QLabel()
        toolbar.addWidget(self.hint)

        toolbar.addStretch()

        self.auto_type_cb = QCheckBox(i18n.tr("settings.auto_type"))
        self.auto_type_cb.setChecked(self.config.get("auto_type", True))
        self.auto_type_cb.stateChanged.connect(self.on_auto_type_changed)
        toolbar.addWidget(self.auto_type_cb)

        self.history_cb = QCheckBox(i18n.tr("settings.history"))
        self.history_cb.setChecked(self.config.get("save_history", False))
        self.history_cb.setToolTip(i18n.tr("settings.history_tip").format(path=str(HISTORY_PATH)))
        self.history_cb.stateChanged.connect(self.on_history_changed)
        toolbar.addWidget(self.history_cb)

        self.punc_cb = QCheckBox(i18n.tr("settings.remove_punc"))
        self.punc_cb.setChecked(self.config.get("remove_punc", False))
        self.punc_cb.stateChanged.connect(self.on_remove_punc_changed)
        self.punc_cb.setVisible(i18n._COMPAT.get(self.config.get("language", "en"), self.config.get("language", "en")) == "zh")
        toolbar.addWidget(self.punc_cb)

        self.clear_btn = QPushButton(_icon("fa5s.trash"), "")
        self.clear_btn.setFixedSize(28, 28)
        self.clear_btn.setToolTip(i18n.tr("btn.clear_history"))
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_history)
        toolbar.addWidget(self.clear_btn)

        self.asr_lang_box = QComboBox()
        self.asr_lang_box.setFixedWidth(120)
        self._populate_asr_lang()
        current_asr = self.config.get("asr_lang", "auto")
        for i, (code, _) in enumerate(ASR_LANGUAGES):
            if code == current_asr:
                self.asr_lang_box.setCurrentIndex(i)
                break
        self.asr_lang_box.currentIndexChanged.connect(self._on_asr_lang_changed)
        toolbar.addWidget(self.asr_lang_box)

        self.lang_btn = QPushButton()
        self.lang_btn.setFlat(True)
        self.lang_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lang_menu = QMenu(self)
        for code, key in ASR_LANGUAGES:
            if code != "auto":
                self._lang_menu.addAction(i18n.tr(key), lambda c=code: self._switch_lang(c))
        self.lang_btn.setMenu(self._lang_menu)
        self._update_lang_label()
        toolbar.addWidget(self.lang_btn)

        main_layout.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                border: none; background: transparent;
                width: 8px; margin: 2px;
            }
            QScrollBar::handle:vertical {
                border-radius: 4px;
                background: palette(mid);
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: palette(dark);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical { background: none; }
        """)

        self.result_container = QWidget()
        self.result_container.setObjectName("resultContainer")
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.setContentsMargins(6, 6, 6, 6)
        self.result_layout.setSpacing(10)
        self.result_layout.addStretch()

        scroll.setWidget(self.result_container)
        main_layout.addWidget(scroll, 1)

        btn_container = QHBoxLayout()
        btn_container.setContentsMargins(0, 8, 0, 4)

        self.about_btn = QPushButton(_icon("fa5s.info-circle"), f"  {i18n.tr('btn.about')}")
        self.about_btn.setFixedHeight(36)
        self.about_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.about_btn.clicked.connect(self.open_about)
        btn_container.addWidget(self.about_btn)

        btn_container.addStretch()

        center = QVBoxLayout()
        center.setSpacing(6)
        self.status_label = QLabel(i18n.tr("btn.loading"))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.status_label.font()
        font.setPointSize(9)
        self.status_label.setFont(font)
        center.addWidget(self.status_label)

        self.btn = QPushButton()
        self.btn.setFixedSize(64, 64)
        self.btn.setEnabled(False)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setIconSize(QSize(28, 28))
        center.addWidget(self.btn, 0, Qt.AlignmentFlag.AlignCenter)
        btn_container.addLayout(center)

        btn_container.addStretch()

        self.exit_btn = QPushButton(_icon("fa5s.sign-out-alt"), f"  {i18n.tr('tray.quit')}")
        self.exit_btn.setFixedSize(110, 36)
        self.exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exit_btn.clicked.connect(self.quit_app)
        btn_container.addWidget(self.exit_btn)

        self._apply_btn_styles()

        main_layout.addLayout(btn_container)

        self.setup_tray()

        self.load_history()

        gh.load()

        self.loader = Loader()
        self.loader.done.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_load_error)
        self.loader.start()

        self._lang_menu.clear()
        for code, key in ASR_LANGUAGES:
            if code != "auto":
                self._lang_menu.addAction(i18n.tr(key), lambda c=code: self._switch_lang(c))
        self._update_lang_label()

        self.overlay = LoadingOverlay(cw)
        self.overlay.raise_()

        self._refresh_icons()

        self._theme_watcher = _ThemeWatcher()
        self._theme_watcher.changed.connect(self._apply_theme)
        QApplication.instance().installNativeEventFilter(self._theme_watcher)
        QApplication.styleHints().colorSchemeChanged.connect(
            lambda _: QTimer.singleShot(50, self._apply_theme))

        self._apply_theme()

        self.setWindowTitle(i18n.tr("app.title"))
        self.winId()

    def setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(_icon("fa5s.microphone"))
        self.tray.setToolTip(i18n.tr("app.title"))
        menu = QMenu()
        self.quit_action = QAction(i18n.tr("tray.quit"), self)
        self.quit_action.triggered.connect(self.quit_app)
        menu.addAction(self.quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def quit_app(self):
        gh.stop()
        self._cleanup_temp_wav()
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait(1000)
        if self.worker and self.worker.isRunning():
            self.worker.wait(1000)
        QApplication.instance().removeNativeEventFilter(self._theme_watcher)
        self.tray.hide()
        QApplication.quit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_bg_label') and self._bg_label:
            cw = self.centralWidget()
            if cw:
                self._bg_label.setGeometry(cw.rect())

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        if self.tray.supportsMessages():
            self.tray.showMessage(
                i18n.tr("app.title"),
                i18n.tr("tray.still_running"),
                QSystemTrayIcon.MessageIcon.Information, 3000)

    def _apply_theme(self):
        t = Theme.current()
        dark = Theme.is_dark()

        cw = self.centralWidget()
        if cw:
            cw.setStyleSheet(f"QWidget#centralWidget {{ background: {t['bg']}; }}")

            # Remove any existing bg overlay
            for child in cw.findChildren(BgLabel, "bg_overlay"):
                child.deleteLater()

            bg_image = self.config.get("bg_image", "")
            if bg_image and Path(bg_image).is_file():
                opacity_val = self.config.get("opacity", 100) / 100.0
                fit_mode = self.config.get("bg_fit", "cover")
                pixmap = QPixmap(str(Path(bg_image).resolve()))
                if not pixmap.isNull():
                    bg_label = BgLabel(pixmap, fit_mode=fit_mode, parent=cw)
                    bg_label.setObjectName("bg_overlay")
                    bg_label.setGeometry(cw.rect())
                    effect = QGraphicsOpacityEffect()
                    effect.setOpacity(opacity_val)
                    bg_label.setGraphicsEffect(effect)
                    bg_label.lower()
                    bg_label.show()
                    self._bg_label = bg_label
            else:
                self._bg_label = None

            cw.repaint()

        self._apply_btn_styles()

        _apply_title_bar(self, dark)

        for i in range(self.result_layout.count()):
            w = self.result_layout.itemAt(i).widget()
            if w:
                w.repaint()

        self.setWindowIcon(_icon("fa5s.microphone"))
        self.tray.setIcon(_icon("fa5s.microphone"))
        self._refresh_icons()

    def _refresh_icons(self):
        if not hasattr(self, 'settings_btn'):
            return
        self.settings_btn.setIcon(_icon("fa5s.cog"))
        self.clear_btn.setIcon(_icon("fa5s.trash"))
        self.exit_btn.setIcon(_icon("fa5s.sign-out-alt"))
        self.about_btn.setIcon(_icon("fa5s.info-circle"))
        self.lang_btn.setIcon(_icon("fa5s.language"))
        if self.btn.isEnabled() and self.pipeline:
            self.btn.setIcon(_icon("fa5s.microphone", "#ffffff"))
        self._apply_btn_styles()

    def _apply_btn_styles(self):
        accent = Theme.accent()
        t = Theme.current()
        r = 32

        style = (
            "QPushButton {"
            f"  border:none; border-radius:{r}px;"
            f"  background:{accent.name()}; color:#ffffff;"
            "}"
            "QPushButton:hover {"
            f"  background:{accent.lighter(115).name()};"
            "}"
            "QPushButton:pressed {"
            f"  background:{accent.darker(110).name()};"
            "}"
            "QPushButton:disabled {"
            f"  background:{t['border']}; color:{t['sub_text']};"
            "}"
        )
        self.btn.setStyleSheet(style)

    def load_history(self):
        items = History.load()
        for item in items:
            text = item.get("text", "")
            if text:
                widget = ResultItem(text)
                self.result_layout.insertWidget(0, widget)

    def open_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def open_settings(self):
        gh.stop()
        dlg = SettingsDialog(self.config, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            Config.save(self.config)
            gh.save(self.config.get("hotkey", "f2"))
            self._apply_theme()
        gh.start(on_down=self.hotkey_pressed.emit, on_up=self.hotkey_released.emit)
        self.update_hint()

    def _populate_asr_lang(self):
        self.asr_lang_box.clear()
        for code, key in ASR_LANGUAGES:
            self.asr_lang_box.addItem(i18n.tr(key), code)

    def _on_asr_lang_changed(self, idx):
        code = self.asr_lang_box.itemData(idx)
        self.config["asr_lang"] = code
        Config.save(self.config)

    def update_hint(self):
        hk = gh.get()
        self.hint.setText(i18n.tr("hint.hold").format(key=hk))
        self.hint.setStyleSheet("")

    def on_auto_type_changed(self, state):
        self.config["auto_type"] = (state == Qt.CheckState.Checked.value)
        Config.save(self.config)

    def on_history_changed(self, state):
        self.config["save_history"] = (state == Qt.CheckState.Checked.value)
        Config.save(self.config)

    def on_remove_punc_changed(self, state):
        self.config["remove_punc"] = (state == Qt.CheckState.Checked.value)
        Config.save(self.config)

    def clear_history(self):
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle(i18n.tr("dialog.clear_title"))
        box.setText(i18n.tr("dialog.clear_msg"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setWindowIcon(_icon("fa5s.microphone"))
        box.setIconPixmap(_icon("fa5s.microphone").pixmap(32, 32))
        reply = box.exec()
        if reply == QMessageBox.StandardButton.Yes:
            History.save([])
            while self.result_layout.count() > 1:
                item = self.result_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _switch_lang(self, lang):
        self.config["language"] = lang
        Config.save(self.config)
        i18n.load(lang)
        self._lang_menu.clear()
        for code, key in ASR_LANGUAGES:
            if code != "auto":
                self._lang_menu.addAction(i18n.tr(key), lambda c=code: self._switch_lang(c))
        self.setWindowTitle(i18n.tr("app.title"))
        self.settings_btn.setText(f"  {i18n.tr('btn.settings')}")
        self.auto_type_cb.setText(i18n.tr("settings.auto_type"))
        self.history_cb.setText(i18n.tr("settings.history"))
        self.history_cb.setToolTip(i18n.tr("settings.history_tip").format(path=str(HISTORY_PATH)))
        self.punc_cb.setText(i18n.tr("settings.remove_punc"))
        self.punc_cb.setVisible(i18n._COMPAT.get(lang, lang) == "zh")
        self.clear_btn.setToolTip(i18n.tr("btn.clear_history"))
        self.about_btn.setText(f"  {i18n.tr('btn.about')}")
        self.exit_btn.setText(f"  {i18n.tr('tray.quit')}")
        self.quit_action.setText(i18n.tr("tray.quit"))
        self._populate_asr_lang()
        current_asr = self.config.get("asr_lang", "auto")
        for i in range(self.asr_lang_box.count()):
            if self.asr_lang_box.itemData(i) == current_asr:
                self.asr_lang_box.setCurrentIndex(i)
                break
        self._update_lang_label()
        self.update_hint()

    def _update_lang_label(self):
        lang = self.config.get("language", "en")
        code = i18n._COMPAT.get(lang, lang).upper()
        self.lang_btn.setIcon(_icon("fa5s.language"))
        self.lang_btn.setText(f"  {code}")

    def on_model_loaded(self, pipeline):
        self.pipeline = pipeline
        self.status_label.setText(i18n.tr("btn.start"))
        self.btn.setIcon(_icon("fa5s.microphone", "#ffffff"))
        self.btn.setEnabled(True)
        self.btn.pressed.connect(self.on_press)
        self.btn.released.connect(self.on_release)
        self.update_hint()
        self.setFocus()
        gh.start(on_down=self.hotkey_pressed.emit, on_up=self.hotkey_released.emit)
        self.overlay.hide()

    def on_load_error(self, msg):
        self.overlay.hide()
        self.status_label.setText(i18n.tr("status.load_failed"))
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, i18n.tr("app.title"),
            i18n.tr("status.load_error_msg").format(error=msg))

    def on_press(self):
        if self.debounce.isActive():
            self.debounce.stop()
            return
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait(1000)
        if self.recorder:
            try: self.recorder.finished.disconnect()
            except Exception: pass
            self.recorder.deleteLater()
            self.recorder = None
        self.recorder = Recorder()
        self.recorder.finished.connect(self.on_recorded)
        self.recorder.start()
        self.status_label.setText(i18n.tr("btn.stop"))
        self.btn.setIcon(_icon("fa5s.stop", "#ffffff"))

    def on_release(self):
        if self.recorder and self.recorder.isRunning():
            self.debounce.start(DEBOUNCE_MS)

    def on_debounce_end(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.btn.setEnabled(False)
            self.status_label.setText(i18n.tr("status.transcribing"))
            self.btn.setIcon(_icon("fa5s.spinner", "#ffffff"))

    def on_recorded(self, path):
        if self.worker and self.worker.isRunning():
            self._pending_wavs.append(path)
            n = len(self._pending_wavs)
            self.status_label.setText(i18n.tr("status.transcribing") + f" ({n})")
            self.btn.setEnabled(True)
            return
        self._start_infer(path)

    def _start_infer(self, path):
        self.status_label.setText(i18n.tr("status.transcribing"))
        self.btn.setIcon(_icon("fa5s.spinner", "#ffffff"))
        self.btn.setEnabled(False)
        if self.worker:
            try: self.worker.done.disconnect()
            except Exception: pass
            try: self.worker.error.disconnect()
            except Exception: pass
            self.worker.deleteLater()
        self._temp_wav = path
        self.worker = InferWorker(self.pipeline, path, self.config.get("asr_lang", "auto"))
        self.worker.done.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_result(self, text):
        self._cleanup_temp_wav()
        self.status_label.setText(i18n.tr("btn.start"))
        self.btn.setIcon(_icon("fa5s.microphone", "#ffffff"))
        self.btn.setEnabled(True)
        if text.strip():
            lang = self.config.get("language", "en")
            real = i18n._COMPAT.get(lang, lang)
            if real == "zh" and self.config.get("remove_punc", False):
                import re
                _PUNC = '，。！？、；：""''（）【】《》…—.!?;:\"\'()[]{}<>~·～—'
                text = re.sub('[' + re.escape(_PUNC) + ']', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
            if self.config.get("save_history", False):
                History.add(text)
            while self.result_layout.count() > 101:
                item = self.result_layout.takeAt(self.result_layout.count() - 2)
                if item and item.widget():
                    item.widget().deleteLater()
            item_widget = ResultItem(text)
            self.result_layout.insertWidget(0, item_widget)
            if self.auto_type_cb.isChecked():
                QGuiApplication.clipboard().setText(text)
                QTimer.singleShot(100, send_paste)
        self._process_next()

    def on_error(self, msg):
        self._cleanup_temp_wav()
        self.status_label.setText(i18n.tr("btn.start"))
        self.btn.setIcon(_icon("fa5s.microphone", "#ffffff"))
        self.btn.setEnabled(True)
        self._process_next()

    def _process_next(self):
        if self._pending_wavs:
            path = self._pending_wavs.pop(0)
            self._start_infer(path)

    def _cleanup_temp_wav(self):
        if self._temp_wav:
            try: Path(self._temp_wav).unlink(missing_ok=True)
            except Exception: pass
            self._temp_wav = None
        if self.recorder and not self.recorder.isRunning():
            self.recorder.wait(500)
            self.recorder.deleteLater()
            self.recorder = None
        if self.worker and not self.worker.isRunning():
            self.worker.deleteLater()
            self.worker = None

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and self.btn.isEnabled():
            self.on_press()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self.on_release()


if __name__ == "__main__":
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(sys.argv)
        app.setStyle("windows11")

        font = app.font()
        font.setFamilies(["Segoe UI", "Microsoft YaHei", "sans-serif"])
        font.setPointSize(9)
        app.setFont(font)

        app.setWindowIcon(_icon("fa5s.microphone"))
        i18n.load("en")
        logger.info("App started")
        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"App exited with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
