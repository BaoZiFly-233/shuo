# ── 主题 ──────────────────────────────────────────────────────────────

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import QObject, QTimer, Signal, QAbstractNativeEventFilter
import ctypes


class Theme:
    """显式颜色主题。不依赖 QPalette 传播（在 Win11 下不可靠）。"""

    # 亮色
    LIGHT = {
        "bg":         "#ffffff",
        "surface":    "#f3f3f3",
        "border":     "#e0e0e0",
        "text":       "#1e1e1e",
        "sub_text":   "#666666",
        "overlay_bg": "rgba(0,0,0,30)",
    }
    # 暗色
    DARK = {
        "bg":         "#1e1e1e",
        "surface":    "#2d2d2d",
        "border":     "#3d3d3d",
        "text":       "#f0f0f0",
        "sub_text":   "#999999",
        "overlay_bg": "rgba(0,0,0,140)",
    }

    @classmethod
    def is_dark(cls):
        return QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark

    @classmethod
    def current(cls):
        return cls.DARK if cls.is_dark() else cls.LIGHT

    @classmethod
    def accent(cls):
        """系统强调色（来自 QPalette，它对此角色是可靠的）。"""
        pal = QApplication.palette()
        try:
            return pal.color(QPalette.ColorRole.Accent)
        except AttributeError:
            return pal.color(QPalette.ColorRole.Highlight)


class _ThemeWatcher(QObject, QAbstractNativeEventFilter):
    """Win32：监听 WM_SETTINGCHANGE → 注册表变了 → 通知 MainWindow 刷新。"""
    changed = Signal()

    def nativeEventFilter(self, eventType, message):
        if eventType in (b"windows_generic_MSG", b"MSG"):
            msg = int(message)
            if msg in (0x001A, 0x031E):
                # 延迟 100ms 等注册表落定
                QTimer.singleShot(100, self.changed.emit)
        return False, 0


def _apply_title_bar(window, dark):
    try:
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            int(window.winId()),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value))
    except:
        pass
