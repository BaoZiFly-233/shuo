import json, threading, time, ctypes, logging
from pathlib import Path
from pynput import mouse, keyboard

_log = logging.getLogger("shuo")

CONFIG_DIR = Path.home() / ".shuo"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_HOTKEY = "xbutton1"

_hotkey = DEFAULT_HOTKEY
_listener = None
_listener_lock = threading.RLock()
_on_down = None
_on_up = None
_hotkey_active = False
_last_down = 0.0
DEBOUNCE_MS = 300

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return True

def load():
    global _hotkey
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
                _hotkey = cfg.get("hotkey", DEFAULT_HOTKEY)
        except: pass

def save(hotkey=None):
    if hotkey:
        global _hotkey
        _hotkey = hotkey
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
        except: pass
    cfg["hotkey"] = _hotkey
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def get():
    return _hotkey

def start(on_down=None, on_up=None):
    global _listener, _on_down, _on_up, _hotkey_active, _last_down
    _on_down = on_down
    _on_up = on_up
    _hotkey_active = False
    _last_down = 0.0

    parts = _hotkey.split("+")
    key_name = parts[-1]
    is_mouse = key_name in ("xbutton1", "xbutton2")

    with _listener_lock:
        stop()
        time.sleep(0.05)

        if not is_admin():
            _log.warning("Not running as administrator — hotkey may not work "
                         "in elevated windows (Task Manager, UAC prompts, etc.)")

        if is_mouse:
            btn = mouse.Button.x2 if key_name == "xbutton2" else mouse.Button.x1
            def _on_click(x, y, b, pressed):
                global _hotkey_active, _last_down
                if b != btn: return
                if pressed and not _hotkey_active:
                    now = time.monotonic()
                    if now - _last_down < DEBOUNCE_MS / 1000.0:
                        return
                    _hotkey_active = True
                    _last_down = now
                    if _on_down: _on_down()
                elif not pressed and _hotkey_active:
                    _hotkey_active = False
                    if _on_up: _on_up()
            ml = mouse.Listener(on_click=_on_click)
            ml.start()
            _listener = (ml,)
        else:
            needs_ctrl = "ctrl" in parts
            needs_shift = "shift" in parts
            needs_alt = "alt" in parts
            _pressed = set()

            def _on_press(key):
                global _hotkey_active, _last_down
                if isinstance(key, keyboard.Key):
                    k = key.name.lower()
                else:
                    try: k = key.char.lower()
                    except: return True
                if k in ("ctrl", "ctrl_l", "ctrl_r"):
                    _pressed.add("ctrl")
                elif k in ("shift", "shift_l", "shift_r"):
                    _pressed.add("shift")
                elif k in ("alt", "alt_l", "alt_r"):
                    _pressed.add("alt")
                elif k == key_name and not _hotkey_active:
                    if (needs_ctrl and "ctrl" not in _pressed): return True
                    if (needs_shift and "shift" not in _pressed): return True
                    if (needs_alt and "alt" not in _pressed): return True
                    now = time.monotonic()
                    if now - _last_down < DEBOUNCE_MS / 1000.0:
                        return True
                    _hotkey_active = True
                    _last_down = now
                    if _on_down: _on_down()
                return True

            def _on_release(key):
                global _hotkey_active
                if isinstance(key, keyboard.Key):
                    k = key.name.lower()
                else:
                    try: k = key.char.lower()
                    except: return True
                if k in ("ctrl", "ctrl_l", "ctrl_r"):
                    _pressed.discard("ctrl")
                elif k in ("shift", "shift_l", "shift_r"):
                    _pressed.discard("shift")
                elif k in ("alt", "alt_l", "alt_r"):
                    _pressed.discard("alt")
                elif k == key_name and _hotkey_active:
                    _hotkey_active = False
                    if _on_up: _on_up()
                return True

            kl = keyboard.Listener(on_press=_on_press, on_release=_on_release)
            kl.start()
            _listener = (kl,)

def stop():
    global _listener
    with _listener_lock:
        if _listener:
            for l in _listener:
                l.stop()
                l.join(timeout=2.0)
            _listener = None
