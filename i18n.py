import json, os

_current = {}
_fallback = {}

_COMPAT = {"zh_CN": "zh"}

def load(lang="en"):
    _current.clear()
    _fallback.clear()
    locale_dir = os.path.join(os.path.dirname(__file__), "locales")
    base = os.path.join(locale_dir, "en.json")
    if os.path.exists(base):
        with open(base, encoding="utf-8") as f:
            _fallback.update(json.load(f))
    real = _COMPAT.get(lang, lang)
    path = os.path.join(locale_dir, f"{real}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _current.update(json.load(f))

def tr(key):
    return _current.get(key, _fallback.get(key, key))
