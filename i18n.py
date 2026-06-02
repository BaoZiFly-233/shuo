import json, os

_current = {}

def load(lang="zh_CN"):
    _current.clear()
    path = os.path.join(os.path.dirname(__file__), "locales", f"{lang}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _current.update(json.load(f))

def tr(key):
    return _current.get(key, key)
