import re, json, os
from pathlib import Path

locale_dir = Path(__file__).parent / "locales"
scan_dirs = [Path(__file__).parent]

keys = set()
for sd in scan_dirs:
    for py in sd.rglob("*.py"):
        if "locales" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(r'i18n\.tr\(["\']([^"\']+)["\']\)', text):
            keys.add(m.group(1))

for lang_file in locale_dir.glob("*.json"):
    with open(lang_file, encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for k in sorted(keys):
        if k not in data:
            data[k] = ""
            changed = True
    for k in list(data.keys()):
        if k not in keys:
            del data[k]
            changed = True
    if changed:
        with open(lang_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")
        print(f"Updated {lang_file.name}")
    else:
        print(f"{lang_file.name} up to date")
