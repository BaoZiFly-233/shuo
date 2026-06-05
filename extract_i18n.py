import re, json, os
from pathlib import Path

locale_dir = Path(__file__).parent / "locales"
scan_dirs = [Path(__file__).parent]

# Dynamic keys referenced via variables at runtime
DYNAMIC_KEYS = {
    "asr_lang.auto", "asr_lang.zh", "asr_lang.yue", "asr_lang.en",
    "asr_lang.ja", "asr_lang.de", "asr_lang.ko", "asr_lang.ru",
    "asr_lang.fr", "asr_lang.pt", "asr_lang.ar", "asr_lang.it",
    "asr_lang.es", "asr_lang.hi", "asr_lang.id", "asr_lang.th",
    "asr_lang.tr", "asr_lang.uk", "asr_lang.vi", "asr_lang.cs",
    "asr_lang.da", "asr_lang.fil", "asr_lang.fi", "asr_lang.is",
    "asr_lang.ms", "asr_lang.no", "asr_lang.pl", "asr_lang.sv",
}

keys = set(DYNAMIC_KEYS)
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
