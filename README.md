# 说 · Shuo

[中文](./README.cn.md)

On-device speech recognition GUI powered by Qwen3-ASR-0.6B ONNX CPU pipeline. No GPU, no PyTorch, no cloud.

## Project Structure

```
├── docs/                       # Documentation
├── locales/                    # Translation files (28 languages)
├── Qwen3-ASR-0.6B-ONNX-CPU/    # ASR model (download first)
├── asr_gui.py                  # Speech recognition GUI
├── theme.py                    # Theme manager (light/dark + background image)
├── config.py                   # Config / history / logging
├── global_hotkey.py            # Global hotkey listener (pynput)
├── benchmark.py                # Performance benchmark
├── extract_i18n.py             # Extract translation keys
├── i18n.py                     # Internationalization module
├── mel_filters.npy             # Precomputed Mel filterbank
└── onnx_inference.py           # ASR inference pipeline
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model
hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU --local-dir Qwen3-ASR-0.6B-ONNX-CPU

# Launch GUI
python asr_gui.py
```

## Dependencies

| Library | Purpose |
|---|---|
| PySide6 | GUI framework |
| qtawesome | Font Awesome icons |
| pynput | Global hotkey listener |
| sounddevice | Audio recording |
| onnxruntime | ONNX model inference |
| numpy | Numerical computing |
| librosa | Audio processing / Mel spectrogram |
| tokenizers | Text tokenization |
| transformers | Tokenizer loading |

## Configuration

App creates `~/.shuo/` on first launch:

| File | Description |
|---|---|
| `config.json` | Language, hotkey, auto-type, background image, opacity settings |
| `history.json` | Recognition history (max 500 entries) |
| `shuo.log` | Runtime log (UTF-8) |

- Default hotkey: F2, customizable in Settings
- Background image with adjustable opacity and fit modes (Cover / Contain / Tile / Center)
- Window opacity slider (10%–100%)

## Packaging

```bash
pip install pyinstaller

# Windows (semicolons)
pyinstaller --windowed --name Shuo --icon=shuo.ico -y --add-data "locales;locales" --add-data "Qwen3-ASR-0.6B-ONNX-CPU;Qwen3-ASR-0.6B-ONNX-CPU" --add-data "mel_filters.npy;." --add-data "theme.py;." --add-data "config.py;." --hidden-import onnxruntime --hidden-import tokenizers --hidden-import transformers --hidden-import sounddevice --hidden-import numpy --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py

# macOS / Linux (colons)
pyinstaller --windowed --name Shuo --icon=shuo.ico -y --add-data "locales:locales" --add-data "Qwen3-ASR-0.6B-ONNX-CPU:Qwen3-ASR-0.6B-ONNX-CPU" --add-data "mel_filters.npy:." --add-data "theme.py:." --add-data "config.py:." --hidden-import onnxruntime --hidden-import tokenizers --hidden-import transformers --hidden-import sounddevice --hidden-import numpy --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py
```

Output: `dist/Shuo/` (~3 GB, includes ONNX model)

## i18n

GUI supports 28 languages. Default is English.

```bash
python extract_i18n.py
```

## License

Code: GPL-3.0

### Qt / PySide6

This app uses PySide6 (Qt for Python) under **LGPL v3**.
Dynamic linking in proprietary software is permitted without open-sourcing your code.
Full license: https://www.gnu.org/licenses/lgpl-3.0.html

### Third-party Libraries

| Library | License |
|---|---|
| qtawesome | MIT |
| pynput | LGPL-3.0 |
| sounddevice | MIT |
| onnxruntime | MIT |
| numpy | BSD-3 |
| librosa | ISC |
| tokenizers | Apache-2.0 |

Model weights: copyright belongs to original authors. See [NOTICE](./NOTICE).
