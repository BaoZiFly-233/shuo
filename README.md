# 说 · Shuo

[中文](./README.cn.md)

On-device speech recognition GUI powered by Qwen3-ASR-0.6B ONNX CPU pipeline. No GPU, no PyTorch, no cloud.

## Project Structure

```
├── docs/                       # Documentation
├── locales/                    # Translation files
├── Qwen3-ASR-0.6B-ONNX-CPU/   # ASR model (download first)
├── asr_gui.py                  # Speech recognition GUI
├── benchmark.py                # Performance benchmark
├── extract_i18n.py             # Extract translation keys from source
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
| PyAudio | Audio recording |
| onnxruntime | ONNX model inference |
| numpy | Numerical computing |
| librosa | Audio processing / Mel spectrogram |
| tokenizers | Text tokenization |

## Configuration

App creates `~/.shuo/` on first launch:

| File | Description |
|---|---|
| `config.json` | Language, hotkey, auto-type settings |
| `history.json` | Recognition history (max 500) |
| `shuo.log` | Runtime log (UTF-8) |

- Default hotkey: mouse side button (back), customizable via top-left button
- Language switch: top-right dropdown (中文 / English)

## Packaging

```bash
pip install pyinstaller

# Windows（CMD，分号分隔）
pyinstaller --windowed --name Shuo --icon=shuo.ico -y --add-data "locales;locales" --add-data "Qwen3-ASR-0.6B-ONNX-CPU;Qwen3-ASR-0.6B-ONNX-CPU" --add-data "mel_filters.npy;." --hidden-import onnxruntime --hidden-import tokenizers --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py

# macOS / Linux（冒号分隔）
pyinstaller --windowed --name Shuo --icon=shuo.ico -y --add-data "locales:locales" --add-data "Qwen3-ASR-0.6B-ONNX-CPU:Qwen3-ASR-0.6B-ONNX-CPU" --add-data "mel_filters.npy:." --hidden-import onnxruntime --hidden-import tokenizers --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py
```

输出：`dist/Shuo/`（约 3 GB，含 ONNX 模型 + scipy）

## i18n

GUI supports Chinese and English. Switch from the top-right dropdown.

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
| PyAudio | MIT |
| onnxruntime | MIT |
| numpy | BSD-3 |
| librosa | ISC |
| tokenizers | Apache-2.0 |

Model weights: copyright belongs to original authors. See [NOTICE](./NOTICE).
