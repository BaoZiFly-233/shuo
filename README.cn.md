# shuo

[English](./README.md)

## 项目文件

```
├── docs/                       # 文档
├── locales/                    # 翻译文件
├── Qwen3-ASR-0.6B-ONNX-CPU/    # ASR 模型（需先下载）
├── asr_gui.py                  # 语音识别 GUI 主程序
├── theme.py                    # 主题管理（亮/暗色 + 背景图）
├── config.py                   # 配置 / 历史 / 日志
├── global_hotkey.py            # 全局热键监听 (pynput)
├── benchmark.py                # 性能测试
├── extract_i18n.py             # 提取翻译键
├── i18n.py                     # 国际化模块
├── mel_filters.npy             # 预计算 Mel 滤波器
└── onnx_inference.py           # ASR 推理管线
```

## 下载模型

```bash
# 国内用户加镜像
HF_ENDPOINT=https://hf-mirror.com hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU --local-dir Qwen3-ASR-0.6B-ONNX-CPU

# 或 Powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU --local-dir Qwen3-ASR-0.6B-ONNX-CPU
```

## 安装依赖

```bash
pip install -r requirements.txt
```

| 库 | 用途 |
|---|---|
| PySide6 | GUI 框架 |
| qtawesome | Font Awesome 图标 |
| pynput | 全局热键监听 |
| sounddevice | 音频录制 |
| onnxruntime | ONNX 模型推理 |
| numpy | 数值计算 |
| librosa | 音频处理 / Mel 频谱 |
| tokenizers | 文本分词 |
| transformers | 分词器加载 |

## 启动 GUI

```bash
python asr_gui.py
```

## 配置

应用启动后自动创建 `~/.shuo/` 目录：

| 文件 | 说明 |
|---|---|
| `config.json` | 语言、快捷键、自动输入等配置 |
| `history.json` | 识别历史（最多 500 条） |
| `shuo.log` | 运行日志（UTF-8） |

- 默认快捷键：F2，可在设置中自定义
- 支持设置窗口透明度（10%-100%）和背景图片

## 打包

```bash
pip install pyinstaller

# Windows（CMD，分号分隔）
pyinstaller --windowed --name Shuo --icon=shuo.ico -y --add-data "locales;locales" --add-data "Qwen3-ASR-0.6B-ONNX-CPU;Qwen3-ASR-0.6B-ONNX-CPU" --add-data "mel_filters.npy;." --add-data "theme.py;." --add-data "config.py;." --hidden-import onnxruntime --hidden-import tokenizers --hidden-import transformers --hidden-import sounddevice --hidden-import numpy --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py

# macOS / Linux（冒号分隔）
pyinstaller --windowed --name Shuo --icon=shuo.ico -y --add-data "locales:locales" --add-data "Qwen3-ASR-0.6B-ONNX-CPU:Qwen3-ASR-0.6B-ONNX-CPU" --add-data "mel_filters.npy:." --add-data "theme.py:." --add-data "config.py:." --hidden-import onnxruntime --hidden-import tokenizers --hidden-import transformers --hidden-import sounddevice --hidden-import numpy --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py
```

输出：`dist/Shuo/`（约 3 GB，含 ONNX 模型）

## 国际化

GUI 支持中/英文切换，默认中文。新增翻译键：

```bash
python extract_i18n.py
```

## License

代码：GPL-3.0

### Qt / PySide6

本应用使用 PySide6（Qt for Python），依据 **LGPL v3** 协议发布。
可在专有软件中动态链接，无需开源。完整协议：https://www.gnu.org/licenses/lgpl-3.0.html

### 第三方库

| 库 | 协议 |
|---|---|
| qtawesome | MIT |
| pynput | LGPL-3.0 |
| sounddevice | MIT |
| onnxruntime | MIT |
| numpy | BSD-3 |
| librosa | ISC |
| tokenizers | Apache-2.0 |

模型权重版权归原作者所有，详见 [NOTICE](./NOTICE)。
