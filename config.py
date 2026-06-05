# ── 用户目录配置 ──────────────────────────────────────────────────────

import json
import logging
from pathlib import Path
from datetime import datetime


USER_DIR = Path.home() / ".shuo"
USER_DIR.mkdir(exist_ok=True)
CONFIG_PATH = USER_DIR / "config.json"
HISTORY_PATH = USER_DIR / "history.json"
LOG_PATH = USER_DIR / "shuo.log"

# 日志配置（UTF-8）
file_handler = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("shuo")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

DEFAULT_CONFIG = {
    "hotkey": "f2",
    "language": "en",
    "asr_lang": "auto",
    "auto_type": True,
    "remove_punc": False,
    "save_history": False,
    "bg_image": "",
    "bg_fit": "cover",
    "opacity": 100,
}


class Config:
    """配置管理"""

    @staticmethod
    def load():
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    # 合并默认值
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in cfg:
                            cfg[k] = v
                    return cfg
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
        return DEFAULT_CONFIG.copy()

    @staticmethod
    def save(cfg):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            logger.info("配置已保存")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")


class History:
    """历史记录管理"""

    @staticmethod
    def load():
        if HISTORY_PATH.exists():
            try:
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载历史失败: {e}")
        return []

    @staticmethod
    def save(items):
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存历史失败: {e}")

    @staticmethod
    def add(text):
        items = History.load()
        items.append({
            "text": text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # 最多保留 500 条
        if len(items) > 500:
            items = items[-500:]
        History.save(items)
        return items
