#!/usr/bin/env python3
"""God AI v3.0 - 共通設定・定数モジュール"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- パス定義 ---
BASE_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = BASE_DIR / "core"
MEMORY_DIR = BASE_DIR / "memory"
ENV_PATH = CORE_DIR / ".env"
IDENTITY_PATH = MEMORY_DIR / "identity.md"
STATE_PATH = MEMORY_DIR / "state.json"
JOURNAL_PATH = MEMORY_DIR / "journal.md"
BENY_PATH = MEMORY_DIR / "beny.md"
CONVERSATIONS_PATH = MEMORY_DIR / "conversations.json"
CONVERSATIONS_ARCHIVE_PATH = MEMORY_DIR / "conversations_archive.json"
GOD_PY_PATH = CORE_DIR / "god.py"
PID_FILE = CORE_DIR / "god.pid"

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("god")

# --- .env読み込み ---
def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        log.error(f".env not found: {path}")
        sys.exit(1)
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

ENV = load_env(ENV_PATH)
TELEGRAM_TOKEN = ENV.get("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_KEY = ENV.get("ANTHROPIC_API_KEY", "")
BENY_CHAT_ID = ENV.get("BENY_CHAT_ID", "")
GOOGLE_AI_KEY = ENV.get("GOOGLE_AI_API_KEY", "")

for name, val in [("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN), ("ANTHROPIC_API_KEY", ANTHROPIC_KEY),
                   ("BENY_CHAT_ID", BENY_CHAT_ID), ("GOOGLE_AI_API_KEY", GOOGLE_AI_KEY)]:
    if not val:
        log.error(f"Missing env: {name}")
        sys.exit(1)

# --- Telegram API Base URL ---
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# --- 定数 ---
REFLECTION_INTERVAL = 1800  # 秒（30分）
SELF_GROWTH_INTERVAL = 600  # 秒（10分）
HEAVY_KEYWORDS = ["コード", "作って", "LP", "HTML", "修正", "プログラム", "書いて", "実装", "スクリプト"]
