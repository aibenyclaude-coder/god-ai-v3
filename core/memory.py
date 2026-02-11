#!/usr/bin/env python3
"""God AI v3.0 - メモリ管理モジュール"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from config import (
    STATE_PATH, JOURNAL_PATH, IDENTITY_PATH,
    CONVERSATIONS_PATH, CONVERSATIONS_ARCHIVE_PATH, log
)

# --- asyncio.Lock（並行書き込み保護）---
_write_lock: asyncio.Lock | None = None

def get_write_lock() -> asyncio.Lock:
    global _write_lock
    if _write_lock is None:
        try:
            _write_lock = asyncio.Lock()
        except RuntimeError:
            _write_lock = asyncio.Lock()
    return _write_lock

def init_write_lock():
    """メインループで明示的にLockを初期化"""
    global _write_lock
    _write_lock = asyncio.Lock()

# --- ファイル読み込み ---
def read_file(path: Path, tail: int = 0) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if tail > 0:
        lines = text.splitlines()
        return "\n".join(lines[-tail:])
    return text

# --- State管理 ---
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return {"status": "unknown", "current_task": None, "last_reflection": None,
                "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}

def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

# --- Journal管理 ---
def append_journal(text: str):
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{text}\n")

async def safe_append_journal(text: str):
    async with get_write_lock():
        append_journal(text)

def load_journal(tail: int = 0) -> str:
    return read_file(JOURNAL_PATH, tail=tail)

# --- Conversations管理 ---
def load_conversations() -> list:
    if CONVERSATIONS_PATH.exists():
        try:
            return json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return []

def load_conversations_archive() -> list:
    if CONVERSATIONS_ARCHIVE_PATH.exists():
        try:
            return json.loads(CONVERSATIONS_ARCHIVE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return []

def save_conversations_archive(archive: list):
    CONVERSATIONS_ARCHIVE_PATH.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")

def save_conversations(convos: list):
    # 重要な会話を判定してアーカイブに保存
    archive = load_conversations_archive()
    important_keywords = ["エラー", "失敗", "重要", "バグ", "修正", "致命的", "警告", "問題"]

    for conv in convos:
        text_lower = conv.get("text", "").lower()
        if any(kw in text_lower for kw in important_keywords):
            # 既にアーカイブにあるか確認（重複防止）
            if not any(a.get("time") == conv.get("time") and a.get("text") == conv.get("text") for a in archive):
                archive.append({
                    "time": conv.get("time"),
                    "from": conv.get("from"),
                    "text": conv.get("text"),
                    "importance": "high",
                    "archived_at": datetime.now(timezone.utc).isoformat()
                })

    # アーカイブは最新500件まで保持
    if len(archive) > 500:
        # 効率化のため、スライシングで保持する件数を減らす
        archive = archive[-500:]
    try:
        save_conversations_archive(archive)
    except Exception as e:
        log.error(f"Failed to save conversations archive: {e}")
        # アーカイブ保存失敗時も、最新の会話は保存を試みる

    # 最新50件のみ保持
    convos = convos[-50:]
    try:
        CONVERSATIONS_PATH.write_text(json.dumps(convos, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to save current conversations: {e}")

# --- Identity読み込み ---
def load_identity() -> str:
    return read_file(IDENTITY_PATH)
