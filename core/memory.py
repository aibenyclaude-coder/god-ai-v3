#!/usr/bin/env python3
"""God AI v3.0 - メモリ管理モジュール"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from config import (
    STATE_PATH, JOURNAL_PATH, IDENTITY_PATH,
    CONVERSATIONS_PATH, CONVERSATIONS_ARCHIVE_PATH, MEMORY_DIR, log
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
    """
    Loads the AI's state from the state file.
    If the file is not found or corrupted, returns a default state.
    Implements a retry mechanism for file operations.
    """
    state_path = STATE_PATH
    memory_dir = MEMORY_DIR
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            if not state_path.exists():
                log.warning(f"State file not found at {state_path}. Attempting to restore from backup.")
                if memory_dir is not None:
                    backup_path = memory_dir / "state.json.bak"
                    if backup_path.exists():
                        log.info(f"Restoring state from backup: {backup_path}")
                        try:
                            return json.loads(backup_path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            log.error(f"Failed to restore state from backup {backup_path}: {e}")
                
                log.info("Returning default state.")
                return {"status": "unknown", "current_task": None, "last_reflection": None,
                        "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}
            
            state_content = state_path.read_text(encoding="utf-8")
            return json.loads(state_content)
        
        except json.JSONDecodeError:
            log.error(f"State file at {state_path} is corrupted on attempt {attempt + 1}/{max_retries}. Attempting recovery.")
            if attempt < max_retries - 1:
                log.info(f"Retrying in {retry_delay} seconds...")
                asyncio.sleep(retry_delay)  # Use asyncio.sleep if in an async context, otherwise time.sleep
                retry_delay *= 2  # Exponential backoff
                continue

            # Recovery attempt after all retries fail
            try:
                corrupted_state_content = state_path.read_text(encoding="utf-8")
                if memory_dir is not None:
                    timestamp = datetime.now(timezone.utc).isoformat().replace(':', '-').replace('+', '_')
                    backup_filename = f"state.json.corrupted.{timestamp}.bak"
                    backup_path = memory_dir / backup_filename
                    backup_path.write_text(corrupted_state_content, encoding="utf-8")
                    log.info(f"Corrupted state file backed up to {backup_path}")
                else:
                    log.warning("MEMORY_DIR not defined. Cannot back up corrupted state file.")
            except Exception as e:
                log.error(f"Failed to create backup of corrupted state file: {e}")

            log.info("Returning default state after corruption.")
            return {"status": "unknown", "current_task": None, "last_reflection": None,
                    "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}
        
        except FileNotFoundError:
            log.error(f"State file not found at {state_path} on attempt {attempt + 1}/{max_retries}.")
            if attempt < max_retries - 1:
                log.info(f"Retrying in {retry_delay} seconds...")
                asyncio.sleep(retry_delay) # Use asyncio.sleep if in an async context, otherwise time.sleep
                retry_delay *= 2
                continue
            else:
                log.info("Returning default state after file not found.")
                return {"status": "unknown", "current_task": None, "last_reflection": None,
                        "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}

        except Exception as e:
            log.error(f"An unexpected error occurred while loading state from {state_path} on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                log.info(f"Retrying in {retry_delay} seconds...")
                asyncio.sleep(retry_delay) # Use asyncio.sleep if in an async context, otherwise time.sleep
                retry_delay *= 2
                continue
            else:
                log.error("Returning default state due to unexpected error.")
                return {"status": "unknown", "current_task": None, "last_reflection": None,
                        "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}
    
    # Fallback in case loop finishes without returning (should not happen with `continue` and `return`)
    log.error("Fallback: load_state exited loop without returning a value. Returning default state.")
    return {"status": "unknown", "current_task": None, "last_reflection": None,
            "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}

def save_state(state: dict):
    """
    Saves the AI's state to the state file.
    Validates for full-width characters that could break JSON/Python syntax.
    """
    try:
        # Convert state to JSON string first, ensuring non-ASCII characters are preserved
        state_str = json.dumps(state, ensure_ascii=False, indent=2)
        
        # Define problematic characters and their replacements.
        # This dictionary can be expanded to cover a wider range of characters if issues arise.
        # For now, focusing on commonly problematic full-width characters.
        replacements = {
            '【': '[',  # Replace opening full-width bracket with ASCII equivalent
            '】': ']',  # Replace closing full-width bracket with ASCII equivalent
            '！': '!',  # Replace full-width exclamation mark
            '？': '?',  # Replace full-width question mark
            '：': ':',  # Replace full-width colon
            '；': ';',  # Replace full-width semicolon
            '，': ',',  # Replace full-width comma
            '．': '.',  # Replace full-width period
            '（': '(',  # Replace full-width parenthesis open
            '）': ')',  # Replace full-width parenthesis close
            '｛': '{',  # Replace full-width brace open
            '｝': '}',  # Replace full-width brace close
            '「': '"',  # Replace full-width quote open
            '」': '"',  # Replace full-width quote close
            '‘': "'",  # Replace full-width apostrophe open
            '’': "'",  # Replace full-width apostrophe close
            '…': '...', # Replace full-width ellipsis
        }
        
        # Apply replacements to the JSON string
        for char, replacement in replacements.items():
            state_str = state_str.replace(char, replacement)

        # Write the cleaned string to the state file
        STATE_PATH.write_text(state_str, encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to save state to {STATE_PATH}: {e}")

async def safe_save_state(state: dict):
    """
    Safely saves the AI's state with retry logic, atomic writes,
    and backup creation to prevent data corruption or loss.
    Uses a lock to prevent concurrent writes, writes to a temp file
    first then atomically renames to prevent partial writes, and
    retries with exponential backoff on disk I/O errors.
    """
    max_retries = 3
    retry_delay = 0.5

    async with get_write_lock():
        for attempt in range(max_retries):
            try:
                # Create a .bak backup of the current state file before overwriting
                if STATE_PATH.exists():
                    backup_path = STATE_PATH.parent / (STATE_PATH.name + ".bak")
                    try:
                        backup_path.write_text(
                            STATE_PATH.read_text(encoding="utf-8"),
                            encoding="utf-8"
                        )
                    except Exception as bak_err:
                        log.warning(f"Failed to create .bak backup before save: {bak_err}")

                # Delegate JSON serialization and sanitization to save_state-compatible logic
                state_str = json.dumps(state, ensure_ascii=False, indent=2)

                replacements = {
                    '\u3010': '[', '\u3011': ']',
                    '\uff01': '!', '\uff1f': '?',
                    '\uff1a': ':', '\uff1b': ';',
                    '\uff0c': ',', '\uff0e': '.',
                    '\uff08': '(', '\uff09': ')',
                    '\uff5b': '{', '\uff5d': '}',
                    '\u300c': '"', '\u300d': '"',
                    '\u2018': "'", '\u2019': "'",
                    '\u2026': '...',
                }
                for char, replacement in replacements.items():
                    state_str = state_str.replace(char, replacement)

                # Write to a temporary file in the same directory, then atomically rename
                dir_path = STATE_PATH.parent
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(dir_path), suffix=".tmp", prefix="state_"
                )
                try:
                    os.write(fd, state_str.encode("utf-8"))
                    os.fsync(fd)
                    os.close(fd)
                    fd = -1
                    os.replace(tmp_path, str(STATE_PATH))
                except BaseException:
                    if fd >= 0:
                        os.close(fd)
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise

                # Success - exit retry loop
                return

            except OSError as e:
                log.error(
                    f"Disk I/O error saving state (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log.error(
                        "All retries exhausted for safe_save_state. State may not be saved."
                    )

            except Exception as e:
                log.error(f"Failed to save state to {STATE_PATH}: {e}")
                return

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

# --- Journal管理 ---
def append_journal(text: str):
    """Journalファイルにテキストを追記する"""
    try:
        with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{text}\n")
    except FileNotFoundError:
        log.error(f"Journal file not found at {JOURNAL_PATH}. Creating it.")
        try:
            JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
                f.write(f"{text}\n")
        except Exception as e:
            log.error(f"Failed to create journal file and write: {e}")
    except Exception as e:
        log.error(f"Failed to append to journal file {JOURNAL_PATH}: {e}")

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
