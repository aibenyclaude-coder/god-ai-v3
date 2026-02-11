#!/usr/bin/env python3
"""God AI v3.0 - AI思考モジュール"""
from __future__ import annotations

import asyncio
import subprocess

import httpx

from config import GOOGLE_AI_KEY, log

# --- 脳の使い分けカウンタ ---
gemini_count = 0
claude_count = 0

# --- Gemini API ---
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_AI_KEY}"

async def think_gemini(prompt: str) -> tuple[str, str]:
    """Geminiで思考。戻り値: (テキスト, 脳の名前)"""
    global gemini_count
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                GEMINI_URL,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 2048},
                },
                timeout=60,
            )
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            gemini_count += 1
            return (text, "Gemini 2.5 Flash")
    except Exception as e:
        log.error(f"Gemini failed: {e}, falling back to Claude CLI")
        text, _ = await think_claude(prompt)
        return (text, "Claude CLI (fallback)")

# --- Claude CLI ---
async def think_claude(prompt: str) -> tuple[str, str]:
    """Claude CLIで思考（会話用、タイムアウト120秒、リトライ強化）。戻り値: (テキスト, 脳の名前)"""
    global claude_count
    loop = asyncio.get_running_loop()
    for attempt in range(3):
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["claude", "--print", "-p", prompt],
                    capture_output=True, text=True, timeout=280,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                claude_count += 1
                return (result.stdout.strip(), "Claude CLI")
            log.error(f"Claude CLI attempt {attempt+1}: returncode={result.returncode}, stderr={result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log.error(f"Claude CLI attempt {attempt+1}: timeout (280s)")
        except Exception as e:
            log.error(f"Claude CLI attempt {attempt+1}: {e}")
        if attempt < 2:
            await asyncio.sleep(3)
    raise RuntimeError("Claude CLI failed after 3 attempts (timeout=280s)")

async def think_claude_heavy(prompt: str) -> tuple[str, str]:
    """Claude CLIで重い処理（自己改善用、タイムアウト280秒、リトライ強化）。戻り値: (テキスト, 脳の名前)"""
    global claude_count
    loop = asyncio.get_running_loop()
    for attempt in range(3):
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["claude", "--print", "-p", prompt],
                    capture_output=True, text=True, timeout=280,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                claude_count += 1
                return (result.stdout.strip(), "Claude CLI")
            log.error(f"Claude CLI heavy attempt {attempt+1}: returncode={result.returncode}, stderr={result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log.error(f"Claude CLI heavy attempt {attempt+1}: timeout (280s)")
        except Exception as e:
            log.error(f"Claude CLI heavy attempt {attempt+1}: {e}")
        if attempt < 2:
            await asyncio.sleep(5)
    raise RuntimeError("Claude CLI heavy failed after 3 attempts (timeout=280s)")

# --- 統合思考関数 ---
async def think(prompt: str, heavy: bool = False) -> tuple[str, str]:
    """統合思考関数。戻り値: (テキスト, 脳の名前)"""
    if heavy:
        return await think_claude(prompt)
    return await think_gemini(prompt)

def is_heavy(message: str) -> bool:
    from config import HEAVY_KEYWORDS
    return any(kw in message for kw in HEAVY_KEYWORDS)

def get_brain_counts() -> tuple[int, int]:
    """(gemini_count, claude_count) を返す"""
    return gemini_count, claude_count
