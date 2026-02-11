#!/usr/bin/env python3
"""God AI v3.0 - AI思考モジュール"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import shutil

import httpx

from config import GOOGLE_AI_KEY, STATE_PATH, log
from memory import load_state, safe_save_state

# Claude CLIのパス（nohup環境でもフルパスで呼べるように）
CLAUDE_PATH = shutil.which("claude") or "/opt/homebrew/bin/claude"

# nohup環境用のPATH設定（node, npmなどが見つかるように）
def _get_cli_env() -> dict:
    """Claude CLI実行用の環境変数を取得"""
    env = os.environ.copy()
    # Homebrew paths for macOS (Intel and Apple Silicon)
    homebrew_paths = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/homebrew/sbin", "/usr/local/sbin"]
    current_path = env.get("PATH", "")
    for p in homebrew_paths:
        if p not in current_path:
            current_path = f"{p}:{current_path}"
    env["PATH"] = current_path
    return env

CLI_ENV = _get_cli_env()

# --- 脳の使い分けカウンタ（state.jsonに永続化） ---
def _load_brain_counts() -> tuple[int, int]:
    """state.jsonからカウンターを読み込む"""
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return state.get("gemini_count", 0), state.get("claude_count", 0)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0, 0

async def _save_brain_counts(gemini: int, claude: int):
    """state.jsonにカウンターを保存（排他制御あり）"""
    state = load_state()
    state["gemini_count"] = gemini
    state["claude_count"] = claude
    await safe_save_state(state)

# メモリ上のカウンタ（起動時にstate.jsonから読み込み）
_gemini_count, _claude_count = _load_brain_counts()

# --- Gemini API ---
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_AI_KEY}"

async def think_gemini(prompt: str, max_tokens: int = 4096) -> tuple[str, str]:
    """Geminiで思考（429リトライ付き）。戻り値: (テキスト, 脳の名前)

    Args:
        prompt: プロンプト
        max_tokens: 最大出力トークン数（デフォルト4096、コード生成時は8192推奨）
    """
    global _gemini_count
    max_retries = 3
    retry_delay = 8

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    GEMINI_URL,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": max_tokens},
                    },
                    timeout=60,
                )
                data = resp.json()

                # 429 Rate Limit チェック
                if resp.status_code == 429:
                    log.warning(f"Gemini 429 rate limit, attempt {attempt+1}/{max_retries}, waiting {retry_delay}s...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Gemini 429 rate limit exceeded after retries")

                if "error" in data:
                    raise Exception(f"Gemini API error: {data['error']}")

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                _gemini_count += 1
                await _save_brain_counts(_gemini_count, _claude_count)
                return (text, f"Gemini {GEMINI_MODEL}")
        except Exception as e:
            if attempt < max_retries - 1 and "429" in str(e):
                log.warning(f"Gemini error (attempt {attempt+1}): {e}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                continue
            log.error(f"Gemini failed: {e}, falling back to Claude CLI")
            text, _ = await think_claude(prompt)
            return (text, "Claude CLI (fallback)")

# --- Claude CLI ---
async def think_claude(prompt: str, timeout: int = 120) -> tuple[str, str]:
    """Claude CLIで思考（会話用、段階的タイムアウト対応）。戻り値: (テキスト, 脳の名前)

    Args:
        prompt: プロンプト
        timeout: タイムアウト秒数（デフォルト120秒、コード生成時は280秒推奨）

    段階的リトライ戦略:
    - 1回目: 同じプロンプトでリトライ
    - 2回目: プロンプトを短縮してリトライ（最後の1000文字のみ）
    - 3回目: Geminiにフォールバック
    """
    global _claude_count
    loop = asyncio.get_running_loop()
    original_prompt = prompt

    for attempt in range(3):
        # 2回目以降はプロンプトを短縮
        if attempt == 1:
            # プロンプトを最後の1000文字に短縮
            if len(prompt) > 1000:
                prompt = "...(省略)...\n" + original_prompt[-1000:]
                log.warning(f"Claude CLI attempt {attempt+1}: プロンプトを短縮 ({len(original_prompt)} -> {len(prompt)}文字)")

        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt, t=timeout: subprocess.run(
                    [CLAUDE_PATH, "--print", "-p", p],
                    capture_output=True, text=True, timeout=t,
                    env=CLI_ENV,
                ),
            )
            # stderrがあればログ記録（原因特定用）
            if result.stderr and result.stderr.strip():
                log.error(f"Claude CLI stderr: {result.stderr[:500]}")

            if result.returncode == 0 and result.stdout.strip():
                _claude_count += 1
                await _save_brain_counts(_gemini_count, _claude_count)
                return (result.stdout.strip(), "Claude CLI")
            # 詳細なエラーログ（stdout/stderr両方、プロンプト長も）
            stdout_full = result.stdout if result.stdout else "(empty)"
            stderr_full = result.stderr if result.stderr else "(empty)"
            log.error(f"Claude CLI attempt {attempt+1}: returncode={result.returncode}, prompt_len={len(prompt)}, timeout={timeout}s")
            log.error(f"  stdout: {stdout_full[:500]}")
            log.error(f"  stderr: {stderr_full[:500]}")
        except subprocess.TimeoutExpired:
            log.error(f"Claude CLI attempt {attempt+1}: timeout ({timeout}s)")
            # 3回目のタイムアウトはGeminiにフォールバック
            if attempt == 2:
                log.warning("Claude CLI 3回タイムアウト、Geminiにフォールバック")
                try:
                    text, _ = await think_gemini(original_prompt[-2000:] if len(original_prompt) > 2000 else original_prompt)
                    return (text, "Gemini (Claude fallback)")
                except Exception as e:
                    log.error(f"Gemini fallback failed: {e}")
                    raise RuntimeError(f"Claude CLI and Gemini both failed: {e}")
        except Exception as e:
            log.error(f"Claude CLI attempt {attempt+1}: {e}")
        if attempt < 2:
            await asyncio.sleep(3)
    raise RuntimeError(f"Claude CLI failed after 3 attempts (timeout={timeout}s)")

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
    """(gemini_count, claude_count) を返す（state.jsonから永続化された値）"""
    return _gemini_count, _claude_count


# --- アクション意図検出 ---
def detect_action_intent(message: str) -> dict:
    """アクション意図を検出。戻り値: {"needs_action": bool, "action_type": str, "target": str}

    「〇〇しろ」「作れ」「書け」などの命令形 -> needs_action=True
    「〇〇って何？」「教えて」「説明して」などの質問 -> needs_action=False
    """
    result = {"needs_action": False, "action_type": "", "target": ""}

    # アクションキーワード（命令形）
    action_patterns = [
        ("追記しろ", "append"), ("追記して", "append"), ("追加しろ", "append"), ("追加して", "append"),
        ("更新しろ", "update"), ("更新して", "update"), ("変更しろ", "update"), ("変更して", "update"),
        ("削除しろ", "delete"), ("削除して", "delete"), ("消して", "delete"), ("消しろ", "delete"),
        ("作れ", "create"), ("作って", "create"), ("作成しろ", "create"), ("作成して", "create"),
        ("書け", "write"), ("書いて", "write"),
        ("実行しろ", "execute"), ("実行して", "execute"), ("走らせろ", "execute"), ("走らせて", "execute"),
        ("送れ", "send"), ("送って", "send"), ("投稿しろ", "send"), ("投稿して", "send"),
        ("バックアップしろ", "backup"), ("バックアップして", "backup"),
    ]

    # 質問キーワード（アクション不要）
    question_patterns = [
        "って何", "とは何", "とは？", "って？", "ってなに",
        "教えて", "説明して", "わかる？", "わかりますか",
        "どう思う", "どうですか", "どうなってる", "どうなっていますか",
        "ある？", "ありますか", "できる？", "できますか",
        "なぜ", "何故", "どうして", "理由は",
    ]

    msg_lower = message.lower()

    # 質問パターンに該当する場合はアクション不要
    for qp in question_patterns:
        if qp in message:
            return result

    # アクションパターンをチェック
    for pattern, action_type in action_patterns:
        if pattern in message:
            result["needs_action"] = True
            result["action_type"] = action_type
            # ターゲットを抽出（パターンの前の部分）
            idx = message.find(pattern)
            if idx > 0:
                target_part = message[:idx].strip()
                # 「〇〇に」「〇〇を」などの助詞を除去
                for suffix in ["に", "を", "へ", "の", "は"]:
                    if target_part.endswith(suffix):
                        target_part = target_part[:-1]
                result["target"] = target_part[-50:]  # 最大50文字
            return result

    return result
