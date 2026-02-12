#!/usr/bin/env python3
"""God AI v3.0 - AI思考モジュール"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import shutil

import httpx

from config import GOOGLE_AI_KEY, GLM_API_KEY, STATE_PATH, log
from memory import load_state, safe_save_state

# --- 特殊例外クラス ---
class ClaudeSessionExpired(Exception):
    """Claude CLIのセッション切れ（再ログイン必要）"""
    pass

class AIUnavailable(Exception):
    """Gemini + Claude両方使用不可"""
    pass

# --- AI一時停止状態 ---
_ai_paused_until: float = 0  # Unix timestamp（0 = 停止中でない）
AI_PAUSE_DURATION = 300  # 5分間

def is_ai_paused() -> bool:
    """AI呼び出しが一時停止中か"""
    import time
    return time.time() < _ai_paused_until

def pause_ai():
    """AI呼び出しを5分間停止"""
    global _ai_paused_until
    import time
    _ai_paused_until = time.time() + AI_PAUSE_DURATION
    log.warning(f"AI呼び出しを{AI_PAUSE_DURATION}秒間停止")

def get_ai_pause_remaining() -> int:
    """停止残り秒数（0 = 停止中でない）"""
    import time
    remaining = int(_ai_paused_until - time.time())
    return max(0, remaining)

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

    # Get HOME from environment, default if not set
    home_dir = os.environ.get("HOME")
    if home_dir:
        env["HOME"] = home_dir
    else:
        # Fallback to a safe default if HOME is not set
        env["HOME"] = "/Users/default_user"
        log.warning("HOME environment variable not set, using default: /Users/default_user")

    # Add AI status information if relevant
    if is_ai_paused():
        remaining = get_ai_pause_remaining()
        env["AI_STATUS"] = f"PAUSED_UNTIL_{remaining}"
    else:
        env["AI_STATUS"] = "RUNNING"

    return env

CLI_ENV = _get_cli_env()

# --- 脳の使い分けカウンタ（state.jsonに永続化） ---
def _load_brain_counts() -> tuple[int, int, int]:
    """Load counters from state.json. If the file does not exist or is invalid, return default values."""
    try:
        # Attempt to read and parse the state file.
        state_text = STATE_PATH.read_text(encoding="utf-8")
        state = json.loads(state_text)
        # Return counts, defaulting to 0 if keys are missing.
        return state.get("gemini_count", 0), state.get("glm_count", 0), state.get("claude_count", 0)
    except FileNotFoundError:
        # If the state file doesn't exist, log this as a normal startup condition and return defaults.
        log.info(f"State file not found at {STATE_PATH}. Initializing counts to zero.")
        return 0, 0, 0
    except json.JSONDecodeError:
        # If the file content is not valid JSON, log an error and return defaults.
        log.error(f"Error decoding JSON from {STATE_PATH}. File content might be corrupted. Initializing counts to zero.")
        return 0, 0, 0
    except Exception as e:
        # Catch any other unexpected exceptions during file loading.
        log.error(f"An unexpected error occurred while loading brain counts from {STATE_PATH}: {e}. Initializing counts to zero.")
        return 0, 0, 0

async def _save_brain_counts(gemini: int, glm: int, claude: int):
    """state.jsonにカウンターを保存（排他制御あり）"""
    state = load_state()
    state["gemini_count"] = gemini
    state["glm_count"] = glm
    state["claude_count"] = claude
    await safe_save_state(state)

# メモリ上のカウンタ（起動時にstate.jsonから読み込み）
_gemini_count, _glm_count, _claude_count = _load_brain_counts()

# --- Gemini API ---
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_AI_KEY}"

async def think_gemini(prompt: str, max_tokens: int = 4096) -> tuple[str, str]:
    """Geminiで思考（429/503リトライ付き）。戻り値: (テキスト, 脳の名前)

    フォールバック順序: Gemini → GLM-4 → Claude CLI

    Args:
        prompt: プロンプト
        max_tokens: 最大出力トークン数（デフォルト4096、コード生成時は8192推奨）
    """
    global _gemini_count
    max_retries = 3
    initial_retry_delay = 8  # Initial delay in seconds
    max_retry_delay = 60  # Maximum delay in seconds
    retry_delay = initial_retry_delay

    # Context window management: Limit prompt length to avoid exceeding model limits
    # and to prioritize recent information. A reasonable limit could be around 16000 tokens
    # for Gemini 2.5 Flash Lite, leaving room for the model's response.
    # Adjust this value based on actual model constraints and performance.
    MAX_PROMPT_TOKENS = 16000
    # A rough estimate for tokenization, can be refined with a proper tokenizer if needed.
    APPROX_TOKENS_PER_CHAR = 0.25

    if len(prompt) * APPROX_TOKENS_PER_CHAR > MAX_PROMPT_TOKENS:
        log.warning(f"Prompt too long ({len(prompt)} chars), truncating for Gemini. Original length: {len(prompt)}")
        # Simple truncation, prioritizing the end of the prompt.
        # More sophisticated methods could involve summarization or keyword extraction.
        prompt = prompt[int(MAX_PROMPT_TOKENS / APPROX_TOKENS_PER_CHAR) - 1000:] # Keep a buffer

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
                resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                data = resp.json()

                if "candidates" not in data or not data["candidates"]:
                    log.error(f"Gemini API invalid response (attempt {attempt+1}): No candidates found. Response: {data}")
                    raise Exception("Gemini API returned no candidates")

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                _gemini_count += 1
                await _save_brain_counts(_gemini_count, _glm_count, _claude_count)
                # Reset retry delay on success
                retry_delay = initial_retry_delay
                return (text, f"Gemini {GEMINI_MODEL}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503):
                error_code = e.response.status_code
                log.warning(f"Gemini {error_code} error (attempt {attempt+1}/{max_retries}), waiting {retry_delay}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                    continue
                else:
                    log.error(f"Gemini {error_code} error after retries.")
                    raise Exception(f"Gemini {error_code} error after retries") from e
            else:
                # Handle other HTTP errors
                log.error(f"Gemini HTTP error (attempt {attempt+1}): {e.response.status_code} - {e.response.text}", exc_info=True)
                raise e # Re-raise to be caught by the general exception handler
        except Exception as e:
            # Catch any other exceptions (network issues, JSON parsing, etc.)
            log.error(f"Gemini failed (attempt {attempt+1}): {e}", exc_info=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                # Exponential backoff
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue
            else:
                # Gemini complete failure → GLM-4 fallback
                log.error("Gemini failed after all retries, falling back to GLM-4.")
                try:
                    text, brain = await think_glm(prompt)
                    return (text, f"{brain} (fallback)")
                except Exception as glm_error:
                    log.error(f"GLM-4 failed: {glm_error}, falling back to Claude CLI")
                    # GLM failure → Claude CLI fallback
                    try:
                        text, _ = await think_claude(prompt)
                        return (text, "Claude CLI (fallback)")
                    except ClaudeSessionExpired:
                        # All failed → AI pause
                        pause_ai()
                        raise AIUnavailable("Gemini failed + GLM failed + Claude CLI session expired. Conversation unavailable.")
                    except Exception as claude_error:
                        log.error(f"Claude CLI failed during fallback: {claude_error}")
                        raise AIUnavailable(f"Gemini and GLM failed, Claude CLI also failed: {claude_error}")

    # Should not reach here if max_retries is handled correctly, but as a safeguard:
    log.error("Unexpected exit from Gemini think loop.")
    raise AIUnavailable("An unexpected error occurred during Gemini processing.")

# --- GLM-4 API (智谱AI) ---
GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

async def think_glm(prompt: str, max_tokens: int = 4096) -> tuple[str, str]:
    """GLM-4で思考（Geminiのバックアップ）。戻り値: (テキスト, 脳の名前)

    Args:
        prompt: プロンプト
        max_tokens: 最大出力トークン数
    """
    global _glm_count
    if not GLM_API_KEY:
        raise Exception("GLM_API_KEY not configured")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                GLM_URL,
                headers={
                    "Authorization": f"Bearer {GLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )
            data = resp.json()

            if resp.status_code != 200:
                raise Exception(f"GLM API error: {resp.status_code} - {data}")

            if "choices" not in data or not data["choices"]:
                raise Exception(f"GLM API invalid response: {data}")

            text = data["choices"][0]["message"]["content"]
            _glm_count += 1
            await _save_brain_counts(_gemini_count, _glm_count, _claude_count)
            return (text, "GLM-4")
    except Exception as e:
        log.error(f"GLM-4 failed: {e}")
        raise

# --- Claude CLI ---
async def think_claude(prompt: str, timeout: int = 120) -> tuple[str, str]:
    """Claude CLIで思考（会話用、段階的タイムアウト対応）。戻り値: (テキスト, 脳の名前)

    Args:
        prompt: プロンプト
        timeout: タイムアウト秒数（デフォルト120秒、コード生成時は280秒推奨）

    段階的リトライ戦略:
    - 1回目: 同じプロンプトでリトライ
    - 2回目: プロンプトを短縮してリトライ（最後の1000文字のみ）
    - 3回目: Claude CLIのセットアップを試行し、成功すればリトライ。失敗したらGeminiにフォールバック。
    """
    global _claude_count
    loop = asyncio.get_running_loop()
    original_prompt = prompt

    # Shorten prompt for LP content generation if it's not already specific
    # This ensures the AI focuses on the Coconala promotion aspect.
    coconala_lp_instruction = (
        "[GOAL: Drive traffic to Coconala for service inquiries.] "
        "Generate compelling content that persuades users to visit Coconala. "
        "Specifically highlight the benefits of using Coconala for finding "
        "and inquiring about services. Focus on calls to action that direct users to "
        "the Coconala platform. Ensure the output is direct, engaging, and conversion-focused.\n\n"
    )
    # Check if the prompt already contains specific LP/Coconala instructions.
    # If not, prepend our specific instruction.
    if "coconala" not in original_prompt.lower() and "lp" not in original_prompt.lower() and "landing page" not in original_prompt.lower():
        log.info("Prepending Coconala LP instruction to Claude prompt.")
        modified_prompt = coconala_lp_instruction + original_prompt
    else:
        modified_prompt = original_prompt
        log.info("Prompt already contains LP/Coconala related terms, not prepending specific instruction.")


    # Incremental timeout settings and retry intervals.
    timeouts = [60, 90, 120]
    sleep_between_retries = 5

    for attempt in range(3):
        current_timeout = timeouts[attempt]

        # Shorten prompt on subsequent attempts to reduce chances of timeout.
        current_prompt = modified_prompt
        if attempt >= 1 and len(modified_prompt) > 1000:
            current_prompt = "...(truncated)...\n" + modified_prompt[-1000:]
            log.warning(f"Claude CLI attempt {attempt+1}: Prompt truncated ({len(modified_prompt)} -> {len(current_prompt)} chars)")

        try:
            # Execute Claude CLI command in a separate thread to avoid blocking the event loop.
            result = await loop.run_in_executor(
                None,
                lambda p=current_prompt, t=current_timeout: subprocess.run(
                    [CLAUDE_PATH, "--print", "-p", p],
                    capture_output=True, text=True, timeout=t,
                    env=CLI_ENV,
                ),
            )
            # Log stderr if present, for debugging.
            if result.stderr and result.stderr.strip():
                log.error(f"Claude CLI stderr (attempt {attempt+1}): {result.stderr[:500]}")

            # Check for successful execution and non-empty stdout.
            if result.returncode == 0 and result.stdout.strip():
                _claude_count += 1
                await _save_brain_counts(_gemini_count, _glm_count, _claude_count)
                log.info("Claude CLI success.")
                return (result.stdout.strip(), "Claude CLI")

            # If Claude CLI failed, check for specific error conditions.
            stdout_full = result.stdout if result.stdout else "(empty)"
            stderr_full = result.stderr if result.stderr else "(empty)"

            # Detect specific failure conditions.
            if result.returncode != 0 or not result.stdout.strip():
                # Fallback to Gemini if Claude CLI returns an error or empty output.
                log.error(f"Claude CLI failed (attempt {attempt+1}): returncode={result.returncode}, stdout_empty={not result.stdout.strip()}. Falling back to Gemini.")
                try:
                    text, brain = await think_gemini(original_prompt) # Use original prompt for Gemini fallback
                    log.info(f"Gemini fallback success from Claude CLI failure.")
                    return (text, f"{brain} (fallback)")
                except Exception as gemini_fallback_e:
                    log.error(f"Gemini fallback failed: {gemini_fallback_e}", exc_info=True)
                    # If Gemini also fails, raise an AIUnavailable exception.
                    raise AIUnavailable(f"Claude CLI failed and Gemini fallback also failed: {gemini_fallback_e}") from gemini_fallback_e

            # Detect specific session expired conditions.
            if result.returncode == 1 and ("Error: You are not logged in" in stdout_full or "Not logged in" in stdout_full or "Session expired" in stdout_full):
                log.error("Claude CLI: Session expired or not logged in.")
                setup_success = False
                try:
                    log.info("Attempting to run 'claude setup-token' to re-authenticate.")
                    setup_process = await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(
                            [CLAUDE_PATH, "setup-token"],
                            capture_output=True, text=True, timeout=120,
                            env=CLI_ENV,
                        )
                    )
                    if setup_process.returncode == 0:
                        log.info("Claude CLI setup-token completed successfully.")
                        setup_success = True
                    else:
                        log.error(f"Claude CLI setup-token failed: {setup_process.stderr}")
                except subprocess.TimeoutExpired:
                    log.error("Claude CLI setup-token timed out.")
                except Exception as e:
                    log.error(f"An error occurred during Claude CLI setup-token: {e}")

                if setup_success:
                    # Retry the original prompt after successful setup.
                    log.info("Retrying Claude CLI with original prompt after successful setup-token.")
                    continue # Restarts the for loop for the next attempt.
                else:
                    # If setup failed, raise ClaudeSessionExpired to trigger fallback.
                    raise ClaudeSessionExpired("Claude CLI session expired and setup-token failed or timed out.")

            # Log detailed error information for debugging for non-session-expired errors.
            log.error(f"Claude CLI attempt {attempt+1}: returncode={result.returncode}, prompt_len={len(current_prompt)}, timeout={current_timeout}s")
            log.error(f"  stdout: {stdout_full[:500]}")
            log.error(f"  stderr: {stderr_full[:500]}")

        except subprocess.TimeoutExpired:
            log.error(f"Claude CLI attempt {attempt+1}: timeout ({current_timeout}s)")
            # On timeout, fallback to Gemini.
            log.warning("Claude CLI timed out, falling back to Gemini.")
            try:
                text, brain = await think_gemini(original_prompt)
                log.info(f"Gemini fallback success from Claude CLI timeout.")
                return (text, f"{brain} (fallback)")
            except Exception as e:
                log.error(f"Gemini fallback failed: {e}", exc_info=True)
                raise AIUnavailable(f"Claude CLI timed out and Gemini fallback failed: {e}") from e
        except ClaudeSessionExpired as e:
            log.error(f"Claude CLI session expired (attempt {attempt+1}): {e}")
            # Attempt to re-authenticate if it's not the last attempt.
            if attempt < 2:
                setup_success = False
                try:
                    log.info("Attempting to run 'claude setup-token' to re-authenticate after session expired.")
                    setup_process = await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(
                            [CLAUDE_PATH, "setup-token"],
                            capture_output=True, text=True, timeout=120,
                            env=CLI_ENV,
                        )
                    )
                    if setup_process.returncode == 0:
                        log.info("Claude CLI setup-token completed successfully.")
                        setup_success = True
                    else:
                        log.error(f"Claude CLI setup-token failed: {setup_process.stderr}")
                except subprocess.TimeoutExpired:
                    log.error("Claude CLI setup-token timed out.")
                except Exception as e:
                    log.error(f"An error occurred during Claude CLI setup-token: {e}")

                if setup_success:
                    log.info("Retrying Claude CLI with original prompt after successful setup-token.")
                    continue # Restarts the for loop for the next attempt.
                else:
                    log.error("Claude CLI setup-token failed. Falling back to Gemini.")
                    try:
                        text, brain = await think_gemini(original_prompt) # Use original prompt for Gemini fallback
                        return (text, f"{brain} (fallback)")
                    except Exception as gemini_fallback_e:
                        log.error(f"Gemini fallback failed: {gemini_fallback_e}", exc_info=True)
                        raise AIUnavailable(f"Claude CLI session expired and setup failed. Gemini fallback also failed: {gemini_fallback_e}") from gemini_fallback_e
            else:
                # If it's the last attempt and ClaudeSessionExpired, fallback to Gemini.
                log.warning("Claude CLI session expired on last attempt, falling back to Gemini.")
                try:
                    text, brain = await think_gemini(original_prompt)
                    return (text, f"{brain} (fallback)")
                except Exception as e:
                    log.error(f"Gemini fallback failed: {e}", exc_info=True)
                    raise AIUnavailable(f"Claude CLI session expired and last attempt failed. Gemini fallback also failed: {e}") from e

        except Exception as e:
            log.error(f"Claude CLI attempt {attempt+1}: {e}")

        # Wait before the next retry.
        if attempt < 2:
            await asyncio.sleep(sleep_between_retries)

    # If all attempts fail, raise AIUnavailable.
    log.error("Claude CLI failed after all attempts.")
    raise AIUnavailable(f"Claude CLI failed after 3 attempts (timeouts={timeouts}s).")

# --- 統合思考関数 ---
async def think(prompt: str, heavy: bool = False, bias_to_x_lp: bool = False) -> tuple[str, str]:
    """Unified thinking function. Returns: (text, brain_name)

    Args:
        prompt (str): The input prompt for the AI model.
        heavy (bool, optional): If True, prioritizes Claude CLI for longer tasks. Defaults to False.
        bias_to_x_lp (bool, optional): If True, biases the prompt towards X posts and LP proposals. Defaults to False.

    Raises:
        AIUnavailable: If the AI is paused or if all underlying AI models fail.
    """
    # Check if AI is paused before proceeding.
    if is_ai_paused():
        remaining = get_ai_pause_remaining()
        raise AIUnavailable(f"AI is currently paused (remaining {remaining} seconds).")

    # Dynamically prepend instructions based on goals or explicit bias.
    augmented_prompt = prompt
    prepended_instruction = False

    # 1. Check if bias_to_x_lp is explicitly True.
    if bias_to_x_lp:
        x_lp_instruction = (
            "[PRIORITY] Focus on generating content specifically for X (formerly Twitter) posts and "
            "Landing Page (LP) proposals. Prioritize conciseness, engagement, and conversion-oriented "
            "language suitable for these platforms. Ensure output is ready for immediate use.\n\n"
        )
        augmented_prompt = x_lp_instruction + augmented_prompt
        log.info("bias_to_x_lp is True, prepended specific instruction to prompt.")
        prepended_instruction = True

    # 2. Check if current goal is revenue generation and prepend LP priority instruction.
    # This check should only apply if bias_to_x_lp was not already explicitly set.
    try:
        from config import GOALS_PATH
        if GOALS_PATH.exists() and not prepended_instruction:
            goals_text = GOALS_PATH.read_text(encoding="utf-8")
            # Detect revenue-focused phase from goals
            revenue_keywords = ["LP", "coconala", "revenue", "earning", "income"]
            is_revenue_phase = any(kw.lower() in goals_text.lower() for kw in revenue_keywords)
            if is_revenue_phase:
                state = load_state()
                current_task = state.get("current_task")
                # Only prepend LP instruction when no specific task is set
                # or when the task is related to revenue/content generation
                revenue_task_keywords = ["lp", "revenue", "coconala", "content", "x post", "landing"]
                task_is_revenue = (
                    current_task is None
                    or (isinstance(current_task, str) and any(k in current_task.lower() for k in revenue_task_keywords))
                )
                if task_is_revenue:
                    lp_instruction = (
                        "[PRIORITY] Current goal is revenue generation via LP content creation. "
                        "When generating content, prioritize high-quality LP (landing page) material, "
                        "persuasive copy, and actionable output suitable for Coconala service delivery.\n\n"
                    )
                    augmented_prompt = lp_instruction + augmented_prompt
                    log.info("Revenue generation goal detected, prepended LP priority instruction to prompt.")
                    prepended_instruction = True # Mark as prepended to avoid double-prefixing
    except Exception as e:
        log.warning(f"Failed to check revenue goal status: {e}")

    # Auto-detect LP/landing page/copywriting tasks and upgrade to heavy mode
    # Claude CLI produces higher-quality long-form content for these tasks
    if not heavy:
        lp_task_keywords = [
            "landing page", "lp", "copywriting", "sales copy",
            "persuasive", "conversion", "headline", "cta",
            "call to action", "lead magnet", "sales letter",
            "product description", "service page",
        ]
        prompt_lower = prompt.lower() # Use original prompt for detection
        if any(kw in prompt_lower for kw in lp_task_keywords):
            heavy = True
            log.info("LP/copywriting task detected in prompt, upgrading to heavy mode (Claude CLI) for better quality.")

    # Attempt to use AI models, with fallbacks and specific error handling.
    try:
        if heavy:
            # For heavy tasks, prioritize Claude CLI, then Gemini.
            try:
                return await think_claude(augmented_prompt)
            except ClaudeSessionExpired:
                log.warning("Claude CLI session expired, falling back to Gemini.")
                # If Claude session expired, fallback to Gemini immediately.
                return await think_gemini(augmented_prompt)
            except Exception as e:
                # Catch any other exceptions from think_claude and fallback to Gemini.
                log.error(f"Error in think_claude during heavy task: {e}. Falling back to Gemini.", exc_info=True)
                return await think_gemini(augmented_prompt)
        else:
            # For non-heavy tasks, prioritize Gemini, then GLM-4, then Claude CLI.
            try:
                return await think_gemini(augmented_prompt)
            except Exception as e:
                # If Gemini fails, try GLM-4.
                log.error(f"Gemini failed for non-heavy task: {e}. Falling back to GLM-4.", exc_info=True)
                try:
                    text, brain = await think_glm(augmented_prompt)
                    return (text, f"{brain} (fallback)")
                except Exception as glm_error:
                    # If GLM-4 also fails, try Claude CLI.
                    log.error(f"GLM-4 also failed: {glm_error}. Falling back to Claude CLI.", exc_info=True)
                    try:
                        text, brain = await think_claude(augmented_prompt)
                        return (text, f"{brain} (fallback)")
                    except ClaudeSessionExpired:
                        log.error("Claude CLI session expired during fallback for non-heavy task.")
                        raise AIUnavailable("AI is unavailable. All models failed, Claude CLI session expired.") from glm_error
                    except Exception as claude_error:
                        log.error(f"Claude CLI failed during fallback for non-heavy task: {claude_error}", exc_info=True)
                        raise AIUnavailable(f"AI is unavailable. All models failed, including Claude CLI fallback.") from claude_error

    except AIUnavailable as e:
        # Re-raise AIUnavailable exceptions that originate from the underlying AI functions.
        log.error(f"AI Unavailable: {e}")
        raise e
    except Exception as e:
        # Catch any other unexpected exceptions during the AI thinking process and attempt fallbacks.
        log.error(f"An unexpected error occurred during AI thinking: {e}", exc_info=True)
        # If a general unexpected error occurred, try to use the most robust fallback (Claude CLI, then Gemini).
        try:
            log.warning("Attempting fallback to Claude CLI due to unexpected error.")
            return await think_claude(augmented_prompt)
        except ClaudeSessionExpired:
            log.warning("Claude CLI session expired during general fallback, attempting Gemini.")
            try:
                return await think_gemini(augmented_prompt)
            except Exception as gemini_fallback_e:
                log.error(f"Gemini fallback also failed: {gemini_fallback_e}", exc_info=True)
                raise AIUnavailable(f"AI is unavailable. Unexpected error occurred and all fallback models failed: {e}") from gemini_fallback_e
        except Exception as claude_fallback_e:
            log.error(f"Claude CLI fallback also failed: {claude_fallback_e}", exc_info=True)
            raise AIUnavailable(f"AI is unavailable. Unexpected error occurred and Claude CLI fallback failed: {e}") from claude_fallback_e

def is_heavy(message: str) -> bool:
    from config import HEAVY_KEYWORDS
    return any(kw in message for kw in HEAVY_KEYWORDS)

def get_brain_counts() -> tuple[int, int, int]:
    """(gemini_count, glm_count, claude_count) を返す（state.jsonから永続化された値）"""
    return _gemini_count, _glm_count, _claude_count

def is_heavy(message: str) -> bool:
    from config import HEAVY_KEYWORDS
    return any(kw in message for kw in HEAVY_KEYWORDS)

def get_brain_counts() -> tuple[int, int, int]:
    """(gemini_count, glm_count, claude_count) を返す（state.jsonから永続化された値）"""
    return _gemini_count, _glm_count, _claude_count


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
