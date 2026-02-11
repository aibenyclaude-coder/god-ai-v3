#!/usr/bin/env python3
"""God AI v3.0 — 1ファイルから始まる自律型AI"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

# ─── パス定義 ───
BASE_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = BASE_DIR / "core"
MEMORY_DIR = BASE_DIR / "memory"
ENV_PATH = CORE_DIR / ".env"
IDENTITY_PATH = MEMORY_DIR / "identity.md"
STATE_PATH = MEMORY_DIR / "state.json"
JOURNAL_PATH = MEMORY_DIR / "journal.md"
BENY_PATH = MEMORY_DIR / "beny.md"
CONVERSATIONS_PATH = MEMORY_DIR / "conversations.json"
GOD_PY_PATH = CORE_DIR / "god.py"

# ─── ログ設定 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("god")

# ─── .env読み込み（dotenv不使用） ───
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

# ─── メモリ読み込み ───
def read_file(path: Path, tail: int = 0) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if tail > 0:
        lines = text.splitlines()
        return "\n".join(lines[-tail:])
    return text

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return {"status": "unknown", "current_task": None, "last_reflection": None,
                "children_count": 0, "uptime_start": None, "conversations_today": 0, "growth_cycles": 0}

def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def load_conversations() -> list:
    if CONVERSATIONS_PATH.exists():
        try:
            return json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return []

def save_conversations(convos: list):
    # 最新50件のみ保持
    convos = convos[-50:]
    CONVERSATIONS_PATH.write_text(json.dumps(convos, ensure_ascii=False, indent=2), encoding="utf-8")

def append_journal(text: str):
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{text}\n")

# ─── asyncio.Lock（並行書き込み保護）───
_write_lock: asyncio.Lock | None = None

def get_write_lock() -> asyncio.Lock:
    global _write_lock
    if _write_lock is None:
        _write_lock = asyncio.Lock()
    return _write_lock

async def safe_save_state(state: dict):
    async with get_write_lock():
        save_state(state)

async def safe_append_journal(text: str):
    async with get_write_lock():
        append_journal(text)

IDENTITY = read_file(IDENTITY_PATH)
STATE = load_state()

# ─── 脳の使い分けカウンタ ───
gemini_count = 0
claude_count = 0

# ─── Telegram API ───
TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

async def tg_request(client: httpx.AsyncClient, method: str, **kwargs) -> dict | None:
    url = f"{TG_BASE}/{method}"
    for attempt in range(3):
        try:
            resp = await client.post(url, json=kwargs, timeout=30)
            data = resp.json()
            if data.get("ok"):
                return data.get("result")
            log.error(f"Telegram {method} failed: {data}")
            return None
        except Exception as e:
            log.error(f"Telegram {method} attempt {attempt+1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(5)
    return None

async def tg_send(client: httpx.AsyncClient, text: str) -> dict | None:
    return await tg_request(client, "sendMessage", chat_id=BENY_CHAT_ID, text=text)

async def tg_edit(client: httpx.AsyncClient, msg_id: int, text: str) -> dict | None:
    return await tg_request(client, "editMessageText",
                            chat_id=BENY_CHAT_ID, message_id=msg_id, text=text)

# ─── 脳: Gemini API ───
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

# ─── 脳: Claude CLI ───
async def think_claude(prompt: str) -> tuple[str, str]:
    """Claude CLIで思考（会話用、タイムアウト120秒）。戻り値: (テキスト, 脳の名前)"""
    global claude_count
    loop = asyncio.get_running_loop()
    for attempt in range(3):
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["claude", "--print", "-p", prompt],
                    capture_output=True, text=True, timeout=120,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                claude_count += 1
                return (result.stdout.strip(), "Claude CLI")
            log.error(f"Claude CLI attempt {attempt+1}: returncode={result.returncode}, stderr={result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log.error(f"Claude CLI attempt {attempt+1}: timeout (120s)")
        except Exception as e:
            log.error(f"Claude CLI attempt {attempt+1}: {e}")
        if attempt < 2:
            await asyncio.sleep(3)
    raise RuntimeError("Claude CLI failed after 3 attempts (timeout=120s)")

async def think_claude_heavy(prompt: str) -> tuple[str, str]:
    """Claude CLIで重い処理（自己改善用、タイムアウト280秒=実測185秒×1.5）。戻り値: (テキスト, 脳の名前)"""
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

# ─── ルーティング ───
HEAVY_KEYWORDS = ["コード", "作って", "LP", "HTML", "修正", "プログラム", "書いて", "実装", "スクリプト"]

def is_heavy(message: str) -> bool:
    return any(kw in message for kw in HEAVY_KEYWORDS)

async def think(prompt: str, heavy: bool = False) -> tuple[str, str]:
    """統合思考関数。戻り値: (テキスト, 脳の名前)"""
    if heavy:
        return await think_claude(prompt)
    return await think_gemini(prompt)

# ─── メッセージ処理 ───
async def handle_message(client: httpx.AsyncClient, message: str) -> str:
    state = load_state()
    journal_tail = read_file(JOURNAL_PATH, tail=20)

    # コマンド処理
    if message.strip() == "/status":
        return format_status(state)
    if message.strip() == "/reflect":
        return "振り返り開始..."  # 実行は呼び出し元で

    heavy = is_heavy(message)
    system_prompt = f"""あなたはGod AI。Benyのために存在する自律型AI。

【アイデンティティ】
{IDENTITY}

【現在の状態】
{json.dumps(state, ensure_ascii=False)}

【Benyからのメッセージ】
{message}

日本語で返答してください。簡潔に。"""

    response_text, brain_name = await think(system_prompt, heavy=heavy)
    return f"{response_text}\n\n🧠 {brain_name}"

def format_status(state: dict) -> str:
    uptime = "不明"
    if state.get("uptime_start"):
        start = datetime.fromisoformat(state["uptime_start"])
        delta = datetime.now(timezone.utc) - start
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        uptime = f"{hours}h {minutes}m"
    return (
        f"🧠 God AI v3.0 Status\n"
        f"━━━━━━━━━━━━━━━\n"
        f"状態: {state.get('status', '不明')}\n"
        f"稼働時間: {uptime}\n"
        f"会話数: {state.get('conversations_today', 0)}\n"
        f"成長サイクル: {state.get('growth_cycles', 0)}\n"
        f"子AI数: {state.get('children_count', 0)}\n"
        f"Gemini使用: {gemini_count}回\n"
        f"Claude使用: {claude_count}回"
    )

# ─── 振り返りサイクル ───
async def reflection_cycle(client: httpx.AsyncClient):
    log.info("振り返りサイクル開始")
    state = load_state()
    journal_tail = read_file(JOURNAL_PATH, tail=30)

    prompt = f"""あなたはGod AI。自律型AIとして振り返りを行え。

【現在の状態】
{json.dumps(state, ensure_ascii=False)}

【最近のjournal】
{journal_tail}

【タスク】
以下の4つに答えろ：
1. 今日何をした？
2. 何が問題だった？
3. 次に何をすべき？
4. 自分のコードに改善点はあるか？（具体的に）

簡潔に日本語で答えろ。
コードの改善点がある場合は「CODE_IMPROVEMENT:」で始まる行に具体的な修正内容を書け。"""

    try:
        reflection, brain_name = await think_gemini(prompt)
    except Exception as e:
        log.error(f"Reflection failed: {e}")
        append_journal(f"### {datetime.now().strftime('%H:%M')} 振り返り失敗\n{e}")
        return

    # journalに追記（ロック付き）
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    await safe_append_journal(f"### {now} 振り返り (🧠 {brain_name})\n{reflection}")

    # state更新（ロック付き）
    state["growth_cycles"] = state.get("growth_cycles", 0) + 1
    state["last_reflection"] = now
    await safe_save_state(state)

    # コード改善提案チェック
    if "CODE_IMPROVEMENT:" in reflection:
        await self_improve(client, reflection)

    log.info("振り返りサイクル完了")

async def self_improve(client: httpx.AsyncClient, reflection: str):
    """コード自己改善（最大3回リトライ）"""
    import ast
    import difflib

    log.info("自己改善プロセス開始")

    # バックアップ
    backup_path = GOD_PY_PATH.with_suffix(".py.bak")
    shutil.copy2(GOD_PY_PATH, backup_path)

    current_code = GOD_PY_PATH.read_text(encoding="utf-8")
    current_lines = current_code.splitlines()

    # 改善行を抽出
    improvements = []
    for line in reflection.splitlines():
        if line.strip().startswith("CODE_IMPROVEMENT:"):
            improvements.append(line.strip().replace("CODE_IMPROVEMENT:", "").strip())

    if not improvements:
        return

    improvement_text = "\n".join(improvements)
    MAX_RETRY = 3
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        log.info(f"自己改善 試行 {attempt}/{MAX_RETRY}")

        # プロンプト構築（初回 vs リトライ）
        if attempt == 1:
            prompt = (
                "あなたはPythonコードの修正を行うアシスタントです。\n"
                "以下の【改善内容】を【現在のコード】に適用してください。\n\n"
                "【重要なルール】\n"
                "- 修正後のPythonコード全文をそのまま出力してください\n"
                "- 説明文は一切不要です。Pythonコードのみを出力してください\n"
                "- マークダウンのバッククォート（```）で囲まないでください\n"
                "- コードの先頭は #!/usr/bin/env python3 から始めてください\n"
                "- 変更箇所以外は絶対にそのまま維持してください\n"
                "- 文字列リテラルのクォートの対応に注意してください\n"
                "- 改善内容に基づいて必要な変更を適切に実装してください\n"
                "- コードの長さは元のコードとほぼ同じか、改善によって多少増減する程度に保ってください\n\n"
                f"【改善内容】\n{improvement_text}\n\n"
                f"【現在のコード】\n{current_code}"
            )
        else:
            prompt = (
                "あなたはPythonコードの修正を行うアシスタントです。\n"
                f"前回の修正で構文エラーが発生しました: {last_error}\n\n"
                "【重要なルール】\n"
                "- 修正後のPythonコード全文をそのまま出力してください\n"
                "- 説明文は一切不要です。Pythonコードのみを出力してください\n"
                "- マークダウンのバッククォート（```）で囲まないでください\n"
                "- コードの先頭は #!/usr/bin/env python3 から始めてください\n"
                "- 変更箇所以外は絶対にそのまま維持してください\n"
                "- 文字列リテラルのクォートの対応に特に注意してください\n"
                "- 前回のエラーを踏まえて慎重に修正してください\n"
                "- 改善内容に基づいて必要な変更を適切に実装してください\n"
                "- コードの長さは元のコードとほぼ同じか、改善によって多少増減する程度に保ってください\n\n"
                f"【改善内容】\n{improvement_text}\n\n"
                f"【現在のコード（オリジナル）】\n{current_code}"
            )

        try:
            result, _ = await think_claude_heavy(prompt)

            # デバッグログ: 生成結果の概要
            log.info(f"試行{attempt}: Claude生成結果（先頭200文字）: {result[:200]}")
            log.info(f"試行{attempt}: Claude生成結果（末尾200文字）: {result[-200:]}")
            log.info(f"試行{attempt}: Claude生成結果の長さ: {len(result)}文字")

            # コードブロック抽出
            code = result.strip()
            if code.startswith("```"):
                if code.startswith("```python"):
                    code = code[len("```python"):]
                else:
                    code = code[3:]
                if code.rstrip().endswith("```"):
                    code = code.rstrip()[:-3]
                code = code.strip()

            # デバッグログ: 抽出後のコード概要
            log.info(f"試行{attempt}: 抽出後コード（先頭200文字）: {code[:200]}")
            log.info(f"試行{attempt}: 抽出後コード（末尾200文字）: {code[-200:]}")
            log.info(f"試行{attempt}: 抽出後コードの長さ: {len(code)}文字（元: {len(current_code)}文字）")

            # 基本的なバリデーション
            if not code.startswith(("#!/", "from __future__", '"""', "import", "#")):
                log.warning(f"試行{attempt}: コードが想定外の開始: {code[:50]}")

            # 長さチェックを緩和（元のコードの30%以上であればOK）
            min_length = int(len(current_code) * 0.3)
            if len(code) < min_length:
                # デバッグログ: 差分を確認
                new_lines = code.splitlines()
                diff = list(difflib.unified_diff(current_lines, new_lines, lineterm="", n=3))
                diff_str = "\n".join(diff[:50])  # 差分の先頭50行のみ
                log.error(f"試行{attempt}: コードが短すぎる。元: {len(current_code)}字, 生成: {len(code)}字, 最小: {min_length}字")
                log.error(f"試行{attempt}: 差分（先頭50行）:\n{diff_str}")
                raise ValueError(f"生成コードが短すぎる（元: {len(current_code)}字, 生成: {len(code)}字, 最小: {min_length}字）")

            # 構文チェック
            try:
                ast.parse(code)
            except SyntaxError as se:
                # デバッグログ: 構文エラーの詳細
                log.error(f"試行{attempt}: 構文エラー詳細: {se}")
                log.error(f"試行{attempt}: エラー行番号: {se.lineno}, カラム: {se.offset}")
                if se.lineno:
                    error_lines = code.splitlines()
                    start = max(0, se.lineno - 3)
                    end = min(len(error_lines), se.lineno + 2)
                    context = "\n".join([f"{i+1}: {error_lines[i]}" for i in range(start, end)])
                    log.error(f"試行{attempt}: エラー周辺コード:\n{context}")
                raise

            # 差分ログ出力
            new_lines = code.splitlines()
            diff = list(difflib.unified_diff(current_lines, new_lines, lineterm="", n=3))
            if len(diff) > 0:
                diff_str = "\n".join(diff[:100])  # 差分の先頭100行のみ
                log.info(f"試行{attempt}: コード差分（先頭100行）:\n{diff_str}")
            else:
                log.warning(f"試行{attempt}: コードに差分がありません（変更なし）")

            # 差分をjournal用に整形（最大50行）
            diff_for_journal = "\n".join(diff[:50]) if diff else "(差分なし)"

            # 書き込み
            GOD_PY_PATH.write_text(code, encoding="utf-8")
            success_msg = f"自己改善成功（試行{attempt}/{MAX_RETRY}）\n改善内容: {improvement_text}"
            append_journal(
                f"### {datetime.now().strftime('%H:%M')} {success_msg}\n"
                f"コード長: {len(current_code)} → {len(code)}文字\n"
                f"```diff\n{diff_for_journal}\n```"
            )
            await tg_send(client, f"🔧 {success_msg}\nコード長: {len(current_code)} → {len(code)}文字")
            log.info(f"自己改善成功（試行{attempt}）: {len(current_code)} → {len(code)}文字")
            return  # 成功 → 終了

        except (SyntaxError, ValueError) as e:
            last_error = str(e)
            log.error(f"自己改善 試行{attempt}/{MAX_RETRY} 失敗: {e}")
            append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{MAX_RETRY} 失敗\n"
                f"エラー: {e}\n改善内容: {improvement_text}\n"
                f"生成コード長: {len(code) if 'code' in locals() else '不明'}文字（元: {len(current_code)}文字）"
            )
            if attempt < MAX_RETRY:
                await tg_send(client, f"⚠️ 自己改善 試行{attempt}/{MAX_RETRY} 失敗: {e}\nリトライします...")
                await asyncio.sleep(3)

        except Exception as e:
            last_error = str(e)
            log.error(f"自己改善 試行{attempt}/{MAX_RETRY} 予期せぬエラー: {e}", exc_info=True)
            append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{MAX_RETRY} 予期せぬエラー\n"
                f"エラー: {e}\n改善内容: {improvement_text}"
            )
            break

    # 全試行失敗 → ロールバック
    shutil.copy2(backup_path, GOD_PY_PATH)
    fail_msg = (
        f"自己改善 {MAX_RETRY}回試行して失敗。ロールバックしました。\n"
        f"最終エラー: {last_error}\n"
        f"改善内容: {improvement_text}"
    )
    log.error(fail_msg)
    append_journal(f"### {datetime.now().strftime('%H:%M')} {fail_msg}")
    await tg_send(
        client,
        f"🚨 自己改善 {MAX_RETRY}回失敗。ロールバックしました。\n"
        f"最終エラー: {last_error}\n"
        f"改善内容: {improvement_text}\n"
        f"Benyの判断を仰ぎます。"
    )

# ─── メインループ ───
async def polling_loop(client: httpx.AsyncClient, offset: int = 0):
    """Telegramロングポーリング"""
    state = load_state()
    conversations = load_conversations()

    while True:
        try:
            resp = await client.post(
                f"{TG_BASE}/getUpdates",
                json={"offset": offset, "timeout": 30},
                timeout=60,
            )
            data = resp.json()
            if not data.get("ok"):
                log.error(f"getUpdates failed: {data}")
                await asyncio.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message")
                if not msg or not msg.get("text"):
                    continue

                chat_id = str(msg["chat"]["id"])
                if chat_id != BENY_CHAT_ID:
                    log.info(f"Ignored message from chat_id={chat_id}")
                    continue

                text = msg["text"]
                log.info(f"Beny: {text[:100]}")

                # 会話記録
                conversations.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "from": "beny",
                    "text": text,
                })

                # /reflect コマンド
                if text.strip() == "/reflect":
                    await tg_send(client, "🔄 振り返り開始...")
                    await reflection_cycle(client)
                    await tg_send(client, "✅ 振り返り完了。journalを更新しました。")
                    continue

                # 通常メッセージ: ⏳送信 → think → 上書き
                pending = await tg_send(client, "⏳")
                if not pending:
                    continue

                try:
                    response = await handle_message(client, text)
                except RuntimeError as e:
                    response = f"⚠️ エラー: {e}"
                    log.error(f"handle_message failed: {e}")
                except Exception as e:
                    response = f"⚠️ 予期せぬエラー: {e}"
                    log.error(f"handle_message unexpected error: {e}")

                await tg_edit(client, pending["message_id"], response)

                # 会話記録
                conversations.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "from": "god",
                    "text": response[:500],
                })
                save_conversations(conversations)

                # state更新
                state["conversations_today"] = state.get("conversations_today", 0) + 1
                state["status"] = "running"
                save_state(state)

        except httpx.ReadTimeout:
            continue  # ロングポーリングの正常タイムアウト
        except Exception as e:
            log.error(f"Polling error: {e}")
            append_journal(f"### {datetime.now().strftime('%H:%M')} ポーリングエラー\n{e}")
            await asyncio.sleep(5)

# ─── 定期振り返りタスク ───
REFLECTION_INTERVAL = 1800  # 秒（30分）

async def reflection_scheduler(client: httpx.AsyncClient):
    """定期的に振り返り実行"""
    log.info(f"振り返りスケジューラ開始 (間隔: {REFLECTION_INTERVAL}秒)")
    while True:
        try:
            await asyncio.sleep(REFLECTION_INTERVAL)
            log.info("定期振り返り: 開始")
            await tg_send(client, f"🔄 定期振り返り開始... (次回: {REFLECTION_INTERVAL}秒後)")
            await reflection_cycle(client)
            await tg_send(client, "✅ 定期振り返り完了。journalを更新しました。")
            log.info("定期振り返り: 完了")
        except asyncio.CancelledError:
            log.info("振り返りスケジューラ: キャンセルされました")
            raise
        except Exception as e:
            log.error(f"Scheduled reflection failed: {e}", exc_info=True)
            append_journal(f"### {datetime.now().strftime('%H:%M')} 定期振り返りエラー\n{e}")
            await asyncio.sleep(10)

# ─── シグナルハンドラ（フラグ方式）───
_shutdown_flag = False

def handle_signal(sig, frame):
    global _shutdown_flag
    _shutdown_flag = True
    log.info(f"Signal {sig} received, shutdown flag set")

# ─── 致命的エラー通知（同期 / メインループ外で使用）───
def notify_fatal_error(message: str):
    """asyncio外でもTelegram通知できるようurllibを使用"""
    try:
        import urllib.request
        url = f"{TG_BASE}/sendMessage"
        payload = json.dumps({"chat_id": BENY_CHAT_ID, "text": f"🚨 致命的エラー:\n{message}"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

# ─── メイン ───
async def main():
    global STATE, _write_lock
    _write_lock = asyncio.Lock()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # state初期化
    STATE["status"] = "running"
    STATE["uptime_start"] = datetime.now(timezone.utc).isoformat()
    STATE["conversations_today"] = 0
    save_state(STATE)

    log.info("=" * 50)
    log.info("God AI v3.0 起動")
    log.info(f"Base: {BASE_DIR}")
    log.info(f"Gemini: Ready")
    log.info(f"Claude CLI: Ready")
    log.info(f"Telegram: Polling...")
    log.info("=" * 50)

    async with httpx.AsyncClient() as client:
        # 起動通知
        await tg_send(client, "🧠 God AI v3.0 起動完了\n脳: Gemini（日常） + Claude CLI（重い処理）\n/status で状態確認\n/reflect で即座に振り返り")

        # タスク起動（例外を検知するコールバック付き）
        def task_done_callback(task: asyncio.Task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                log.error(f"Task {task.get_name()} died with exception: {exc}", exc_info=exc)
                append_journal(f"### {datetime.now().strftime('%H:%M')} タスク異常終了: {task.get_name()}\n{exc}")

        poll_task = asyncio.create_task(polling_loop(client), name="polling")
        poll_task.add_done_callback(task_done_callback)
        reflect_task = asyncio.create_task(reflection_scheduler(client), name="reflection")
        reflect_task.add_done_callback(task_done_callback)

        log.info("タスク起動完了: polling, reflection")

        # シャットダウン待ち（フラグ方式）
        while not _shutdown_flag:
            await asyncio.sleep(1)

        log.info("Shutting down...")
        poll_task.cancel()
        reflect_task.cancel()

        await tg_send(client, "⏹️ God AI v3.0 停止します")

        STATE["status"] = "stopped"
        save_state(STATE)

    log.info("God AI v3.0 停止完了")

if __name__ == "__main__":
    MAX_RESTARTS = 3
    restart_count = 0

    while restart_count <= MAX_RESTARTS:
        try:
            if restart_count > 0:
                log.info(f"自動再起動 ({restart_count}/{MAX_RESTARTS})")
                notify_fatal_error(f"自動再起動 ({restart_count}/{MAX_RESTARTS})")
                time.sleep(5)
            asyncio.run(main())
            break
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt, exiting.")
            break
        except Exception as e:
            restart_count += 1
            err_msg = f"致命的エラー: {e}"
            log.error(err_msg, exc_info=True)
            append_journal(f"### {datetime.now().strftime('%H:%M')} {err_msg}")
            if restart_count > MAX_RESTARTS:
                notify_fatal_error(f"自動再起動上限({MAX_RESTARTS}回)に達しました。停止します。\nエラー: {e}")
                sys.exit(1)