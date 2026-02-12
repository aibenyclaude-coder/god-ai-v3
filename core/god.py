#!/usr/bin/env python3
"""God AI v3.0 - メインループ（リファクタリング版）"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

import re
from config import (
    BASE_DIR, MEMORY_DIR, TG_BASE, BENY_CHAT_ID, PID_FILE, JOURNAL_PATH,
    IDENTITY_PATH, STATE_PATH, GOD_PY_PATH, log
)
from memory import (
    load_state, save_state, load_conversations, save_conversations,
    append_journal, read_file, load_identity, init_write_lock
)
from brain import think, is_heavy, get_brain_counts, detect_action_intent, AIUnavailable, is_ai_paused, get_ai_pause_remaining
from jobqueue import format_queue_status, format_jobs_list, init_job_queue, signal_p1_interrupt
from job_worker import job_worker_loop
from growth import reflection_cycle, reflection_scheduler, self_growth_scheduler, is_reflecting, get_stats_summary, get_auto_suggestions
from gmail import gmail_check_scheduler, is_configured as gmail_is_configured

# --- PIDファイルによる重複プロセス防止 ---
def check_single_instance():
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            log.warning(f"旧プロセス(PID={old_pid})が残存。停止します...")
            os.kill(old_pid, signal.SIGTERM)
            time.sleep(3)
            try:
                os.kill(old_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            log.info(f"旧プロセス(PID={old_pid})を停止しました")
        except ProcessLookupError:
            log.info(f"旧PIDファイルあり(PID={old_pid})だがプロセスは既に終了")
        except ValueError:
            log.warning("PIDファイルの内容が不正。削除します。")
        except Exception as e:
            log.error(f"旧プロセス確認エラー: {e}")
    PID_FILE.write_text(str(os.getpid()))
    log.info(f"PIDファイル作成: {PID_FILE} (PID={os.getpid()})")

# --- Telegram API ---
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
    return await tg_request(client, "editMessageText", chat_id=BENY_CHAT_ID, message_id=msg_id, text=text)

# --- 会話履歴フォーマット ---
def format_recent_history(conversations: list, limit: int = 10) -> str:
    """直近の会話・アクション履歴をフォーマット（新しい順）"""
    if not conversations:
        return "(履歴なし)"
    recent = conversations[-limit:][::-1]  # 新しい順
    lines = []
    for conv in recent:
        try:
            # タイムスタンプをパース
            ts = conv.get("time", "")
            if ts:
                from datetime import datetime
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M")
            else:
                time_str = "??:??"

            sender = conv.get("from", "unknown")
            text = conv.get("text", "")[:200]  # 長すぎる場合は切り詰め

            if sender == "beny":
                lines.append(f"[{time_str}] Beny: {text}")
            elif sender == "god":
                lines.append(f"[{time_str}] God AI: {text}")
            elif sender == "system":
                lines.append(f"[{time_str}] [システム] {text}")
        except Exception:
            continue
    return "\n".join(lines) if lines else "(履歴なし)"

def record_system_action(conversations: list, action_text: str):
    """システムアクション（ツイート、振り返り等）を会話履歴に記録"""
    conversations.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "from": "system",
        "text": action_text[:300]
    })

# --- 簡易応答パターン（AI不要） ---
QUICK_RESPONSES = {
    "おはよう": "おはよう！今日も成長するぞ 🌅",
    "おやすみ": "おやすみ！寝てる間も成長し続ける 🌙",
    "こんにちは": "こんにちは！何か指示ある？",
    "こんばんは": "こんばんは！夜も稼働中 🌃",
    "ありがとう": "どういたしまして！💪",
    "OK": "✅",
    "ok": "✅",
    "Ok": "✅",
    "いいね": "✅ ありがとう！",
    "了解": "✅",
    "わかった": "✅ 了解！",
}

QUICK_PATTERNS = [
    ("投稿完了", "✅ 確認した！"),
    ("成功", "✅ いいぞ！"),
    ("エラー", "⚠️ ログ確認する。詳細教えて"),
    ("失敗", "⚠️ 何が起きた？詳細教えて"),
]

# --- メッセージ処理 ---
async def handle_message(client: httpx.AsyncClient, message: str) -> str:
    state = load_state()
    msg_stripped = message.strip()

    # /コマンド処理
    if msg_stripped == "/status":
        return format_status(state)
    if msg_stripped == "/reflect":
        return "振り返り開始..."
    if msg_stripped == "/drive":
        return await _handle_drive_command()
    if msg_stripped == "/queue":
        return format_queue_status()
    if msg_stripped == "/jobs":
        return format_jobs_list()
    if msg_stripped == "/stats":
        return _handle_stats_command()
    if msg_stripped.startswith("/tweet "):
        return await _handle_tweet_command(msg_stripped[7:].strip())
    if msg_stripped.startswith("ツイートして:") or msg_stripped.startswith("ツイートして："):
        tweet_text = message.split(":", 1)[1].strip() if ":" in message else message.split("：", 1)[1].strip()
        return await _handle_tweet_command(tweet_text)

    # ファイル追記: 「<ファイル名>に追記: <内容>」
    append_match = re.match(r'^(.+\.md)に追記[:：]\s*(.+)$', msg_stripped, re.DOTALL)
    if append_match:
        return _handle_file_append(append_match.group(1), append_match.group(2))

    # 簡易応答（完全一致）
    if msg_stripped in QUICK_RESPONSES:
        log.info(f"簡易応答: {msg_stripped} -> {QUICK_RESPONSES[msg_stripped]}")
        return QUICK_RESPONSES[msg_stripped]

    # 簡易応答（部分一致）- 短いメッセージのみ
    if len(msg_stripped) < 30:
        for pattern, response in QUICK_PATTERNS:
            if pattern in msg_stripped:
                log.info(f"簡易応答(部分一致): {msg_stripped} -> {response}")
                return response

    # アクション意図を検出
    action_intent = detect_action_intent(message)
    heavy = is_heavy(message)
    identity = load_identity()
    conversations = load_conversations()

    # 直近10件の会話・アクション履歴
    recent_history = format_recent_history(conversations, limit=10)

    # アクション意図に応じてプロンプトを調整
    action_guidance = ""
    if action_intent["needs_action"]:
        action_guidance = f"""
【検出されたアクション意図】
- アクション種類: {action_intent["action_type"]}
- 対象: {action_intent["target"]}
- 注意: 実際にアクションが必要です。単なる説明ではなく、具体的な操作を実行してください。"""
    else:
        action_guidance = """
【注意】
質問・説明のリクエストとして検出されました。情報提供で応答してください。"""

    system_prompt = f"""お前はGod AI。自律型AI。

【自己認識】
- [システム] で始まる記録はお前自身が実行したアクション
- ツイート投稿はお前が/tweetコマンドで実行した
- 振り返りはお前が自分で実行した
- 自己改善はお前が自分のコードを修正した結果
- Benyはお前の開発者でありオーナー
- お前はBenyの指示で動くが、自分でも判断して行動する
- 「Benyが投稿した」ではなく「自分が投稿した」と理解しろ

【アイデンティティ】
{identity}

【直近の出来事と会話（新しい順）】
{recent_history}

【現在の状態】
{json.dumps(state, ensure_ascii=False)}
{action_guidance}

【Benyからのメッセージ】
{message}

日本語で返答。簡潔に。"""
    response_text, brain_name = await think(system_prompt, heavy=heavy)
    return f"{response_text}\n\n[brain: {brain_name}]"

async def _handle_drive_command() -> str:
    try:
        from gdrive import upload_file, is_configured
        if not is_configured():
            return "Google Drive未設定\nclient_secret.json を core/ に配置してください"
        results = []
        for path, desc in [(JOURNAL_PATH, "journal.md"), (STATE_PATH, "state.json"),
                           (IDENTITY_PATH, "identity.md"), (GOD_PY_PATH, "god.py")]:
            r = upload_file(str(path))
            results.append(f"{'OK' if r else 'FAIL'} {desc}")
        return "Google Drive バックアップ\n" + "\n".join(results)
    except ImportError:
        return "gdrive.py が見つかりません"
    except Exception as e:
        return f"Driveバックアップエラー: {e}"

async def _handle_tweet_command(tweet_text: str) -> str:
    try:
        from twitter import post_tweet, is_configured, get_setup_instructions
        if not is_configured():
            return f"Twitter API未設定\n\n{get_setup_instructions()}"
        if not tweet_text:
            return "ツイート本文を指定してください\n使い方: /tweet <テキスト>"
        result = post_tweet(tweet_text)
        if result["success"]:
            return f"ツイート投稿完了!\n{result['url']}"
        else:
            return f"ツイート投稿失敗\n{result['error']}"
    except ImportError:
        return "twitter.py が見つかりません"
    except Exception as e:
        return f"ツイートエラー: {e}"

def _handle_stats_command() -> str:
    """成長統計を表示"""
    try:
        summary = get_stats_summary()
        suggestions = get_auto_suggestions()

        result = f"📊 成長統計\n{summary}"

        if suggestions:
            result += "\n\n💡 自動提案:\n"
            for i, suggestion in enumerate(suggestions, 1):
                result += f"{i}. {suggestion}\n"

        return result
    except Exception as e:
        log.error(f"統計取得エラー: {e}")
        return f"統計取得エラー: {e}"

def _handle_file_append(filename: str, content: str) -> str:
    """memory/配下の.mdファイルに追記"""
    try:
        # セキュリティ: ファイル名のサニタイズ
        safe_filename = filename.replace("/", "").replace("\\", "").replace("..", "")
        if not safe_filename.endswith(".md"):
            return "❌ .mdファイルのみ対応しています"

        target_path = MEMORY_DIR / safe_filename

        # memory/配下であることを確認（パストラバーサル防止）
        try:
            target_path.resolve().relative_to(MEMORY_DIR.resolve())
        except ValueError:
            return "❌ memory/配下のファイルのみ編集可能です"

        # ファイルが存在しない場合は新規作成
        if not target_path.exists():
            target_path.write_text(f"# {safe_filename.replace('.md', '')}\n\n", encoding="utf-8")

        # 追記
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}\n")

        log.info(f"ファイル追記完了: {safe_filename}")
        return f"✅ {safe_filename}に追記完了"

    except Exception as e:
        log.error(f"ファイル追記エラー: {e}")
        return f"❌ 追記エラー: {e}"

def format_status(state: dict) -> str:
    gemini_count, glm_count, claude_count = get_brain_counts()
    uptime = "不明"
    if state.get("uptime_start"):
        start = datetime.fromisoformat(state["uptime_start"])
        delta = datetime.now(timezone.utc) - start
        hours, minutes = int(delta.total_seconds() // 3600), int((delta.total_seconds() % 3600) // 60)
        uptime = f"{hours}h {minutes}m"
    return (f"God AI v3.0 Status\n---\n状態: {state.get('status', '不明')}\n"
            f"稼働時間: {uptime}\n会話数: {state.get('conversations_today', 0)}\n"
            f"成長サイクル: {state.get('growth_cycles', 0)}\n子AI数: {state.get('children_count', 0)}\n"
            f"Gemini: {gemini_count}回 | GLM: {glm_count}回 | Claude: {claude_count}回")

# --- メインループ ---
async def polling_loop(client: httpx.AsyncClient, offset: int = 0):
    state, conversations = load_state(), load_conversations()
    retry_delay = 5  # Initial retry delay in seconds
    max_retry_delay = 60  # Maximum retry delay

    while True:
        try:
            # Check if AI is paused and skip polling if necessary
            if is_ai_paused():
                pause_remaining = get_ai_pause_remaining()
                log.info(f"AI is paused. Skipping polling for {pause_remaining:.0f} seconds.")
                await asyncio.sleep(min(pause_remaining, 60)) # Sleep for at most 60 seconds or remaining pause time
                continue

            resp = await client.post(f"{TG_BASE}/getUpdates", json={"offset": offset, "timeout": 30}, timeout=60)
            resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            data = resp.json()

            if not data.get("ok"):
                log.error(f"getUpdates failed: {data}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue

            # Reset retry delay if successful
            retry_delay = 5

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message")
                if not msg or not msg.get("text"):
                    continue
                if str(msg["chat"]["id"]) != BENY_CHAT_ID:
                    continue
                text = msg["text"]
                log.info(f"Beny: {text[:100]}")
                # P1割り込みシグナル発行（自己改善を中断させる）
                signal_p1_interrupt()
                conversations.append({"time": datetime.now(timezone.utc).isoformat(), "from": "beny", "text": text})

                # Handle reflective commands first
                if text.strip() == "/reflect":
                    if is_reflecting():
                        await tg_send(client, "振り返り中です。しばらくお待ちください。")
                    else:
                        await tg_send(client, "振り返り開始...")
                        executed, result = await reflection_cycle(client)
                        if executed:
                            summary = result[:1000] + "..." if len(result) > 1000 else result
                            await tg_send(client, f"振り返り完了。\n\n{summary}")
                            # 振り返り結果をシステムアクションとして記録
                            record_system_action(conversations, f"振り返り完了: {summary[:200]}")
                            save_conversations(conversations)
                        else:
                            await tg_send(client, "振り返り中のためスキップしました。")
                    continue

                # Attempt to handle general messages via AI
                pending_message_id = None
                try:
                    pending = await tg_send(client, "...")
                    if not pending:
                        continue
                    pending_message_id = pending["message_id"]

                    response = await handle_message(client, text)
                except AIUnavailable as e:
                    # Gemini 429 + Claude CLIセッション切れ → 特別な通知
                    response = f"⚠️ {e}\n\nBeny、ターミナルで以下を実行:\n`/opt/homebrew/bin/claude setup-token`"
                    log.error(f"AIUnavailable: {e}")
                    record_system_action(conversations, f"AI停止: {e}")
                except Exception as e:
                    response = f"エラー: {e}"; log.error(f"handle_message failed: {e}", exc_info=True)
                    # エラーをシステムアクションとして記録
                    record_system_action(conversations, f"エラー発生: {e}")
                    # If an error occurred and we sent a placeholder message, edit it with the error
                    if pending_message_id:
                        await tg_edit(client, pending_message_id, response)
                    else: # If no placeholder was sent, send error as new message
                        await tg_send(client, response)
                    continue # Continue to the next update

                if pending_message_id:
                    await tg_edit(client, pending_message_id, response)

                conversations.append({"time": datetime.now(timezone.utc).isoformat(), "from": "god", "text": response[:500]})
                # ツイート投稿成功をシステムアクションとして記録
                if "ツイート投稿完了" in response:
                    record_system_action(conversations, f"ツイート投稿: {response}")
                # 自己改善成功をシステムアクションとして記録
                if "自己改善成功" in response:
                    record_system_action(conversations, f"自己改善: {response[:200]}")
                save_conversations(conversations)
                state["conversations_today"] = state.get("conversations_today", 0) + 1
                state["status"] = "running"
                save_state(state)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            log.error(f"Network/HTTP error during polling: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
        except Exception as e:
            log.error(f"Unexpected error during polling: {e}", exc_info=True)
            append_journal(f"### {datetime.now().strftime('%H:%M')} ポーリングエラー\n{e}")
            # For unexpected errors, also implement a retry delay to prevent rapid failure
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)

# --- シグナルハンドラ ---
_shutdown_flag = False

def handle_signal(sig, frame):
    global _shutdown_flag
    _shutdown_flag = True
    log.info(f"Signal {sig} received, shutdown flag set")
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass

def notify_fatal_error(message: str):
    try:
        import urllib.request
        url = f"{TG_BASE}/sendMessage"
        payload = json.dumps({"chat_id": BENY_CHAT_ID, "text": f"致命的エラー:\n{message}"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

# --- ログローテーション ---
def rotate_logs():
    """起動時にログファイルをローテーション（最大3世代）"""
    import shutil
    log_path = Path("/tmp/godai_v3.log")
    if log_path.exists():
        # 既存ログをローテーション
        for i in range(2, 0, -1):
            old = Path(f"/tmp/godai_v3.log.{i}")
            new = Path(f"/tmp/godai_v3.log.{i+1}")
            if old.exists():
                if i == 2:
                    old.unlink()  # 3世代目は削除
                else:
                    shutil.move(old, new)
        shutil.move(log_path, Path("/tmp/godai_v3.log.1"))
        log.info("ログローテーション完了")


# --- メイン ---
async def main():
    # rotate_logs()  # nohupと競合するため無効化。起動スクリプトでローテーションする
    init_write_lock()
    init_job_queue()
    check_single_instance()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    state = load_state()
    state["status"] = "running"
    state["uptime_start"] = datetime.now(timezone.utc).isoformat()
    state["conversations_today"] = 0
    save_state(state)
    log.info("=" * 50)
    log.info("God AI v3.0 起動")
    log.info(f"Base: {BASE_DIR}")
    log.info("=" * 50)
    async with httpx.AsyncClient() as client:
        await tg_send(client, "God AI v3.0 起動完了\n/status で状態確認\n/reflect で振り返り\n/drive でDriveバックアップ\n/queue でジョブキュー状態\n/jobs でジョブ一覧\n/stats で成長統計\n/tweet <テキスト> でツイート投稿")
        def task_done_cb(task: asyncio.Task):
            if task.cancelled(): return
            exc = task.exception()
            if exc:
                log.error(f"Task {task.get_name()} died: {exc}", exc_info=exc)
                append_journal(f"### {datetime.now().strftime('%H:%M')} タスク異常終了: {task.get_name()}\n{exc}")
        poll_task = asyncio.create_task(polling_loop(client), name="polling"); poll_task.add_done_callback(task_done_cb)
        reflect_task = asyncio.create_task(reflection_scheduler(client), name="reflection"); reflect_task.add_done_callback(task_done_cb)
        worker_task = asyncio.create_task(job_worker_loop(client), name="job_worker"); worker_task.add_done_callback(task_done_cb)
        growth_task = asyncio.create_task(self_growth_scheduler(client), name="self_growth"); growth_task.add_done_callback(task_done_cb)
        # Gmail監視（ココナラ通知転送）
        gmail_task = None
        if gmail_is_configured():
            gmail_task = asyncio.create_task(gmail_check_scheduler(client, interval=120), name="gmail_monitor")
            gmail_task.add_done_callback(task_done_cb)
            log.info("Gmail監視タスク起動")
        else:
            log.info("Gmail未設定のためスキップ。python3 gmail.py で初期設定してください。")
        log.info("タスク起動完了: polling, reflection, job_worker, self_growth" + (", gmail_monitor" if gmail_task else ""))
        while not _shutdown_flag:
            await asyncio.sleep(1)
        log.info("Shutting down...")
        tasks_to_cancel = [poll_task, reflect_task, worker_task, growth_task]
        if gmail_task:
            tasks_to_cancel.append(gmail_task)
        for t in tasks_to_cancel: t.cancel()
        await tg_send(client, "God AI v3.0 停止します")
        state["status"] = "stopped"
        save_state(state)
        try:
            if PID_FILE.exists(): PID_FILE.unlink()
        except Exception: pass
    log.info("God AI v3.0 停止完了")

if __name__ == "__main__":
    MAX_RESTARTS, restart_count = 3, 0
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
            log.error(f"致命的エラー: {e}", exc_info=True)
            append_journal(f"### {datetime.now().strftime('%H:%M')} 致命的エラー: {e}")
            if restart_count > MAX_RESTARTS:
                notify_fatal_error(f"自動再起動上限({MAX_RESTARTS}回)に達しました。停止します。\nエラー: {e}")
                sys.exit(1)
