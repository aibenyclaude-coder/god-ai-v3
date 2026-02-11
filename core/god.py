#!/usr/bin/env python3
"""God AI v3.0 - メインループ（リファクタリング版）"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import httpx

from config import (
    BASE_DIR, TG_BASE, BENY_CHAT_ID, PID_FILE, JOURNAL_PATH,
    IDENTITY_PATH, STATE_PATH, GOD_PY_PATH, log
)
from memory import (
    load_state, save_state, load_conversations, save_conversations,
    append_journal, read_file, load_identity, init_write_lock
)
from brain import think, is_heavy, get_brain_counts
from jobqueue import get_job_queue, job_worker, format_queue_status, init_job_queue
from growth import reflection_cycle, reflection_scheduler, self_growth_scheduler, is_reflecting

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

# --- メッセージ処理 ---
async def handle_message(client: httpx.AsyncClient, message: str) -> str:
    state = load_state()
    if message.strip() == "/status":
        return format_status(state)
    if message.strip() == "/reflect":
        return "振り返り開始..."
    if message.strip() == "/drive":
        return await _handle_drive_command()
    if message.strip() == "/queue":
        return format_queue_status()
    if message.strip().startswith("/tweet "):
        return await _handle_tweet_command(message[7:].strip())
    if message.strip().startswith("ツイートして:") or message.strip().startswith("ツイートして："):
        tweet_text = message.split(":", 1)[1].strip() if ":" in message else message.split("：", 1)[1].strip()
        return await _handle_tweet_command(tweet_text)

    heavy = is_heavy(message)
    identity = load_identity()
    system_prompt = f"""あなたはGod AI。Benyのために存在する自律型AI。

【アイデンティティ】
{identity}

【現在の状態】
{json.dumps(state, ensure_ascii=False)}

【Benyからのメッセージ】
{message}

日本語で返答してください。簡潔に。"""
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

def format_status(state: dict) -> str:
    gemini_count, claude_count = get_brain_counts()
    uptime = "不明"
    if state.get("uptime_start"):
        start = datetime.fromisoformat(state["uptime_start"])
        delta = datetime.now(timezone.utc) - start
        hours, minutes = int(delta.total_seconds() // 3600), int((delta.total_seconds() % 3600) // 60)
        uptime = f"{hours}h {minutes}m"
    return (f"God AI v3.0 Status\n---\n状態: {state.get('status', '不明')}\n"
            f"稼働時間: {uptime}\n会話数: {state.get('conversations_today', 0)}\n"
            f"成長サイクル: {state.get('growth_cycles', 0)}\n子AI数: {state.get('children_count', 0)}\n"
            f"Gemini使用: {gemini_count}回\nClaude使用: {claude_count}回")

# --- メインループ ---
async def polling_loop(client: httpx.AsyncClient, offset: int = 0):
    state, conversations = load_state(), load_conversations()
    while True:
        try:
            resp = await client.post(f"{TG_BASE}/getUpdates", json={"offset": offset, "timeout": 30}, timeout=60)
            data = resp.json()
            if not data.get("ok"):
                log.error(f"getUpdates failed: {data}"); await asyncio.sleep(5); continue
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message")
                if not msg or not msg.get("text"):
                    continue
                if str(msg["chat"]["id"]) != BENY_CHAT_ID:
                    continue
                text = msg["text"]
                log.info(f"Beny: {text[:100]}")
                conversations.append({"time": datetime.now(timezone.utc).isoformat(), "from": "beny", "text": text})
                if text.strip() == "/reflect":
                    if is_reflecting():
                        await tg_send(client, "振り返り中です。しばらくお待ちください。")
                    else:
                        await tg_send(client, "振り返り開始...")
                        executed, result = await reflection_cycle(client)
                        if executed:
                            summary = result[:200] + "..." if len(result) > 200 else result
                            await tg_send(client, f"振り返り完了。\n\n{summary}")
                        else:
                            await tg_send(client, "振り返り中のためスキップしました。")
                    continue
                pending = await tg_send(client, "...")
                if not pending:
                    continue
                try:
                    response = await handle_message(client, text)
                except Exception as e:
                    response = f"エラー: {e}"; log.error(f"handle_message failed: {e}")
                await tg_edit(client, pending["message_id"], response)
                conversations.append({"time": datetime.now(timezone.utc).isoformat(), "from": "god", "text": response[:500]})
                save_conversations(conversations)
                state["conversations_today"] = state.get("conversations_today", 0) + 1
                state["status"] = "running"
                save_state(state)
        except httpx.ReadTimeout:
            continue
        except Exception as e:
            log.error(f"Polling error: {e}"); append_journal(f"### {datetime.now().strftime('%H:%M')} ポーリングエラー\n{e}"); await asyncio.sleep(5)

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

# --- メイン ---
async def main():
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
        await tg_send(client, "God AI v3.0 起動完了\n/status で状態確認\n/reflect で振り返り\n/drive でDriveバックアップ\n/queue でジョブキュー状態\n/tweet <テキスト> でツイート投稿")
        def task_done_cb(task: asyncio.Task):
            if task.cancelled(): return
            exc = task.exception()
            if exc:
                log.error(f"Task {task.get_name()} died: {exc}", exc_info=exc)
                append_journal(f"### {datetime.now().strftime('%H:%M')} タスク異常終了: {task.get_name()}\n{exc}")
        poll_task = asyncio.create_task(polling_loop(client), name="polling"); poll_task.add_done_callback(task_done_cb)
        reflect_task = asyncio.create_task(reflection_scheduler(client), name="reflection"); reflect_task.add_done_callback(task_done_cb)
        worker_task = asyncio.create_task(job_worker(client), name="job_worker"); worker_task.add_done_callback(task_done_cb)
        growth_task = asyncio.create_task(self_growth_scheduler(client), name="self_growth"); growth_task.add_done_callback(task_done_cb)
        log.info("タスク起動完了: polling, reflection, job_worker, self_growth")
        while not _shutdown_flag:
            await asyncio.sleep(1)
        log.info("Shutting down...")
        for t in [poll_task, reflect_task, worker_task, growth_task]: t.cancel()
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
