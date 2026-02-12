#!/usr/bin/env python3
"""God AI v3.0 - ジョブキュー（データ層・永続化・分散対応）

ジョブ定義スキーマ:
{
  "task_id": "improve-abc123",
  "type": "self_improve | reflect | tweet | lp_create | conversation",
  "priority": "P1 | P2 | P3",
  "status": "queued | running | success | failed | cancelled",
  "worker_id": "local",
  "input": {
    "target_file": "core/brain.py",
    "read_files": [],
    "instruction": "",
    "constraints": []
  },
  "output": {
    "patches": [],
    "test_result": null,
    "changelog": "",
    "result_status": null
  },
  "meta": {
    "created_at": "ISO datetime",
    "started_at": null,
    "completed_at": null,
    "estimated_seconds": 60,
    "backup_path": null,
    "depends_on": [],
    "retry_count": 0,
    "max_retries": 3,
    "error_history": []
  }
}
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import log, JOB_QUEUE_PATH

# --- 優先度定数 ---
PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3}

# --- P1割り込みイベント ---
_p1_interrupt_event: asyncio.Event | None = None


def init_job_queue():
    """メインループで明示的に初期化"""
    global _p1_interrupt_event
    _p1_interrupt_event = asyncio.Event()


def get_p1_interrupt_event() -> asyncio.Event:
    global _p1_interrupt_event
    if _p1_interrupt_event is None:
        _p1_interrupt_event = asyncio.Event()
    return _p1_interrupt_event


def signal_p1_interrupt():
    """P1ジョブ到着を通知（P3タスクの中断用）"""
    evt = get_p1_interrupt_event()
    evt.set()
    log.info("P1割り込みシグナル発行")


def clear_p1_interrupt():
    """P1割り込みシグナルをクリア"""
    get_p1_interrupt_event().clear()


def is_p1_interrupted() -> bool:
    """P1割り込みが発生しているか"""
    return get_p1_interrupt_event().is_set()


# --- ファイルベース永続化 ---
def load_queue() -> list[dict]:
    """job_queue.json からジョブ一覧を読み込む。
    整合性チェックを強化し、不正なエントリはログに記録して除外する。
    """
    if JOB_QUEUE_PATH.exists():
        try:
            data = json.loads(JOB_QUEUE_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                log.warning(f"ジョブキューのフォーマットが不正です: {JOB_QUEUE_PATH}")
                return []

            valid_jobs = []
            for i, job in enumerate(data):
                if not isinstance(job, dict):
                    log.warning(f"ジョブキューの {i} 番目のエントリは不正な型です: {job}")
                    continue

                # 必須キーのチェック
                required_keys = ["task_id", "type", "priority", "status", "input", "output", "meta"]
                if not all(key in job for key in required_keys):
                    log.warning(f"ジョブキューの {i} 番目のエントリに必要なキーが不足しています: {job.get('task_id', 'N/A')}")
                    continue

                # input内の必須キーチェック
                input_job = job.get("input", {})
                if not all(key in input_job for key in ["target_file", "read_files", "instruction", "constraints"]):
                    log.warning(f"ジョブキューの {i} 番目のエントリの input に必要なキーが不足しています: {job.get('task_id', 'N/A')}")
                    continue

                # meta内の必須キーチェック
                meta_job = job.get("meta", {})
                if not all(key in meta_job for key in ["created_at", "estimated_seconds", "retry_count", "max_retries", "error_history"]):
                    log.warning(f"ジョブキューの {i} 番目のエントリの meta に必要なキーが不足しています: {job.get('task_id', 'N/A')}")
                    continue

                # 型チェック (一部)
                if not isinstance(job["task_id"], str) or not isinstance(job["type"], str) or not isinstance(job["priority"], str) or not isinstance(job["status"], str):
                    log.warning(f"ジョブキューの {i} 番目のエントリの必須文字列フィールドの型が不正です: {job.get('task_id', 'N/A')}")
                    continue
                if not isinstance(job["input"].get("read_files"), list) or not isinstance(job["input"].get("instruction"), str) or not isinstance(job["input"].get("constraints"), list):
                    log.warning(f"ジョブキューの {i} 番目のエントリの input フィールドの型が不正です: {job.get('task_id', 'N/A')}")
                    continue
                if not isinstance(job["meta"].get("error_history"), list):
                    log.warning(f"ジョブキューの {i} 番目のエントリの meta.error_history の型が不正です: {job.get('task_id', 'N/A')}")
                    continue

                valid_jobs.append(job)

            if len(valid_jobs) < len(data):
                # 整形して保存し直すことで、不正なエントリを削除
                save_queue(valid_jobs)

            return valid_jobs

        except json.JSONDecodeError as e:
            log.warning(f"ジョブキューのJSONデコード失敗: {e} in {JOB_QUEUE_PATH}")
        except FileNotFoundError:
            pass  # ファイルが存在しない場合は空リストを返す
        except Exception as e:
            log.error(f"ジョブキュー読み込み中に予期せぬエラー: {e}", exc_info=True)

    return []


def save_queue(jobs: list[dict]):
    """save_queue.py にジョブ一覧を保存（最新200件）
    atomic writeを実装し、書き込み中のプロセス中断によるデータ破損を防ぐ。
    """
    if len(jobs) > 200:
        jobs = jobs[-200:]

    temp_path = JOB_QUEUE_PATH.with_suffix(".json.tmp")
    try:
        # 一時ファイルに書き込む
        temp_path.write_text(
            json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 既存ファイルを置き換える（アトミック操作）
        temp_path.replace(JOB_QUEUE_PATH)
    except Exception as e:
        log.error(f"ジョブキュー保存失敗: {e}")
        # 一時ファイルが残っていたら削除
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as rm_e:
                log.error(f"一時ジョブキューファイルの削除失敗: {rm_e}")


# --- CRUD操作 ---
def is_duplicate_job(job_type: str, target_file: str, instruction: str) -> bool:
    """同じtarget_fileかつ同じinstructionのジョブがqueued/runningで既にあるかチェック"""
    if not target_file or not instruction:
        return False
    jobs = load_queue()
    for j in jobs:
        if j["status"] not in ("queued", "running"):
            continue
        if j["type"] != job_type:
            continue
        j_target = j.get("input", {}).get("target_file", "")
        j_instruction = j.get("input", {}).get("instruction", "")
        if j_target == target_file and j_instruction == instruction:
            return True
    return False


def get_queued_job_summaries() -> list[dict]:
    """キュー内のqueued/runningジョブのサマリーを返す（振り返りプロンプト用）"""
    jobs = load_queue()
    summaries = []
    for j in jobs:
        if j["status"] in ("queued", "running"):
            summaries.append({
                "target_file": j.get("input", {}).get("target_file", ""),
                "instruction": j.get("input", {}).get("instruction", "")[:100],
                "status": j["status"],
            })
    return summaries


def get_recent_failed_jobs(limit: int = 3) -> list[dict]:
    """直近の失敗ジョブのサマリーを返す（振り返りプロンプト用）"""
    jobs = load_queue()
    failed = [j for j in jobs if j["status"] == "failed"]
    failed.sort(key=lambda j: j.get("meta", {}).get("completed_at", ""), reverse=True)
    summaries = []
    for j in failed[:limit]:
        last_error = ""
        errors = j.get("meta", {}).get("error_history", [])
        if errors:
            last_error = errors[-1].get("error", "")[:100]
        summaries.append({
            "target_file": j.get("input", {}).get("target_file", ""),
            "instruction": j.get("input", {}).get("instruction", "")[:100],
            "error": last_error,
        })
    return summaries


def create_job(
    job_type: str,
    priority: str = "P3",
    target_file: str = "",
    read_files: list[str] | None = None,
    instruction: str = "",
    constraints: list[str] | None = None,
    estimated_seconds: int = 60,
    max_retries: int = 3,
    depends_on: list[str] | None = None,
    worker_id: str = "local",
) -> dict | None:
    """ジョブを作成してキューに追加。作成されたジョブ辞書を返す。重複時はNone。"""
    # 重複ジョブ防止
    if is_duplicate_job(job_type, target_file, instruction):
        log.info(f"重複ジョブのためスキップ: {target_file}")
        return None

    task_id = f"{job_type}-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    job = {
        "task_id": task_id,
        "type": job_type,
        "priority": priority,
        "status": "queued",
        "worker_id": worker_id,
        "input": {
            "target_file": target_file,
            "read_files": read_files or [],
            "instruction": instruction,
            "constraints": constraints or [],
        },
        "output": {
            "patches": [],
            "test_result": None,
            "changelog": "",
            "result_status": None,
        },
        "meta": {
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "estimated_seconds": estimated_seconds,
            "backup_path": None,
            "depends_on": depends_on or [],
            "retry_count": 0,
            "max_retries": max_retries,
            "error_history": [],
        },
    }

    jobs = load_queue()
    jobs.append(job)
    save_queue(jobs)
    log.info(f"ジョブ作成: {task_id} type={job_type} priority={priority} target={target_file}")

    # P1が来たらP3を中断
    if priority == "P1":
        signal_p1_interrupt()

    return job


def get_job(task_id: str) -> dict | None:
    """task_idでジョブを取得"""
    for job in load_queue():
        if job["task_id"] == task_id:
            return job
    return None


def update_job(task_id: str, **updates) -> dict | None:
    """ジョブを更新。更新後のジョブを返す。

    使用例:
        update_job("improve-abc123", status="running")
        update_job("improve-abc123", status="success", output={...})
    """
    jobs = load_queue()
    for job in jobs:
        if job["task_id"] == task_id:
            for key, value in updates.items():
                if key in ("status",):
                    job[key] = value
                    if value == "running":
                        job["meta"]["started_at"] = datetime.now(timezone.utc).isoformat()
                    elif value in ("success", "failed", "cancelled"):
                        job["meta"]["completed_at"] = datetime.now(timezone.utc).isoformat()
                elif key == "output":
                    if isinstance(value, dict):
                        job["output"].update(value)
                elif key == "worker_id":
                    job["worker_id"] = value
                elif key == "error":
                    job["meta"]["error_history"].append({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "error": str(value),
                    })
                    job["meta"]["retry_count"] = job["meta"].get("retry_count", 0) + 1
                elif key == "backup_path":
                    job["meta"]["backup_path"] = value
            save_queue(jobs)
            return job
    log.warning(f"ジョブ更新失敗: {task_id} が見つかりません")
    return None


def get_next_queued_job() -> dict | None:
    """優先度が最も高い(P1>P2>P3)、statusがqueuedのジョブを取得。
    depends_onが未完了のジョブはスキップ。
    同一target_fileでrunning中のジョブがあるものもスキップ（ファイルロック）。
    """
    jobs = load_queue()
    queued = [j for j in jobs if j["status"] == "queued"]
    if not queued:
        return None

    # 依存チェック: depends_onの全ジョブがsuccess/cancelledであること
    completed_ids = {j["task_id"] for j in jobs if j["status"] in ("success", "cancelled")}
    running_targets = {
        j["input"]["target_file"]
        for j in jobs
        if j["status"] == "running" and j["input"].get("target_file")
    }

    eligible = []
    for job in queued:
        deps = job["meta"].get("depends_on", [])
        if deps and not all(d in completed_ids for d in deps):
            continue
        # ファイルロック: running中のジョブと同じtarget_fileは待機
        tf = job["input"].get("target_file", "")
        if tf and tf in running_targets:
            continue
        eligible.append(job)

    if not eligible:
        return None

    # 優先度順ソート (P1=1, P2=2, P3=3)、同優先度ならcreated_at順
    eligible.sort(key=lambda j: (PRIORITY_ORDER.get(j["priority"], 9), j["meta"]["created_at"]))
    return eligible[0]


def is_file_locked(target_file: str) -> bool:
    """指定target_fileがrunning中のジョブで使用されているか"""
    if not target_file:
        return False
    jobs = load_queue()
    return any(
        j["status"] == "running" and j["input"].get("target_file") == target_file
        for j in jobs
    )


def get_recent_jobs(limit: int = 10) -> list[dict]:
    """直近のジョブ一覧を返す（新しい順）"""
    jobs = load_queue()
    return list(reversed(jobs[-limit:]))


def count_by_status() -> dict[str, int]:
    """ステータスごとのジョブ数を返す"""
    jobs = load_queue()
    counts: dict[str, int] = {}
    for job in jobs:
        s = job.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    return counts


# --- 表示用フォーマッタ ---
def format_queue_status() -> str:
    """/queue コマンド用: キューの状態表示"""
    jobs = load_queue()
    counts = count_by_status()

    # 実行中ジョブ
    running = [j for j in jobs if j["status"] == "running"]
    running_str = "なし"
    if running:
        rj = running[0]
        running_str = f"{rj['type']} ({rj['priority']}) [{rj['task_id']}@{rj['worker_id']}]"

    queued_count = counts.get("queued", 0)
    success_count = counts.get("success", 0)
    failed_count = counts.get("failed", 0)

    return (
        f"Job Queue Status\n"
        f"---\n"
        f"待機中: {queued_count}件\n"
        f"実行中: {running_str}\n"
        f"成功: {success_count}件\n"
        f"失敗: {failed_count}件\n"
        f"合計: {len(jobs)}件"
    )


def format_jobs_list() -> str:
    """/jobs コマンド用: 直近10件のジョブ一覧"""
    recent = get_recent_jobs(10)
    if not recent:
        return "ジョブ履歴なし"

    lines = ["直近のジョブ一覧\n---"]
    for job in recent:
        status_icon = {
            "queued": "--", "running": "..", "success": "OK",
            "failed": "NG", "cancelled": "XX"
        }.get(job["status"], "??")

        # 所要時間計算
        duration = ""
        meta = job.get("meta", {})
        if meta.get("started_at") and meta.get("completed_at"):
            try:
                start = datetime.fromisoformat(meta["started_at"])
                end = datetime.fromisoformat(meta["completed_at"])
                secs = int((end - start).total_seconds())
                duration = f" {secs}s"
            except Exception:
                pass

        target = job["input"].get("target_file", "")
        target_str = f" -> {target}" if target else ""
        lines.append(
            f"[{status_icon}] {job['task_id']} | {job['type']} ({job['priority']}){target_str}{duration}"
        )

    return "\n".join(lines)
