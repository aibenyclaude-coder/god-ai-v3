#!/usr/bin/env python3
"""God AI v3.0 - ジョブキューモジュール"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Coroutine

from config import log
from memory import append_journal

# --- Priority Enum ---
class Priority(IntEnum):
    """ジョブ優先度（数値が小さいほど高優先度）"""
    P1_URGENT = 1    # 緊急: 会話応答
    P2_NORMAL = 2    # 通常: 振り返り
    P3_BACKGROUND = 3  # 背景: 自己改善


@dataclass(order=True)
class Job:
    """優先度付きジョブ"""
    priority: int
    created_at: float = field(compare=False)
    job_id: str = field(compare=False)
    job_type: str = field(compare=False)
    handler: Callable[..., Coroutine[Any, Any, Any]] = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    description: str = field(default="", compare=False)


class JobQueue:
    """asyncio.PriorityQueueベースのジョブキュー"""

    def __init__(self):
        self._queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._current_job: Job | None = None
        self._completed_count: dict[str, int] = {"P1": 0, "P2": 0, "P3": 0}
        self._failed_count: int = 0

    async def put(self, job: Job):
        """ジョブをキューに追加"""
        await self._queue.put(job)
        log.info(f"Job queued: {job.job_type} (P{job.priority}) - {job.description}")

    async def get(self) -> Job:
        """次のジョブを取得（優先度順）"""
        job = await self._queue.get()
        self._current_job = job
        return job

    def task_done(self):
        """現在のジョブ完了をマーク"""
        if self._current_job:
            priority_key = f"P{self._current_job.priority}"
            self._completed_count[priority_key] = self._completed_count.get(priority_key, 0) + 1
            self._current_job = None
        self._queue.task_done()

    def mark_failed(self):
        """ジョブ失敗をマーク"""
        self._failed_count += 1
        self._current_job = None

    def qsize(self) -> int:
        return self._queue.qsize()

    def get_status(self) -> dict:
        """キューの状態を返す"""
        return {
            "queue_size": self._queue.qsize(),
            "current_job": {
                "type": self._current_job.job_type,
                "priority": f"P{self._current_job.priority}",
                "description": self._current_job.description,
            } if self._current_job else None,
            "completed": self._completed_count.copy(),
            "failed": self._failed_count,
        }


# グローバルジョブキュー
_job_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """ジョブキューを取得"""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


def init_job_queue():
    """メインループで明示的にジョブキューを初期化"""
    global _job_queue
    _job_queue = JobQueue()


async def create_job(
    priority: Priority,
    job_type: str,
    handler: Callable[..., Coroutine[Any, Any, Any]],
    args: tuple = (),
    kwargs: dict | None = None,
    description: str = "",
) -> str:
    """ジョブを作成してキューに追加"""
    job = Job(
        priority=int(priority),
        created_at=time.time(),
        job_id=str(uuid.uuid4())[:8],
        job_type=job_type,
        handler=handler,
        args=args,
        kwargs=kwargs or {},
        description=description,
    )
    await get_job_queue().put(job)
    return job.job_id


async def job_worker(client):
    """ジョブキューからジョブを取得して実行"""
    log.info("ジョブワーカー開始")
    queue = get_job_queue()

    while True:
        try:
            job = await queue.get()
            log.info(f"ジョブ実行開始: {job.job_type} (P{job.priority}) - {job.description}")

            try:
                await job.handler(*job.args, **job.kwargs)
                queue.task_done()
                log.info(f"ジョブ完了: {job.job_type}")
            except Exception as e:
                queue.mark_failed()
                log.error(f"ジョブ失敗: {job.job_type} - {e}", exc_info=True)
                append_journal(f"### {datetime.now().strftime('%H:%M')} ジョブ失敗: {job.job_type}\n{e}")

        except asyncio.CancelledError:
            log.info("ジョブワーカー: キャンセルされました")
            raise
        except Exception as e:
            log.error(f"ジョブワーカーエラー: {e}", exc_info=True)
            await asyncio.sleep(5)


def format_queue_status() -> str:
    """ジョブキューの状態をフォーマット"""
    queue = get_job_queue()
    status = queue.get_status()

    current_job_str = "なし"
    if status["current_job"]:
        cj = status["current_job"]
        current_job_str = f"{cj['type']} ({cj['priority']})"

    completed = status["completed"]
    return (
        f"Job Queue Status\n"
        f"---\n"
        f"待機中ジョブ: {status['queue_size']}件\n"
        f"実行中: {current_job_str}\n"
        f"---\n"
        f"完了済み:\n"
        f"  P1 (緊急/会話): {completed.get('P1', 0)}件\n"
        f"  P2 (通常/振り返り): {completed.get('P2', 0)}件\n"
        f"  P3 (背景/自己改善): {completed.get('P3', 0)}件\n"
        f"失敗: {status['failed']}件"
    )
