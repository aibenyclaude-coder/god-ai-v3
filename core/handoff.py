#!/usr/bin/env python3
"""God AI v3.0 - HANDOFF.md自動生成モジュール

HANDOFFファイルは新しいセッションやエージェントへの引き継ぎ用。
振り返りサイクルの末尾で自動生成される。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from config import (
    IDENTITY_PATH, STATE_PATH, JOURNAL_PATH, MEMORY_DIR, log
)

HANDOFF_PATH = MEMORY_DIR / "HANDOFF.md"


def _read_identity_summary(max_lines: int = 50) -> str:
    """identity.mdの要約（最初のmax_lines行）を取得"""
    try:
        if not IDENTITY_PATH.exists():
            return "(identity.md not found)"
        lines = IDENTITY_PATH.read_text(encoding="utf-8").splitlines()
        summary_lines = lines[:max_lines]
        if len(lines) > max_lines:
            summary_lines.append(f"\n... (残り {len(lines) - max_lines} 行省略)")
        return "\n".join(summary_lines)
    except Exception as e:
        log.error(f"Failed to read identity.md: {e}")
        return f"(Error reading identity.md: {e})"


def _read_state() -> dict:
    """state.jsonを読み込み"""
    try:
        if not STATE_PATH.exists():
            return {}
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Failed to read state.json: {e}")
        return {"error": str(e)}


def _read_journal_tail(lines: int = 20) -> str:
    """journal.mdの直近lines行を取得"""
    try:
        if not JOURNAL_PATH.exists():
            return "(journal.md not found)"
        all_lines = JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
        tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "\n".join(tail_lines)
    except Exception as e:
        log.error(f"Failed to read journal.md: {e}")
        return f"(Error reading journal.md: {e})"


def _extract_completed_tasks(state: dict) -> list[str]:
    """stateから完了タスク一覧を抽出"""
    tasks = []

    # growth_cyclesから推測
    cycles = state.get("growth_cycles", 0)
    if cycles > 0:
        tasks.append(f"振り返りサイクル: {cycles}回完了")

    # conversations_todayから
    convos = state.get("conversations_today", 0)
    if convos > 0:
        tasks.append(f"今日の会話: {convos}件処理")

    # current_taskがあれば表示
    current = state.get("current_task")
    if current:
        tasks.append(f"現在のタスク: {current}")

    # completed_tasksキーがあれば追加
    completed = state.get("completed_tasks", [])
    if isinstance(completed, list):
        tasks.extend(completed)

    return tasks if tasks else ["(完了タスクなし)"]


def _extract_issues_and_todos(journal_text: str) -> tuple[list[str], list[str]]:
    """journalから既知問題とTODOを抽出"""
    issues = []
    todos = []

    lines = journal_text.splitlines()

    for line in lines:
        line_lower = line.lower()

        # 問題・エラー検出
        if any(kw in line_lower for kw in ["エラー", "失敗", "問題", "error", "failed", "bug"]):
            # 過度に長い行は短縮
            issue = line.strip()
            if len(issue) > 150:
                issue = issue[:150] + "..."
            if issue and issue not in issues:
                issues.append(issue)

        # TODO検出
        if any(kw in line_lower for kw in ["todo", "次に", "すべき", "必要", "should"]):
            todo = line.strip()
            if len(todo) > 150:
                todo = todo[:150] + "..."
            if todo and todo not in todos:
                todos.append(todo)

        # CODE_IMPROVEMENT検出（未実装の改善提案）
        if "CODE_IMPROVEMENT:" in line:
            improvement = line.split("CODE_IMPROVEMENT:", 1)[1].strip()
            if len(improvement) > 150:
                improvement = improvement[:150] + "..."
            if improvement and improvement not in todos:
                todos.append(f"[改善提案] {improvement}")

    # 最新の5件に絞る
    issues = issues[-5:] if len(issues) > 5 else issues
    todos = todos[-5:] if len(todos) > 5 else todos

    return issues, todos


def generate() -> str:
    """HANDOFF.mdを生成してパスを返す"""
    log.info("HANDOFF.md生成開始")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 各セクションのデータ収集
    identity_summary = _read_identity_summary(50)
    state = _read_state()
    journal_tail = _read_journal_tail(20)
    completed_tasks = _extract_completed_tasks(state)
    issues, todos = _extract_issues_and_todos(journal_tail)

    # HANDOFF.md組み立て
    content = f"""# God AI HANDOFF

**生成日時**: {now}

---

## 1. アイデンティティ概要

```markdown
{identity_summary}
```

---

## 2. 現在の状態 (state.json)

```json
{json.dumps(state, ensure_ascii=False, indent=2)}
```

---

## 3. 直近のjournal (最新20行)

```markdown
{journal_tail}
```

---

## 4. 完了タスク一覧

{chr(10).join(f"- {task}" for task in completed_tasks)}

---

## 5. 既知の問題

{chr(10).join(f"- {issue}" for issue in issues) if issues else "- (現在認識している問題なし)"}

---

## 6. TODO / 次のアクション

{chr(10).join(f"- {todo}" for todo in todos) if todos else "- (明示的なTODOなし)"}

---

## 7. 引き継ぎノート

- 振り返りサイクル: {state.get('growth_cycles', 0)}回完了
- 最終振り返り: {state.get('last_reflection', 'N/A')}
- システム状態: {state.get('status', 'unknown')}
- 起動時刻: {state.get('uptime_start', 'N/A')}

**注意事項**:
- このファイルは振り返りサイクルで自動更新される
- 詳細はmemory/journal.mdを参照
- アイデンティティの完全版はmemory/identity.mdを参照

---

*Generated by God AI v3.0 - HANDOFF Module*
"""

    # ファイル書き込み
    try:
        HANDOFF_PATH.write_text(content, encoding="utf-8")
        log.info(f"HANDOFF.md生成完了: {HANDOFF_PATH}")
    except Exception as e:
        log.error(f"HANDOFF.md書き込み失敗: {e}")
        return ""

    # Google Driveアップロード（存在すれば）
    try:
        from gdrive import upload_file, is_configured
        if is_configured():
            result = upload_file(str(HANDOFF_PATH))
            if result:
                log.info(f"HANDOFF.md Driveアップロード完了: {result.get('name')}")
            else:
                log.warning("HANDOFF.md Driveアップロード失敗")
    except ImportError:
        log.debug("gdrive module not available, skipping Drive upload")
    except Exception as e:
        log.warning(f"HANDOFF.md Driveアップロードエラー: {e}")

    return str(HANDOFF_PATH)


# --- CLI テスト ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("HANDOFF.md生成テスト...")
    path = generate()
    if path:
        print(f"生成成功: {path}")
        print("\n--- 内容プレビュー (最初の50行) ---")
        content = Path(path).read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines()[:50]):
            print(line)
    else:
        print("生成失敗")
