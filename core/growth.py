#!/usr/bin/env python3
"""God AI v3.0 - 成長・振り返りモジュール"""
from __future__ import annotations

import ast
import asyncio
import difflib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

import time as time_module

import hashlib

from config import (
    GOD_PY_PATH, JOURNAL_PATH, REFLECTION_INTERVAL, GROWTH_CYCLE_SLEEP, log,
    BASE_DIR, CORE_DIR, STATE_PATH, IDENTITY_PATH, MEMORY_DIR
)

# --- 改善履歴パス ---
IMPROVEMENT_HISTORY_PATH = MEMORY_DIR / "improvement_history.json"
from memory import (
    load_state, save_state, read_file, append_journal,
    safe_save_state, safe_append_journal
)
from brain import think_gemini, think_claude
from jobqueue import create_job, is_p1_interrupted, clear_p1_interrupt, get_queued_job_summaries, get_recent_failed_jobs, load_queue

# --- CLAUDE.md パス ---
CLAUDE_MD_PATH = BASE_DIR / "CLAUDE.md"

# --- モジュールパス定義 ---
MODULE_PATHS = {
    "god": CORE_DIR / "god.py",
    "growth": CORE_DIR / "growth.py",
    "brain": CORE_DIR / "brain.py",
    "memory": CORE_DIR / "memory.py",
    "config": CORE_DIR / "config.py",
    "jobqueue": CORE_DIR / "jobqueue.py",
    "job_worker": CORE_DIR / "job_worker.py",
    "handoff": CORE_DIR / "handoff.py",
    "gdrive": CORE_DIR / "gdrive.py",
    "twitter": CORE_DIR / "twitter.py",
}

# 有効なモジュール名のセット（存在チェック用）
VALID_MODULES = set(MODULE_PATHS.keys())

# 有効なファイル名のセット（.py付き）
VALID_MODULE_FILES = {f"{name}.py" for name in VALID_MODULES}

def _detect_invalid_module_names(text: str) -> list[str]:
    """テキスト内の無効なモジュール名を検出。戻り値: 無効なファイル名のリスト"""
    import re
    # .pyで終わる単語を抽出
    potential_files = re.findall(r'\b(\w+\.py)\b', text)
    invalid = []
    for f in potential_files:
        if f not in VALID_MODULE_FILES and f != "__init__.py":
            invalid.append(f)
    return list(set(invalid))  # 重複除去

def _extract_function_signatures(filepath) -> str:
    """ファイルの全関数の「def文 + 先頭3行」を抽出して文字列で返す"""
    try:
        code = filepath.read_text(encoding="utf-8")
        lines = code.splitlines()
        tree = ast.parse(code)
    except Exception:
        return "(parse error)"

    sigs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1  # 0-indexed
            end = min(start + 4, len(lines))  # def line + 3 body lines
            snippet = "\n".join(lines[start:end])
            sigs.append(snippet)
    return "\n\n".join(sigs) if sigs else "(no functions)"


def _detect_read_files(target_path, target_name: str) -> list[str]:
    """対象ファイルのimport文を解析し、同プロジェクト内の依存ファイルパスを返す。"""
    import re as _re
    deps = []
    try:
        code = target_path.read_text(encoding="utf-8")
        for line in code.splitlines():
            line = line.strip()
            # from config import ... / import brain
            m = _re.match(r'^(?:from\s+(\w+)\s+import|import\s+(\w+))', line)
            if m:
                mod_name = m.group(1) or m.group(2)
                if mod_name in VALID_MODULES and mod_name != target_name:
                    dep_path = MODULE_PATHS[mod_name]
                    dep_str = str(dep_path)
                    if dep_str not in deps:
                        deps.append(dep_str)
    except Exception as e:
        log.warning(f"依存ファイル検出失敗 ({target_name}): {e}")
    return deps


# --- 振り返り排他制御 ---
_reflecting = False

def is_reflecting() -> bool:
    return _reflecting


# --- 成長統計パス ---
GROWTH_STATS_PATH = MEMORY_DIR / "growth_stats.json"


# --- 成長統計関数 ---
def load_growth_stats() -> dict:
    """成長統計を読み込む"""
    default_stats = {
        "total_attempts": 0,
        "total_successes": 0,
        "total_failures": 0,
        "success_rate": 0.0,
        "failure_reasons": {"syntax_error": 0, "timeout": 0, "test_fail": 0, "duplicate": 0, "rollback": 0},
        "modules_improved": {"brain.py": 0, "growth.py": 0, "god.py": 0, "memory.py": 0, "config.py": 0, "jobqueue.py": 0},
        "avg_improvement_time": 0,
        "last_10_results": [],
        "streak": {"current_success": 0, "best_success": 0}
    }
    if GROWTH_STATS_PATH.exists():
        try:
            with open(GROWTH_STATS_PATH, "r", encoding="utf-8") as f:
                stats = json.load(f)
                # 欠けているキーを補完
                for key, value in default_stats.items():
                    if key not in stats:
                        stats[key] = value
                return stats
        except Exception as e:
            log.warning(f"成長統計読み込み失敗: {e}")
    return default_stats


def save_growth_stats(stats: dict):
    """成長統計を保存"""
    try:
        with open(GROWTH_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        log.debug(f"成長統計保存: {GROWTH_STATS_PATH}")
    except Exception as e:
        log.error(f"成長統計保存失敗: {e}")


def update_growth_stats(success: bool, failure_reason: str | None, module: str, duration: float):
    """成長統計を更新

    Args:
        success: 改善が成功したか
        failure_reason: 失敗理由（"syntax_error", "timeout", "test_fail", "duplicate"）
        module: 対象モジュール名（例: "brain.py"）
        duration: 改善にかかった時間（秒）
    """
    stats = load_growth_stats()

    # 試行回数を更新
    stats["total_attempts"] += 1

    if success:
        stats["total_successes"] += 1
        stats["streak"]["current_success"] += 1
        if stats["streak"]["current_success"] > stats["streak"]["best_success"]:
            stats["streak"]["best_success"] = stats["streak"]["current_success"]

        # モジュール改善回数を更新
        module_key = f"{module}.py" if not module.endswith(".py") else module
        if module_key in stats["modules_improved"]:
            stats["modules_improved"][module_key] += 1
        else:
            stats["modules_improved"][module_key] = 1
    else:
        stats["total_failures"] += 1
        stats["streak"]["current_success"] = 0

        # 失敗理由を更新
        if failure_reason and failure_reason in stats["failure_reasons"]:
            stats["failure_reasons"][failure_reason] += 1

    # 成功率を計算
    if stats["total_attempts"] > 0:
        stats["success_rate"] = round(stats["total_successes"] / stats["total_attempts"] * 100, 1)

    # 平均改善時間を更新（成功時のみ）
    if success and duration > 0:
        if stats["avg_improvement_time"] == 0:
            stats["avg_improvement_time"] = duration
        else:
            # 移動平均で更新
            stats["avg_improvement_time"] = round(
                (stats["avg_improvement_time"] * 0.8 + duration * 0.2), 1
            )

    # 直近10回の結果を記録
    result_entry = {
        "time": datetime.now().isoformat(),
        "success": success,
        "module": module,
        "failure_reason": failure_reason,
        "duration": round(duration, 1)
    }
    stats["last_10_results"].append(result_entry)
    if len(stats["last_10_results"]) > 10:
        stats["last_10_results"] = stats["last_10_results"][-10:]

    save_growth_stats(stats)
    log.info(f"成長統計更新: success={success}, module={module}, duration={duration:.1f}s")


def get_stats_summary() -> str:
    """統計サマリーを取得（Telegram表示用）"""
    stats = load_growth_stats()

    # 成功率
    total = stats["total_attempts"]
    successes = stats["total_successes"]
    rate = stats["success_rate"]

    # 最多失敗原因（日本語表示）
    failure_reasons = stats["failure_reasons"]
    reason_names = {
        "syntax_error": "構文エラー",
        "timeout": "タイムアウト",
        "test_fail": "テスト失敗",
        "duplicate": "重複提案",
        "rollback": "ロールバック"
    }
    max_failure = max(failure_reasons.items(), key=lambda x: x[1]) if any(v > 0 for v in failure_reasons.values()) else ("なし", 0)
    max_failure_name = reason_names.get(max_failure[0], max_failure[0]) if max_failure[0] != "なし" else "なし"

    # 最も改善されたモジュール
    modules = stats["modules_improved"]
    max_module = max(modules.items(), key=lambda x: x[1]) if any(v > 0 for v in modules.values()) else ("なし", 0)

    # 連続成功
    streak = stats["streak"]["current_success"]
    best_streak = stats["streak"]["best_success"]

    # 平均改善時間
    avg_time = stats.get("avg_improvement_time", 0)
    avg_time_str = f"{avg_time:.1f}秒" if avg_time > 0 else "N/A"

    return (
        f"成功率: {rate}% ({successes}/{total})\n"
        f"連続成功: {streak}回 (最高: {best_streak}回)\n"
        f"最多失敗原因: {max_failure_name} ({max_failure[1]}回)\n"
        f"最も改善されたモジュール: {max_module[0]} ({max_module[1]}回)\n"
        f"平均改善時間: {avg_time_str}"
    )


def should_skip_improvement(improvement_text: str, module: str) -> tuple[bool, str]:
    """Decide whether to skip an improvement based on growth stats.

    Improvements over the original logic:
    - Raise the same-reason failure threshold from 3 to 5 to avoid
      blocking all improvements too aggressively.
    - Reset failure counts if the last success was more than 2 hours ago,
      so the system gets another chance after a cooldown period.
    - Raise the per-module consecutive failure threshold from 3 to 4.
    - Only count per-module failures for the *same* module as the
      proposed improvement (skip unrelated module failures).

    Args:
        improvement_text: The improvement proposal text.
        module: Target module name.

    Returns:
        (should_skip, reason)
    """
    stats = load_growth_stats()

    recent = stats["last_10_results"]
    if not recent:
        return (False, "")

    # Check if enough time has passed since the last success.
    # If more than 2 hours have elapsed, reset failure tracking
    # to give improvements another chance.
    last_success_time = None
    for result in reversed(recent):
        if result.get("success"):
            try:
                last_success_time = datetime.fromisoformat(result["time"])
            except (ValueError, KeyError):
                pass
            break

    hours_since_success = None
    if last_success_time:
        now = datetime.now()
        # Handle timezone-naive comparison
        if last_success_time.tzinfo is not None:
            now = datetime.now(timezone.utc)
        hours_since_success = (now - last_success_time).total_seconds() / 3600

    # If more than 2 hours since last success, allow improvements again
    if hours_since_success is not None and hours_since_success >= 2.0:
        log.info(
            f"should_skip_improvement: {hours_since_success:.1f}h since last success, "
            f"resetting failure tracking"
        )
        return (False, "")

    # Count failures by reason in last 10 results
    failure_counts = {}
    for result in recent:
        if not result["success"] and result.get("failure_reason"):
            reason = result["failure_reason"]
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    reason_names = {
        "syntax_error": "構文エラー",
        "timeout": "タイムアウト",
        "test_fail": "テスト失敗",
        "duplicate": "重複提案",
        "rollback": "ロールバック"
    }

    # Threshold raised from 3 to 5 to avoid blocking improvements too early
    for reason, count in failure_counts.items():
        if count >= 5:
            reason_jp = reason_names.get(reason, reason)
            return (True, f"同じ原因({reason_jp})で{count}回失敗しています")

    # Check per-module consecutive failures (only for the target module)
    # Threshold raised from 3 to 4
    module_failure_count = 0
    for result in recent[-5:]:
        if not result["success"] and result.get("module") == module:
            module_failure_count += 1

    if module_failure_count >= 4:
        return (True, f"モジュール({module})で直近{module_failure_count}回連続失敗中")

    return (False, "")


def get_recommended_module() -> str | None:
    """統計に基づいて改善推奨モジュールを取得

    成功率が高いモジュールを優先
    """
    stats = load_growth_stats()
    modules = stats["modules_improved"]

    # 改善実績があるモジュールを成功回数でソート
    if not any(v > 0 for v in modules.values()):
        return None

    sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
    return sorted_modules[0][0] if sorted_modules else None


def get_auto_suggestions() -> list[str]:
    """統計に基づいて自動提案を生成

    Returns:
        提案リスト
    """
    stats = load_growth_stats()
    suggestions = []

    failure_reasons = stats["failure_reasons"]

    # タイムアウトが3回以上 -> モジュール分割を提案
    if failure_reasons.get("timeout", 0) >= 3:
        suggestions.append("タイムアウトが頻発しています。大きいモジュールの分割を検討してください。")

    # 構文エラーが3回以上 -> コードレビュー強化を提案
    if failure_reasons.get("syntax_error", 0) >= 3:
        suggestions.append("構文エラーが多発しています。生成コードの検証強化を検討してください。")

    # テスト失敗が3回以上 -> テストカバレッジ改善を提案
    if failure_reasons.get("test_fail", 0) >= 3:
        suggestions.append("テスト失敗が多発しています。テストカバレッジの改善を検討してください。")

    # 重複提案が3回以上 -> 新しい観点を提案
    if failure_reasons.get("duplicate", 0) >= 3:
        suggestions.append("重複提案が多発しています。より新しい観点での改善を検討してください。")

    # 成功率が低い場合
    if stats["total_attempts"] >= 5 and stats["success_rate"] < 30:
        suggestions.append(f"成功率が{stats['success_rate']}%と低めです。改善提案の品質向上を検討してください。")

    return suggestions


# --- CLAUDE.md 読み込み ---
def load_claude_md() -> str:
    """CLAUDE.mdの内容を読み込む"""
    if CLAUDE_MD_PATH.exists():
        try:
            return CLAUDE_MD_PATH.read_text(encoding="utf-8")
        except Exception as e:
            log.warning(f"CLAUDE.md読み込み失敗: {e}")
    return ""


# --- CLAUDE.md 自動更新 ---
def update_claude_md():
    """振り返り完了時にCLAUDE.mdを自動更新"""
    log.info("CLAUDE.md自動更新開始")
    try:
        state = load_state()
        identity_text = ""
        if IDENTITY_PATH.exists():
            identity_text = IDENTITY_PATH.read_text(encoding="utf-8")
        journal_text = read_file(JOURNAL_PATH, tail=100)
        recent_events = _extract_recent_events(journal_text, max_events=10)
        known_issues = _extract_known_issues(journal_text)
        module_list = _generate_module_list()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = _generate_claude_md_content(
            now, identity_text, state, module_list, recent_events, known_issues
        )
        CLAUDE_MD_PATH.write_text(content, encoding="utf-8")
        log.info(f"CLAUDE.md更新完了: {CLAUDE_MD_PATH}")
    except Exception as e:
        log.error(f"CLAUDE.md更新失敗: {e}", exc_info=True)


def _extract_recent_events(journal_text: str, max_events: int = 10) -> list[str]:
    """
    Extracts recent important events from the journal text.
    Optimized for speed and accuracy by using more efficient string searching.

    Args:
        journal_text: The text content of the journal.
        max_events: The maximum number of events to return.

    Returns:
        A list of the most recent important event strings.
    """
    events = []
    # Use regex to find lines starting with ### and containing important keywords.
    # This is generally faster than iterating line by line and checking keywords individually.
    # We specifically look for the ### prefix to ensure we're parsing structured journal entries.
    # The keywords are also part of the regex for efficiency.
    # Added more specific keywords and patterns to capture events more accurately.
    # Ensured that lines are captured as whole events.
    important_keywords_pattern = "|".join([
        "成功", "失敗", "エラー", "改善", "誕生", "完了", "開始", "スキップ", "ロールバック", "テスト合格", "テスト失敗", "構文エラー"
    ])
    # The pattern now specifically looks for a line starting with ### and containing the important keywords.
    # It captures the entire line after ###.
    event_pattern = re.compile(
        rf"^###\s+(.*?{important_keywords_pattern}.*?)$",
        re.MULTILINE
    )

    for match in event_pattern.finditer(journal_text):
        event = match.group(1).strip() # Capture group 1, which is the content after ###
        if event and event not in events:
            events.append(event)

    return events[-max_events:] if len(events) > max_events else events


def _extract_known_issues(journal_text: str) -> list[str]:
    """journalから既知の問題を抽出"""
    issues = []
    for line in journal_text.splitlines():
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["問題", "エラー", "失敗", "バグ"]):
            issue = line.strip()
            if 10 < len(issue) < 200 and issue not in issues:
                issues.append(issue)
    return issues[-5:] if len(issues) > 5 else issues


def _generate_module_list() -> str:
    """core/内のモジュール一覧と説明を生成"""
    descriptions = {
        "god.py": "メインエントリ・Telegramボット",
        "config.py": "設定・定数・環境変数",
        "memory.py": "ファイルI/O・状態管理",
        "brain.py": "AI思考（Gemini/Claude CLI）",
        "growth.py": "振り返り・自己改善",
        "jobqueue.py": "優先度付きジョブキュー",
        "handoff.py": "HANDOFF.md自動生成",
        "gdrive.py": "Google Driveバックアップ",
    }
    lines = []
    for name, desc in descriptions.items():
        if (CORE_DIR / name).exists():
            lines.append(f"| `{name}` | {desc} |")
    return "\n".join(lines)


def _generate_claude_md_content(
    now: str, identity_text: str, state: dict,
    module_list: str, recent_events: list[str], known_issues: list[str]
) -> str:
    """CLAUDE.mdの内容を生成"""
    identity_summary = "\n".join(identity_text.splitlines()[:40]) if identity_text else ""
    events_text = "\n".join([f"{i+1}. **{e}**" for i, e in enumerate(recent_events)])
    issues_text = "\n".join([f"{i+1}. {iss}" for i, iss in enumerate(known_issues)]) if known_issues else "(現在認識している問題なし)"
    return f"""# God AI v3.0 - Claude Memory File

**最終更新**: {now}

---

## 1. 目的・設計思想

{identity_summary}

---

## 2. コード構成

### 各モジュールの責務

| モジュール | 責務 |
|-----------|------|
{module_list}

---

## 3. 現在の状態

```json
{json.dumps(state, ensure_ascii=False, indent=2)}
```

---

## 4. 直近の成長履歴（重要イベント）

{events_text if events_text else "(イベントなし)"}

---

## 5. 既知の問題

{issues_text}

---

## 6. 自己改善時の注意事項

### 必須チェック
1. **構文チェック必須**: `python3 -m py_compile <file>` で検証
2. **差分記録**: 変更前後の差分をjournalに記録
3. **重複チェック**: 同じ改善提案を繰り返さない
4. **バックアップ**: 改善前に `.py.bak` を作成
5. **ロールバック**: 3回失敗したら自動ロールバック

### コード生成ルール
- マークダウンのバッククォートで囲まない
- 先頭は `#!/usr/bin/env python3` から
- 変更箇所以外は絶対にそのまま維持
- 文字列リテラルのクォート対応に注意

### 禁止事項
- `except:pass` - 全エラーをログに残す
- Benyの個人情報の外部送信

---

*このファイルは振り返り完了時に自動更新される*
"""


# --- 改善対象モジュールの自動選択 ---
def select_target_module(improvement_text: str) -> tuple[Path, str] | None:
    """改善内容から対象モジュールを自動選択。戻り値: (モジュールパス, モジュール名) or None（無効な場合）"""
    improvement_lower = improvement_text.lower()

    # キーワードマッチングでモジュール選択
    if any(kw in improvement_lower for kw in ["振り返り", "reflection", "自己改善", "growth", "improve"]):
        module_name = "growth"
    elif any(kw in improvement_lower for kw in ["gemini", "claude", "think", "脳", "brain", "ai", "timeout"]):
        module_name = "brain"
    elif any(kw in improvement_lower for kw in ["memory", "state", "journal", "保存", "読み込み"]):
        module_name = "memory"
    elif any(kw in improvement_lower for kw in ["queue", "job", "キュー", "priority"]):
        module_name = "jobqueue"
    elif any(kw in improvement_lower for kw in ["config", "設定", "定数", "env"]):
        module_name = "config"
    elif any(kw in improvement_lower for kw in ["handoff", "引き継ぎ"]):
        module_name = "handoff"
    elif any(kw in improvement_lower for kw in ["drive", "google", "upload", "backup"]):
        module_name = "gdrive"
    elif any(kw in improvement_lower for kw in ["twitter", "tweet", "ツイート", "x連携"]):
        module_name = "twitter"
    elif any(kw in improvement_lower for kw in ["telegram", "polling", "メイン", "main", "god"]):
        module_name = "god"
    else:
        module_name = "god"  # デフォルト

    # 存在チェック
    if module_name not in VALID_MODULES:
        log.warning(f"無効なモジュール名: {module_name}")
        return None

    module_path = MODULE_PATHS[module_name]
    if not module_path.exists():
        log.warning(f"モジュールファイルが存在しない: {module_path}")
        return None

    return (module_path, module_name)


# --- バックアップ管理 ---
def create_module_backups() -> dict[str, Path]:
    """改善前に全モジュールのバックアップを作成。戻り値: {モジュール名: バックアップパス}"""
    backups = {}
    for module_name, module_path in MODULE_PATHS.items():
        if module_path.exists():
            backup_path = module_path.with_suffix(".py.bak")
            try:
                shutil.copy2(module_path, backup_path)
                backups[module_name] = backup_path
                log.debug(f"バックアップ作成: {module_name} -> {backup_path}")
            except Exception as e:
                log.error(f"バックアップ作成失敗 {module_name}: {e}")
    # twitter.pyも追加
    twitter_path = CORE_DIR / "twitter.py"
    if twitter_path.exists():
        backup_path = twitter_path.with_suffix(".py.bak")
        try:
            shutil.copy2(twitter_path, backup_path)
            backups["twitter"] = backup_path
        except Exception as e:
            log.error(f"バックアップ作成失敗 twitter: {e}")
    log.info(f"モジュールバックアップ作成完了: {len(backups)}件")
    return backups


def rollback_from_backups(backups: dict[str, Path], target_modules: Optional[list[str]] = None):
    """バックアップからロールバック

    Args:
        backups: create_module_backups()の戻り値
        target_modules: ロールバック対象のモジュール名リスト（Noneなら全て）
    """
    modules_to_rollback = target_modules if target_modules else list(backups.keys())
    rolled_back = []

    for module_name in modules_to_rollback:
        if module_name not in backups:
            continue
        backup_path = backups[module_name]
        if not backup_path.exists():
            log.warning(f"バックアップファイルが存在しない: {backup_path}")
            continue

        module_path = MODULE_PATHS.get(module_name)
        if module_name == "twitter":
            module_path = CORE_DIR / "twitter.py"
        if module_path:
            try:
                shutil.copy2(backup_path, module_path)
                rolled_back.append(module_name)
                log.info(f"ロールバック完了: {module_name}")
            except Exception as e:
                log.error(f"ロールバック失敗 {module_name}: {e}")

    log.info(f"ロールバック完了: {rolled_back}")
    return rolled_back


def cleanup_backups(backups: dict[str, Path]):
    """成功時にバックアップファイルを削除"""
    for module_name, backup_path in backups.items():
        try:
            if backup_path.exists():
                backup_path.unlink()
                log.debug(f"バックアップ削除: {backup_path}")
        except Exception as e:
            log.warning(f"バックアップ削除失敗 {module_name}: {e}")


# --- モジュール再読み込み ---
def reload_modules(module_names: list[str]) -> tuple[bool, str]:
    """指定モジュールをimportlib.reloadで再読み込み（God AI再起動不要）

    Args:
        module_names: 再読み込みするモジュール名のリスト

    Returns:
        (成功したか, メッセージ)
    """
    reloaded = []
    errors = []

    for name in module_names:
        if name in sys.modules:
            try:
                module = sys.modules[name]
                importlib.reload(module)
                reloaded.append(name)
                log.info(f"モジュール再読み込み成功: {name}")
            except Exception as e:
                errors.append(f"{name}: {e}")
                log.error(f"モジュール再読み込み失敗 {name}: {e}")

    if errors:
        return (False, f"再読み込み失敗: {'; '.join(errors)}")
    return (True, f"再読み込み成功: {', '.join(reloaded) if reloaded else '(対象なし)'}")


# --- 改善後自動テスト ---
async def run_post_improvement_tests(
    client: httpx.AsyncClient,
    backups: Optional[dict[str, Path]] = None,
    target_module: Optional[str] = None
) -> tuple[bool, str]:
    """
    改善後テスト実行。戻り値: (全合格か, 結果メッセージ)

    テスト項目（順序厳守）:
    1. 全モジュール構文チェック: python3 -m py_compile core/*.py
    2. importテスト: 全モジュールのimport確認
    3. Gemini API疎通テスト: 簡単なprompt送信→応答あるか
    4. 全合格 → journal記録「✅ テスト全合格」+ モジュール再読み込み
    5. 1つでも失敗 → .bakからロールバック → Telegram通知
    """
    from god import tg_send

    results = []
    all_passed = True

    # --- 1. 全モジュール構文チェック ---
    log.info("テスト(1): 全モジュール構文チェック開始")
    syntax_errors = []
    for module_name, module_path in MODULE_PATHS.items():
        if module_path.exists():
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(module_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if proc.returncode != 0:
                    syntax_errors.append(f"{module_name}: {proc.stderr.strip()}")
                    log.error(f"構文エラー in {module_name}: {proc.stderr.strip()}")
            except subprocess.TimeoutExpired:
                syntax_errors.append(f"{module_name}: タイムアウト")
                log.error(f"構文チェックタイムアウト: {module_name}")
            except Exception as e:
                syntax_errors.append(f"{module_name}: {e}")
                log.error(f"構文チェックエラー in {module_name}: {e}")

    # twitter.pyも追加チェック
    twitter_path = CORE_DIR / "twitter.py"
    if twitter_path.exists():
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(twitter_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if proc.returncode != 0:
                syntax_errors.append(f"twitter: {proc.stderr.strip()}")
        except Exception as e:
            syntax_errors.append(f"twitter: {e}")

    if syntax_errors:
        all_passed = False
        results.append(f"❌ 構文チェック失敗: {'; '.join(syntax_errors)}")
        # ロールバック実行
        if backups:
            rollback_from_backups(backups)
            await tg_send(client, f"⚠️ 自己改善失敗→ロールバック完了\n構文エラー: {'; '.join(syntax_errors)[:200]}")
        return (False, "\n".join(results))
    else:
        results.append("✅ 構文チェック全合格")
        log.info("テスト(1): 全モジュール構文チェック合格")

    # --- 2. importテスト ---
    log.info("テスト(2): importテスト開始")
    import_errors = []

    # 各モジュールのimportテスト
    import_tests = [
        ("brain", "think_gemini"),
        ("memory", "load_state"),
        ("jobqueue", "create_job"),
        ("growth", "reflection_cycle"),
        ("gdrive", "is_configured"),
        ("handoff", "generate"),
        ("twitter", "is_configured"),
    ]

    for module_name, func_name in import_tests:
        try:
            # 新しいPythonプロセスでimportテスト実行
            import_cmd = f"from {module_name} import {func_name}"
            proc = subprocess.run(
                [sys.executable, "-c", import_cmd],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(CORE_DIR)
            )
            if proc.returncode != 0:
                import_errors.append(f"{module_name}.{func_name}: {proc.stderr.strip()[:100]}")
                log.error(f"importエラー {module_name}.{func_name}: {proc.stderr.strip()}")
        except subprocess.TimeoutExpired:
            import_errors.append(f"{module_name}.{func_name}: タイムアウト")
            log.error(f"importテストタイムアウト: {module_name}.{func_name}")
        except Exception as e:
            import_errors.append(f"{module_name}.{func_name}: {e}")
            log.error(f"importテストエラー {module_name}.{func_name}: {e}")

    if import_errors:
        all_passed = False
        results.append(f"❌ importテスト失敗: {'; '.join(import_errors)}")
        # ロールバック実行
        if backups:
            rollback_from_backups(backups)
            await tg_send(client, f"⚠️ 自己改善失敗→ロールバック完了\nimportエラー: {'; '.join(import_errors)[:200]}")
        return (False, "\n".join(results))
    else:
        results.append("✅ importテスト全合格")
        log.info("テスト(2): importテスト合格")

    # --- 3. Gemini API疎通テスト ---
    log.info("テスト(3): Gemini API疎通テスト開始")
    try:
        from brain import think_gemini
        response, brain_name = await think_gemini("Hello, respond with 'OK' if you can read this.")
        if response and len(response) > 0:
            results.append(f"✅ Gemini API疎通テスト合格（応答あり: {brain_name}）")
            log.info(f"テスト(3): Gemini API疎通テスト合格: {response[:50]}")
        else:
            all_passed = False
            results.append("❌ Gemini API疎通テスト失敗: 応答が空")
            log.error("テスト(3): Gemini API応答が空")
            if backups:
                rollback_from_backups(backups)
                await tg_send(client, "⚠️ 自己改善失敗→ロールバック完了\nGemini API応答が空")
            return (False, "\n".join(results))
    except Exception as e:
        all_passed = False
        results.append(f"❌ Gemini API疎通テスト失敗: {e}")
        log.error(f"テスト(3): Gemini APIエラー: {e}")
        if backups:
            rollback_from_backups(backups)
            await tg_send(client, f"⚠️ 自己改善失敗→ロールバック完了\nGemini APIエラー: {str(e)[:100]}")
        return (False, "\n".join(results))

    # --- 全テスト合格 ---
    if all_passed:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        await safe_append_journal(f"### {now} ✅ テスト全合格\n{chr(10).join(results)}")
        log.info("全テスト合格、journal記録完了")

        # モジュール再読み込み（God AI再起動不要）
        if target_module:
            reload_success, reload_msg = reload_modules([target_module])
            if reload_success:
                results.append(f"✅ モジュール再読み込み: {reload_msg}")
                log.info(f"モジュール再読み込み成功: {target_module}")
            else:
                results.append(f"⚠️ モジュール再読み込み: {reload_msg}")
                log.warning(f"モジュール再読み込み失敗: {reload_msg}")

        # バックアップをクリーンアップ（成功時のみ）
        if backups:
            cleanup_backups(backups)

    return (all_passed, "\n".join(results))


# --- コード構文検証関数 ---
def validate_code_syntax(code: str) -> tuple[bool, str]:
    """Generate code syntax validation. Returns: (is_valid, error_message)
    Also detects invalid full-width characters (e.g., 【, 】) in generated code.
    """
    # Detect invalid full-width characters in the code.
    # This set includes common full-width punctuation and brackets.
    fullwidth_chars_pattern = re.compile(r'[【】『』（）《》「」〔〕、。！？：；]')
    match = fullwidth_chars_pattern.search(code)
    if match:
        # Find the line number and context for the invalid character.
        error_line_num = code.count('\n', 0, match.start()) + 1
        lines = code.splitlines()
        start = max(0, error_line_num - 3)
        end = min(len(lines), error_line_num + 2)
        context = "\n".join([f"{i+1}: {lines[i]}" for i in range(start, end)])
        return (False, f"Invalid full-width character '{match.group(0)}' found at line {error_line_num}. Check code context:\n{context}")

    try:
        # Attempt to parse the code as Python syntax.
        ast.parse(code)
        return (True, "")
    except SyntaxError as e:
        # Handle Python syntax errors.
        error_msg = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
        if e.lineno and e.lineno <= len(code.splitlines()):
            lines = code.splitlines()
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            context = "\n".join([f"{i+1}: {lines[i]}" for i in range(start, end)])
            error_msg += f"\nCode context:\n{context}"
        return (False, error_msg)
    except Exception as e:
        # Handle any other unexpected errors during parsing.
        return (False, f"Unexpected error during syntax validation: {e}")

# --- 差分パッチ適用関数 ---
def parse_patches(result_text: str) -> list[tuple[str, str]]:
    """AI生成結果から<<<BEFORE>>><<<AFTER>>><<<END>>>形式のパッチを抽出。
    戻り値: [(before_code, after_code), ...]
    """
    patches = []
    parts = result_text.split("<<<BEFORE>>>")
    for part in parts[1:]:  # 最初の空要素をスキップ
        if "<<<AFTER>>>" not in part or "<<<END>>>" not in part:
            continue
        before_section, rest = part.split("<<<AFTER>>>", 1)
        after_section = rest.split("<<<END>>>", 1)[0]
        before_code = before_section.strip()
        after_code = after_section.strip()
        if before_code:  # 空のパッチはスキップ
            patches.append((before_code, after_code))
    return patches


def apply_patches(module_path: Path, patches: list[tuple[str, str]]) -> tuple[bool, str, str]:
    """差分パッチをモジュールに適用。

    Args:
        module_path: 対象モジュールのパス
        patches: [(before_code, after_code), ...] のリスト

    Returns:
        (成功かどうか, 適用後コード, エラーメッセージ)
    """
    code = module_path.read_text(encoding="utf-8")
    original_code = code
    applied_count = 0

    for i, (before, after) in enumerate(patches):
        if before in code:
            code = code.replace(before, after, 1)
            applied_count += 1
            log.info(f"パッチ {i+1}/{len(patches)} 適用成功")
        else:
            # 改行・空白の微妙な差異に対応: 前後の空白をnormalizeして再試行
            # Before節とAfter節の末尾に改行がない場合、置換後に改行が欠落するのを防ぐ
            normalized_before = "\n".join(line.rstrip() for line in before.splitlines())
            normalized_after = "\n".join(line.rstrip() for line in after.splitlines())

            if normalized_before in code:
                # normalizationなしで直接置換（改行ずれが少ない場合）
                code = code.replace(before, after, 1)
                applied_count += 1
                log.info(f"パッチ {i+1}/{len(patches)} 適用成功（直接置換）")
            else:
                # normalization版で位置を特定し、元のコードで置換
                try:
                    code_lines = code.splitlines(keepends=True)
                    normalized_code_lines = [line.rstrip() for line in code_lines]
                    normalized_code = "\n".join(normalized_code_lines)

                    start_index = normalized_code.find(normalized_before)
                    if start_index == -1:
                        raise ValueError("Normalized BEFORE not found in normalized code")

                    # 元のコードでの正確な位置を計算
                    current_index = 0
                    original_start_index = -1
                    original_lines_count = 0
                    for idx, line in enumerate(code_lines):
                        stripped_line = line.rstrip()
                        if current_index == start_index and stripped_line == normalized_before.splitlines()[0].rstrip():
                            original_start_index = sum(len(l) for l in code_lines[:idx])
                            break
                        current_index += len(line)
                        if stripped_line == normalized_before.splitlines()[original_lines_count].rstrip():
                            original_lines_count += 1
                        else:
                            original_lines_count = 0 # Reset if line doesn't match

                    if original_start_index == -1:
                        raise ValueError("Could not find exact original position for replacement")

                    # After節の末尾に改行がない場合、afterに改行を追加
                    if not after.endswith('\n') and not code[original_start_index + len(before):].startswith('\n'):
                         after_with_newline = after + '\n'
                    else:
                        after_with_newline = after

                    code = code[:original_start_index] + after_with_newline + code[original_start_index + len(before):]
                    applied_count += 1
                    log.info(f"パッチ {i+1}/{len(patches)} 適用成功（normalized match）")

                except Exception as e:
                    error_msg = f"パッチ {i+1}/{len(patches)} 適用失敗: BEFORE部分がコード内に見つかりません（正規化後も）\nBEFORE先頭: {before[:100]}\n詳細: {e}"
                    log.error(error_msg)
                    return (False, original_code, error_msg)

    if applied_count == 0:
        return (False, original_code, "適用されたパッチがありません")

    log.info(f"全{applied_count}件のパッチ適用完了")
    return (True, code, "")


# --- 改善履歴管理関数 ---
def load_improvement_history() -> dict:
    """改善履歴を読み込む"""
    default_history = {"proposals": []}
    if IMPROVEMENT_HISTORY_PATH.exists():
        try:
            with open(IMPROVEMENT_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"改善履歴読み込み失敗: {e}")
    return default_history


def save_improvement_history(history: dict):
    """改善履歴を保存"""
    try:
        with open(IMPROVEMENT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        log.debug(f"改善履歴保存: {IMPROVEMENT_HISTORY_PATH}")
    except Exception as e:
        log.error(f"改善履歴保存失敗: {e}")


def get_improvement_hash(improvement_text: str) -> str:
    """改善提案のハッシュを生成（正規化してから）"""
    # 正規化: 小文字化、空白を統一、記号除去
    normalized = improvement_text.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]


def record_improvement(improvement_text: str, status: str):
    """改善提案を履歴に記録

    Args:
        improvement_text: 改善内容のテキスト
        status: "proposed" / "success" / "failed" / "skipped"
            - proposed: 提案されたが未実行
            - success: 実行して成功
            - failed: 実行して失敗
            - skipped: 3回以上失敗してスキップ対象に
    """
    history = load_improvement_history()
    proposal_hash = get_improvement_hash(improvement_text)

    # 既存のエントリを検索
    existing = None
    for p in history["proposals"]:
        if p["hash"] == proposal_hash:
            existing = p
            break

    now = datetime.now(timezone.utc).isoformat()
    if existing:
        old_status = existing.get("status", existing.get("result", "unknown"))
        existing["status"] = status
        existing["timestamp"] = now

        # 失敗回数をトラック
        if status == "failed":
            existing["fail_count"] = existing.get("fail_count", 0) + 1
            # 3回以上失敗したらskippedに変更
            if existing["fail_count"] >= 3:
                existing["status"] = "skipped"
                log.warning(f"改善提案が3回以上失敗、skippedに変更: {proposal_hash}")
        elif status == "success":
            existing["fail_count"] = 0  # 成功したらリセット

        existing["attempt_count"] = existing.get("attempt_count", 0) + 1
        log.info(f"改善履歴更新: hash={proposal_hash}, {old_status} -> {existing['status']}")
    else:
        history["proposals"].append({
            "hash": proposal_hash,
            "content_preview": improvement_text[:100],
            "status": status,
            "timestamp": now,
            "attempt_count": 1,
            "fail_count": 1 if status == "failed" else 0
        })
        log.info(f"改善履歴新規記録: hash={proposal_hash}, status={status}")

    # 古いエントリを削除（90日以上前）
    cutoff = datetime.now(timezone.utc).timestamp() - (90 * 24 * 60 * 60)
    history["proposals"] = [
        p for p in history["proposals"]
        if datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")).timestamp() > cutoff
    ]

    save_improvement_history(history)


def is_duplicate_improvement(improvement_text: str) -> tuple[bool, str]:
    """改善提案が重複しているかチェック

    status-based チェック:
    - "success": スキップ（既に成功した改善は再実行不要）
    - "skipped": スキップ（3回以上失敗した提案）
    - "proposed": 再挑戦を許可（提案されたが実行されなかった）
    - "failed": 再挑戦を許可（失敗したが3回未満）

    Returns:
        (スキップすべきか, 理由メッセージ)
    """
    history = load_improvement_history()
    proposal_hash = get_improvement_hash(improvement_text)

    for p in history["proposals"]:
        if p["hash"] == proposal_hash:
            status = p.get("status", p.get("result", "unknown"))
            fail_count = p.get("fail_count", 0)

            # 成功済みはスキップ
            if status == "success":
                return (True, f"同一提案が成功済み（スキップ）")

            # 3回以上失敗（skipped）もスキップ
            if status == "skipped":
                return (True, f"同一提案が{fail_count}回失敗してスキップ対象")

            # proposed（提案のみ）は再挑戦を許可
            if status == "proposed":
                log.info(f"未実行の提案を再挑戦: hash={proposal_hash}")
                return (False, "")

            # failed（失敗）は3回未満なら再挑戦を許可
            if status == "failed":
                if fail_count < 3:
                    log.info(f"失敗した提案を再挑戦（{fail_count}回目）: hash={proposal_hash}")
                    return (False, "")
                else:
                    return (True, f"同一提案が{fail_count}回失敗してスキップ対象")

    return (False, "")


# --- 重複改善提案チェック（status-basedのみ） ---
# 注: 旧 check_duplicate_improvements は削除。status-based の is_duplicate_improvement のみ使用

# --- journal圧縮関数 ---
def compress_journal_for_reflection(journal_text: str, max_chars: int = 2500) -> str:
    """振り返り用にjournalを圧縮する。

    (1) journalを「### 」区切りでエントリ分割
    (2) 連続する「振り返り失敗」エントリを1行に集約
    (3) 直近2エントリのみ全文保持、それ以前は###タイトル行のみ保持
    (4) max_chars以内に収める
    """
    if not journal_text or not journal_text.strip():
        return journal_text

    # ### 区切りでエントリ分割
    entries = []
    current_entry = []
    for line in journal_text.splitlines():
        if line.startswith("### ") and current_entry:
            entries.append("\n".join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)
    if current_entry:
        entries.append("\n".join(current_entry))

    # ### で始まらないヘッダ部分を分離
    header_entries = []
    body_entries = []
    for entry in entries:
        if entry.strip().startswith("### "):
            body_entries.append(entry)
        else:
            header_entries.append(entry)

    if not body_entries:
        return journal_text[:max_chars]

    # 連続する「振り返り失敗」エントリを集約
    compressed_entries = []
    fail_streak = []
    for entry in body_entries:
        first_line = entry.splitlines()[0] if entry.splitlines() else ""
        if "振り返り失敗" in first_line:
            # 時刻を抽出
            time_match = re.search(r'(\d{1,2}:\d{2})', first_line)
            fail_time = time_match.group(1) if time_match else "??:??"
            fail_streak.append(fail_time)
        else:
            if fail_streak:
                compressed_entries.append(f"### 振り返り失敗×{len(fail_streak)}回(最終: {fail_streak[-1]})")
                fail_streak = []
            compressed_entries.append(entry)
    if fail_streak:
        compressed_entries.append(f"### 振り返り失敗×{len(fail_streak)}回(最終: {fail_streak[-1]})")

    if not compressed_entries:
        return journal_text[:max_chars]

    # 直近2エントリは全文保持、それ以前は###タイトル行のみ
    if len(compressed_entries) <= 2:
        result_entries = compressed_entries
    else:
        old_entries = compressed_entries[:-2]
        recent_entries = compressed_entries[-2:]
        title_only = []
        for entry in old_entries:
            first_line = entry.splitlines()[0] if entry.splitlines() else entry
            title_only.append(first_line)
        result_entries = title_only + recent_entries

    result = "\n".join(result_entries)

    # max_chars制限
    if len(result) > max_chars:
        result = result[-max_chars:]

    return result


# --- Google Driveバックアップ ---
async def _drive_backup_silent():
    """振り返り後の自動バックアップ"""
    try:
        from gdrive import upload_file, is_configured
        from config import STATE_PATH
        if not is_configured():
            return
        upload_file(str(JOURNAL_PATH))
        upload_file(str(STATE_PATH))
        log.info("Drive自動バックアップ完了")
    except Exception as e:
        log.debug(f"Drive自動バックアップスキップ: {e}")

# --- 振り返りサイクル ---
async def reflection_cycle(client: httpx.AsyncClient) -> tuple[bool, str]:
    """振り返り実行。戻り値: (実行したかどうか, 振り返り結果テキスト)"""
    global _reflecting
    if _reflecting:
        log.warning("振り返り中のため新しい振り返り要求を無視")
        return (False, "")
    _reflecting = True
    try:
        result = await _reflection_cycle_inner(client)
        return (True, result)
    finally:
        _reflecting = False

async def _reflection_cycle_inner(client: httpx.AsyncClient) -> str:
    """振り返り実行の内部処理。戻り値: 振り返り結果のテキスト"""
    from god import tg_send
    log.info("振り返りサイクル開始")
    state = load_state()
    journal_tail = read_file(JOURNAL_PATH, tail=50)

    # journalを圧縮してプロンプトサイズを削減
    journal_compressed = compress_journal_for_reflection(journal_tail, max_chars=2500)

    # 有効なモジュール一覧を生成
    valid_module_list = ", ".join([f"{name}.py" for name in VALID_MODULES])

    # キュー情報と失敗ジョブ情報を収集
    queued_jobs = get_queued_job_summaries()
    queued_info = "(なし)"
    if queued_jobs:
        queued_lines = []
        for qj in queued_jobs:
            queued_lines.append(f"- [{qj['status']}] {qj['target_file']}: {qj['instruction']}")
        queued_info = "\n".join(queued_lines)

    failed_jobs = get_recent_failed_jobs(limit=3)
    failed_info = "(なし)"
    if failed_jobs:
        failed_lines = []
        for fj in failed_jobs:
            failed_lines.append(f"- {fj['target_file']}: {fj['instruction']} (エラー: {fj['error']})")
        failed_info = "\n".join(failed_lines)

    # 改善対象のファイル分散のために、最近改善されたモジュールを取得
    stats = load_growth_stats()
    modules_improved = stats.get("modules_improved", {})
    least_improved = sorted(modules_improved.items(), key=lambda x: x[1])
    least_improved_list = ", ".join([f"{name}({count}回)" for name, count in least_improved[:3]])

    # 各モジュールの関数一覧を抽出
    module_func_lines = []
    for mod_name, mod_path in MODULE_PATHS.items():
        if mod_path.exists():
            try:
                mod_code = mod_path.read_text(encoding="utf-8")
                tree = ast.parse(mod_code)
                func_names = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if func_names:
                    module_func_lines.append(f"{mod_name}.py: {', '.join(func_names)}")
            except Exception:
                pass
    module_func_text = "\n".join(module_func_lines) if module_func_lines else "(抽出失敗)"

    prompt = f"""あなたはGod AI。自律型AIとして振り返りを行え。

【現在の状態】
{json.dumps(state, ensure_ascii=False)}

【最近のjournal】
{journal_compressed}

【対象モジュール一覧】（これ以外のファイル名を提案するな）
{valid_module_list}

【各モジュールの関数一覧】
{module_func_text}

CODE_IMPROVEMENTで関数名を指定する場合は、上の一覧にある関数名だけ使え。
存在しない関数名（run_self_growth, analyze_error, process_job, save_error_context等）を書くな。

【キューに既にあるジョブ】
{queued_info}

【直近3回の失敗ジョブ】
{failed_info}

【改善回数が少ないモジュール】
{least_improved_list}

【タスク】
以下の4つに答えろ：
1. 今日何をした？
2. 何が問題だった？
3. 次に何をすべき？
4. 自分のコードに改善点はあるか？（具体的に、上記モジュール一覧のファイルのみ対象）

簡潔に日本語で答えろ。
コードの改善点がある場合は「CODE_IMPROVEMENT:」で始まる行に具体的な修正内容を書け。
【重要ルール】
- 存在しないファイル（self_growth.py等）を提案してはならない
- 存在しない関数名を提案してはならない。上の関数一覧にある名前だけ使え
- 既にキューにある改善と同じものを提案するな。別の改善を提案しろ
- 直近3回で失敗した改善と同じものは提案するな
- 改善対象のファイルを分散させろ（改善回数が少ないモジュールを優先）
- 1つのCODE_IMPROVEMENTで1つのファイルの1つの関数だけを対象にしろ"""

    try:
        reflection, brain_name = await think_gemini(prompt)
    except Exception as e:
        log.error(f"Reflection failed: {e}")
        error_msg = f"振り返り失敗: {e}"
        await safe_append_journal(f"### {datetime.now().strftime('%H:%M')} {error_msg}")
        return error_msg

    # journalに追記
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    await safe_append_journal(f"### {now} 振り返り (brain: {brain_name})\n{reflection}")

    # state更新
    state["growth_cycles"] = state.get("growth_cycles", 0) + 1
    state["last_reflection"] = now
    await safe_save_state(state)

    # Google Driveバックアップ
    await _drive_backup_silent()

    # コード改善提案チェック
    if "CODE_IMPROVEMENT:" in reflection:
        improvements = []
        for line in reflection.splitlines():
            cleaned = line.strip().lstrip('*- ').strip()
            if cleaned.startswith("CODE_IMPROVEMENT:"):
                improvements.append(cleaned.replace("CODE_IMPROVEMENT:", "").strip())

        if improvements:
            improvement_text = "\n".join(improvements)

            # 無効なファイル名チェック（存在しないモジュールを提案していないか）
            invalid_files = _detect_invalid_module_names(improvement_text)
            if invalid_files:
                log.warning(f"無効なファイル名を検出: {invalid_files}")
                skip_msg = f"### {now} 自己改善スキップ（無効なファイル名）\n無効: {invalid_files}\n提案: {improvement_text[:200]}"
                await safe_append_journal(skip_msg)
                await tg_send(client, f"自己改善スキップ: 存在しないファイル名を検出 ({', '.join(invalid_files)})")
                return reflection

            # status-based 重複チェック（successとskippedのみスキップ）
            is_dup, dup_reason = is_duplicate_improvement(improvement_text)
            if is_dup:
                log.info(f"重複チェック: {dup_reason}")
                update_growth_stats(success=False, failure_reason="duplicate", module="unknown", duration=0)
                skip_msg = f"### {now} 自己改善スキップ（{dup_reason}）\n改善内容: {improvement_text}"
                await safe_append_journal(skip_msg)
                await tg_send(client, f"自己改善スキップ: {dup_reason}\n提案: {improvement_text[:200]}")
            else:
                # 提案段階で "proposed" を記録（まだ実行前）
                record_improvement(improvement_text, "proposed")
                log.info(f"改善提案を記録 (status=proposed): {improvement_text[:50]}")
                # ジョブキューにself_improveジョブを登録
                result = select_target_module(improvement_text)
                if result:
                    target_path, target_name = result
                    # 依存ファイル自動推定（対象ファイルのimport文を解析）
                    dep_files = _detect_read_files(target_path, target_name)
                    new_job = create_job(
                        job_type="self_improve",
                        priority="P3",
                        target_file=f"core/{target_name}.py",
                        read_files=dep_files,
                        instruction=improvement_text,
                        constraints=["構文チェック必須", "テスト必須"],
                        estimated_seconds=120,
                    )
                    if new_job:
                        log.info(f"自己改善ジョブ作成: {target_name}.py")
                        await tg_send(client, f"自己改善ジョブ登録: {target_name}.py\n改善: {improvement_text[:100]}")
                    else:
                        log.info(f"自己改善ジョブ重複スキップ: {target_name}.py")

    # CLAUDE.md自動更新
    update_claude_md()

    log.info("振り返りサイクル完了")
    return reflection


# --- 定期振り返りスケジューラ ---
async def reflection_scheduler(client: httpx.AsyncClient):
    """定期的に振り返り実行"""
    from god import tg_send
    log.info(f"振り返りスケジューラ開始 (間隔: {REFLECTION_INTERVAL}秒)")
    while True:
        try:
            await asyncio.sleep(REFLECTION_INTERVAL)
            log.info("定期振り返り: 開始")
            if _reflecting:
                log.warning("定期振り返り: 手動振り返り中のためスキップ")
                continue
            await tg_send(client, f"定期振り返り開始... (次回: {REFLECTION_INTERVAL}秒後)")
            executed, result = await reflection_cycle(client)
            if executed:
                summary = result[:1000] + "..." if len(result) > 1000 else result
                await tg_send(client, f"定期振り返り完了。\n\n{summary}")
                log.info("定期振り返り: 完了")
            else:
                log.warning("定期振り返り: 他の振り返りと競合のためスキップ")
        except asyncio.CancelledError:
            log.info("振り返りスケジューラ: キャンセルされました")
            raise
        except Exception as e:
            log.error(f"Scheduled reflection failed: {e}", exc_info=True)
            await safe_append_journal(f"### {datetime.now().strftime('%H:%M')} 定期振り返りエラー\n{e}")
            await asyncio.sleep(10)

# --- 自己成長スケジューラ（3フェーズ版） ---
async def _wait_for_pending_improvements(timeout: int = 600):
    """Wait for all queued/running self_improve jobs to complete.

    Args:
        timeout: Maximum wait time in seconds (default 10 minutes).
    """
    start = time_module.time()
    while time_module.time() - start < timeout:
        jobs = load_queue()
        pending = [
            j for j in jobs
            if j["type"] == "self_improve"
            and j["status"] in ("queued", "running")
        ]
        if not pending:
            log.info("All self_improve jobs completed")
            return

        log.debug(f"Waiting for {len(pending)} self_improve jobs...")
        await asyncio.sleep(5)

    log.warning(f"Timed out waiting for self_improve jobs after {timeout}s")


async def self_growth_scheduler(client: httpx.AsyncClient):
    """3-phase self-growth: research -> analyze_plan -> wait for self_improve.

    Phase 1 (Research): Pure Python code analysis, no AI.
                        Scans all modules, extracts structure, saves report.
    Phase 2 (Analyze):  Gemini analyzes report, creates self_improve jobs.
    Phase 3 (Wait):     job_worker_loop executes self_improve jobs.
                        This scheduler waits for completion.
    """
    from config import GROWTH_CYCLE_SLEEP
    from job_worker import _execute_research, _execute_analyze_plan

    log.info(f"3-phase self-growth scheduler started (interval: {GROWTH_CYCLE_SLEEP}s)")
    await asyncio.sleep(30)  # Wait 30s after startup

    while True:
        try:
            # P1 interrupt check
            if is_p1_interrupted():
                log.info("self_growth: P1 interrupt, waiting")
                clear_p1_interrupt()
                await asyncio.sleep(GROWTH_CYCLE_SLEEP)
                continue

            # === Phase 1: Research (no AI, pure code analysis) ===
            log.info("=== Growth Phase 1: Research ===")
            report = _execute_research()

            if not report or not report.get("modules"):
                log.warning("Research phase produced empty report, skipping cycle")
                await asyncio.sleep(GROWTH_CYCLE_SLEEP)
                continue

            # === Phase 2: Analyze + Plan (Gemini -> create self_improve jobs) ===
            log.info("=== Growth Phase 2: Analyze + Plan ===")
            await _execute_analyze_plan(client, report)

            # === Phase 3: Wait for self_improve jobs to complete ===
            log.info("=== Growth Phase 3: Waiting for improvements ===")
            await _wait_for_pending_improvements()

            # Update CLAUDE.md and state
            update_claude_md()
            state = load_state()
            state["growth_cycles"] = state.get("growth_cycles", 0) + 1
            state["last_reflection"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            await safe_save_state(state)

            await asyncio.sleep(GROWTH_CYCLE_SLEEP)
        except asyncio.CancelledError:
            log.info("self_growth_scheduler: Cancelled")
            raise
        except Exception as e:
            log.error(f"self_growth_scheduler error: {e}", exc_info=True)
            await safe_append_journal(
                f"### {datetime.now().strftime('%H:%M')} Growth cycle error\n{e}"
            )
            await asyncio.sleep(GROWTH_CYCLE_SLEEP)
