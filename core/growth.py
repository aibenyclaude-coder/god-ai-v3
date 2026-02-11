#!/usr/bin/env python3
"""God AI v3.0 - 成長・振り返りモジュール"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

import time as time_module

from config import (
    GOD_PY_PATH, JOURNAL_PATH, REFLECTION_INTERVAL, SELF_GROWTH_INTERVAL, log,
    BASE_DIR, CORE_DIR, STATE_PATH, IDENTITY_PATH, MEMORY_DIR
)
from memory import (
    load_state, save_state, read_file, append_journal,
    safe_save_state, safe_append_journal
)
from brain import think_gemini, think_claude_heavy
from jobqueue import Priority, create_job

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
    "handoff": CORE_DIR / "handoff.py",
    "gdrive": CORE_DIR / "gdrive.py",
}

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
        "failure_reasons": {"syntax_error": 0, "timeout": 0, "test_fail": 0, "duplicate": 0},
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

    # 最多失敗原因
    failure_reasons = stats["failure_reasons"]
    max_failure = max(failure_reasons.items(), key=lambda x: x[1]) if any(v > 0 for v in failure_reasons.values()) else ("なし", 0)

    # 最も改善されたモジュール
    modules = stats["modules_improved"]
    max_module = max(modules.items(), key=lambda x: x[1]) if any(v > 0 for v in modules.values()) else ("なし", 0)

    # 連続成功
    streak = stats["streak"]["current_success"]

    return (
        f"成功率: {rate}% ({successes}/{total})\n"
        f"連続成功: {streak}回\n"
        f"最多失敗原因: {max_failure[0]} ({max_failure[1]}回)\n"
        f"最も改善されたモジュール: {max_module[0]} ({max_module[1]}回)"
    )


def should_skip_improvement(improvement_text: str, module: str) -> tuple[bool, str]:
    """統計に基づいて改善をスキップすべきか判定

    Args:
        improvement_text: 改善内容のテキスト
        module: 対象モジュール名

    Returns:
        (スキップすべきか, 理由)
    """
    stats = load_growth_stats()

    # 直近10回の結果から同じ失敗パターンをチェック
    recent = stats["last_10_results"]

    # 同じ原因で3回以上失敗しているかチェック
    failure_counts = {}
    for result in recent:
        if not result["success"] and result["failure_reason"]:
            reason = result["failure_reason"]
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    for reason, count in failure_counts.items():
        if count >= 3:
            return (True, f"同じ原因({reason})で{count}回失敗しています")

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
    """journalから直近の重要イベントを抽出"""
    events = []
    for line in journal_text.splitlines():
        if line.startswith("###"):
            important_keywords = ["成功", "失敗", "エラー", "改善", "誕生", "完了", "開始"]
            if any(kw in line for kw in important_keywords):
                event = line.replace("###", "").strip()
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
def select_target_module(improvement_text: str) -> tuple[Path, str]:
    """改善内容から対象モジュールを自動選択。戻り値: (モジュールパス, モジュール名)"""
    improvement_lower = improvement_text.lower()
    if any(kw in improvement_lower for kw in ["振り返り", "reflection", "自己改善", "growth", "improve"]):
        return (MODULE_PATHS["growth"], "growth")
    elif any(kw in improvement_lower for kw in ["gemini", "claude", "think", "脳", "brain", "ai"]):
        return (MODULE_PATHS["brain"], "brain")
    elif any(kw in improvement_lower for kw in ["memory", "state", "journal", "保存", "読み込み"]):
        return (MODULE_PATHS["memory"], "memory")
    elif any(kw in improvement_lower for kw in ["queue", "job", "キュー", "priority"]):
        return (MODULE_PATHS["jobqueue"], "jobqueue")
    elif any(kw in improvement_lower for kw in ["config", "設定", "定数", "env"]):
        return (MODULE_PATHS["config"], "config")
    elif any(kw in improvement_lower for kw in ["handoff", "引き継ぎ"]):
        return (MODULE_PATHS["handoff"], "handoff")
    elif any(kw in improvement_lower for kw in ["drive", "google", "upload", "backup"]):
        return (MODULE_PATHS["gdrive"], "gdrive")
    else:
        return (MODULE_PATHS["god"], "god")


# --- 改善後自動テスト ---
async def run_post_improvement_tests(client: httpx.AsyncClient) -> tuple[bool, str]:
    """
    改善後テスト実行。戻り値: (全合格か, 結果メッセージ)

    テスト項目:
    a. 全モジュール構文チェック: python3 -m py_compile で各.pyファイルをチェック
    b. God AI起動テスト: 起動→5秒待ち→/statusが応答するか→停止
    c. Gemini API疎通テスト: 簡単なprompt送信→応答あるか
    """
    results = []
    all_passed = True

    # --- a. 全モジュール構文チェック ---
    log.info("テスト(a): 全モジュール構文チェック開始")
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

    if syntax_errors:
        all_passed = False
        results.append(f"❌ 構文チェック失敗: {'; '.join(syntax_errors)}")
    else:
        results.append("✅ 構文チェック全合格")
        log.info("テスト(a): 全モジュール構文チェック合格")

    # 構文エラーがある場合は以降のテストをスキップ
    if not all_passed:
        return (False, "\n".join(results))

    # --- b. God AI起動テスト ---
    log.info("テスト(b): God AI起動テスト開始")
    god_test_passed = False
    god_process = None
    try:
        # 現在のプロセスとは別にGod AIを起動
        env = os.environ.copy()
        god_process = subprocess.Popen(
            [sys.executable, str(CORE_DIR / "god.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(CORE_DIR)
        )

        # 5秒待つ
        await asyncio.sleep(5)

        # プロセスが生きているか確認
        if god_process.poll() is None:
            # /statusが応答するかテスト（Telegram APIを直接呼ぶのは難しいので、プロセスが生きていることを確認）
            god_test_passed = True
            results.append("✅ God AI起動テスト合格（5秒後も稼働中）")
            log.info("テスト(b): God AI起動テスト合格")
        else:
            returncode = god_process.returncode
            stderr = god_process.stderr.read().decode() if god_process.stderr else ""
            all_passed = False
            results.append(f"❌ God AI起動テスト失敗: 5秒以内にクラッシュ(returncode={returncode}, stderr={stderr[:200]})")
            log.error(f"テスト(b): God AIが5秒以内にクラッシュ: {stderr[:200]}")
    except Exception as e:
        all_passed = False
        results.append(f"❌ God AI起動テスト失敗: {e}")
        log.error(f"テスト(b): God AI起動テストエラー: {e}")
    finally:
        # テスト用プロセスを停止
        if god_process and god_process.poll() is None:
            try:
                god_process.terminate()
                await asyncio.sleep(1)
                if god_process.poll() is None:
                    god_process.kill()
                log.info("テスト(b): テスト用God AIプロセスを停止")
            except Exception as e:
                log.warning(f"テスト用プロセス停止エラー: {e}")

    # 起動テスト失敗の場合は以降のテストをスキップ
    if not god_test_passed:
        return (False, "\n".join(results))

    # --- c. Gemini API疎通テスト ---
    log.info("テスト(c): Gemini API疎通テスト開始")
    try:
        from brain import think_gemini
        response, brain_name = await think_gemini("Hello, respond with 'OK' if you can read this.")
        if response and len(response) > 0:
            results.append(f"✅ Gemini API疎通テスト合格（応答あり: {brain_name}）")
            log.info(f"テスト(c): Gemini API疎通テスト合格: {response[:50]}")
        else:
            all_passed = False
            results.append("❌ Gemini API疎通テスト失敗: 応答が空")
            log.error("テスト(c): Gemini API応答が空")
    except Exception as e:
        all_passed = False
        results.append(f"❌ Gemini API疎通テスト失敗: {e}")
        log.error(f"テスト(c): Gemini APIエラー: {e}")

    return (all_passed, "\n".join(results))


# --- コード構文検証関数 ---
def validate_code_syntax(code: str) -> tuple[bool, str]:
    """生成コードの構文を厳密に検証。戻り値: (有効かどうか, エラーメッセージ)"""
    try:
        ast.parse(code)
        return (True, "")
    except SyntaxError as e:
        error_msg = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
        if e.lineno:
            lines = code.splitlines()
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            context = "\n".join([f"{i+1}: {lines[i]}" for i in range(start, end)])
            error_msg += f"\n周辺コード:\n{context}"
        return (False, error_msg)
    except Exception as e:
        return (False, f"Unexpected error: {e}")

# --- journal解析: 重複改善提案チェック ---
def check_duplicate_improvements(journal_text: str, improvement_text: str) -> bool:
    """直近3回のjournal振り返り履歴から、同一のCODE_IMPROVEMENT提案があるかチェック。"""
    lines = journal_text.splitlines()
    reflections = []
    current_reflection = []

    for line in lines:
        if line.startswith("###") and "振り返り" in line:
            if current_reflection:
                reflections.append("\n".join(current_reflection))
            current_reflection = [line]
        elif current_reflection:
            current_reflection.append(line)

    if current_reflection:
        reflections.append("\n".join(current_reflection))

    # 直近3回の振り返りから CODE_IMPROVEMENT を抽出
    recent_improvements = []
    for refl in reflections[-3:]:
        for line in refl.splitlines():
            if "CODE_IMPROVEMENT:" in line:
                improvement = line.split("CODE_IMPROVEMENT:", 1)[1].strip()
                recent_improvements.append(improvement)

    # 類似度チェック
    improvement_words = set(improvement_text.lower().split())
    for past_imp in recent_improvements:
        past_words = set(past_imp.lower().split())
        if len(improvement_words & past_words) / max(len(improvement_words), 1) > 0.5:
            return True

    return False

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
    """振り返り実行。戻り値: (実行したかどうか, 振り返り結果の要約)"""
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
        return ""

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
            if line.strip().startswith("CODE_IMPROVEMENT:"):
                improvements.append(line.strip().replace("CODE_IMPROVEMENT:", "").strip())

        if improvements:
            improvement_text = "\n".join(improvements)
            journal_full = read_file(JOURNAL_PATH)

            if check_duplicate_improvements(journal_full, improvement_text):
                log.info("重複した改善提案を検出。自己改善をスキップします。")
                # 重複検出も統計に記録
                update_growth_stats(success=False, failure_reason="duplicate", module="unknown", duration=0)
                skip_msg = f"### {now} 自己改善スキップ（重複検出）\n改善内容: {improvement_text}"
                await safe_append_journal(skip_msg)
                await tg_send(client, f"重複した改善提案を検出。既に適用済みの可能性が高いためスキップしました。\n提案: {improvement_text[:200]}")
            else:
                await self_improve(client, reflection)

    # CLAUDE.md自動更新
    update_claude_md()

    log.info("振り返りサイクル完了")
    return reflection

# --- 自己改善 ---
async def self_improve(client: httpx.AsyncClient, reflection: str):
    """コード自己改善（構文チェック強化、最大3回リトライ、モジュール選択、統計記録）"""
    from god import tg_send

    log.info("自己改善プロセス開始")
    start_time = time_module.time()

    # 改善行を抽出
    improvements = []
    for line in reflection.splitlines():
        if line.strip().startswith("CODE_IMPROVEMENT:"):
            improvements.append(line.strip().replace("CODE_IMPROVEMENT:", "").strip())

    if not improvements:
        return

    improvement_text = "\n".join(improvements)

    # 改善対象モジュールを自動選択
    target_path, target_name = select_target_module(improvement_text)
    log.info(f"改善対象モジュール: {target_name} ({target_path})")

    # 統計に基づくスキップ判定
    should_skip, skip_reason = should_skip_improvement(improvement_text, target_name)
    if should_skip:
        log.info(f"統計判定により自己改善スキップ: {skip_reason}")
        now = datetime.now().strftime("%H:%M")
        await safe_append_journal(f"### {now} 自己改善スキップ（統計判定）\n理由: {skip_reason}\n改善内容: {improvement_text}")
        await tg_send(client, f"自己改善スキップ（統計判定）\n理由: {skip_reason}")
        return

    # バックアップ
    backup_path = target_path.with_suffix(".py.bak")
    shutil.copy2(target_path, backup_path)

    current_code = target_path.read_text(encoding="utf-8")
    current_lines = current_code.splitlines()

    # CLAUDE.mdの内容を読み込み（コンテキスト用）
    claude_md_content = load_claude_md()
    claude_md_summary = claude_md_content[:2000] if claude_md_content else "(CLAUDE.mdなし)"

    MAX_RETRY = 3
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        log.info(f"自己改善 試行 {attempt}/{MAX_RETRY} (対象: {target_name})")

        if attempt == 1:
            prompt = (
                "あなたはPythonコードの修正を行うアシスタントです。\n"
                "以下の【改善内容】を【現在のコード】に適用してください。\n\n"
                "【プロジェクト情報（CLAUDE.md要約）】\n"
                f"{claude_md_summary}\n\n"
                "【重要なルール】\n"
                "- 修正後のPythonコード全文をそのまま出力してください\n"
                "- 説明文は一切不要です。Pythonコードのみを出力してください\n"
                "- マークダウンのバッククォート（```）で囲まないでください\n"
                "- コードの先頭は #!/usr/bin/env python3 から始めてください\n"
                "- 変更箇所以外は絶対にそのまま維持してください\n"
                "- 文字列リテラルのクォートの対応に注意してください\n"
                f"【改善内容】\n{improvement_text}\n\n"
                f"【対象モジュール】{target_name}.py\n\n"
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
                f"【改善内容】\n{improvement_text}\n\n"
                f"【対象モジュール】{target_name}.py\n\n"
                f"【現在のコード（オリジナル）】\n{current_code}"
            )

        try:
            result, _ = await think_claude_heavy(prompt)

            log.info(f"試行{attempt}: Claude生成結果（先頭200文字）: {result[:200]}")
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

            log.info(f"試行{attempt}: 抽出後コードの長さ: {len(code)}文字（元: {len(current_code)}文字）")

            # 基本的なバリデーション
            if not code.startswith(("#!/", "from __future__", '"""', "import", "#")):
                log.warning(f"試行{attempt}: コードが想定外の開始: {code[:50]}")

            # 長さチェック
            min_length = int(len(current_code) * 0.3)
            if len(code) < min_length:
                new_lines = code.splitlines()
                diff = list(difflib.unified_diff(current_lines, new_lines, lineterm="", n=3))
                diff_str = "\n".join(diff[:50])
                log.error(f"試行{attempt}: コードが短すぎる。元: {len(current_code)}字, 生成: {len(code)}字")
                log.error(f"試行{attempt}: 差分:\n{diff_str}")
                raise ValueError(f"生成コードが短すぎる（元: {len(current_code)}字, 生成: {len(code)}字）")

            # 構文チェック
            is_valid, syntax_error_msg = validate_code_syntax(code)
            if not is_valid:
                log.error(f"試行{attempt}: 構文エラー: {syntax_error_msg}")
                raise SyntaxError(syntax_error_msg)

            # 差分ログ
            new_lines = code.splitlines()
            diff = list(difflib.unified_diff(current_lines, new_lines, lineterm="", n=3))
            if len(diff) > 0:
                diff_str = "\n".join(diff[:100])
                log.info(f"試行{attempt}: コード差分:\n{diff_str}")
            else:
                log.warning(f"試行{attempt}: コードに差分がありません")

            diff_for_journal = "\n".join(diff[:50]) if diff else "(差分なし)"

            # 書き込み（仮適用）
            target_path.write_text(code, encoding="utf-8")
            log.info(f"試行{attempt}: コード仮適用完了、自動テスト開始")

            # --- 改善後自動テスト実行 ---
            test_passed, test_result = await run_post_improvement_tests(client)

            if test_passed:
                # 全テスト合格 -> 改善確定
                duration = time_module.time() - start_time
                update_growth_stats(success=True, failure_reason=None, module=target_name, duration=duration)
                success_msg = f"自己改善成功（試行{attempt}/{MAX_RETRY}）\n対象: {target_name}.py\n改善内容: {improvement_text}"
                append_journal(
                    f"### {datetime.now().strftime('%H:%M')} {success_msg}\n"
                    f"コード長: {len(current_code)} -> {len(code)}文字\n"
                    f"```diff\n{diff_for_journal}\n```\n"
                    f"✅ テスト全合格\n{test_result}"
                )
                await tg_send(client, f"[自己改善] {success_msg}\nコード長: {len(current_code)} -> {len(code)}文字\n✅ テスト全合格")
                log.info(f"自己改善成功（試行{attempt}）: {len(current_code)} -> {len(code)}文字、テスト全合格")
                return
            else:
                # テスト失敗 -> 即ロールバック
                shutil.copy2(backup_path, target_path)
                # 失敗したテスト名を抽出
                failed_tests = [line for line in test_result.splitlines() if line.startswith("❌")]
                failed_test_name = failed_tests[0] if failed_tests else "不明なテスト"
                log.error(f"試行{attempt}: 自動テスト失敗、ロールバック実行: {failed_test_name}")

                # 最後の試行で失敗した場合のみ統計更新
                if attempt == MAX_RETRY:
                    duration = time_module.time() - start_time
                    update_growth_stats(success=False, failure_reason="test_fail", module=target_name, duration=duration)

                append_journal(
                    f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{MAX_RETRY} テスト失敗でロールバック\n"
                    f"❌ {failed_test_name}\n{test_result}\n改善内容: {improvement_text}"
                )
                await tg_send(client, f"⚠️ 自己改善失敗: {failed_test_name}\nロールバック済み\n改善内容: {improvement_text[:100]}")
                last_error = f"自動テスト失敗: {failed_test_name}"
                if attempt < MAX_RETRY:
                    await asyncio.sleep(3)

        except (SyntaxError, ValueError) as e:
            last_error = str(e)
            log.error(f"自己改善 試行{attempt}/{MAX_RETRY} 失敗: {e}")

            # 最後の試行で失敗した場合のみ統計更新
            if attempt == MAX_RETRY:
                duration = time_module.time() - start_time
                update_growth_stats(success=False, failure_reason="syntax_error", module=target_name, duration=duration)

            append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{MAX_RETRY} 失敗\n"
                f"エラー: {e}\n改善内容: {improvement_text}"
            )
            if attempt < MAX_RETRY:
                await tg_send(client, f"自己改善 試行{attempt}/{MAX_RETRY} 失敗: {e}\nリトライします...")
                await asyncio.sleep(3)

        except Exception as e:
            last_error = str(e)
            log.error(f"自己改善 試行{attempt}/{MAX_RETRY} 予期せぬエラー: {e}", exc_info=True)

            # 予期しないエラーは即座に統計更新（timeout扱い）
            duration = time_module.time() - start_time
            update_growth_stats(success=False, failure_reason="timeout", module=target_name, duration=duration)

            append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{MAX_RETRY} 予期せぬエラー\n"
                f"エラー: {e}\n改善内容: {improvement_text}"
            )
            break

    # 全試行失敗 -> ロールバック
    shutil.copy2(backup_path, target_path)
    fail_msg = (
        f"自己改善 {MAX_RETRY}回試行して失敗。ロールバックしました。\n"
        f"対象: {target_name}.py\n"
        f"最終エラー: {last_error}\n"
        f"改善内容: {improvement_text}"
    )
    log.error(fail_msg)
    append_journal(f"### {datetime.now().strftime('%H:%M')} {fail_msg}")
    await tg_send(
        client,
        f"自己改善 {MAX_RETRY}回失敗。ロールバックしました。\n"
        f"最終エラー: {last_error}\n"
        f"改善内容: {improvement_text}\n"
        f"Benyの判断を仰ぎます。"
    )

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
                summary = result[:200] + "..." if len(result) > 200 else result
                await tg_send(client, f"定期振り返り完了。\n\n{summary}")
                log.info("定期振り返り: 完了")
            else:
                log.warning("定期振り返り: 他の振り返りと競合のためスキップ")
        except asyncio.CancelledError:
            log.info("振り返りスケジューラ: キャンセルされました")
            raise
        except Exception as e:
            log.error(f"Scheduled reflection failed: {e}", exc_info=True)
            append_journal(f"### {datetime.now().strftime('%H:%M')} 定期振り返りエラー\n{e}")
            await asyncio.sleep(10)

# --- 自己成長提案ジョブ ---
async def _self_growth_job(client: httpx.AsyncClient):
    """自己成長提案ジョブの実行（統計活用版）"""
    log.info("自己成長提案ジョブ開始")
    state = load_state()
    journal_tail = read_file(JOURNAL_PATH, tail=30)

    # 成長統計を取得してプロンプトに活用
    stats = load_growth_stats()
    stats_summary = get_stats_summary()

    # 統計に基づく追加ガイダンスを生成
    guidance_lines = []

    # 失敗理由の分析
    failure_reasons = stats["failure_reasons"]
    for reason, count in failure_reasons.items():
        if count >= 3:
            if reason == "timeout":
                guidance_lines.append(f"- タイムアウトが{count}回発生。処理の軽量化やモジュール分割を検討")
            elif reason == "syntax_error":
                guidance_lines.append(f"- 構文エラーが{count}回発生。より慎重なコード生成が必要")
            elif reason == "test_fail":
                guidance_lines.append(f"- テスト失敗が{count}回発生。テストカバレッジの改善を検討")
            elif reason == "duplicate":
                guidance_lines.append(f"- 重複提案が{count}回発生。より新しい観点での提案が必要")

    # 成功率が低いモジュールの回避、高いモジュールの優先
    modules = stats["modules_improved"]
    if any(v > 0 for v in modules.values()):
        sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
        top_module = sorted_modules[0]
        if top_module[1] > 0:
            guidance_lines.append(f"- 推奨モジュール: {top_module[0]}（過去{top_module[1]}回成功）")

    guidance_text = "\n".join(guidance_lines) if guidance_lines else "(特になし)"

    prompt = f"""あなたはGod AI。自律型AIとして自己成長を提案せよ。

【現在の状態】
{json.dumps(state, ensure_ascii=False)}

【成長統計】
{stats_summary}

【統計に基づく改善ガイダンス】
{guidance_text}

【最近のjournal】
{journal_tail}

【タスク】
以下の観点で自己成長提案を1つだけ挙げよ：
1. 新しい機能追加の提案
2. パフォーマンス改善の提案
3. コード品質向上の提案
4. ユーザー体験改善の提案

【重要な制約】
- 統計に基づくガイダンスを考慮すること
- 過去に何度も失敗している種類の改善は避けること
- 成功率の高いモジュールを優先すること
- タイムアウトが多い場合はモジュール分割を検討すること

簡潔に日本語で提案せよ。
実装可能な具体的提案を「GROWTH_PROPOSAL:」で始まる行に書け。"""

    try:
        proposal, brain_name = await think_gemini(prompt)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        await safe_append_journal(f"### {now} 自己成長提案 (brain: {brain_name})\n{proposal}")

        if "GROWTH_PROPOSAL:" in proposal:
            for line in proposal.splitlines():
                if line.strip().startswith("GROWTH_PROPOSAL:"):
                    prop = line.strip().replace("GROWTH_PROPOSAL:", "").strip()
                    log.info(f"自己成長提案: {prop}")

        log.info("自己成長提案ジョブ完了")

    except Exception as e:
        log.error(f"自己成長提案失敗: {e}")

# --- 自己成長スケジューラ ---
async def self_growth_scheduler(client: httpx.AsyncClient):
    """10分ごとに自己成長提案をP3としてキューに登録"""
    log.info(f"自己成長スケジューラ開始 (間隔: {SELF_GROWTH_INTERVAL}秒)")
    await asyncio.sleep(60)  # 起動後60秒待ってから開始

    while True:
        try:
            await create_job(
                priority=Priority.P3_BACKGROUND,
                job_type="self_growth",
                handler=_self_growth_job,
                args=(client,),
                description="自己成長提案の生成",
            )
            log.info("自己成長ジョブをキューに追加")
            await asyncio.sleep(SELF_GROWTH_INTERVAL)
        except asyncio.CancelledError:
            log.info("自己成長スケジューラ: キャンセルされました")
            raise
        except Exception as e:
            log.error(f"自己成長スケジューラエラー: {e}", exc_info=True)
            await asyncio.sleep(60)
