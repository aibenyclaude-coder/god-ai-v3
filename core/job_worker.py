#!/usr/bin/env python3
"""God AI v3.0 - ジョブワーカー（実行エンジン）

ジョブキュー（jobqueue.py）からジョブを取り出して実行する。
対応ジョブタイプ:
- self_improve: 自己改善（関数単位置換→テスト→git commit）
- continuous_growth: 連続成長サイクル（振り返り→改善ジョブ作成）
"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import re
import shutil
import subprocess
import sys
import time as time_module
from datetime import datetime
from pathlib import Path

import httpx

from config import log, BASE_DIR, CORE_DIR, IDENTITY_PATH, JOURNAL_PATH, MEMORY_DIR
from jobqueue import (
    get_next_queued_job, update_job, create_job,
    is_p1_interrupted, clear_p1_interrupt,
    get_queued_job_summaries, get_recent_failed_jobs,
)
from memory import append_journal, safe_append_journal, load_state, safe_save_state, read_file
from brain import think_gemini, think_claude
from growth import (
    select_target_module,
    create_module_backups,
    rollback_from_backups,
    cleanup_backups,
    validate_code_syntax,
    run_post_improvement_tests,
    reload_modules,
    load_claude_md,
    update_growth_stats,
    load_growth_stats,
    record_improvement,
    is_duplicate_improvement,
    should_skip_improvement,
    _detect_invalid_module_names,
    _detect_read_files,
    MODULE_PATHS,
)


# ---------------------------------------------------------------------------
# 0. 関数単位置換ロジック
# ---------------------------------------------------------------------------
def parse_function_replacement(result_text: str) -> dict | None:
    """AI生成結果から<<<TARGET_FILE>>><<<FUNCTION_NAME>>><<<NEW_FUNCTION>>><<<END>>>形式を抽出。
    戻り値: {"target_file": str, "function_name": str, "new_function": str} or None
    """
    if "<<<TARGET_FILE>>>" not in result_text or "<<<FUNCTION_NAME>>>" not in result_text:
        return None
    if "<<<NEW_FUNCTION>>>" not in result_text or "<<<END>>>" not in result_text:
        return None

    try:
        after_target = result_text.split("<<<TARGET_FILE>>>", 1)[1]
        target_file_raw, after_func = after_target.split("<<<FUNCTION_NAME>>>", 1)
        func_name_raw, after_new = after_func.split("<<<NEW_FUNCTION>>>", 1)
        new_function_raw = after_new.split("<<<END>>>", 1)[0]

        target_file = target_file_raw.strip()
        function_name = func_name_raw.strip()
        new_function = new_function_raw.rstrip("\n")
        # 先頭の空行1つだけ除去（コード先頭の改行）
        if new_function.startswith("\n"):
            new_function = new_function[1:]

        if not target_file or not function_name or not new_function:
            return None

        return {
            "target_file": target_file,
            "function_name": function_name,
            "new_function": new_function,
        }
    except (ValueError, IndexError):
        return None


def find_and_replace_function(code: str, function_name: str, new_function: str) -> tuple[bool, str, str]:
    """コード内の指定関数を新しい関数で置換する。
    トップレベル関数、クラスメソッド（インデント付きdef）、async def 全対応。

    Args:
        code: 対象ファイルの全コード
        function_name: 置換対象の関数名
        new_function: 置換後の関数コード全体

    Returns:
        (成功かどうか, 置換後コード, エラーメッセージ)
    """
    # 関数定義の開始を検出（トップレベルまたはインデント付き）
    # async def / def 両対応、デコレータも含む
    pattern = re.compile(
        r'^([ \t]*)(async\s+)?def\s+' + re.escape(function_name) + r'\s*\(',
        re.MULTILINE,
    )

    match = pattern.search(code)
    if not match:
        return (False, code, f"関数 '{function_name}' がコード内に見つかりません")

    # デコレータを含む場合、関数の開始をデコレータまで遡る
    func_def_start = match.start()
    func_start = func_def_start
    code_before = code[:func_def_start]
    before_lines = code_before.split("\n")
    # 末尾から空行を飛ばしてデコレータを探す
    i = len(before_lines) - 1
    while i >= 0:
        stripped = before_lines[i].strip()
        if stripped.startswith("@"):
            # デコレータ行→funcの開始を遡る
            func_start = sum(len(l) + 1 for l in before_lines[:i])
            i -= 1
        elif stripped == "":
            # 空行はスキップ（デコレータの前の空行は含めない）
            break
        else:
            break

    func_indent = match.group(1)  # 関数のインデントレベル
    indent_len = len(func_indent)

    # 関数の終了位置を検出:
    # 同じかそれより浅いインデントの非空行、またはファイル末尾
    lines = code[func_def_start:].split("\n")
    func_end_offset = 0

    for i, line in enumerate(lines):
        if i == 0:
            # 関数定義行自体はスキップ
            func_end_offset += len(line) + 1
            continue

        stripped = line.rstrip()

        # 空行はスキップ
        if not stripped:
            func_end_offset += len(line) + 1
            continue

        # 実際のコード行のインデントを計算
        content_indent = len(line) - len(line.lstrip())

        if content_indent <= indent_len:
            # 同じかそれより浅いインデントの非空行 = 関数外
            break

        func_end_offset += len(line) + 1

    func_end = func_def_start + func_end_offset

    # 末尾の余分な空行を1つだけ残す
    while func_end > func_start and func_end < len(code) and code[func_end - 1] == "\n" and func_end - 2 >= func_start and code[func_end - 2] == "\n":
        func_end -= 1

    # 新しい関数の末尾に改行を確保
    new_func_text = new_function.rstrip("\n") + "\n"

    new_code = code[:func_start] + new_func_text + code[func_end:]

    return (True, new_code, "")


# ---------------------------------------------------------------------------
# 1. テスト関数群（6ステップ自己改善用）
# ---------------------------------------------------------------------------

def syntax_test(code: str) -> tuple[bool, str]:
    """構文テスト: ast.parseでコードの構文をチェック"""
    try:
        ast.parse(code)
        return (True, "構文OK")
    except SyntaxError as e:
        return (False, f"SyntaxError L{e.lineno}: {e.msg}")


def import_test(file_path: Path) -> tuple[bool, str]:
    """importテスト: 別プロセスで対象ファイルをimportできるか確認"""
    module_name = file_path.stem
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True, text=True, timeout=30,
            cwd=str(file_path.parent),
        )
        if proc.returncode == 0:
            return (True, f"import {module_name} OK")
        return (False, f"import失敗: {proc.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        return (False, "importテスト タイムアウト")
    except Exception as e:
        return (False, f"importテスト例外: {e}")


def function_exists_test(code: str, func_name: str) -> tuple[bool, str]:
    """関数存在テスト: 指定関数がコード内にdef定義されているか"""
    pattern = re.compile(
        r'^[ \t]*(async\s+)?def\s+' + re.escape(func_name) + r'\s*\(',
        re.MULTILINE,
    )
    if pattern.search(code):
        return (True, f"関数 '{func_name}' 存在OK")
    return (False, f"関数 '{func_name}' がコード内に見つかりません")


def execution_test(file_path: Path) -> tuple[bool, str]:
    """実行テスト: python3 -m py_compile で対象ファイルをコンパイルチェック"""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return (True, f"py_compile {file_path.name} OK")
        return (False, f"py_compile失敗: {proc.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        return (False, "py_compile タイムアウト")
    except Exception as e:
        return (False, f"py_compile例外: {e}")


def diff_test(before_code: str, after_code: str) -> tuple[bool, str, list[str]]:
    """差分テスト: before/afterに差分があるか確認（差分なし=絶対に失敗）
    Returns: (差分あり, メッセージ, diff行リスト)
    """
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()
    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm="", n=3))
    if not diff:
        return (False, "コードに変更なし（差分ゼロ）", [])
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return (True, f"差分あり (+{added}/-{removed}行)", diff)


async def judge_improvement(
    instruction: str,
    diff_lines: list[str],
    before_results: dict[str, tuple[bool, str]],
    after_results: dict[str, tuple[bool, str]],
) -> tuple[bool, str]:
    """Geminiにbefore/afterを比較させて改善品質を判定。
    Returns: (承認するか, 判定理由)
    """
    # リグレッション検出: beforeで合格→afterで不合格のテストがないか
    regressions = []
    for test_name, (before_ok, _) in before_results.items():
        if before_ok and test_name in after_results:
            after_ok, after_msg = after_results[test_name]
            if not after_ok:
                regressions.append(f"{test_name}: {after_msg}")
    if regressions:
        return (False, f"リグレッション検出: {'; '.join(regressions)}")

    # diff_test必須チェック
    if "diff_test" in after_results:
        d_ok, d_msg = after_results["diff_test"]
        if not d_ok:
            return (False, f"差分テスト失敗: {d_msg}")

    # Geminiに改善品質を判定させる
    diff_text = "\n".join(diff_lines[:60])
    prompt = (
        "コード改善の品質を判定しろ。\n\n"
        f"【改善指示】\n{instruction}\n\n"
        f"【コード差分】\n{diff_text}\n\n"
        "【判定基準】\n"
        "1. 差分は改善指示の内容に合致しているか？\n"
        "2. 不要な変更（空白だけ、コメントだけ等）ではないか？\n"
        "3. 明らかなバグを含んでいないか？\n\n"
        "「APPROVE」（承認）か「REJECT: 理由」（却下: 理由）の1行だけで答えろ。"
    )
    try:
        response, _ = await think_gemini(prompt, max_tokens=200)
        first_line = response.strip().split("\n")[0]
        if "APPROVE" in first_line.upper():
            return (True, "Gemini判定: 承認")
        elif "REJECT" in first_line.upper():
            reason = first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
            return (False, f"Gemini判定: 却下 - {reason}")
        else:
            return (True, "Gemini判定: 不明応答のため差分ありで承認")
    except Exception as e:
        log.warning(f"Gemini判定失敗: {e}")
        return (True, f"Gemini判定スキップ（{e}）、差分ありのため承認")


def parse_plan(result_text: str) -> dict | None:
    """<<<PLAN>>>...<<<END_PLAN>>>形式をパース。
    Returns: {"target_file", "function_name", "change_description", "measurement_method"} or None
    """
    if "<<<PLAN>>>" not in result_text or "<<<END_PLAN>>>" not in result_text:
        return None
    try:
        plan_text = result_text.split("<<<PLAN>>>", 1)[1].split("<<<END_PLAN>>>", 1)[0].strip()
        plan = {}
        for line in plan_text.splitlines():
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                plan[key.strip().lower().replace(" ", "_")] = value.strip()
        if "target_file" not in plan or "function_name" not in plan:
            return None
        return {
            "target_file": plan.get("target_file", ""),
            "function_name": plan.get("function_name", ""),
            "change_description": plan.get("change_description", ""),
            "measurement_method": plan.get("measurement_method", "syntax_test, diff_test"),
        }
    except (ValueError, IndexError):
        return None


async def _generate_code_with_cli(prompt: str, timeout: int = 120) -> str:
    """Claude CLIでコード生成。高品質なコード出力用。

    Args:
        prompt: コード生成プロンプト
        timeout: タイムアウト秒数（デフォルト120秒）

    Returns:
        生成されたテキスト（stdout）

    Raises:
        RuntimeError: CLI実行失敗時
        subprocess.TimeoutExpired: タイムアウト時
    """
    loop = asyncio.get_running_loop()
    env = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": "/Users/beny",
    }

    def _run():
        proc = subprocess.run(
            ["/opt/homebrew/bin/claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
        if not proc.stdout.strip():
            raise RuntimeError("Claude CLI returned empty output")
        return proc.stdout

    return await loop.run_in_executor(None, _run)


def _count_consecutive_errors(module: str) -> int:
    """Count consecutive failures for a module from growth_stats last_10_results.

    Resets the count to 0 if:
    - The most recent result for this module was a success
    - The last failure for this module occurred more than 1 hour ago
    This prevents transient errors from permanently blocking a module.
    """
    stats = load_growth_stats()
    recent = stats.get("last_10_results", [])
    count = 0

    # Find the most recent result for this module
    latest_module_result = None
    for r in reversed(recent):
        if r.get("module") == module:
            latest_module_result = r
            break

    # If the most recent result for this module was a success, no consecutive errors
    if latest_module_result and latest_module_result.get("success"):
        return 0

    # Check if the last failure is older than 1 hour (3600 seconds)
    if latest_module_result and not latest_module_result.get("success"):
        timestamp_str = latest_module_result.get("timestamp", "")
        if timestamp_str:
            try:
                from datetime import datetime, timezone
                last_fail_time = datetime.fromisoformat(timestamp_str)
                now = datetime.now(timezone.utc)
                # Handle naive datetime by assuming UTC
                if last_fail_time.tzinfo is None:
                    last_fail_time = last_fail_time.replace(tzinfo=timezone.utc)
                elapsed_seconds = (now - last_fail_time).total_seconds()
                if elapsed_seconds > 3600:
                    return 0
            except (ValueError, TypeError):
                pass

    # Count consecutive failures from the end of the list
    for r in reversed(recent):
        if not r.get("success") and r.get("module") == module:
            count += 1
        else:
            break
    return count


def _extract_function_names(code: str) -> list[str]:
    """ast.parseでコード内の全def文の関数名を抽出する"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def _run_all_tests(code: str, file_path: Path, func_name: str = "") -> dict[str, tuple[bool, str]]:
    """全テストを実行して結果辞書を返す"""
    results = {}
    results["syntax_test"] = syntax_test(code)
    results["execution_test"] = execution_test(file_path)
    results["import_test"] = import_test(file_path)
    if func_name:
        results["function_exists_test"] = function_exists_test(code, func_name)
    return results


# ---------------------------------------------------------------------------
# 2. メインワーカーループ
# ---------------------------------------------------------------------------
async def job_worker_loop(client: httpx.AsyncClient):
    """ジョブキューをポーリングし、ジョブを順次実行する。"""
    log.info("job_worker_loop 開始")
    while True:
        try:
            job = get_next_queued_job()
            if job is None:
                await asyncio.sleep(1)
                continue

            task_id = job["task_id"]
            priority = job.get("priority", "P3")

            # P1ジョブ実行前にP1割り込みシグナルをクリア
            if priority == "P1":
                clear_p1_interrupt()

            log.info(f"ジョブ取得: {task_id} type={job['type']} priority={priority}")
            await execute_job(client, job)

        except asyncio.CancelledError:
            log.info("job_worker_loop: キャンセルされました")
            raise
        except Exception as e:
            log.error(f"job_worker_loop エラー: {e}", exc_info=True)
            await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# 2. ジョブ実行ディスパッチャ
# ---------------------------------------------------------------------------
async def execute_job(client: httpx.AsyncClient, job: dict) -> bool:
    """ジョブタイプに応じて処理を分岐し、結果をupdate_jobで記録する。"""
    task_id = job["task_id"]
    job_type = job["type"]

    # ステータスを running に更新
    update_job(task_id, status="running")

    try:
        if job_type == "self_improve":
            result = await _execute_self_improve(client, job)
        elif job_type == "continuous_growth":
            result = await _execute_continuous_growth(client, job)
        elif job_type == "research":
            report = _execute_research()
            result = bool(report and report.get("modules"))
        elif job_type == "analyze_plan":
            report_path = MEMORY_DIR / "research_report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = {}
            result = await _execute_analyze_plan(client, report)
        else:
            log.warning(f"未知のジョブタイプ: {job_type} (task_id={task_id})")
            update_job(task_id, status="failed", error=f"未知のジョブタイプ: {job_type}")
            return False

        if result:
            update_job(task_id, status="success", output={"completed": True})
        else:
            update_job(task_id, status="failed", error="ジョブが成功条件を満たさなかった")
        return result

    except Exception as e:
        log.error(f"ジョブ実行エラー: {task_id}: {e}", exc_info=True)
        update_job(task_id, status="failed", error=str(e))
        return False


# ---------------------------------------------------------------------------
# 4. 6ステップ自己改善
# ---------------------------------------------------------------------------
async def _execute_self_improve(client: httpx.AsyncClient, job: dict) -> bool:
    """6ステップ自己改善プロセス:
    Step 0: 準備（対象モジュール特定・バリデーション）
    Step 1: 計画（Geminiに改善計画を立てさせる）
    Step 2: 事前計測（現在のコードでテスト実行）
    Step 3: コード変更（Geminiにコード生成させる）→リトライあり
    Step 4: 事後計測（変更後のコードでテスト実行）
    Step 5: 判定（before/afterを比較、Gemini判定）
    Step 6: 記録（成功: git commit / 失敗: ロールバック）
    """
    from god import tg_send

    task_id = job["task_id"]
    job_input = job.get("input", {})
    instruction = job_input.get("instruction", "")
    target_file = job_input.get("target_file", "")

    if not instruction:
        log.warning(f"self_improve: instructionが空 (task_id={task_id})")
        return False

    start_time = time_module.time()

    # ===== Step 0: 準備 =====
    log.info(f"[Step 0] 準備開始 (task_id={task_id})")

    if target_file:
        target_name = Path(target_file).stem
        target_path = MODULE_PATHS.get(target_name)
        if target_path is None or not target_path.exists():
            result = select_target_module(instruction)
            if result is None:
                log.warning(f"self_improve: 無効な改善対象モジュール (task_id={task_id})")
                await tg_send(client, f"自己改善スキップ: 無効な対象モジュール ({task_id})")
                return False
            target_path, target_name = result
    else:
        result = select_target_module(instruction)
        if result is None:
            log.warning(f"self_improve: 無効な改善対象モジュール (task_id={task_id})")
            await tg_send(client, f"自己改善スキップ: 無効な対象モジュール ({task_id})")
            return False
        target_path, target_name = result

    log.info(f"[Step 0] 対象={target_name} ({target_path})")

    # 無効なファイル名チェック
    invalid_files = _detect_invalid_module_names(instruction)
    if invalid_files:
        log.warning(f"self_improve: 無効なファイル名を検出: {invalid_files}")
        await safe_append_journal(
            f"### {datetime.now().strftime('%H:%M')} 自己改善スキップ（無効なファイル名）\n"
            f"無効: {invalid_files}\n提案: {instruction[:200]}"
        )
        await tg_send(client, f"自己改善スキップ: 存在しないファイル名 ({', '.join(invalid_files)})")
        return False

    # 重複チェック
    is_dup, dup_reason = is_duplicate_improvement(instruction)
    if is_dup:
        log.info(f"self_improve: 重複: {dup_reason}")
        update_growth_stats(success=False, failure_reason="duplicate", module=target_name, duration=0)
        await safe_append_journal(
            f"### {datetime.now().strftime('%H:%M')} 自己改善スキップ（{dup_reason}）\n"
            f"改善内容: {instruction}"
        )
        return False

    # 統計に基づくスキップ判定
    should_skip, skip_reason = should_skip_improvement(instruction, target_name)
    if should_skip:
        log.info(f"self_improve: 統計判定スキップ: {skip_reason}")
        await safe_append_journal(
            f"### {datetime.now().strftime('%H:%M')} 自己改善スキップ（統計判定）\n"
            f"理由: {skip_reason}\n改善内容: {instruction}"
        )
        return False

    # 提案記録
    record_improvement(instruction, "proposed")

    # バックアップ
    backups = create_module_backups()
    backup_path = target_path.with_suffix(".py.bak")
    current_code = target_path.read_text(encoding="utf-8")

    # コンテキスト収集
    claude_md_text = load_claude_md() or "(CLAUDE.mdなし)"

    identity_text = "(identity.mdなし)"
    try:
        if IDENTITY_PATH.exists():
            identity_text = IDENTITY_PATH.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"identity.md読み込み失敗: {e}")

    journal_recent = "(journalなし)"
    try:
        journal_recent = read_file(JOURNAL_PATH, tail=10) or "(journalなし)"
    except Exception as e:
        log.warning(f"journal読み込み失敗: {e}")

    read_files = job_input.get("read_files", [])
    dep_files_text = ""
    for rf in read_files:
        rf_path = Path(rf) if Path(rf).is_absolute() else CORE_DIR / rf
        if rf_path.exists():
            try:
                rf_lines = rf_path.read_text(encoding="utf-8").splitlines()[:30]
                dep_files_text += f"\n--- {rf_path.name} (先頭30行) ---\n" + "\n".join(rf_lines) + "\n"
            except Exception as e:
                log.warning(f"依存ファイル読み込み失敗 {rf}: {e}")
    if not dep_files_text:
        dep_files_text = "(なし)"

    constraints = job_input.get("constraints", [])
    constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "- 構文チェック必須\n- テスト必須"

    max_retries = job.get("meta", {}).get("max_retries", 3)
    last_error = None

    # ===== Step 1: 計画 =====
    log.info("[Step 1] 計画作成中...")

    # 対象ファイルの関数一覧を抽出
    existing_functions = _extract_function_names(current_code)
    func_list_text = ", ".join(existing_functions) if existing_functions else "(抽出失敗)"

    plan_prompt = (
        f"You are God AI. Create an improvement plan for the following code.\n\n"
        f"[Target file: {target_name}.py]\n{current_code}\n\n"
        f"[Functions that exist in this file]\n{func_list_text}\n\n"
        f"[Improvement instruction]\n{instruction}\n\n"
        f"Output ONLY in the following format. No other output is accepted.\n\n"
        f"<<<PLAN>>>\n"
        f"target_file: filename (e.g. brain.py)\n"
        f"function_name: name of function to modify (MUST be from the list above)\n"
        f"change_description: what to change specifically (1-2 sentences)\n"
        f"measurement_method: syntax_test, import_test, diff_test\n"
        f"<<<END_PLAN>>>\n\n"
        f"ABSOLUTE RULES:\n"
        f"- function_name MUST be one from the list above. Only choose from existing functions.\n"
        f"- Do NOT propose a function that does not exist. Creating new functions is FORBIDDEN.\n"
        f"- If you want to add new logic, add it INSIDE an existing function.\n"
        f"- Target exactly ONE function.\n"
    )

    plan = None
    try:
        try:
            plan_result = await _generate_code_with_cli(plan_prompt, timeout=60)
            log.info("[Step 1] Claude CLIで計画生成")
        except Exception as cli_err:
            log.warning(f"[Step 1] Claude CLI失敗 ({cli_err})、Geminiへ")
            plan_result, _ = await think_gemini(plan_prompt, max_tokens=1024)
        plan = parse_plan(plan_result)
        if plan:
            log.info(f"[Step 1] 計画OK: func={plan['function_name']}, change={plan['change_description'][:80]}")
        else:
            log.error("[Step 1] 計画パース失敗（<<<PLAN>>>形式が不正）")
    except Exception as e:
        log.error(f"[Step 1] 計画生成失敗: {e}")

    if plan is None:
        duration = time_module.time() - start_time
        update_growth_stats(success=False, failure_reason="syntax_error", module=target_name, duration=duration)
        record_improvement(instruction, "failed")
        await safe_append_journal(
            f"### {datetime.now().strftime('%H:%M')} 自己改善失敗（計画生成失敗）\n"
            f"対象: {target_name}.py\n改善内容: {instruction}"
        )
        await tg_send(client, f"自己改善失敗: Step 1計画生成に失敗\n対象: {target_name}.py")
        return False

    plan_func_name = plan["function_name"]
    plan_description = plan["change_description"]

    # Step 1 検証: 指定された関数名が対象ファイルに実在するか確認
    if plan_func_name and existing_functions:
        if plan_func_name not in existing_functions:
            err_msg = f"指定された関数 '{plan_func_name}' は対象ファイル {target_name}.py に存在しません。存在する関数: {func_list_text}"
            log.error(f"[Step 1] {err_msg}")
            duration = time_module.time() - start_time
            update_growth_stats(success=False, failure_reason="syntax_error", module=target_name, duration=duration)
            record_improvement(instruction, "failed")
            await safe_append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善失敗（計画の関数名が不正）\n"
                f"{err_msg}\n改善内容: {instruction}"
            )
            await tg_send(client, f"自己改善失敗: {err_msg[:200]}")
            return False

    # ===== Step 2: 事前計測 =====
    log.info("[Step 2] 事前計測...")
    before_results = _run_all_tests(current_code, target_path, plan_func_name)

    before_summary = ", ".join(f"{tn}={'OK' if ok else 'NG'}" for tn, (ok, _) in before_results.items())
    log.info(f"[Step 2] 事前計測結果: {before_summary}")

    before_failures = [k for k, (ok, _) in before_results.items() if not ok]
    if before_failures:
        log.warning(f"[Step 2] 事前テスト失敗あり: {before_failures}（続行）")

    # ===== Step 3-5: コード変更→事後計測→判定（リトライループ） =====
    for attempt in range(1, max_retries + 1):
        log.info(f"[Step 3] コード変更 試行{attempt}/{max_retries}")

        error_context = ""
        if attempt > 1 and last_error:
            error_context = f"Previous attempt failed with error: {last_error}\nFix this error in your new attempt.\n\n"

        plan_context = ""
        if plan:
            plan_context = (
                f"[Improvement plan]\n"
                f"Target function: {plan['function_name']}\n"
                f"Change: {plan['change_description']}\n\n"
            )

        prompt = (
            f"You are God AI. Improve the following code based on the information below.\n"
            f"Output ONLY valid Python code. Do NOT include any Japanese characters in the code.\n"
            f"All comments MUST be in English.\n\n"
            f"{error_context}"
            f"[Context]\n{claude_md_text}\n\n"
            f"[Purpose]\n{identity_text}\n\n"
            f"[Recent events]\n{journal_recent}\n\n"
            f"{plan_context}"
            f"[Functions in this file]\n{func_list_text}\n\n"
            f"RULE: FUNCTION_NAME must be from the list above. Do NOT use non-existent function names.\n\n"
            f"[Target file: {target_name}.py]\n{current_code}\n\n"
            f"[Dependency imports]\n{dep_files_text}\n\n"
            f"[Improvement instruction]\n{instruction}\n\n"
            f"[Constraints]\n{constraints_text}\n\n"
            "[Output format]\n"
            "Choose ONE function from the list above and output in this exact format:\n\n"
            "<<<TARGET_FILE>>>\nfilename\n"
            "<<<FUNCTION_NAME>>>\nfunction name (from the list above)\n"
            "<<<NEW_FUNCTION>>>\ncomplete modified function\n"
            "<<<END>>>\n\n"
            "RULES:\n"
            "- Modify exactly ONE function per improvement\n"
            "- Output the COMPLETE function (not a diff)\n"
            "- Keep the same indentation as the original\n"
            "- Output ONLY the format above. No explanations.\n"
            "- Use the exact function name from the def statement\n"
            "- Make actual code changes. No change = failure\n"
            "- All comments must be in English. No Japanese in code.\n"
            "- Do NOT include Japanese brackets or symbols in code\n"
            "- Output only valid Python syntax\n"
        )

        try:
            # --- Step 3: Claude CLIでコード生成（1st）→ Geminiフォールバック ---
            brain_used = ""
            try:
                gen_result = await _generate_code_with_cli(prompt, timeout=120)
                brain_used = "claude-cli"
                log.info(f"[Step 3] 試行{attempt}: Claude CLI成功")
            except Exception as cli_err:
                log.warning(f"[Step 3] 試行{attempt}: Claude CLI失敗 ({cli_err})、Geminiへ")
                gen_result, brain_used = await think_gemini(prompt, max_tokens=8192)
                log.info(f"[Step 3] 試行{attempt}: Geminiフォールバック成功（{brain_used}）")

            replacement = parse_function_replacement(gen_result)
            if not replacement:
                raise ValueError("関数置換形式が見つかりません")

            func_name = replacement["function_name"]
            new_func = replacement["new_function"]
            log.info(f"[Step 3] 試行{attempt}: 関数 '{func_name}' を置換")

            # 関数名が対象ファイルに実在するか確認
            if existing_functions and func_name not in existing_functions:
                raise ValueError(f"関数 '{func_name}' は対象ファイルに存在しません。存在する関数: {func_list_text}")

            replace_ok, new_code, replace_error = find_and_replace_function(current_code, func_name, new_func)
            if not replace_ok:
                raise ValueError(f"関数置換失敗: {replace_error}")

            # Step 3内の構文チェック（書き込み前）
            s_ok, s_msg = syntax_test(new_code)
            if not s_ok:
                raise SyntaxError(s_msg)

            # ファイルに仮適用
            target_path.write_text(new_code, encoding="utf-8")
            log.info(f"[Step 3] 試行{attempt}: コード仮適用完了")

            # --- Step 4: 事後計測 ---
            log.info(f"[Step 4] 事後計測...")
            after_results = _run_all_tests(new_code, target_path, func_name)

            # diff_test（必須）
            d_ok, d_msg, d_lines = diff_test(current_code, new_code)
            after_results["diff_test"] = (d_ok, d_msg)

            after_summary = ", ".join(f"{tn}={'OK' if ok else 'NG'}" for tn, (ok, _) in after_results.items())
            log.info(f"[Step 4] 事後計測結果: {after_summary}")

            # --- Step 5: 判定 ---
            log.info("[Step 5] 判定...")

            # 致命的テスト失敗チェック
            critical_failures = []
            for tn in ["syntax_test", "execution_test", "diff_test"]:
                if tn in after_results:
                    a_ok, a_msg = after_results[tn]
                    if not a_ok:
                        critical_failures.append(f"{tn}: {a_msg}")

            if critical_failures:
                shutil.copy2(backup_path, target_path)
                error_msg = f"事後テスト失敗: {'; '.join(critical_failures)}"
                log.error(f"[Step 5] 試行{attempt}: {error_msg}")
                last_error = error_msg

                if attempt == max_retries:
                    fr = "syntax_error" if "syntax_test" in str(critical_failures) else "test_fail"
                    duration = time_module.time() - start_time
                    update_growth_stats(success=False, failure_reason=fr, module=target_name, duration=duration)

                await safe_append_journal(
                    f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{max_retries} 判定失敗\n"
                    f"エラー: {error_msg}\n改善内容: {instruction}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(3)
                continue

            # Gemini判定
            approved, judge_reason = await judge_improvement(instruction, d_lines, before_results, after_results)
            log.info(f"[Step 5] 判定結果: approved={approved}, reason={judge_reason}")

            if not approved:
                shutil.copy2(backup_path, target_path)
                last_error = f"Gemini却下: {judge_reason}"
                log.warning(f"[Step 5] 試行{attempt}: {last_error}")

                if attempt == max_retries:
                    duration = time_module.time() - start_time
                    update_growth_stats(success=False, failure_reason="test_fail", module=target_name, duration=duration)

                await safe_append_journal(
                    f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{max_retries} Gemini却下\n"
                    f"理由: {judge_reason}\n改善内容: {instruction}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(3)
                continue

            # ===== Step 6: 記録（成功） =====
            log.info("[Step 6] 記録（成功）")
            duration = time_module.time() - start_time
            update_growth_stats(success=True, failure_reason=None, module=target_name, duration=duration)
            record_improvement(instruction, "success")

            diff_for_journal = "\n".join(d_lines[:50])
            test_summary = " | ".join(f"{tn}:{'OK' if ok else 'NG'}" for tn, (ok, _) in after_results.items())

            success_msg = (
                f"自己改善成功（試行{attempt}/{max_retries}）\n"
                f"対象: {target_name}.py / 関数: {func_name}\n"
                f"改善内容: {instruction}"
            )
            await safe_append_journal(
                f"### {datetime.now().strftime('%H:%M')} {success_msg}\n"
                f"判定: {judge_reason}\n"
                f"テスト: {test_summary}\n"
                f"コード長: {len(current_code)} -> {len(new_code)}文字\n"
                f"```diff\n{diff_for_journal}\n```"
            )
            await tg_send(
                client,
                f"[自己改善成功] {target_name}.py / {func_name}\n"
                f"改善: {instruction[:100]}\n"
                f"判定: {judge_reason}\nテスト: {test_summary}"
            )

            _git_commit(target_path, target_name, instruction)
            reload_modules([target_name])

            update_job(task_id, output={
                "completed": True,
                "target": target_name,
                "function": func_name,
                "code_diff_chars": len(new_code) - len(current_code),
                "attempts": attempt,
                "judge_reason": judge_reason,
            })
            return True

        except (SyntaxError, ValueError) as e:
            last_error = str(e)
            log.error(f"[Step 3] 試行{attempt}/{max_retries} 失敗: {e}")

            if backup_path.exists():
                shutil.copy2(backup_path, target_path)

            if attempt == max_retries:
                duration = time_module.time() - start_time
                update_growth_stats(success=False, failure_reason="syntax_error", module=target_name, duration=duration)

            await safe_append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{max_retries} 失敗\n"
                f"エラー: {e}\n改善内容: {instruction}"
            )
            update_job(task_id, error=str(e))
            if attempt < max_retries:
                await tg_send(client, f"自己改善 試行{attempt}/{max_retries} 失敗: {e}\nリトライ...")
                await asyncio.sleep(3)

        except Exception as e:
            last_error = str(e)
            log.error(f"[Step 3] 試行{attempt}/{max_retries} 予期せぬエラー: {e}", exc_info=True)

            if backup_path.exists():
                shutil.copy2(backup_path, target_path)

            duration = time_module.time() - start_time
            update_growth_stats(success=False, failure_reason="timeout", module=target_name, duration=duration)
            await safe_append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{max_retries} 予期せぬエラー\n"
                f"エラー: {e}\n改善内容: {instruction}"
            )
            update_job(task_id, error=str(e))
            break

    # ===== 全試行失敗 =====
    if backup_path.exists():
        shutil.copy2(backup_path, target_path)
    record_improvement(instruction, "failed")

    analysis = await _analyze_error(last_error or "不明", instruction, target_name)

    fail_msg = (
        f"自己改善 {max_retries}回試行して失敗。ロールバック完了。\n"
        f"対象: {target_name}.py\n最終エラー: {last_error}\n改善内容: {instruction}"
    )
    log.error(fail_msg)
    await safe_append_journal(f"### {datetime.now().strftime('%H:%M')} {fail_msg}\n根本分析: {analysis}")

    error_count = _count_consecutive_errors(target_name)
    if error_count >= 3:
        await tg_send(
            client,
            f"同じモジュール({target_name})で{error_count}回連続失敗。別の改善に移ります。\n"
            f"最終エラー: {last_error}\n根本分析: {analysis[:200]}"
        )
    else:
        await safe_append_journal(f"### {datetime.now().strftime('%H:%M')} エラー分析結果\n{analysis}")

    return False


# ---------------------------------------------------------------------------
# 4. 連続成長サイクル
# ---------------------------------------------------------------------------
async def _execute_continuous_growth(client: httpx.AsyncClient, job: dict) -> bool:
    """continuous_growth ジョブを実行する。振り返り→改善ジョブ自動作成。"""
    from god import tg_send

    # P1割り込みチェック
    if is_p1_interrupted():
        log.info("continuous_growth: P1割り込みにより中断")
        clear_p1_interrupt()
        return True  # 割り込みは正常終了扱い

    # 振り返り実行
    from growth import reflection_cycle
    executed, reflection_text = await reflection_cycle(client)

    if not executed:
        log.info("continuous_growth: 振り返りスキップ（他の振り返りと競合）")
        return True  # 競合は正常終了扱い

    # CODE_IMPROVEMENT が見つかった場合 → self_improve ジョブを作成
    # (注: reflection_cycle 内で直接 self_improve を呼ぶ既存ロジックがある。
    #  将来 growth.py 側が修正されて直接呼ばなくなった場合にここで作成する)
    if "CODE_IMPROVEMENT:" in reflection_text:
        improvements = []
        for line in reflection_text.splitlines():
            if line.strip().startswith("CODE_IMPROVEMENT:"):
                improvements.append(line.strip().replace("CODE_IMPROVEMENT:", "").strip())

        if improvements:
            improvement_text = "\n".join(improvements)

            # ターゲットモジュール推定
            result = select_target_module(improvement_text)
            target_file = ""
            dep_files = []
            if result is not None:
                target_file = str(result[0])
                dep_files = _detect_read_files(result[0], result[1])

            # self_improve ジョブを作成（P3）
            new_job = create_job(
                job_type="self_improve",
                priority="P3",
                target_file=target_file,
                read_files=dep_files,
                instruction=improvement_text,
                constraints=["構文チェック必須", "テスト必須"],
                max_retries=3,
            )
            if new_job:
                log.info(f"continuous_growth: self_improve ジョブ作成: {new_job['task_id']}")
            else:
                log.info("continuous_growth: 重複ジョブのためスキップ")

    log.info("continuous_growth: 完了")
    return True


# ---------------------------------------------------------------------------
# 5. エラー根本分析
# ---------------------------------------------------------------------------
async def _analyze_error(error: str, improvement_text: str, module: str) -> str:
    """Geminiにエラーの原因と解決策を聞く。"""
    prompt = f"""あなたはコードデバッグの専門家。以下のエラーの原因と解決策を分析せよ。

【エラー内容】
{error}

【試みた改善内容】
{improvement_text}

【対象モジュール】
{module}.py

【タスク】
1. このエラーの根本原因は何か？
2. 同じ失敗を繰り返さないためにどうすべきか？
3. 代替アプローチはあるか？

簡潔に日本語で答えろ。各項目を1-2文で。"""

    try:
        analysis, brain_name = await think_gemini(prompt)
        log.info(f"エラー根本分析完了 (brain: {brain_name})")
        return analysis
    except Exception as e:
        log.error(f"エラー根本分析失敗: {e}")
        return f"分析失敗: {e}"


# ---------------------------------------------------------------------------
# 6. git自動commit（内部ヘルパー）
# ---------------------------------------------------------------------------
def _git_commit(target_path: Path, target_name: str, instruction: str):
    """改善成功時にgit add / commit / push を実行する。"""
    try:
        commit_summary = instruction[:50].replace("\n", " ")
        subprocess.run(
            ["git", "add", str(target_path)],
            cwd=str(BASE_DIR), check=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", f"self-improve: {target_name} - {commit_summary}"],
            cwd=str(BASE_DIR), check=True, timeout=30,
        )
        subprocess.run(
            ["git", "push"],
            cwd=str(BASE_DIR), check=False, timeout=60,
        )
        log.info(f"git自動commit完了: {target_name}")
    except subprocess.CalledProcessError as e:
        log.warning(f"git commit失敗（変更なしの可能性）: {e}")
    except Exception as e:
        log.error(f"git自動commit失敗: {e}")


# ---------------------------------------------------------------------------
# 7. Research (Phase 1: コード解析、AI不使用)
# ---------------------------------------------------------------------------
def _execute_research() -> dict:
    """Phase 1: Analyze codebase without AI.

    Scans all .py files in CORE_DIR using ast.parse.
    Extracts: file info, function signatures, dependencies, error logs.
    Saves result to memory/research_report.json.
    Returns the report dict.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "modules": {},
        "dependencies": {},
        "recent_errors": [],
    }

    # Scan all .py files in CORE_DIR
    for py_file in sorted(CORE_DIR.glob("*.py")):
        module_name = py_file.stem
        try:
            code = py_file.read_text(encoding="utf-8")
            lines = code.splitlines()
            line_count = len(lines)

            # Parse AST
            tree = ast.parse(code)

            # Extract function signatures
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno - 1
                    def_line = lines[start_line] if start_line < len(lines) else ""

                    # Get first line of docstring
                    docstring = ast.get_docstring(node) or ""
                    if docstring:
                        docstring = docstring.strip().splitlines()[0][:100]

                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "signature": def_line.strip(),
                        "docstring": docstring,
                    })

            # Extract imports for dependency map
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            report["modules"][module_name] = {
                "file": str(py_file),
                "line_count": line_count,
                "functions": functions,
                "imports": imports,
            }

            # Build dependency map (internal dependencies only)
            internal_deps = [imp for imp in imports if imp in MODULE_PATHS]
            report["dependencies"][module_name] = internal_deps

        except Exception as e:
            log.warning(f"Research: Failed to analyze {py_file.name}: {e}")
            report["modules"][module_name] = {
                "file": str(py_file),
                "error": str(e),
            }

    # Extract ERROR/WARNING from log file
    log_path = Path("/tmp/godai_v3.log")
    if log_path.exists():
        try:
            log_lines = log_path.read_text(encoding="utf-8").splitlines()
            recent_lines = log_lines[-100:]
            for line in recent_lines:
                if "ERROR" in line or "WARNING" in line:
                    report["recent_errors"].append(line.strip()[:200])
        except Exception as e:
            log.warning(f"Research: Failed to read log file: {e}")

    # Save report
    report_path = MEMORY_DIR / "research_report.json"
    try:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"Research report saved: {report_path} ({len(report['modules'])} modules)")
    except Exception as e:
        log.error(f"Research: Failed to save report: {e}")

    return report


# ---------------------------------------------------------------------------
# 8. Analyze + Plan (Phase 2: Geminiで分析→self_improveジョブ作成)
# ---------------------------------------------------------------------------
async def _execute_analyze_plan(client: httpx.AsyncClient, report: dict) -> bool:
    """Phase 2: Analyze research report with Gemini and create self_improve jobs.

    Reads the research report, asks Gemini to identify real problems,
    validates function names against actual code, then creates self_improve jobs.
    Returns True if at least one job was created.
    """
    from god import tg_send

    if not report or not report.get("modules"):
        log.warning("analyze_plan: Empty research report")
        return False

    # Build a compact summary of the codebase for Gemini
    module_summaries = []
    for mod_name, mod_info in report["modules"].items():
        if "error" in mod_info:
            continue
        funcs = mod_info.get("functions", [])
        func_entries = [f"{f['name']}(L{f['line']})" for f in funcs]
        module_summaries.append(
            f"  {mod_name}.py ({mod_info['line_count']} lines): "
            f"functions=[{', '.join(func_entries)}]"
        )
    modules_text = "\n".join(module_summaries)

    # Recent errors summary
    errors_text = "\n".join(report.get("recent_errors", [])[-20:]) or "(no recent errors)"

    # Dependencies
    deps_lines = []
    for mod, deps in report.get("dependencies", {}).items():
        if deps:
            deps_lines.append(f"  {mod} -> {', '.join(deps)}")
    deps_text = "\n".join(deps_lines) if deps_lines else "(none)"

    # Get recent failed jobs to avoid repeating
    failed_jobs = get_recent_failed_jobs(limit=5)
    failed_text = "(none)"
    if failed_jobs:
        failed_lines = [f"  - {fj['target_file']}: {fj['instruction']}" for fj in failed_jobs]
        failed_text = "\n".join(failed_lines)

    # Get queued jobs to avoid duplicates
    queued_jobs = get_queued_job_summaries()
    queued_text = "(none)"
    if queued_jobs:
        queued_lines = [f"  - [{qj['status']}] {qj['target_file']}: {qj['instruction']}" for qj in queued_jobs]
        queued_text = "\n".join(queued_lines)

    prompt = (
        "You are God AI's code analyzer. Based on this research report, identify exactly 3 REAL problems "
        "that can be fixed by modifying a SINGLE existing function.\n\n"
        f"[Codebase Structure]\n{modules_text}\n\n"
        f"[Module Dependencies]\n{deps_text}\n\n"
        f"[Recent Errors/Warnings from logs]\n{errors_text}\n\n"
        f"[Already queued improvements (DO NOT duplicate)]\n{queued_text}\n\n"
        f"[Recently failed improvements (DO NOT repeat)]\n{failed_text}\n\n"
        "For each problem, output in this EXACT format (3 items):\n\n"
        "<<<IMPROVEMENT_1>>>\n"
        "module: module_name (without .py)\n"
        "function: existing_function_name\n"
        "description: one-line description of the specific improvement\n"
        "<<<END_1>>>\n\n"
        "<<<IMPROVEMENT_2>>>\n"
        "module: module_name\n"
        "function: existing_function_name\n"
        "description: one-line description\n"
        "<<<END_2>>>\n\n"
        "<<<IMPROVEMENT_3>>>\n"
        "module: module_name\n"
        "function: existing_function_name\n"
        "description: one-line description\n"
        "<<<END_3>>>\n\n"
        "RULES:\n"
        "- Each function name MUST exist in the codebase structure above\n"
        "- Each improvement must target a DIFFERENT module\n"
        "- Focus on real issues visible in the error logs or code structure\n"
        "- Do NOT propose improvements already queued or recently failed\n"
        "- Be specific: name the exact function and what to change\n"
        "- Prioritize: error handling, robustness, performance\n"
    )

    try:
        result, brain_name = await think_gemini(prompt, max_tokens=2048)
        log.info(f"analyze_plan: Gemini response received ({brain_name})")
    except Exception as e:
        log.error(f"analyze_plan: Gemini failed: {e}")
        return False

    # Parse improvements from Gemini response
    improvements_created = 0
    for i in range(1, 4):
        start_tag = f"<<<IMPROVEMENT_{i}>>>"
        end_tag = f"<<<END_{i}>>>"

        if start_tag not in result or end_tag not in result:
            continue

        block = result.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()

        # Parse fields
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip().lower()] = value.strip()

        module_name = fields.get("module", "")
        func_name = fields.get("function", "")
        description = fields.get("description", "")

        if not module_name or not func_name or not description:
            log.warning(f"analyze_plan: Incomplete improvement #{i}: {fields}")
            continue

        # Validate: module exists
        if module_name not in MODULE_PATHS:
            log.warning(f"analyze_plan: Invalid module '{module_name}' in improvement #{i}")
            continue

        module_path = MODULE_PATHS[module_name]
        if not module_path.exists():
            log.warning(f"analyze_plan: Module file not found: {module_path}")
            continue

        # Validate: function exists in module
        try:
            mod_code = module_path.read_text(encoding="utf-8")
            mod_funcs = _extract_function_names(mod_code)
            if func_name not in mod_funcs:
                log.warning(
                    f"analyze_plan: Function '{func_name}' not found in {module_name}.py. "
                    f"Available: {', '.join(mod_funcs[:10])}"
                )
                continue
        except Exception as e:
            log.warning(f"analyze_plan: Failed to read {module_name}.py: {e}")
            continue

        instruction = f"{module_name}.py: {func_name} - {description}"

        # Detect dependencies
        dep_files = _detect_read_files(module_path, module_name)

        # Create self_improve job
        new_job = create_job(
            job_type="self_improve",
            priority="P3",
            target_file=f"core/{module_name}.py",
            read_files=dep_files,
            instruction=instruction,
            constraints=["構文チェック必須", "テスト必須"],
            max_retries=3,
        )

        if new_job:
            improvements_created += 1
            log.info(f"analyze_plan: Created self_improve job #{i}: {module_name}.py/{func_name}")
        else:
            log.info(f"analyze_plan: Duplicate job skipped #{i}: {module_name}.py/{func_name}")

    if improvements_created > 0:
        await tg_send(client, f"Research->Analyze complete: {improvements_created} improvement jobs created")
        await safe_append_journal(
            f"### {datetime.now().strftime('%H:%M')} Research->Analyze complete\n"
            f"Improvement jobs created: {improvements_created}"
        )
    else:
        log.warning("analyze_plan: No valid improvements created")
        await safe_append_journal(
            f"### {datetime.now().strftime('%H:%M')} Research->Analyze: no valid improvements"
        )

    return improvements_created > 0
