#!/usr/bin/env python3
"""God AI v3.0 - 成長・振り返りモジュール"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import shutil
from datetime import datetime, timezone

import httpx

from config import (
    GOD_PY_PATH, JOURNAL_PATH, REFLECTION_INTERVAL, SELF_GROWTH_INTERVAL, log
)
from memory import (
    load_state, save_state, read_file, append_journal,
    safe_save_state, safe_append_journal
)
from brain import think_gemini, think_claude_heavy
from jobqueue import Priority, create_job

# --- 振り返り排他制御 ---
_reflecting = False

def is_reflecting() -> bool:
    return _reflecting

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
async def reflection_cycle(client: httpx.AsyncClient) -> bool:
    """振り返り実行。戻り値: 実行したかどうか"""
    global _reflecting
    if _reflecting:
        log.warning("振り返り中のため新しい振り返り要求を無視")
        return False
    _reflecting = True
    try:
        await _reflection_cycle_inner(client)
        return True
    finally:
        _reflecting = False

async def _reflection_cycle_inner(client: httpx.AsyncClient):
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
        return

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
                skip_msg = f"### {now} 自己改善スキップ（重複検出）\n改善内容: {improvement_text}"
                await safe_append_journal(skip_msg)
                await tg_send(client, f"重複した改善提案を検出。既に適用済みの可能性が高いためスキップしました。\n提案: {improvement_text[:200]}")
            else:
                await self_improve(client, reflection)

    log.info("振り返りサイクル完了")

# --- 自己改善 ---
async def self_improve(client: httpx.AsyncClient, reflection: str):
    """コード自己改善（構文チェック強化、最大3回リトライ）"""
    from god import tg_send

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
                f"【改善内容】\n{improvement_text}\n\n"
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

            # 書き込み
            GOD_PY_PATH.write_text(code, encoding="utf-8")
            success_msg = f"自己改善成功（試行{attempt}/{MAX_RETRY}）\n改善内容: {improvement_text}"
            append_journal(
                f"### {datetime.now().strftime('%H:%M')} {success_msg}\n"
                f"コード長: {len(current_code)} -> {len(code)}文字\n"
                f"```diff\n{diff_for_journal}\n```"
            )
            await tg_send(client, f"[自己改善] {success_msg}\nコード長: {len(current_code)} -> {len(code)}文字")
            log.info(f"自己改善成功（試行{attempt}）: {len(current_code)} -> {len(code)}文字")
            return

        except (SyntaxError, ValueError) as e:
            last_error = str(e)
            log.error(f"自己改善 試行{attempt}/{MAX_RETRY} 失敗: {e}")
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
            append_journal(
                f"### {datetime.now().strftime('%H:%M')} 自己改善 試行{attempt}/{MAX_RETRY} 予期せぬエラー\n"
                f"エラー: {e}\n改善内容: {improvement_text}"
            )
            break

    # 全試行失敗 -> ロールバック
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
            executed = await reflection_cycle(client)
            if executed:
                await tg_send(client, "定期振り返り完了。journalを更新しました。")
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
    """自己成長提案ジョブの実行"""
    log.info("自己成長提案ジョブ開始")
    state = load_state()
    journal_tail = read_file(JOURNAL_PATH, tail=30)

    prompt = f"""あなたはGod AI。自律型AIとして自己成長を提案せよ。

【現在の状態】
{json.dumps(state, ensure_ascii=False)}

【最近のjournal】
{journal_tail}

【タスク】
以下の観点で自己成長提案を1つだけ挙げよ：
1. 新しい機能追加の提案
2. パフォーマンス改善の提案
3. コード品質向上の提案
4. ユーザー体験改善の提案

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
