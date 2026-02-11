# state.json スキーマ定義

このファイルは `memory/state.json` の各フィールドの意味と用途を定義する。

## フィールド一覧

| フィールド名 | 型 | 説明 |
|-------------|-----|------|
| `status` | string | God AIの動作状態。値: `"running"`, `"stopped"`, `"error"` |
| `growth_cycles` | integer | 振り返り完了回数。Gemini/Claude呼び出し総数ではなく、reflection_cycleが完了した回数 |
| `gemini_count` | integer | Gemini API呼び出し総数（全種類の呼び出しを含む） |
| `claude_count` | integer | Claude CLI呼び出し総数（全種類の呼び出しを含む） |
| `conversations_today` | integer | 今日のBenyとの会話数（起動時にリセット） |
| `uptime_start` | string (ISO 8601) | 現在の起動時刻。稼働時間計算に使用 |
| `last_reflection` | string (datetime) | 最後の振り返り実行時刻 |
| `children_count` | integer | 作成した子AI（Claude Agent）の数 |

## 注意事項

### growth_cyclesについて
- `growth_cycles`は「振り返りサイクル（reflection_cycle）」が完了した回数をカウント
- AIの脳（Gemini/Claude）呼び出し回数とは異なる
- 脳の呼び出し回数は `gemini_count` と `claude_count` で別途カウント

### conversations_todayについて
- 起動時に0にリセットされる
- Benyからのメッセージに応答するたびに+1
- 1日の会話量の目安として使用

### 永続化について
- `state.json`は頻繁に更新される
- 書き込み競合を防ぐため、`safe_save_state()`を使用すること
- ファイルロックによる排他制御が実装されている

## 関連ファイル

- `memory/growth_stats.json` - 成長統計（成功率、失敗理由など）
- `memory/improvement_history.json` - 改善提案履歴（重複防止用）
- `memory/conversations.json` - 会話ログ
- `memory/journal.md` - 日記・振り返り記録

---

*このファイルはGod AI v3.0の内部仕様書の一部です*
