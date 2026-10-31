# MCP 書き込み版：システムレベル受け入れ（P2）

[中文](README.md) · [English](README.en.md)

**2026-10-31、1 回の実行で 14/14 通過。** スクリプトは
[`scripts/e2e_write_mcp.py`](../../scripts/e2e_write_mcp.py)、判定基準は計画書 §3.4 で
事前登録した W1–W6（§9.2 により 2026-10-31 に凍結）。

## 実行方法

```bash
python scripts/e2e_write_mcp.py            # 一時ディレクトリ、終了後に削除
python scripts/e2e_write_mcp.py --keep     # 保持して store.json / audit.jsonl を確認
```

実際の `python -m biosnn_mcp write` 子プロセスを起動し、stdin/stdout で JSON-RPC を話します——
本物の MCP クライアントが見るものと同じで、Python API を直接呼んではいません。

## 結果（2026-10-31、Python 3.13 / Windows 11）

| 基準 | 検査項目 | 結果 |
| :--- | :--- | :--- |
| W6 | ツール表は 13 個、読み取り専用の 9 個が先頭 | PASS |
| W1 | 提案が `accepted` と判定される | PASS |
| W1 | 反映後に検索できる | PASS |
| W1 | 監査 3 行がそろう（`proposal.received` / `gate.accepted` / `memory.applied`） | PASS |
| W2 | 重複提案が `rejected`（理由に duplicate） | PASS |
| W2 | 却下後もストアはバイト単位で不変 | PASS |
| W3 | 同じ `proposal_id` の再送で項目が増えない | PASS |
| W4 | 2 件目が承認される | PASS |
| W4 | 取り消しが成功する | PASS |
| W4 | 取り消し後のストアが書き込み前とバイト単位で一致 | PASS |
| W5 | `proposal_id` なしの提案は拒否され、書き込まれない | PASS |
| - | `gate.configure` が効く（1 時間あたりの配額を 0 に） | PASS |
| - | 配額超過の提案は `pending` になり、捨てられない | PASS |
| - | キューの統計が整合（accepted 1 / rejected 1 / reverted 1 / pending 1） | PASS |

## この実行が**示していない**こと

- **Axo は通っていない**：スクリプトは biosnn 側の永続ファイルだけを触ります。Axo の実ライブラリは
  P1–P6 の範囲外です（決定 2）。
- **実モデルは使っていない**：`source` はスクリプトが埋めた値で、`llm` の重みは定数としてしか
  通っていません。qwen3-4b やクラウドモデルが提案を作ったわけではありません。
- **埋め込みはハッシュの代替**（64 次元）：0.90 の重複しきい値が実埋め込みでどう振る舞うかは
  未検証です。実埋め込みを繋いだら、この表を取り直して別途記録します。

## 再現ブロック

| 項目 | 値 |
| :--- | :--- |
| コード版 | `83a4530`（P1）＋本スクリプトを含むコミット |
| コマンド | `python scripts/e2e_write_mcp.py` |
| 結果 | 14/14 通過、終了コード 0 |
## P5：パッケージ検証（2026-10-31）

リリースのためではなく、**同じ wheel に二つの入口が入っていて、どちらも動く**ことの確認です。

| 手順 | 結果 |
| :--- | :--- |
| `uv build --wheel packages/biosnn-mcp` | `biosnn_mcp-0.3.0-py3-none-any.whl` |
| まっさらな venv に導入 | `biosnn-mcp.exe` と `biosnn-mcp-write.exe` の両方が現れる |
| `biosnn-mcp-write --help` | 正常（`--state-dir / --store / --queue / --audit-log`） |
| 導入済みの入口で stdio セッション | `tools/list` は 13 個のツール；提案は `accepted`；検索が当たる；`memory.revert` は `reverted: true`；監査 7 行 |

CI に「書き込み入口の smoke」を一條追加しました：wheel を作り、きれいな venv に入れ、
`biosnn-mcp-write` で propose → accept → revert を通し、13 個のツールと二つの判定を検証します。
**公開はしていません**——バージョン番号と tag はユーザーが決めることです。
## Axo 側の隔離エンドツーエンド（2026-10-31、B2 の穴を埋める）

これまでこのディレクトリは biosnn 側の受け入れしか記録していませんでした。Axo の書き込み段階は
単体テストだけで、入れたプログラムで通しで動かした人はいませんでした。今回それを補い、**すべて
一時ディレクトリ**で実行しました：庫・キュー・監査は `mkdtemp()` の下に置き、終わったら削除します。
書き込み段階のスイッチは `AXO_PROPOSE_SETTING` で上書きしますが、これは `--data-dir` を同時に
指定したときだけ採信されるので、ユーザーの実設定と実庫には一切触れていません。

スクリプト：Axo リポジトリの `tools/_smoke_mcp_write.py`。導入済みの
`dist/_mcp/axo-mcp-write.exe` を動かします。

| 基準 | 結果 |
| :--- | :--- |
| ツール表 7 個 | PASS |
| 提案が accepted と判定される | PASS |
| 反映後に検索できる（複数語のクエリ） | PASS |
| 重複提案が rejected | PASS |
| 同じ `proposal_id` が冪等 | PASS |
| 取り消しで庫が 0 件に戻る | PASS |
| 監査イベントが揃う（received / accepted / applied / revert） | PASS |

**この実行で見つかった二つの実際の問題**

1. **読み取り専用の接続が、たった今書いた項目を見られない**：提案は accepted で `state.query` も
   1 件と出るのに、`memory.search` は空でした。検索は起動時に開いた読み取り専用接続を使い、書き込みは
   別の接続を通るためです。accepted のたびに読み取り接続を開き直すよう直しました。
2. **単語一つのクエリでは当たらない**：同じ記憶に対して `"gate"` は 0 件、`"accepted this line"` は
   1 件。これは検索層の語彙符号化の性質で、書き込み段階の欠陥ではありません——ただしユーザーは
   必ず踏むので記録します。
