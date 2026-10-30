# MCP 書き込み版：システムレベル受け入れ（P2）

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
