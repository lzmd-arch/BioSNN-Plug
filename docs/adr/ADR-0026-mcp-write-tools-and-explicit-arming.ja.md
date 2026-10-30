# ADR-0026：MCP の書き込みツールと明示的な武装（M0 → M1）

[中文](ADR-0026-mcp-write-tools-and-explicit-arming.md) · [English](ADR-0026-mcp-write-tools-and-explicit-arming.en.md)

- **状態**：採択
- **日付**：2026-10-30
- **関連**：[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.ja.md)（M0 の読み取り専用境界）、
  計画書 §七 第 5 フェーズ、[第 5 フェーズのギャップ一覧](../../sweep_results/stage1_5_gap_plan.md)

## Context

[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.ja.md) は M0 を読み取り専用の骨格に固定した：
ツールは 3 つだけで、`trigger.status` は常に未武装。その後 M1〜M3 の要素は実装済み
（mock／実 LLM アダプタ、対話トレース、睡眠統合、スパイク発火、実埋め込み）だが、**そのどれも
MCP ツールになっていない**——外部クライアントは見ることはできても書けず、実際には使えなかった。

ADR-0020 は同時に二つの制約を課している：**LLM は外部推論エンジンにすぎない**（学習経路に入れない、
鍵はリポジトリに入れない）ことと、**すべてのツール呼び出しは記録・再生できなければならない**こと。
ツール面の拡張はこの両方を満たす必要がある。

## Decision

1. **ツール面を 9 個に拡張**し、二層に分ける：
   - 読み取り専用（どのモードでも可）：`state.query`、`memory.search`、`trigger.status`、
     `trigger.poll`；
   - 書き込み（`--allow-write` が必要）：`memory.write`、`memory.consolidate`、`trigger.arm`、
     `trigger.disarm`、`trigger.feed`。
2. **既定は読み取り専用**。書き込みツールも `tools/list` には現れる（発見できる）が、読み取り専用
   モードで呼ぶと `isError: true` を返し、エラー文に「--allow-write が必要」と書く。
   `InMemoryCore(writes_enabled=...)` が能力を決め、サーバは CLI フラグで露出する。
3. **武装には理由が要る**。`trigger.arm` は空でない `reason` 文字列を要求し状態に記録する。
   `disarm` は保留イベントを捨てる。`trigger.feed` は `[0,1]` の顕著性を 1 ステップ分与えて
   パッケージ内の `SpikingTrigger` を進め、`trigger.poll` は排出して返す（**読み取り専用**）。
4. **監査可能**：サーバは `tools/call` ごとに JSONL を 1 行（`ts` / `tool` / `ok` / `error` /
   `arguments`）書き出す。パスは `--audit-log` で指定し、未指定ならメモリ内に留める（テスト用）。
5. **統合は決定論的ハッシュ埋め込み**（`HashEmbedding`）を使い、Ollama の無い環境でも
   `memory.consolidate` が動くようにする。閾値・ラウンド・ノイズ・シードはツール引数。
6. **バージョン 0.2.0**（`biosnn-mcp`）。

## Consequences

- 外部クライアントは**書く・武装する・統合する**ことができるようになり、既定では黙って状態を
  変えない：書くにはスイッチ、武装には理由、呼び出しごとに痕跡；
- `state.query` の `milestone` は `M0` → `M1`、`read_only` はスイッチに従う。テストの
  「M0 は常に読み取り専用」という表明は「読み取り専用モードでは書き込みツールが拒否される」に置換；
- **LLM は依然として未接続**：`llm` は `not_connected` のままで、本 ADR は LLM 呼び出しを
  ツールの背後に置かない；
- **`armed=true` を「自律発火が成立した」と書いてはならない**：武装は状態にすぎず、発火の質は
  M3k–M3p の証拠が決める（合成ストリームは通過、実ストリームは未達）；
- 書き込みツールが変えるのは**メモリ内の代替コア**であり、実記憶庫ではない。Axo アダプタや実庫に
  差し替えるときも同じプロトコルと同じ監査規約を再利用すること。
