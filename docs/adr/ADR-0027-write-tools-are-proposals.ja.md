# ADR-0027：書き込みツールは「提案」であり「書き込み」ではない

[中文](ADR-0027-write-tools-are-proposals.md) · [English](ADR-0027-write-tools-are-proposals.en.md)

- **状態**：採択
- **日付**：2026-10-31
- **関連**：[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.ja.md)（読み取り専用の境界）、
  [ADR-0026](ADR-0026-mcp-write-tools-and-explicit-arming.ja.md)（9 ツールの表面）、
  `packages/biosnn-mcp`

## Context

[ADR-0026](ADR-0026-mcp-write-tools-and-explicit-arming.ja.md) は MCP サーバに 5 つの「書き込み
ツール」を加えた。その実際の作用域は**サーバのメモリ内にある代替コア**であり、実記憶庫には一切
触れない（Axo 側の庫は別途「読み取り専用」で固定。`docs/security.md`：SQLite `mode=ro`、
書き込みはエンジンが拒否）。

ここに名前と挙動の食い違いが残る：`memory.write` と呼ばれるツールが、**どの記憶庫にも書かない**。
本リポジトリの「文書と挙動の不一致はバグ」という規約からすれば、「実は偽物です」と脚注を足す
だけでは済まない。

## Decision

1. **5 つのツールを `propose.` 名前空間へ改名**：`propose.memory.write`、
   `propose.memory.consolidate`、`propose.trigger.arm`、`propose.trigger.disarm`、
   `propose.trigger.feed`。読み取り専用の 4 つ（`state.query`、`memory.search`、
   `trigger.status`、`trigger.poll`）は名前を変えない。
2. **契約を名前に、そしてデータに書く**：`propose.*` の作用域は**恒久的に代替コア**。
   payload には `"scope": "stand-in"`、`state.query` は `"write_scope": "stand-in"` を返す。
3. **書き込みツールは「本採用」しない**。価値は二つ：試用者がツール面を通しで試せること、
   適合性／接続テストに対象があること。**インターフェースの足場**であり、接続待ちの半製品ではない。
4. **将来、提案が実記憶に効く必要が出たら**、これらのツールを接続するのではなく、
   **力学が独占して書く**経路を足す：`propose` キュー＋統合ゲート付き消費——提案を enqueue し、
   力学が処分し、すべて JSONL で監査する。そのときこれらのツールは退役し、
   `biosnn_propose_*` に置き換わる。
5. **ガード**：読み取り専用でないツールは `propose.` で始まり、説明に `stand-in` を含むことを
   テストで表明する。

## Consequences

- 名前が契約：クライアントが読む名前が、実ストアに書かないことを直接示す；
- 書き込みの意味を持つ新ツールは、まず命名ガードを通す必要がある（テストの変更が要る）；
- バージョンは **0.2.0** のまま——改名は初回公開の前なので、公開済みの契約は壊れない；
- Axo 側の読み取り専用決定には影響しない：`axo-mcp` のツール面は本 ADR と無関係で、庫は
  エンジン層で読み取り専用のまま；
- 代償：`propose.*` は `write.*` より冗長。これは意図的——冗長さと引き換えに正直さを得る。
