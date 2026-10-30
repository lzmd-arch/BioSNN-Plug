# biosnn-mcp

[中文](README.md) · [English](README.en.md)

[BioSNN-Plug](https://github.com/lzmd-arch/BioSNN-Plug) の**第 5 フェーズ MCP サーバ**：
stdio 上の JSON-RPC 2.0、サードパーティ依存ゼロで、認知状態・エピソード記憶・睡眠統合・
スパイク発火を MCP ツールとして公開する。

> **境界**：LLM はあくまで**外部推論エンジン**——本パッケージはモデルを呼ばず、API キーを
> メモリに読まず、シナプス学習経路に入らない。ツール面は**既定で読み取り専用**。書き込みには
> 明示的な `--allow-write` が必要で、**呼び出しごとに監査行を 1 行残す**
> （[ADR-0026](../../docs/adr/ADR-0026-mcp-write-tools-and-explicit-arming.ja.md)）。

## インストール

PyPI 未公開のためリポジトリから：

```bash
pip install "biosnn-mcp @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-mcp"
```

このリポジトリ内で動かす場合（未インストールなら `PYTHONPATH` を通す）：

```bash
PYTHONPATH=packages/biosnn-mcp/src python -m biosnn_mcp --allow-write
```

## クイックスタート

```bash
# 読み取り専用（既定）
biosnn-mcp

# 書き込み可 + 監査
biosnn-mcp --allow-write --audit-log ~/biosnn-mcp-audit.jsonl
```

MCP クライアント設定（Claude Desktop など stdio クライアント）：

```json
{
  "mcpServers": {
    "biosnn": {
      "command": "biosnn-mcp",
      "args": ["--allow-write", "--audit-log", "/tmp/biosnn-mcp-audit.jsonl"]
    }
  }
}
```

手書きの JSON-RPC 1 行でも試せる：

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"state.query","arguments":{}}}' | biosnn-mcp
```

## ツール

| ツール | 層 | 引数 | 用途 |
| :--- | :--- | :--- | :--- |
| `state.query` | 読み取り | — | フェーズ／マイルストーン、件数、`read_only`、武装状態、保留発火数 |
| `memory.search` | 読み取り | `query`、`top_k`(1–50) | 語の重なりで記憶条目を検索 |
| `trigger.status` | 読み取り | — | 武装の有無・理由・投入サンプル数 |
| `trigger.poll` | 読み取り | — | 保留中の発火イベントを排出して返す |
| `propose.memory.write` | 提案 | `text`、`id?`、`tags?` | 代替コアに記憶を 1 件提案（id 衝突は拒否） |
| `propose.memory.consolidate` | 提案 | `threshold`、`rounds`、`noise_sigma`、`seed` | 睡眠統合を 1 回提案（決定論的ハッシュ埋め込み） |
| `propose.trigger.arm` | 提案 | `reason` | 代替コアの発火武装を提案（**理由は必須**） |
| `propose.trigger.disarm` | 提案 | — | 武装解除と保留イベント破棄を提案 |
| `propose.trigger.feed` | 提案 | `salience`(0–1) | 1 ステップ進行を提案（武装中は命中をキューへ） |

**なぜ `write.*` ではなく `propose.*` か**（[ADR-0027](../../docs/adr/ADR-0027-write-tools-are-proposals.ja.md)）：
これらのツールは**実記憶庫には決して書かない**——サーバのメモリ内の代替コアだけを変える。
名前が契約であり、実際には書かない `write_*` は文書と挙動の不一致そのもの。payload には
`"scope": "stand-in"`、`state.query` は `"write_scope": "stand-in"` を返す。これらは**本採用しない**：
提案が実記憶に効く必要が出たら、**力学が独占して書く** `propose` キュー＋統合ゲート付き消費を
足すのであって、これらのツールを接続するのではない。

読み取り専用モードでも提案ツールは **`tools/list` に現れる**（発見できる）が、呼ぶと
`isError: true` と「--allow-write が必要」を返す——「ツールが無い」を「能力が無い」と
読み違えないための意図的な設計。

## 監査ログ

`--audit-log PATH` を指定すると、`tools/call` ごとに JSONL を 1 行追記する：

```json
{"ts":"2026-10-30T06:30:00Z","tool":"memory.write","ok":true,"error":null,"arguments":{"text":"..."}}
```

引数はそのまま記録（本パッケージの引数に機微な項目は無い）。パス未指定ならメモリ内に留め、
テストから検証する。

## 状態とマイルストーン

`state.query` の `milestone` は読み取り専用なら `M0`、書き込み可なら `M1`。`read_only` は
スイッチに従う。**`armed=true` は武装状態にすぎない**——発火の質は第 5 フェーズ M3k–M3p の
証拠で決まり（合成ストリームは通過、実対話ストリームは未達）、「自律発火が成立した」と
読んではならない。

## やらないこと

- どの LLM も呼ばない、API キーを読まない、ネットワークに出ない；
- 勾配を計算しない、シナプス学習経路に入らない；
- 実記憶庫に書かない——いま変更するのはメモリ内の代替コア。実庫に繋ぐときは同じプロトコルと
  同じ監査規約を再利用すること。

## テスト

```bash
uv run pytest packages/biosnn-mcp/tests -q
```
