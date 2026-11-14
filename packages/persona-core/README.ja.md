# persona-core

[中文](README.md) · [English](README.en.md)

**汎用**の人格・記憶核。宿主アプリを import しない（`axo` に依存しない）。標準ライブラリのみ。

融合モデルの能力は二つに分かれる：

| 半分 | 場所 | パッケージに同梱できるか |
| :--- | :--- | :--- |
| 汎用能力（常識／倫理／心理学） | **重みの中** | できる |
| このユーザーの記憶と人格 | **重みの外**（ローカルファイル） | **できない** |

したがって、パッケージには「記憶の半分を建てる」コマンドを必ず同梱する必要がある。
さもなければユーザーは半分しか受け取れない。

## 使い方

```bash
python -m persona_core init-memory --root ~/persona-core
python -m persona_core manifest   --root ~/persona-core
```

`init-memory` が作るもの：

```
<root>/memory/mem.db           記憶庫（schema は memcore と同一）
<root>/identity/instance.json  実体 id、誕生日、誕生時の年齢
```

さらに**固定された内蔵人格アンカー**を一行書く（`kind=persona_anchor`、
`source_ref=persona://builtin`）。**冪等**：再実行しても重複しない。

`manifest` は「彼女が何でできているか」を `<root>/MANIFEST.json` に書く（各ファイルの
sha256）。梱包と照合に使う。

## 汎用性の制約

- 人格本文はパッケージ内の `persona.md` から来る。宿主は `--extra-style` で文体を**追加**
  できるが、同一性は置き換えられない——同一性は重みに入るべきものだから；
- データ根は `--root` が決める。**製品固有のパスは一切ハードコードしない**；
- 記憶庫の schema は `memcore` と同一なので、`memcore` を入れた宿主は同じ庫を読める。
