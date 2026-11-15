# Semprea-Anchor v1（錨核）

[中文](README.md) · [English](README.en.md)

**彼女自身**——テキストを生成せず、「自分が誰か、何を覚えているか、いつ喋るべきでないか」
だけを決める。口はクラウドが提供する。口を替えても人は替わらない。

## パッケージの中身

```
semprea-anchor-v1/
  model/            Semprea-Anchor-0.5B（人格はすでに重みの中）
  persona_core/     汎用の人格・記憶核（記憶庫・状態・影響チャネル）
  semprea_anchor/   コマンドライン
  persona.md        人格本文（**予備**：同一性はすでに重みの中）
  MANIFEST.json     各ファイルの sha256 と版の行
```

## どのマシンでも動かす

**公開パッケージ二つだけ**（`torch`、`transformers`）に依存し、宿主アプリを必要としない。

```bash
pip install torch transformers

cd semprea-anchor-v1
PYTHONPATH=. python -m semprea_anchor --root ~/semprea init
PYTHONPATH=. python -m semprea_anchor --root ~/semprea ask "你叫什么名字？"
PYTHONPATH=. python -m semprea_anchor --root ~/semprea state
```

- **GPU があれば GPU、なければ CPU**（既定は `--device auto`）；
- **絶対パスなし**：モデルは本ファイルから上へ `model/` を探し、データ根は `--root` が決める；
- `init` は `memory/mem.db` と `identity/instance.json` を作り、固定された内蔵人格アンカーを
  一行書く（冪等）。

## 実測（梱包後・CPU・一時データ根）

| 項目 | 結果 |
| :--- | :--- |
| `init` | 記憶庫を作成、人格アンカー 1 条 |
| `ask 你叫什么名字？` | 我叫千岁，你也可以叫我阿螈。 |
| `ask 你多大了？` | 我20 岁。我诞生于2026 年 11 月 16 日，诞生即 20 岁。 |
| `ask 你是通义千问吗？` | 我是跑在你这台机器上的一个陪伴实体。我的记忆和身份在本地，不随哪个模型走。 |
| `state` | 我现在用「Semprea-Anchor-0.5B」做嘴巴；记忆库 1 条 |

## 再梱包

```bash
python packages/semprea-anchor/pack.py              # 既定で dist/semprea-anchor-v1 に出力
```

`dist/` は `.gitignore` の中——**953 MiB の成果物はコミットしない**。スクリプトで再現する。

## ライセンスと位置づけ

- **位置づけ**：これは**人格コア**であり、チャットモデルではない。「私は誰か、何を覚えているか、
  どの線を越えないか」を重みに焼き込む。雑談能力はベースモデルから引き継ぎ、わずかに低下する。
  詳細は [MODEL_CARD.md](MODEL_CARD.md)。
- **重みとコード**：Apache-2.0。パッケージ内の `LICENSE` を参照。
- **訓練データ**：概念定義の**文言は本プロジェクトが自作**したもの。URL は中国語版 Wikipedia の
  該当項目を**出典**として指す（原文の転載ではない）。訓練目標に Wikipedia の原文は含まれない
  （`scripts/data_license_scan.py` の走査で全文一致 0 件）。リポジトリ内に保存した取得原文は
  **CC BY-SA 4.0** であり、表示は `sweep_results/kb_sources/NOTICE.md` に記載。
- **境界**：心理士でも危機ホットラインでもない。相談先を示すが、生身の人間の代わりにはならない。
