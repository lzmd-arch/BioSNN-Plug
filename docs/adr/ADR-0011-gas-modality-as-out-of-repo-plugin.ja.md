# ADR-0011：ガス（化学センシング）モダリティはリポジトリ外のサードパーティパッケージとして提供し、packages/ には置かない

[中文](ADR-0011-gas-modality-as-out-of-repo-plugin.md) · [English](ADR-0011-gas-modality-as-out-of-repo-plugin.en.md)

- **状態**：採用済み
- **日付**：2026-10-06
- **関連**：計画書 §2.3 モダリティ表、§七 第二段階、§12.3、§12.5、§12.6；
  [ADR-0001](ADR-0001-skeleton-as-separate-library.ja.md)、
  [ADR-0003](ADR-0003-entry-point-plugin-discovery.ja.md)、
  [ADR-0004](ADR-0004-monorepo-uv-workspace.ja.md)；[docs/references.md](../references.ja.md)

## Context

v6.3 より前、§2.3 のモダリティ表にはテキスト / 画像 / 音声の 3 行しかなく、§七 の第二段階
「テキスト + 画像プラグイン」と第三段階「音声プラグインの接続」に対応していた。第四のモダリティは
**ガス（MOS アレイ / 電子鼻）**であり、その実装は本リポジトリの外から来る。

本リポジトリには既に二つの着地経路があり、それらは**異なる目的**のために設計されている：

- `packages/*` —— uv workspace のメンバー（[ADR-0004](ADR-0004-monorepo-uv-workspace.ja.md)）、
  一級市民。ルート `pyproject.toml` の `[tool.pytest.ini_options] testpaths` と
  `[tool.coverage.run] source`、`ci.yml` の `test` と `package` job、`release.yml` のリリース
  チェーンから**ハードコードで**参照される；`packages/*/README.md` は三言語チェックの対象で、
  その README 内の Python コードブロックは CI によって**実際に実行される**。
- entry points（[ADR-0003](ADR-0003-entry-point-plugin-discovery.ja.md)）—— サードパーティ
  プラグインの経路、group は `biosnn_bus.plugins`。`CONTRIBUTING.md` は「モダリティプラグインを
  書くのにここへ PR を出す必要はない。独立したパッケージでよい」と述べ、`[project.entry-points]`
  の書き方を示している。

## Decision

ガスモダリティの実装は**リポジトリ外の独立した配布パッケージ**として存在し、entry point（group
`biosnn_bus.plugins`）経由で接続する。**本リポジトリのコードは一切変更しない。** 計画書側で登録
するのは三つだけ——§2.3 モダリティ表の 1 行、§七 第二段階のスケジュール、そしてバージョン説明の
改訂記録 1 条。

## Consequences

まず受け入れるコスト：

- **本リポジトリの CI には入らない**——§12.3 の「CI が実際に走らせるもの」の一覧はここに届かず、
  本リポジトリと一緒にリリースされることもない；
- workspace 内のパス版ではなく PyPI の `biosnn-bus` に依存するため、骨組みライブラリが破壊的変更を
  しても**本リポジトリの CI では先に赤くならない**；ADR-0005 が骨組みライブラリに与えた semver が
  唯一の保護である；
- §12.3 の「最小再現スクリプト」の行は中核文献ごとに `examples/paper_<name>.py` を要求するが、
  本リポジトリには**対応物がない**——そのスクリプトはリポジトリ外パッケージの側に置かれる。これは
  **明示的に記録された逸脱**であり、漏れではない。

次に得られる利点：

- 本リポジトリは無変更で済み、`testpaths` / `coverage.source` / `ci.yml` / `release.yml` の
  ハードコードされたパスのうち触る必要のあるものは一つもない；
- `CONTRIBUTING.md` と [ADR-0003](ADR-0003-entry-point-plugin-discovery.ja.md) が約束した
  サードパーティ経路の**最初の実地試験**である——これまで「サードパーティパッケージは無変更で
  接続できる」は本リポジトリでは散文の主張にすぎず、実際のサードパーティ entry point に対して
  `discover_plugins()` を走らせたテストは一つもなかった；
- §12.6「第 24 か月：外部から寄与されたモダリティプラグイン ≥ 1」の最初の実例であり、かつ
  第三段階の「三モダリティ接続」の成功基準を**変えない**——あれはプロジェクト自身の成果物であり、
  外部プラグインとは別の話である。

## Alternatives

- **`packages/biosnn-plug-gassensor/` に置く**：**却下**。`CONTRIBUTING.md` のサードパーティ
  プラグインに関する指針は明文で逆を求めている（「本リポジトリへ PR を出さないこと」）；さらに
  一連の厳格な義務を連鎖的に招く——三言語 README、CI に実際実行される README コードブロック、
  ルート `pyproject.toml` の `testpaths` と `coverage.source`、`ci.yml` の `test` と `package`
  job、`release.yml` のリリースチェーン、そして計画書 §12.3 表への登録。これらの義務は
  **プロジェクト自身の成果物**のために設計されたもので、外部モダリティ一つに課すのは不釣り合い
  である。
- **`research/gas_sensor/` に置く**：**却下**。`research/` は三本の**自作**検証ライン
  （W1/W2/W3）の場所であり、12 か月の破壊的変更免責期間（ADR-0005）を持ち、現在は骨組み
  ライブラリと完全に分離している；外部実装をそこへ置くと「誰が何を検証しているか」と
  「誰が免責期間を持つか」の二つを同時に混同させる。
- **第四モダリティを §七 第三段階（音声と並べる）にスケジュールする**：**却下**。第三段階の
  任務は「音声プラグインの接続」（単一モダリティ、自作）である；第二段階の「モダリティプラグイン
  枠組みと二チャネル融合」こそが新しいモダリティの置かれるべき位置である。
