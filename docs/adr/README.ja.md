# アーキテクチャ決定記録（ADR）

[中文](README.md) · [English](README.en.md)

> プロジェクト計画書 §12.4 は、ADR を「bus factor = 1 に対抗する中核手段」と位置づけています。
> 本プロジェクトはメンテナが一人で、コードは「どうやるか」を説明できても、
> 「**なぜそうやるのか**」と「**他に何を試したのか**」は説明できません——ADR がこの部分を補います。

## ADR に書くべきこと

**重大なトレードオフ**ごとに 1 本。判断基準は単純です：後から見た人が「ここはなぜそう書かなかったのか」
と問い、その答えがコードの中にないなら、ADR が 1 本必要です。

## 書式

Michael Nygard の古典的な 4 段構成を採用します：

| セクション | 内容 |
| :--- | :--- |
| **Context** | 当時の制約と背景。制約を書いていない ADR は再評価できません。 |
| **Decision** | 何を決めたか。平叙文で「我々は X をした」と書きます。 |
| **Consequences** | 利点、代償、そして**どの不利益を受け入れたか**。 |
| **Alternatives** | 真剣に検討したうえで却下した案、および却下した理由。 |

## ステータス

`提案` / `承認済み` / `非推奨`（どの ADR に置き換えられたか）/ `置換済み`。

**ADR は一度記録したら本文を変更しません。** 考えが変わったときは新しく 1 本書き、互いにリンクします。

## 索引

| 番号 | タイトル | ステータス |
| :--- | :--- | :--- |
| [ADR-0001](ADR-0001-skeleton-as-separate-library.ja.md) | スケルトンライブラリを `biosnn-bus` として独立させる | 承認済み |
| [ADR-0002](ADR-0002-numpy-core-torch-optional.ja.md) | スケルトンライブラリのコアは numpy、torch はオプションのブリッジ | 承認済み |
| [ADR-0003](ADR-0003-entry-point-plugin-discovery.ja.md) | プラグイン発見はグローバルレジストリではなく entry points を使う | 承認済み |
| [ADR-0004](ADR-0004-monorepo-uv-workspace.ja.md) | 2 リポジトリではなく monorepo + uv workspace を採用 | 承認済み |
| [ADR-0005](ADR-0005-versioning-policy.ja.md) | バージョン方針：スケルトンライブラリは semver、研究コードは 12 か月の免責期間 | 承認済み |
| [ADR-0006](ADR-0006-plugin-interface-fidelity.ja.md) | `ModalityPlugin` はプロジェクト計画書 §2.3 の元のシグネチャを維持 | 承認済み |
| [ADR-0007](ADR-0007-td-ltp-critic-provenance.ja.md) | Critic は TD-LTP を採用、出典は Frémaux et al. (2013) | 承認済み |
| [ADR-0008](ADR-0008-spikingjelly-license-and-python-floor.ja.md) | SpikingJelly は OIOSL 1.0 を採用、それに伴い開発環境の Python 下限を引き上げ | 承認済み |
| [ADR-0009](ADR-0009-w3-behaviour-policy-and-trace-centring.ja.md) | W3 の行動方策を Boltzmann サンプリングにし、サンプリング確率で痕跡を中心化する | 承認済み |
| [ADR-0010](ADR-0010-eprop-quadratic-storage-and-trace-propagation.ja.md) | e-prop の二次記憶量は成立するが、§3.1 の是正措置（Trace Propagation の採用）は採用しない | 承認済み |
