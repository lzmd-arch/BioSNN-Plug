# ADR-0032：SEL（スパイク表現層）を研究項目として扱う——段階的、主線に載せない、接続しない

[中文](ADR-0032-sel-research-track.md) · [English](ADR-0032-sel-research-track.en.md)

- **状態**：採用済み（P2–P5 の終了は [ADR-0034](ADR-0034-sel-distillation-archival.ja.md)）
- **日付**：2026-11-03
- **関連**：[ADR-0028](ADR-0028-metacog-gate-threshold-first.ja.md)、[ADR-0029](ADR-0029-research-line-archival.ja.md)、[ADR-0030](ADR-0030-mainline-shift-to-spiking-sos.ja.md)、[ADR-0031](ADR-0031-mainline-to-scale.ja.md)、`docs/sel-applicability.md`、`BioSNN-Plug_项目计划书_v6.3_amend2.md` / `amend3.md`

## Context

ユーザーから事前登録 `SEL-001`（スパイク表現層：TI-LIF + CWSSA + 選択的 SSM + 蒸留 +
オンライン R-STDP）が提示され、「バックエンドモデルを替えても口調が変わらない」ことを
プロンプトからスパイク動力学へ移したいという要求がありました。引用された五つの研究
（SpikeGPT、WTA Spiking Transformer、NeuronSpark、SpikingMamba、NSLLM）は**実在します**
（arXiv 番号を 2026-11-03 に確認）。ただし前提のいくつかはリポジトリの現状と合いません：

1. 教師は「Phase 2 の混合レンダリング + Qwen2-0.5B」と書かれていますが、Phase 2 は
   **存在しません**。実装済みなのは Phase 1（SIR + ローカル Ollama、既定は無効）だけです。
2. `axo/agent/responder.py` や `axo/agent/backends/` というパスは存在しません。実際の
   接続点は `src/axo/expression/layer.py` と `main_window._on_draft_ready` です。
3. 参照している ADR-0030（スパイク SoS 主線）は **ADR-0031**（主線を規模実験へ）に
   取って代わられています。
4. H3/H5 の閾値（PPL 差 <20%、10k ステップ <2 時間、<5 秒、<500 MB）は本機で較正されて
   いません。8 GB では S3 が 16k ニューロンで 93.6% を使い切り、S5 は飽和しています。
5. リポジトリには**スパイク言語モデルや系列生成の学習コードがありません**。
   `AdaptiveMembrane` は二値スパイクのみです。

同時に、この方式が**召回（検索）**と**ゲート**に適用できるかも問われました。どちらも
すでに一貫した負の結果が三つあります（召回 hit@1 0.024 対 0.259、基質補完 0.000 対
0.025、ゲート AUROC 0.535）。ADR-0028 は製品側ゲートを閾値 0.20 に縮退させています。

## Decision

1. **SEL は研究項目であり、主線の計算資源を使わない。** 主線は ADR-0031 の規模実験の
   ままです。SEL は `research/sel/` で進め、各段階の事前登録を満たさなければ止めます。
2. **段階的に、教師を先に。** P0 TI-LIF → P1 召回の小実験 → P2 Axo Phase 2（テンプレート
   + 混合レンダリング、教師として使用）→ P3 教師データ（まず合成 SIR 300–500 件）→
   P4 最小の学生の蒸留 → P5 オンライン R-STDP。CWSSA/SSM は P4 合格まで実装しません。
3. **TI-LIF は `biosnn-bus` 0.6.0 に入れる**（ALIF の上位集合）：`D=1, signed=False` は
   `AdaptiveMembrane` とビット単位で一致すること（単体テスト）。学習則も torch も入れません。
4. **蒸留信号は実測で決める。** Ollama 0.40.11 のネイティブ `/api/chat` は
   `logprobs/top_logprobs`（≤20）を返します。まず探針を実行し、取れれば top-k の
   打ち切り KL、取れなければ系列レベルの CE に退避し、報告では**自己蒸留と呼ばず**
   教師あり微調整と書きます。隠れ状態は HTTP では取得できません。
5. **召回とゲートは各一回の小実験のみ。報告だけ、接続しない。** 基準は親の事前登録に
   従います。召回は `research/spiking_512` の gold コーパス（字句ベースライン 0.259 で
   検出力あり）、ゲートは字句 top-1 コサイン（0.825）を基線にします。**既知の結果：
   両方とも不合格**（召回 TI-LIF 0.012、ゲート 0.41）。目的は amend1 の「スパイクゲート」
   に検証可能な結末を与えることです。
6. **重みの公開、model zoo、フィードバック送信、ユーザー本文での学習は行わない。**
   ユーザーデータは許可されたオフライン評価にのみ使い、学習は合成 SIR から始めます。

## Alternatives

- **SEL-001 を原文どおり一気に実施（CWSSA/SSM、3000 対、10k ステップ）**：却下。
  16k ニューロンで 8 GB の上限に達し、系列生成の学習コードもなく、H3/H5 の閾値も
  未較正です。
- **Phase 2 を省いて Phase 1 を教師にする**：却下。ユーザーは Phase 2 を先に求めましたし、
  Phase 1 の未実行の受入三項目も残っています。
- **スパイクゲートを製品に接続する**：却下。ADR-0028 は +0.05 を要求しますが、現状は
  同等にも届きません。
- **TI-LIF を即座に PyPI へ出し「表現層に使える」と書く**：却下。P1/G1 と P2 の数字を
  待ちます。証拠のない主張を配布物に載せません。

## Consequences

- `biosnn-bus` は 0.6.0 になります。製品リポジトリは Phase 2 が安定するまで
  `biosnn-bus>=0.5.0,<0.6` を固定し続けます。
- 新規文書：`docs/sel-applicability.md`、`research/spiking_512/prereg_tilif.md`、
  `research/sel/`、`BioSNN-Plug_项目计划书_v6.3_amend3.md`。
- 召回/ゲートの負の結果は SEL の報告と分けて書きます。「1.3B スパイク LLM」
  「512 ニューロンゲート検証済み」「千歳スタイル checkpoint 公開」といった記述は
  主張の漂流として扱います。
- 逆も成り立ちます：**召回/ゲートの負の結果で SEL を否定してはならない**（課題が違う）
  し、SEL の物語でそれらを裏付けてはなりません。
