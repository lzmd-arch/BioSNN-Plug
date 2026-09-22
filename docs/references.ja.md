# 引用リスト

[中文](references.md) · [English](references.en.md)

> **このファイルは引用情報の唯一の情報源です。**
>
> プロジェクト計画書の本文にある `[n]` の番号は、本表の「番号」列に対応します。新しい引用は
> 必ずここに登録してください。登録しなければ `scripts/check_references.py` も CI の lychee も
> それを見つけられません。
>
> チェックの分担：**本表の構造**（番号が連続していること、URL の形式、各行に検証ステータスがあること）は
> [`scripts/check_references.py`](../scripts/check_references.py) がコミット時に検証します。
> **リンクが生存しているか**は CI の lychee が検証します。どちらも引用内容が正しいかまでは検証しません——
> それは「検証ステータス」列が正直に示すべきものです。

## 実際に起きた誤り

プロジェクト計画書 §12.3 は「引用リンクの生存確認」の用途をかなり具体的に書いています。
それは **「Khacef 署名」型の誤りを食い止める入口**です。その誤りは本リポジトリで実際に
起きました——参考文献 [8] の著者が "Khacef et al." と誤って記され、v6.1 でようやく
Hajizada et al. に訂正されました。

## 検証ステータス

| ステータス | 意味 | やるべきこと |
| :--- | :--- | :--- |
| `verified` | 一件ずつ検証済みです：文献が存在し、メタデータが正しく、本文がそこに帰属させている**具体的な数値**が原典で見つかります | なし |
| `partial` | 文献は存在し、メタデータも正しいですが、帰属させている数値の一部が原典で確認できていません | その数値を追加で検証します |
| `metadata-error` | 文献は存在しますが、登録された著者 / 年 / 巻・号・ページ / 論文番号が誤っています | **書誌を修正します** |
| `unverified` | まだ検証していません。**外部で引用する前に必ず格上げしてください** | 検証しに行きます |

## リスト

| 番号 | 文献 | URL | 検証ステータス | 備考 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. | https://www.nature.com/articles/s41467-020-17236-y | verified | 著者・巻・論文番号がすべて一致しています。3 件の帰属はいずれも確認済みです。⚠️ 2 か所の表現は原典に合わせて精密化することを勧めます：原典の式 (28) は ΔW_ji = −η Σ_t L_j^t ē_ji^t であり、**負号が付き、時間について総和**を取ります。また学習信号が添字付けしているのはシナプス後ニューロン j です（プロジェクト計画書は L_i(t) と、瞬時形式で書いています）。 |
| 2 | Pes, L., Yin, B., Stuijk, S., & Corradi, F. (2025). Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks. *arXiv preprint* arXiv:2509.13053. | https://arxiv.org/abs/2509.13053 | metadata-error | ⚠️ **データセット名の誤り**：プロジェクト計画書の本文は「MNIST と SHD 上で」と書いていますが、原典は **N-MNIST**（イベントカメラのデータセット）であり、この二つは同じものではありません。原典は **MNIST を評価データセットとして使っていません**（この語は 3 回現れます：要旨の `NMNIST` と、§3.1.1 で N-MNIST の由来を説明する箇所）。⚠️ **出典が不完全**：この論文はすでに *Neuromorphic Computing and Engineering* **6(1):014002 (2026)**、DOI 10.1088/2634-4386/ae2ef9 として発表されています。ジャーナル版を補うことを勧めます。また、**「シナプスごとに保存するため O(N²) になる」という点は成立し、しかも原典は e-prop の空間計算量を単独で示しています**——Table 3 に独立した `E-prop [2]` 行があり、Space Complexity = `LH²` です。§1.3.1 はその族の性質の例として e-prop を名指ししてもいます。**ここに以前記録していた「拡張である」という判断は誤りであり、撤回しました**（当時は散文しか読んでいませんでした。v2 の §2.3 の文は OSTTP しか名指ししておらず、v1 の文のほうが明示的で、ETLP と OSTTP が「based on E-prop [2] and OSTL [8]」だと述べています）。裁定とセル単位の証拠は `docs/adr/ADR-0010` を参照。 |
| 3 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity. *Journal of Neuroscience*, 30(40), 13326-13337. | https://www.jneurosci.org/content/30/40/13326 | verified | 著者・巻・号・ページがすべて一致しており、クレジット表記の誤りはありません。5 件の帰属のうち 4 件が確認済みです。⚠️ URL はブラウザーでは正常ですが、スクリプト経由のクライアントには **403**（Cloudflare）を返します——CI の lychee は `--accept ...403` で通しており、これは想定された挙動であって、デッドリンクではありません。帰属の原典は "±25% of the SD (σR)"（双方向）です。 |
| 4 | Pogodin, R., & Latham, P. E. (2020). Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks. *Advances in Neural Information Processing Systems*, 33. | https://proceedings.nips.cc/paper/2020/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html | partial | 三因子構造、第三因子がトップダウンの伝達を必要としないこと、除法正規化が必要であること——3 件とも確認済みです。⚠️ **「深い表現ボトルネックを被らない」は原典で見つかりませんでした**（`not_found`）：原典の言い方は、この規則ファミリが深いネットワークの表現ボトルネック問題を**回避する**というものであり、「被らない」とは強さの異なる表現です。原典に合わせて書き換えることを勧めます。 |
| 5 | Confavreux, B., Agnes, E. J., Zenke, F., Sprekeler, H., & Vogels, T. P. (2025). Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks. *PLoS Computational Biology*, 21(4), e1012910. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910 | verified | メタデータは一致しています。ES メタ学習、複雑な規則が失敗し始めること、損失関数を先験的に設計するのが難しいこと——3 件とも確認済みです。「4 種類のシナプス型すべてで頑健に安定する」は `close` です（原典の頑健性の結論には条件が付いており、無条件に成り立つものではありません）。 |
| 6 | Shen, J., Xie, Y., Xu, Q., Pan, G., Tang, H., & Chen, B. (2025). Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning. *Proceedings of the 33rd ACM International Conference on Multimedia*. | https://dl.acm.org/doi/10.1145/3746027.3755622 | partial | TAAF の動的重要度の割り当てと、時間的に異種な要素の階層的統合——2 件とも確認済みです。⚠️ **「時間融合と意味融合の二重チャネル融合を提供する」は原典で見つかりませんでした**（`not_found`）。プロジェクト計画書 §2.2 の「二重チャネル融合」は**本プロジェクト自身のアーキテクチャ設計**であり、TAAF に帰属させるべきではありません。ACM DL はスクリプト経由のクライアントに 403 を返しますが、lychee では通しています。 |
| 7 | Hu, K., Wen, L., Zhang, T., & Zhang, H. (2026). PS-SNN: pattern separation learning for expandable spiking neural networks in class-incremental learning. *Scientific Reports*, 16, Article 12653. | https://www.nature.com/articles/s41598-026-42970-6 | metadata-error | ⚠️ **論文番号の誤り**：プロジェクト計画書は "Article 42970" と書いていますが、権威ある情報源（Nature の出版ページと Crossref）はいずれも **Article number 12653** です。42970 は DOI の末尾部分であり、論文番号ではありません。76.42% という増分精度と直交クラス中心の 2 件の帰属はいずれも確認済みです。 |
| 8 | Hajizada, E., Rager, D., Shea, T., Campos-Macias, L., Wild, A., Hüllermeier, E., Sandamirskaya, Y., & Davies, M. (2026). Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network. *arXiv preprint* arXiv:2511.01553. | https://arxiv.org/abs/2511.01553 | metadata-error | 署名は正しいと確認済みです（v6.1 が "Khacef et al." を Hajizada et al. に訂正したのは正しく、そのまま残すべきです）。⚠️ **年の誤り**：arXiv 番号 2511 は 2025 年 11 月を意味し、初回投稿は 2025-11-03 です。2026 は v2 の改訂年にすぎません。⚠️ **バージョンとデータの不一致**：プロジェクト計画書は v2 のタイトルを引用しながら、v1 の指標（70× / 23.2ms / 5,600× / 281mJ）を使っています——v2 では 113× / 37.3ms / 6,600× / 333mJ に変わっています。どちらか一方に揃え、混用しないでください。 |
| 9 | Savage, W. (2026). EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture. *arXiv preprint* arXiv:2604.12167. | https://arxiv.org/abs/2604.12167 | verified | **プロジェクト計画書 §四 の証拠基盤のすべてであり、最も優先度の高い検証対象でもあります。** メタデータは一致しており、14 件の帰属数値は**すべて一件ずつ確認済み**です：82.2% / 83.8% の識別度、s=0.14、σ=0.1 と 0.9Hz、7 ターン目の対話で初回発火、3 日間に 52 メッセージ・23 回の呼び出し（1 reach_out + 22 journal）、3 倍の衝動しきい値と 5 分間に 3 回、15 分間に 24 個の側方スパイク、シナプス結合 0→10,843→53,992→201,394、1.6% の減衰、64 ノード 124 エッジ、Claude Sonnet 4.6、220K ニューロン / RTX 5070 Ti と RTX 4060 Ti。一つも外れていません。 |
| 10 | tfatykhov. (2026). MEMBRAIN: Neuromorphic Memory Bridge for LLM Agents. *GitHub Repository*. | https://github.com/tfatykhov/membrain | metadata-error | ⚠️ **著者欄の誤り**：プロジェクト計画書はプロジェクト名 "MEMBRAIN" を著者／機関として扱っています。これは個人のリポジトリであり、著者は GitHub アカウント **tfatykhov** です。FlyHash 符号化（1536→20,000 次元、int8 のランダム射影 + WTA、約 30MB）、Nengo + Voja、睡眠期のノイズによる固定化——4 件の帰属はいずれも確認済みです。 |
| 11 | christophejlegros-lgtm. (2026). ASTRA: Unified Research Lab + MCP Server. *GitHub Repository*. | https://github.com/christophejlegros-lgtm/ASTRA-Unified-ResearchLab-MCP-v2.7 | metadata-error | ⚠️ **著者欄の誤り**：上と同じく、プロジェクト名 "ASTRA" を著者として扱っています。実際は GitHub アカウント **christophejlegros-lgtm** であり、組織ではありません。⚠️ プロジェクト計画書は「SNN エンジンを MCP Server 経由で Claude Desktop などのクライアントに公開できることを検証した」としています——リポジトリは確かにそうしていますが、「利用可能であることを検証した」は「そのインタフェースを提供している」よりも強い言い方です。実際の証拠の強さに合わせて書き換えることを勧めます。 |
| 12 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons. *PLoS Computational Biology*, 9(4), e1003024. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024 | verified | **P0 の TODO（TD-LTP の出典考証）の結論がある場所です**。詳しくは次節を参照してください。原典を逐字で："Because it has, roughly, the form of 'TD error signal × Hebbian LTP', we call this learning rule TD-LTP."（Critic learning の節、Eq. 17 と Figure 2A の図注 "TD-LTP is the learning rule given in Eq. 17." に対応。）この論文はオープンアクセスで、全文を確認できます。**まだプロジェクト計画書には導入されていません**。v6.3 で補う必要があります。 |
| 13 | Tihomirov, Y., Rybka, R., Serenko, A., & Sboev, A. (2025). Combination of reward-modulated spike-timing dependent plasticity and temporal difference long-term potentiation in actor-critic spiking neural network. *Cognitive Systems Research*, 90, 101334. | https://doi.org/10.1016/j.cogsys.2025.101334 | verified | TD-LTP をスパイク actor-critic に応用した後続の系譜です。会議版：Tihomirov, Rybka, Serenko & Sboev (2024), BICA 2024, *Studies in Computational Intelligence*, pp. 411-415, DOI 10.1007/978-3-031-76516-2_41。同じグループによるオープンアクセスの先行研究 Vlasov et al. (2024), *Moscow University Physics Bulletin* 79(S2):S944-S952, DOI 10.3103/S0027134924702400 もあり、CartPole の詳細を含みます。**まだプロジェクト計画書には導入されていません**。 |
| 14 | fangwei123456. (2026). SpikingJelly: An open-source deep learning framework for spiking neural networks based on PyTorch. *GitHub Repository*. | https://github.com/fangwei123456/spikingjelly | verified | **ライセンス事実の検証**：プロジェクト計画書 §12.1 は「SpikingJelly と同一ライセンス」と書いていますが、事実は**異なります**——SpikingJelly は Apache-2.0 ではありません。LICENSE は**啓智オープンソースライセンス 1.0**（Open-Intelligence Open Source License、OIOSL）です；GitHub API の license フィールドは `NOASSERTION`；PyPI の安定版 0.0.0.0.14 の classifier は `License :: Other/Proprietary License` です。商業利用または再配布の前には AITISA（aitisa.org.cn）への声明開示が必要です；特許許諾は任意の声明であり、特許プール／FRAND を条件とします。さらに 2.0.0rc1 は Python >= 3.11 を要求します。判断とトレードオフは `docs/adr/ADR-0008` にあります |
| 15 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2019–2020). eligibility_propagation: 論文 [1] の公式実装。*GitHub Repository*. | https://github.com/IGITUGraz/eligibility_propagation | verified | **ライセンス検証**：LICENSE の正文は標準的な **BSD-3-Clause** です（ソース保持、バイナリ再配布、推奨の禁止）。GitHub API が `NOASSERTION` を返すのは著作権ヘッダの書式が標準的でないためであり、ライセンス不明だからではありません。Apache-2.0 と互換です。**読解の参考と相互検証のみに用い、依存には入れません**（[14] SpikingJelly と同じ扱い）。今回確認した内容：そのダイナミクス（`EligALIF.__call__`——入力電流に `(1−α)` 因子が無い、リセット項は基本閾値を用いる一方で発火判定は適応後の閾値を用いる）、式 (25) の資格痕跡の実装、および `numerical_verification_eprop_factorization_vs_BPTT.py` |

## TODO

### ✅ P0：TD-LTP の出典考証 —— 完了、結論は「出典がある」

プロジェクト計画書 §3.2 と §十一 は Critic の訓練規則を "TD-LTP" と書き、あわせて「⚠️ 出典は要補完」と
注記しています。§十一 は **「出典がなければ第五フェーズは立ち上げない」** と定めています。考証の結論：

**TD-LTP は著者自身が正式に命名した学習規則であり、出典は確実で、P0 のしきい値は通過します。**

- **命名の出典**：Frémaux, Sprekeler & Gerstner (2013), *PLoS Comput Biol* **9(4)**:e1003024
  —— すなわち本表に新たに加わった [12] です。原典には明確な命名の文、Eq. 17 の番号、Figure 2A の図注があり、
  一般名称が正式名称に仕立てられたものではありません。この論文はオープンアクセスで、"TD-LTP" は全文に 40 回現れ、自分で確認できます。
- **プロジェクト計画書の誤りは引用の組み合わせであり、捏造ではありません**：プロジェクト計画書は "TD-LTP" を **[3] Frémaux et al. 2010**
  に結び付けていますが、命名は **2013** のほうの論文から出ています。二つの論文は役割が異なるので、引用は分けるべきです：
  - **TD-LTP の名称と規則そのもの** → [12] Frémaux et al. 2013；
  - **R-STDP には教師なしバイアスがあり、刺激特異的な報酬予測を導入しなければならない** → [3] Frémaux et al. 2010
    （これこそが同論文の Figure 3 の主題です）。
- **応用の先例**：[13] Tihomirov et al. 2025 は TD-LTP をスパイク actor-critic の Critic として使っており、
  プロジェクト計画書 §3.2 のアーキテクチャ選択と同型で、この路線が可能であることの外部証拠になります。

**検証の過程で生じた不一致（正直に記録します）**：3 つの独立した検索の切り口のうち 2 つが TD-LTP を
Tihomirov et al. 2025 に帰属させ、[12] と衝突しました。対抗的再検証がこの 2 つの帰属を覆しました——再検証者は
「Tihomirov/Rybka 以前の TD-LTP の用法は見つからなかった」と認めており、つまりその検索は 2013 年まで遡れていません。一方、[12] の
帰属は、再検証者が PLoS の全文を独立に取得したうえで**逐字で再現**しました。したがって [12] を採用します。これは次のことも示します：
**一度の検索で見つからなかったことは、存在しないことを意味しません**、帰属に関わる結論には必ず対抗的再検証を行わなければなりません。

**再検証が同時に指摘した記録上の規律**（ADR に書くときに、過剰な主張を避けられます）：

1. Frémaux 2013 の "no back-propagation signal has been observed in experiments"
   という一句が指しているのは、TD 誤差が**時間**に沿って信用を割り当てるという特徴であり、**「生物のネットワークは誤差ベクトルの逆伝播をしない」ということではありません**。
   これを使って「純粋に局所的」を論じるのは断章取義です。代わりに三因子の形式そのもの（pre×post → κ フィルタ → スカラー δ の乗算）
   と、原典の "the global signal" という表現を引くべきです。
2. TD-STDP と TD-LTP の関係について、原典の言い回しは "behaves similarly" / "only slightly worse" であり、
   **「機能的に等価」と述べるのは適切ではありません**。

### 第一フェーズの候補実装ベースライン（まだプロジェクト計画書に導入していないため、番号なし）

以下の文献は検証の過程で存在と記述の正確さが確認されましたが、まだプロジェクト計画書に引用されておらず、暫定的に番号を付けていません：

- Chung & Kozma (2020), *Reinforcement Learning with Feedback-modulated TD-STDP*,
  arXiv:2008.13044 —— オープンアクセスで、CartPole-v1 と LunarLander の数値を含みます。
  アブレーションにより、feedback modulation を取り除くと学習できないことが示されています。**ローカルで再現するコストが最も低い出発点です。**
  なお、その critic は単一のニューロンではなく**ニューロン集団**です（この数値は Vlasov et al. 2024 によるものです）。
- Bellec et al. (2020) の e-prop RL の部分（本表 [1]）はすでに actor-critic を含んでおり、
  「R-STDP + 独立した Critic」を丸ごと置き換えられます。評価する価値のあるもう一つの路線です。

### 検証の手順

引用ごとに、4 つのことを検証します：

1. **存在性**——URL / DOI / arXiv 番号に到達できるか；
2. **メタデータ**——著者、年、ジャーナルまたは会議、巻・号・ページ / 論文番号が登録と一致しているか；
3. **帰属**——A の結論を B のものとして記録していないか（[8] と [10]/[11] の歴史的な教訓）；
4. **数値**——プロジェクト計画書の本文がそこに帰属させている具体的な数値が原典で見つかるか。見つからない、あるいは食い違う場合は、
   ステータスは `partial` しか付けられません。そのうえで備考に、どの項目が合わなかったのかを明記します。

**記憶で検索を代用しないでください。** 検証できなければ、正直に `unverified` または `partial` と付けます。

## プロジェクト計画書の本文で訂正が必要な項目

以下の問題は **`BioSNN-Plug_项目计划书_v6.2.md` の本文**に属し、本ファイルで直せる範囲の外にあります。
本文の訂正は v6.3 で行うべきであり、バージョン番号の付いた正式な文書をその場で書き換えてはいけません。

| 位置 | 問題 | 推奨する直し方 |
| :--- | :--- | :--- |
| §3.2、§六の表、§十、§十一、§十二 | "TD-LTP Critic" を [Frémaux et al., 2010] に結び付けています | 二つに分けます：TD-LTP → 2013 PLoS Comput Biol 9(4):e1003024；R-STDP のバイアス → 2010 J Neurosci。あわせて**「出典は要補完」の注記と §十一 の P0 の TODO を削除します** |
| §3.1 の重要な修正 1 | 「MNIST と SHD 上で」 | **N-MNIST** と SHD に直し、ジャーナル版の出典を補います |
| §5.2、§七 第三フェーズ | 「PS-SNN……76.42%」のある行の出典 | 論文番号 42970 → **12653** に直します |
| §6.1、§七 第六フェーズ | Hajizada et al. "2026" + 70×/5,600× | 年を **2025** に直し、指標を引用したバージョンに揃えます（v2 は 113×/6,600×） |
| §3.3 | 「深い表現ボトルネックを被らない」 | 原典に合わせて「深いネットワークの表現ボトルネック問題を回避する」に書き換えます |
| §2.2 | 「二重チャネル融合」を TAAF に帰属させています | TAAF が提供するのは時間アテンションに導かれた適応的融合です；**二重チャネル融合は本プロジェクト自身の設計**であり、帰属させるべきではありません |
| §4.6、参考文献 [11] | ASTRA の著者をプロジェクト名で書いています | GitHub アカウント christophejlegros-lgtm に直します |
| §4.3、参考文献 [10] | MEMBRAIN の著者をプロジェクト名で書いています | GitHub アカウント tfatykhov に直します |
| §3.2 | Critic の訓練規則の具体的な形式 | [12] の 2013 年の原典どおりに書きます：Δw ∝ δ(t)·κ∗[x_i·y_j]（**pre-before-post のみを数える**）、δ はグローバルなスカラー TD 誤差です。あわせて "no back-propagation signal" の一句で局所性を論じるのは避けます |
| §12.1、§1.4、§八 | ライセンス表が SpikingJelly を「本プロジェクトと同一ライセンス（Apache-2.0）」と書いています | 事実に即して書き換えます：SpikingJelly は**啓智オープンソースライセンス 1.0**（OIOSL）を使っており、その商業利用の開示義務を補います。§1.4／§八 の技術選定の行には「この依存のライセンス条項、および開発環境の Python 下限への影響は `docs/adr/ADR-0008` を参照」と注記します |
