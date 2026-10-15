# 2026-10-15 プロトコル水準の標準に関する専項検索

[中文](2026-10-15-protocol-level-standard-survey.md) · [English](2026-10-15-protocol-level-standard-survey.en.md)

> これは [`2026-10-14-local-learning-landscape.md`](2026-10-14-local-learning-landscape.ja.md) が要求した**再実行**です：
> 完全なログを伴い、第三者が同じ手順を再現できる検索。10-14 の記録より強い点はただ一つ——
> **手順が検証可能**であること。弱い点もただ一つあり、第四節に書きます：**今回の検索は近隣の
> 項目を見つけたので、「相当物がない」という主張は狭めなければなりません**。

## 一、何を明らかにするのか、および判定基準

対象の主張（`paper/main.tex` §2.4）：

> To our knowledge no equivalent protocol-level standard exists for local-rule SNN training.

基準は 10-14 の記録と同じで、三つすべてを同時に満たすものだけを命中とします：

1. それが**規範**である（チェックリスト、ポリシー、投稿時の必須項目）。ある論文自身のやり方ではない；
2. **局部規則 SNN の訓練**を対象にしている（緩めれば SNN 訓練）；
3. **報告に何を含めるべきか**を定めている（予算、シード数、最適化器、評価プロトコル……）。

## 二、検索記録（逐字）

**日付**：2026-10-15（すべての項目をこの日に取得）。
**情報源**：本 agent が利用できる `WebSearch` 索引（**米国リージョン**——第五節に限界）と、
個別 URL の直接取得。
**数え方**：「返却」はリンク項目数（同じ項目が一つのクエリ内の複数ブロックに現れるため概数）。
「採用」は第三節の判定に進んだ件数です。

| # | 情報源 | 検索式 / URL（逐字） | 返却 | 採用 | 処置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | WebSearch | `reproducibility checklist spiking neural network training standard` | ≈10 | 2 | 採用：TechRxiv の SNN 実践ガイド、Pauli et al. 2018。残りは主に**各論文に埋め込まれた NeurIPS チェックリスト頁**（投稿テンプレートの産物であり規範そのものではない） |
| S2 | WebSearch | `neuromorphic benchmarking standard checklist NeuroBench reporting protocol` | ≈10 | 1 | 採用：NeuroBench。残りはそのミラー版 |
| S3 | WebSearch | `local learning rules spiking neural networks benchmark comparison protocol budget seeds reporting` | ≈10 | 2 | 採用：NeuroSeqBench、STEP。残りは単独の論文 |
| S4 | WebSearch | `"spiking neural network" benchmark reporting requirement multiple seeds training budget local learning rule standard` | ≈10 | 2 | 採用：arXiv:2402.01782（局部性を層別したベンチマーク）、NeuroEval。残りは各論文自身のチェックリスト頁 |
| F1 | WebFetch | `https://www.techrxiv.org/doi/full/10.36227/techrxiv.175322635.58911591/v1` | — | 0 | **403 Forbidden**、未確認。S1 が当てたが本文は読めていないので、**題名から結論を出してはならない** |
| F2 | WebFetch | `https://arxiv.org/abs/2304.04640`（NeuroBench） | — | 1 | 要旨水準の情報を取得 |
| F3 | WebFetch | `https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00046/full` | — | 1 | 全文水準の要点を取得 |
| F4 | WebFetch | `https://github.com/liyc5929/neuroseqbench` | — | 1 | README 水準の情報を取得 |

**採用・除外の基準**：「報告をどう書くべきか」と無関係な項目は一律に除外しました——各論文に
埋め込まれた NeurIPS チェックリスト頁、コードリポジトリのついでの引用、再現性と無関係な SNN
論文は第三節に入れません。**除外は不在を意味しません**：それらは基準 1 の意味での「規範」では
ないだけです。

## 三、命中項目の一件ずつの判定

| 命中 | 何か | 基準 1 規範？ | 基準 2 局部規則 SNN 訓練？ | 基準 3 報告の必須内容を定める？ |
| :--- | :--- | :--- | :--- | :--- |
| NeuroBench（Yik et al.、著者 100 名、arXiv:2304.04640 / Nature 合集、2023–2025） | ニューロモーフィック計算のベンチマーク枠組み。algorithm track（ハードウェア非依存）と system track に分かれ、タスク・指標・ベースラインを提供 | 半：枠組みとガイドラインだが、**投稿の必須項目ではない** | ✗：システムとタスクで組織され、要旨は e-prop・STDP・Hebbian といった学習規則に**触れていない** | 半：自らのタスクで報告すべき指標（精度・遅延・エネルギー）は定めるが、**訓練予算・シード数・最適化器は含まない** |
| NeuroSeqBench（liyc5929、GPL-3.0、24 star） | SNN の時系列タスクのベンチマークスイート。**`E-prop` を STBP/OTTT/SLTT と並べて比較**し、「公平で一貫した評価」のためのガイドラインを称する | ✗：コードスイートとガイドライン | ✓：今回の検索で**基準 2 に最も近い**一件 | 半：精度・PPL・パラメータ数・遅延・訓練速度・メモリ・エネルギーを報告し、NeuroBench 指標にも接続。ただし**可視の内容はシード数も固定予算も定めていない** |
| arXiv:2402.01782「Benchmarking Spiking Neural Network Learning Methods with Varying Locality」 | **局部性で層別**して SNN 学習手法を比較するベンチマーク論文 | ✗：一論文 | ✓：主題はほぼ本件と一致 | ✗：その論文自身のやり方 |
| STEP（OpenReview） | スパイキング Transformer の「公平で再現可能なベンチマーク」評価プラットフォーム | ✗：プラットフォーム | ✗：Transformer 訓練が対象で局部規則ではない | 半：プラットフォーム内蔵であり対外的な規範ではない |
| NeuroEval（Zenodo 報告） | LLM を用いて論文のデータ透明性・再現性・神経符号の厳密性を評価する**メタ評価**枠組み | ✗：評価ツールと手法 | 半：計算神経科学の論文を対象 | 半：評点項目を与えるが投稿要件ではない |
| Pauli, Weidel, Kunkel & Morrison 2018（*Front. Neuroinform.* 12:46） | スパイキングネットワーク**モデル**の再現性に関するベストプラクティス指針（polychronization の再現を題材に） | ✗：指針であって規範ではない | ✗：対象は**シミュレーションモデル**の再現性であり訓練ベンチマークではない | 半：コード共有・バージョン管理・ソフトウェア版の記録・複数回実行を勧める。⚠️ 自ら**シード感受性**を報告している（87% 対 13% が別の力学状態に落ちる）——これは本件の主張を**支持**する材料であり、反証ではない |
| Pineau et al. 2021（*JMLR* 22(164)、台帳 [27]） | ML 領域のプロトコル水準の実践：コード提出ポリシー、再現チャレンジ、投稿時の必須チェックリスト | ✓ | ✗：ML 全体が対象 | 半：報告要素は定めるが、局部規則訓練に固有の項目は含まない |

## 四、結論：何に当たらなかったか、そしてこの一文を今どう書くか

**三つの基準を同時に満たす項目：0 件。** 最も近いのは **NeuroSeqBench**（基準 2 ✓、基準 3 半）と
**NeuroBench**（基準 1 半、基準 3 半）で、いずれも「ベンチマークスイート＋自前のガイドライン」です。
**投稿で必須、あるいはコミュニティが採用した規範**は一つもなく、局部規則の訓練が報告すべき
予算・シード数・最適化器感受性を定めたものもありません。

したがって**この一文は格上げできますが、格上げ後の文は近隣を名指ししなければなりません**——
「0 件だった」とだけ書くことはできません：

> A search of the sources and query strings recorded in Appendix X on 2026-10-15 returned no
> protocol-level standard for local-rule SNN training. The nearest work is benchmark-oriented:
> suites that fix tasks and metrics for comparing SNN learning algorithms, including e-prop,
> and reproducibility guides aimed at spiking model *simulation*; none of them prescribes the
> budget, seed and optimizer reporting this paper controls.

⚠️ **格上げは未実施です。理由は二つあり、どちらも技術的な理由ではありません**：

1. あの文は近隣を名指しし、名指しすれば**引用**が必要になります（少なくとも NeuroBench、
   NeuroSeqBench、Pauli 2018 の三件）。`paper/refs.bib` 冒頭の規則は「**本ファイルはリポジトリに
   ない引用を新規に追加しない**」であり、まず [`docs/references.md`](../references.ja.md) の台帳に
   登録しメタデータを検証する必要があります——それは独自の裁定を要する一歩です。本ファイルに
   記録することは登録することでは**ありません**：**上の命中項目はいずれも台帳項目ではありません**。
2. 「a search of [情報源] on [日付] with [検索式]」という書き方は、検索仕様を論文側（付録の一節）に
   入れなければ意味を持ちません。そうでなければ査読者に照合する材料がありません。これは §2.4 と
   付録の構造的な変更であり、一文の差し替えではありません。

**この二つが済むまで §2.4 は防御的な表現のままにします。** 10-14 の記録の結論も変わりません。

## 五、限界、および再実行の方法

**網羅ではありません。** 本検索は本 agent が到達できる `WebSearch` 索引（**米国リージョン**）と
個別 URL の取得を用いました。ACM DL / IEEE Xplore / Scopus / Web of Science の庫内検索権限は
**なく**、Google Scholar の完全なページ送りもありません。したがって本ファイルが支えるのは
「**下記の検索式で、この日に、この索引では、見つからなかった**」であり、**「この分野に存在しない」
ではありません**。「網羅的」「システマティックレビュー」といった表現は一切成立しません。

**再実行の方法**：第二節の各行の検索式と URL をそのまま写し、任意の検索エンジンやデータベースで
再実行し、第一節の採用・除外基準を当ててください。基準を書き残したのは、再実行者が**私とは違う**
結論に到達できるようにするためです。

**既知の積み残し**：F1（TechRxiv の「Practical Guidelines for Spiking Neural Networks」）は 403 で
本文未読です。**題名が最も規範らしく見える**一件なので、**次回はこれを優先して確認すべき**です——
もし報告要素を定めているなら、第三節の結論は変わります。
