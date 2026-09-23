# ADR-0010：e-prop の二次記憶量は成立するが、§3.1 の是正措置（Trace Propagation の採用）は採用しない

[中文](ADR-0010-eprop-quadratic-storage-and-trace-propagation.md) · [English](ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md)

- **状態**：採用済み
- **日付**：2026-09-23
- **関連**：計画書 §3.1 キー修正 1、§三、§八 選定表、§十一 リスク表；`research/eprop/`；
  [ADR-0007](ADR-0007-td-ltp-critic-provenance.ja.md)（同じく引用の考証）；
  [docs/references.md](../references.ja.md) 項目 2

## Context

計画書 §3.1「キー修正 1」（`BioSNN-Plug_项目计划书_v6.3.md:161-163`）は二つのことを述べている：

> e-prop の資格痕跡はシナプスごとに保存され、空間計算量はニューロン数に対して**二次的に増大する**
> [Pes et al., 2025]。Traces Propagation（TP）は前向き・メモリ効率が高く・スケーラブルな完全局所
> 学習規則であり、資格痕跡と層ごとの対比損失を組み合わせ、**補助的な層別行列を必要としない**……

そして §八 の選定表はそれを直接の置き換えとして書いている：「e-prop 記憶最適化 | Trace Propagation |
**資格痕跡の記憶を O(N²) から O(N) へ**」——同じ主張は §三 の 104/159 行、アーキテクチャ図（86 行）、
363 行、604 行、リスク表 496/516 行にも現れる。

`docs/references.md` の項目 2 は以前に三つの考証を記録しており、その第 3 条はこう述べていた：
**「原文はその計算量を OSTTP/OSTL に帰属させており、e-prop の空間計算量を個別には与えていない
——計画書がそれを e-prop に帰属させるのは引申（拡大解釈）である。」**

本 ADR は二点を決着させる：**その考証自体が正しいか**、そして **§3.1 を実装すべきか**。

## Decision

### その一：第 3 条の考証は誤りであり、撤回する

arXiv:2509.13053v2 の LaTeXML HTML から Table 3 を**セル単位で**解析した（PDF のテキスト層は表が
ずれており使えない）。v1 と PDF で交差検証済み：

| Model | Update Locking | Weight Transport | Time Local | Space Local | **Space** | Time | Aux |
| :--- | :-: | :-: | :-: | :-: | :--- | :--- | :--- |
| BPTT | ✗ | ✗ | ✗ | ✗ | TLH | TLH² | − |
| **E-prop [2]** | ✗ | ✗ | ✓ | ✗ | **LH²** | **LH²** | **−** |
| E-prop(rnd) [2] | ✗ | ✓ | ✓ | ✗ | **LH²** | LOH | LOH |
| OSTL [8] | ✗ | ✗ | ✓ | ✗ | LH² | LH² | − |
| OSTTP [11] | ✓ | ✓ | ✓ | ✓ | LH² | LOH | LOH |
| TESS [13] | ✓ | ✓ | ✓ | ✓ | LH | LOH | LOH |
| TP (ours) | ✓ | ✓ | ✓ | ✓ | LH | LH | OH |

さらに §1.3.1 は **e-prop を名指ししている**：まず一般式
`ϵ_l^t[i,j] = βϵ_l^{t-1}[i,j] + g(s) f(s)`（Eq. 9、添字はシナプス対 `[i,j]`）を与え、次に

> For instance, the eligibility trace of **E-prop [2]** defines the presynaptic factor as a
> low-pass filtered version of the spiking activity … and the postsynaptic factor as the surrogate
> derivative of the spike function …

と書き、続けて族の性質を述べる：「eligibility traces are stored per synapses, leading to a space
complexity that scales quadratically with the number of neurons, i.e., 𝒪(H²L)」。
**e-prop はその段落で最初に名指しされた例であり、`these solutions` は字面上それを覆う。**

**したがって計画書 §3.1 の前件は成立し、原文の直接の記述である**（「ニューロン数に対して二次、
層数に対して線形」は `LH²` と逐字一致する）。リポジトリの以前の考証は**散文については正しく、
表については誤り**だった——v2 の §2.3 は OSTTP しか名指ししておらず（v1 の文はより明示的で、
ETLP と OSTTP が「based on **E-prop [2]** and OSTL [8]」だと述べている）、散文だけを読むと誤る。
撤回の仕方は Consequences に記す。

### その二：「TP を採用する」という是正措置は**採用しない**——三つの固い根拠

1. **TP は e-prop の省メモリ版ではなく、空間的信用割り当てを置き換える別の規則である。** 論文は
   両者を**別々の行・別々の軸**で並べている（e-prop：時間局所 ✓ / 空間局所 ✗ / LH² / 補助行列なし；
   TP：すべて ✓ / LH / LH / OH）し、TP が e-prop の最適化だとは一言も述べていない。TP は
   **ニューロンごとの**活動痕跡を二本（Eq. 11/12、添字は `(batch, neuron)`。漸化式に ψ_j も
   ε_v/ε_a も e_ij も現れない）と one-hot の目標経路、バッチ×バッチの対比損失（Eq. 13-15）、
   そして更新式（Eq. 18）で**現時刻**に評価する代理勾配を使う。**採用することは W2 の学習器を
   取り替えることであり、そのメモリを最適化することではない。**
2. **W2 のニューロンモデルを覆っていない。** Table 1/2 の TP の結果は**すべて LIF**（SHD の 400/450
   の 2 行も LIF。ALIF の行は ETLP に属する）。**W2 の受け入れ設定は ALIF（β=0.07）。**
3. **論文自身の看板データセットで e-prop のほうが良い。** N-MNIST：`eProp [2] … 97.90` 対
   `TP (ours) … 97.33 ± 0.06`。しかも要旨の「outperforms other fully local learning rules」に
   **e-prop は含まれない**——e-prop は Table 3 で `Partial (time)` 局所とされており、「完全局所」の
   比較集合の外にある。**あの一文を「本プロジェクトが選んだ規則は TP より劣ると証明された」と
   読むのは、向きが逆である。**

（付言、**理由には使わない**：TP は対比損失のためバッチ ≥ 2 を要し、サンプル単位のオンライン更新が
できない。また論文自身の Eq. 25 は TESS に対するメモリ優位が `O > B` のときにのみ成り立つことを
示しており、sMNIST は O=10, B=64 である。しかしこの論法は第三者を標的にしており、しかもこの動作点で
TP のニューロンごとの痕跡は約 0.125 MiB にすぎない。**本当の理由は上の三つである。**）

## Consequences

- **`docs/references.md` 項目 2 の第 3 条を書き換える**（三言語）：「引申」から「前件は成立する。
  原文は Table 3 の `E-prop [2]` 行と §1.3.1 での名指しの両方で与えている。散文（v2 §2.3）は
  OSTTP しか名指ししておらず誤読しやすい——それが当時の誤りの原因だろう」へ。あわせて別の
  一处の表現も締める：原文は `MNIST` という語を一度も使っていないわけでは**ない**（3 回現れる：
  要旨の `NMNIST` と、§3.1.1 で N-MNIST の由来を説明する箇所）。正確には**MNIST を評価データ
  セットとして使っていない**である。
- **`research/eprop/README.md` の境界 1 を書き換える**（三言語）：「Trace Propagation は未実装」
  （負債のように読める）から「考証の結果、§3.1 の是正措置は**本項目に適用されない**。理由は三つ」
  へ。あわせて、**本実装が実際に行うべき唯一のことは実施済み**である旨を記す（下記）。
- **本実装が実際に行ったこと**：`traces.py` の `epsilon_v` は `(batch, n_pre, n_post)` で確保されて
  いたが、その漸化式に**後シナプス添字 j は現れない**——ある時刻で全ての j に対して同じ値である。
  `(batch, n_pre, 1)` にすると**ビット単位で同一**（`torch.equal=True`、`max|diff|=0`）で、
  受け入れ形状での実測壁時計が **−18.2%** になる。`epsilon_a` の係数には `(ρ − β·ψ_j)` が含まれ
  **因数分解できない**ため、この節のメモリの大半は ALIF の適応項が決める。LIF（β=0）では両方とも
  外積に退化し、痕跡状態全体が O(batch·N) まで下がる。この境界はコードのコメントに書いてある。
- **`research/rstdp/README.md` の境界**（三言語）も TP を「本フェーズ未実装」と挙げている——同時に
  直さないと、同一の事柄についてリポジトリの二本の線が別々のことを言うことになる。

## Alternatives

- **§3.1 のとおり TP を実装する**：**却下**。理由は Decision その二。もし実装するなら、それは
  **別の学習規則**として独立に立てるべきであり（e-prop の学習器を置き換えるもの）、「e-prop の
  メモリ最適化」ではない。また先に W2 のニューロンモデルを ALIF から LIF へ移すか、TP の ALIF 版を
  自前で導出する必要がある——論文は与えていない。
- **散文と表の表現だけ直し、実装は触らない**：**却下**——本実装には実際に無駄な O(N²) バッファが
  一つあり（`epsilon_v`）、その修正は無リスク（ビット単位で同一、−18%）である。文書だけ直すのは
  ただの改善を放置することになる。
- **「半局所」へ退いて記憶量を買う**：**該当しない**——それは §十一 の縮退経路であり、この記憶量の
  問題とは無関係である。
