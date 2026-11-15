"""0.5B 局部规则脉冲模型当**状态编码器**：`p` 由它算出来。

## 它从哪来

`sweep_results/lm_semprea_v1/semprea_v1_ckpt.pt`，2026-11-14 训完：
top-1 0.1100 / top-5 0.2193，双平台收敛，4.92B token 曝光，单卡 8 GB 跑了 38.5 小时。
它原本是输出分布级融合的"快半"，四种算子都没赢温度对照之后在 **V0.5** 里退役。
**这里给它换岗位**：不生成文字，只把"当前状态"编码成 `p`。

## 0.5B 到底有多大——先把话说清

| 部件 | 参数 | 占比 |
| :--- | ---: | ---: |
| 词嵌入 `emb`（32000×256） | 8.19M | 1.6% |
| 输入投影 `proj`（256×7812） | 2.00M | 0.4% |
| **层次 softmax 输出树**（31999×15624） | **500.0M** | **97.9%** |
| 合计 | 510.7M | 100% |

⇒ **0.5B 里有 97.9% 是输出树**，承载状态的是 7,812 个循环神经元那 10.7M。
本模块只加载核心、**不建树**：41.7 MiB 而不是 2 GB。
说"0.5B 状态核心"时必须带上这句，否则读者会以为是 0.5B 个神经元。

## 为什么这里是**自己实现的前向**，不是 import 研究层

包要能在任何机器上跑，不能拖 `research/` 进来。训练时 `recurrence="none"`，
所以前向就是一个纯前馈 LIF：

```
current = emb[tok] @ proj
v       = alpha*v + current - z*theta
z       = (v - theta)/theta > 0      （不应期内强制 0）
readout = decay*readout + z
features= (readouts * scales).flatten()
```

`w_rec` / `pre_index` 在 `recurrence="none"` 下**不参与前向**（`_recurrent_current` 直接返回 0），
但为可核对起见一并保留在文件里。`--verify` 会逐元素对比自己实现与研究层的输出。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

P_DIM = 12


def _find(name: str) -> Path:
    """包内运行时从本文件所在目录向上找；仓库内运行时直接命中。"""
    here = Path(__file__).resolve().parent
    for base in (here, *here.parents[:3]):
        cand = base / name
        if cand.exists():
            return cand
    return here / name


def _default_core() -> Path:
    for cand in (
        Path("sweep_results/lm_semprea_v1/semprea_v1_core.pt"),
        _find("snn_core.pt"),
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError("找不到 snn_core.pt / semprea_v1_core.pt")


@torch.no_grad()
def forward_core(tensors: dict, cfg: dict, ids: torch.Tensor) -> torch.Tensor:
    """纯前馈 LIF 前向，返回 `(B, n_taus * n_rec)` 的读出特征。"""
    emb = tensors["emb"].float()
    proj = tensors["proj"].float()
    alpha = float(cfg["alpha"])
    theta = float(cfg["threshold"])
    n_ref = float(cfg["n_refractory"])
    decays = torch.tensor(cfg["readout_decays"], dtype=torch.float32)
    scales = torch.tensor(cfg["readout_scales"], dtype=torch.float32)
    nb = ids.shape[0]
    n_rec = int(cfg["n_rec"])

    v = torch.zeros((nb, n_rec), dtype=torch.float32)
    z = torch.zeros((nb, n_rec), dtype=torch.float32)
    ref = torch.zeros((nb, n_rec), dtype=torch.float32)
    readouts = torch.zeros((nb, decays.numel(), n_rec), dtype=torch.float32)

    for t in range(ids.shape[1]):
        current = emb[ids[:, t]] @ proj
        new_v = alpha * v + current - z * theta
        fresh = ((new_v - theta) / theta > 0).to(torch.float32)
        new_z = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + n_ref * new_z - 1.0, min=0.0, max=n_ref)
        v, z = new_v, new_z
        readouts = readouts * decays.view(1, -1, 1) + z.unsqueeze(1)
    scaled = readouts * scales.view(1, -1, 1)
    return scaled.reshape(nb, -1)


class SNNCoreState:
    """0.5B 脉冲核心当状态编码器。**一次构建、反复查询**（权重冻结，不学习）。"""

    def __init__(
        self,
        core: Path | None = None,
        id_map: Path | None = None,
        tok_dir: str | None = None,
        p_dim: int = P_DIM,
    ) -> None:
        from transformers import AutoTokenizer

        blob = torch.load(core or _default_core(), map_location="cpu", weights_only=False)
        self.tensors = {k: v.float() for k, v in blob["tensors"].items() if k in ("emb", "proj")}
        self.cfg = blob["config"]
        self.step_ckpt = blob.get("step")

        if tok_dir is None:
            # 包内是 ./model；仓库内退回某个已 save_pretrained 的检查点（都带分词器）。
            # **这里只用相对路径**——发布前自检会扫绝对路径。
            for cand in (
                _find("model"),
                Path("sweep_results/persona_train/seeds/s0"),
                Path("sweep_results/persona_train/ckpt_v4"),
            ):
                if cand.is_dir() and (cand / "tokenizer_config.json").is_file():
                    tok_dir = str(cand)
                    break
        if tok_dir is None:
            raise FileNotFoundError("找不到分词器目录（包内应为 ./model）")
        self.tok = AutoTokenizer.from_pretrained(tok_dir)

        if id_map is None:
            for cand in (_find("id_map.json"), Path("sweep_results/lm_semprea_v1/id_map.json")):
                if cand.is_file():
                    id_map = cand
                    break
        if id_map is None:
            raise FileNotFoundError("找不到 id_map.json（本地 id ↔ 千问 id）")
        mapping = json.loads(Path(id_map).read_text(encoding="utf-8"))["ids"]
        self.q2l = {int(q): i for i, q in enumerate(mapping)}

        g = np.random.default_rng(7)
        self.p_dim = p_dim
        self.proj = g.normal(0, 1.0 / math.sqrt(p_dim), (2 * int(self.cfg["n_rec"]), p_dim))

    def _ids(self, text: str, max_len: int) -> list[int]:
        q = self.tok(text, add_special_tokens=False).input_ids[:max_len]
        out = [self.q2l[i] for i in q if i in self.q2l]
        return out or [0]

    def encode(self, texts: list[str], max_len: int = 64) -> np.ndarray:
        lens = [self._ids(t, max_len) for t in texts]
        n_steps = max(len(x) for x in lens)
        ids = torch.zeros((len(texts), n_steps), dtype=torch.long)
        for i, x in enumerate(lens):
            ids[i, : len(x)] = torch.tensor(x)
        feats = forward_core(self.tensors, self.cfg, ids).numpy()
        p = feats @ self.proj
        return (p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-6)).astype(np.float32)


def verify() -> int:
    """逐元素对比自己实现与研究层。**不一致就说明打包出去的核心是错的。**"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research.sel.lm.model import TokenEProp

    enc = SNNCoreState()
    blob = torch.load(_default_core(), map_location="cpu", weights_only=False)
    m = TokenEProp(
        int(enc.cfg["vocab"]),
        int(enc.cfg["n_rec"]),
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=None,
        embed_rank=int(enc.cfg["embed_rank"]),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    for k, v in blob["tensors"].items():
        if hasattr(m, k):
            setattr(m, k, v.float())
    m.w_out = torch.zeros((1, m.n_pad), dtype=torch.float32)
    m._logits = lambda f: torch.zeros((f.shape[0], 1), dtype=f.dtype)

    ids = torch.randint(0, int(enc.cfg["vocab"]), (4, 16))
    tr = m.new_trace(4)
    tgt = torch.full_like(ids, -100)
    for t in range(ids.shape[1]):
        m.step(ids[:, t], tgt[:, t], tr, update=False, all_valid=True)
    ref = m._features(tr)
    got = forward_core(enc.tensors, enc.cfg, ids)
    diff = float((ref - got).abs().max())
    print(f"研究层 {tuple(ref.shape)}  本地实现 {tuple(got.shape)}  最大差 {diff:.3e}")
    print("判：", "✅ 逐元素一致" if diff < 1e-5 else "❌ 不一致")
    return 0 if diff < 1e-5 else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    enc = SNNCoreState()
    n = sum(v.numel() for v in enc.tensors.values())
    print(f"核心参数 {n / 1e6:.2f}M（只有 emb + proj；输出树已剥离）")
    a = enc.encode(["我现在用「Qwen3-4B」做嘴巴；记忆库里有 420 条；现在是离线的；对话 27 轮。"])
    b = enc.encode(["我现在用「DeepSeek」做嘴巴；记忆库里有 12 条；在线；对话 3 轮。"])
    print("p 形状", a.shape, "两条状态差异", float(np.abs(a - b).mean()))
    same = enc.encode(["同一句话。"] * 2)
    print("同输入重复，最大差", float(np.abs(same[0] - same[1]).max()))


# ---------------------------------------------------------------------------
# 完整 0.5B：状态 p 之外，再接上输出树
# ---------------------------------------------------------------------------


def _default_tree_w() -> Path:
    for cand in (
        Path("sweep_results/lm_semprea_v1/semprea_v1_tree_w.pt"),
        _find("snn_tree_w.pt"),
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError("找不到树权重 snn_tree_w.pt")


def _default_tree_paths() -> Path:
    for cand in (
        Path("sweep_results/lm_semprea_v1/semprea_v1_tree.pt"),
        _find("snn_tree.pt"),
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError("找不到树结构 snn_tree.pt")


class SNNFull(SNNCoreState):
    """**完整 0.5B**：核心给状态 `p`，输出树给下一 token 的分布。

    树权重 fp32 是 1.86 GiB；`weight_dtype="bf16"` 减半。V0.5 实测过 bf16 的口径：
    top-1 10/10 不变、最大 logprob 差 0.0071（见 Semprea-4.5B 的 manifest）。

    它**不是**聊天模型：top-1 只有 0.11。它能做的是"在 32k 子集上给出一个分布"，
    用来做状态编码、困惑度、或与别的模型的分布级融合（后者在 V0.5 四测四负）。
    """

    def __init__(
        self,
        tree_w: Path | None = None,
        tree_paths: Path | None = None,
        weight_dtype: str = "fp32",
        **kw,
    ) -> None:
        super().__init__(**kw)
        try:  # 仓库内
            from research.sel.lm.hierarchical import HuffmanPaths, path_probabilities
        except ImportError:  # 打包后是平铺目录
            from hierarchical import HuffmanPaths, path_probabilities

        self._path_probabilities = path_probabilities
        d = torch.load(
            _default_tree_paths() if tree_paths is None else tree_paths,
            map_location="cpu",
            weights_only=False,
        )
        self.paths = HuffmanPaths(
            nodes=d["nodes"],
            bits=d["bits"],
            length=d["length"],
            n_nodes=int(d["n_nodes"]),
            right_share=d["right_share"],
        )
        w = torch.load(
            _default_tree_w() if tree_w is None else tree_w,
            map_location="cpu",
            weights_only=False,
        )
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[weight_dtype]
        self.tree_w = w["__tree_weight"].to(dt)
        self.tree_b = w["__tree_bias"].float()

    def features(self, texts: list[str], max_len: int = 64) -> torch.Tensor:
        lens = [self._ids(t, max_len) for t in texts]
        n_steps = max(len(x) for x in lens)
        ids = torch.zeros((len(texts), n_steps), dtype=torch.long)
        for i, x in enumerate(lens):
            ids[i, : len(x)] = torch.tensor(x)
        return forward_core(self.tensors, self.cfg, ids)

    @torch.no_grad()
    def next_logprobs(
        self, texts: list[str], top_k: int = 20, max_len: int = 64
    ) -> list[list[tuple[int, float]]]:
        """每个输入返回 [(本地词表 id, logprob), ...]，按概率降序。"""
        feats = self.features(texts, max_len)
        probs = self._path_probabilities(feats, self.tree_w, self.paths, self.tree_b)
        vals, idx = probs.topk(min(top_k, probs.shape[1]), dim=-1)
        return [
            [(int(i), float(math.log(max(float(v), 1e-30)))) for v, i in zip(vr, ir, strict=False)]
            for vr, ir in zip(vals, idx, strict=False)
        ]
