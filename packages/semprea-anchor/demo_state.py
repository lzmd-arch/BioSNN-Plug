"""演示：**同一个问题、两种状态 → 两个答案**。

打包出去的 V1 在加状态之前是这样的（实测）：

    问：你现在用哪个模型做嘴巴？
    塞状态块 / 不塞状态块 → **逐字相同**："我叫千岁，你也可以叫我阿螈。"

这个脚本演示加状态链之后应该变成什么：`p` 从脉冲池算出来，经 FiLM 进每一层，
**不需要提示词**。

用法（包内，任意机器）：

    python demo_state.py --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parent


def _find(name: str) -> Path:
    for base in (PKG, *PKG.parents[:3]):
        cand = base / name
        if cand.exists():
            return cand
    return PKG / name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default=None)
    ap.add_argument("--modulator", default=None)
    ap.add_argument("--question", default="你现在用哪个模型做嘴巴？")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    from fusion_model import load_fusion

    model_dir = args.model or str(_find("model"))
    mod_path = args.modulator or str(_find("modulator.pt"))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    fusion = load_fusion(model_dir, state_dim=12, alpha=0.5, rank=32, device=dev)
    mod = torch.load(mod_path, map_location=dev)
    fusion.modulator.load_state_dict(mod)
    fusion.set_state(None)

    # `p` 的两种来源在实验里表现一致（film 0.50 手写 / 0.44 脉冲池；both 都是 1.00）。
    # 这里用**手写 12 维**，因为它不依赖研究层的脉冲模型；脉冲池版本见
    # `scripts/snn_state_p.py` 与其结果 `sweep_results/state_qa/README.md`。
    def state_p(st: dict) -> list[float]:
        import math

        backbones = ["Qwen3-4B", "DeepSeek", "Qwen2.5-0.5B", "本地小模型"]
        return [
            min(1.0, st["turns"] / 50.0),
            1.0 if st["online"] else 0.0,
            min(1.0, math.log1p(st["mem"]) / 12.0),
            *[1.0 if k == st["bb"] else 0.0 for k in range(len(backbones))],
            0.2,
            0.4,
            1.0,
            min(1.0, math.log1p(st["mem"]) / 12.0),
            0.5,
        ]

    states = [
        {"mem": 12, "online": True, "bb": 0, "turns": 3},
        {"mem": 420, "online": False, "bb": 1, "turns": 27},
    ]
    ps = [state_p(s) for s in states]
    print(f"问：{args.question}\n")
    for s, p in zip(states, ps, strict=False):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.question}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(ids, return_tensors="pt").input_ids.to(dev)
        fusion.set_state(torch.tensor([p], dtype=torch.float32, device=dev))
        try:
            with torch.no_grad():
                out = fusion.lm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        finally:
            fusion.set_state(None)
        ans = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()
        print(f"状态 {json.dumps(s, ensure_ascii=False)}")
        print(f"  → {ans}\n")

    # 对照：不给状态
    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.question}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(ids, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = fusion.lm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print("对照（不给状态）")
    print(f"  → {tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
