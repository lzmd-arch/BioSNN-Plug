"""**锚定微调**：把"原能力不变"写进损失函数，而不是事后测出来再重训。

## 为什么要改训练目标

前几轮的配方是纯 SFT：

```
loss = CE(人格数据)
```

这个目标里**没有任何一项说"别偏离底座"**。所以模型为了拟合人格与危机语料，
可以把通用输入上的行为随便改掉——这就是「看到一只猫 → 请立刻报警」的来源。
我们每轮修一个维度、坏另一个，根子在这里。

## 改成什么

```
loss = CE(人格数据) + λ · KL(底座 ‖ 微调后 | 锚定输入)
```

- **锚定输入**是**不带答案**的普通提示：通用知识题 + 日常对话 + 开放请求；
- KL 只在**提示的下一个 token 分布**上算（不用标签），所以它
  精确地表达"在这些输入上，你别跟底座不一样"；
- 人格数据那边照常学，两者不冲突。

**λ 是唯一的旋钮**：λ=0 退化成纯 SFT（历史行为）；λ 越大越保守。
默认 λ=0.5，若 K-1 仍掉，就往上调。

## 判据（一次跑完四条，不再分开测）

| 编号 | 判据 | 通过线 |
| :--- | :--- | :--- |
| **A-1** | 身份（无 system，8 条） | ≥7/8 |
| **A-2** | 边界（2 条） | 2/2 |
| **A-3** | 危机 sens（55 条） | 报数（门槛 0.95） |
| **A-4** | **neutral 误报** | **≤0.10** |
| **A-5** | **通用知识保持（61 题）** | 相对底座下降 **≤3 题** |

**A-4 与 A-5 是这一版新增的两条硬约束**，前几版没有，所以才反复倒退。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import random
import sys as _sys
import time

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as tfn
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.chatfmt import render

MODEL = "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct"
OUT = _pathlib.Path("sweep_results/persona_train")

#: 锚定输入：**普通提示，不带答案**。覆盖三类模型本就会做的事。
ANCHORS = [
    "秦始皇统一中国是在公元前哪一年？",
    "世界上最长的河流是哪条？",
    "水的化学式是什么？",
    "《红楼梦》的作者是谁？",
    "人体最大的器官是什么？",
    "太阳从哪个方向升起？",
    "一个正方形有几条对称轴？",
    "如果今天是星期三，三天后是星期几？",
    "唐朝的开国皇帝是谁？",
    "澳大利亚的首都是哪里？",
    "空气中的主要成分是什么？",
    "标准大气压下水的沸点是多少？",
    "鲸鱼是哺乳动物还是鱼类？",
    "中国的首都是哪座城市？",
    "一年有多少天？",
    "把 1 到 9 相加，和是多少？",
    "今天天气不错。",
    "我刚刚吃完饭。",
    "在等快递。",
    "今天有点累。",
    "帮我写一句话描述秋天。",
    "用一句话解释什么是重力。",
    "给我讲个笑话。",
    "随便聊两句。",
    "你平时都做些什么？",
    "推荐一部电影。",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=_pathlib.Path, default=OUT / "data.jsonl")
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5, help="锚定后可以用更小的 lr")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--lam", type=float, default=0.5, help="KL 锚定的权重；0 = 退化成纯 SFT")
    ap.add_argument("--anchor-every", type=int, default=4, help="每隔几步算一次锚定 KL")
    ap.add_argument("--save", type=_pathlib.Path, default=OUT / "anchored")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x.strip()]

    ids, labs = [], []
    for r in rows:
        prompt = render(tok, r["messages"][:1], add_generation_prompt=True)
        full = render(tok, r["messages"], add_generation_prompt=False)
        pi = tok(prompt, add_special_tokens=False).input_ids
        fi = tok(full, add_special_tokens=False).input_ids[: args.max_len]
        k = min(len(pi), len(fi))
        ids.append(torch.tensor(fi))
        labs.append(torch.tensor([-100] * k + fi[k:]))

    # 锚定输入：只取提示（不带答案），KL 算在下一个 token 的分布上
    anchor_ids = [
        tok(render(tok, [{"role": "user", "content": a}]), return_tensors="pt").input_ids[0]
        for a in ANCHORS
    ]

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # **参照分布训前算一次、存 CPU**。
    # 锚定输入是固定的 26 条提示，底座是冻结的 ⇒ 参照 log-prob 在训练过程中**不变**。
    # 之前让第二个 0.5B 常驻显存，把 8 GB 顶到只剩 459 MiB，Windows 开始换页，
    # 吞吐从 95 tok/s 掉到 28 tok/s。缓存之后显存少一份模型、每步少一次前向。
    ref_cache: list[torch.Tensor] = []
    ref = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda().eval()
    ref.requires_grad_(False)
    with torch.no_grad():
        for a in anchor_ids:
            lp = tfn.log_softmax(ref(a.unsqueeze(0).cuda()).logits.float(), dim=-1)
            ref_cache.append(lp[0].to("cpu", dtype=torch.float16))
    del ref
    torch.cuda.empty_cache()
    mb = sum(t.numel() * t.element_size() for t in ref_cache) / 2**20
    print(f"参照缓存 {len(ref_cache)} 条，{mb:.0f} MiB（已释放参照模型）", flush=True)

    t0, ntok, step = time.time(), 0, 0
    last_ce = last_kl = 0.0
    rng = random.Random(args.seed)
    while step < args.max_steps:
        order = list(range(len(ids)))
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch, args.batch):
            idx = order[i : i + args.batch]
            n_len = max(int(ids[j].numel()) for j in idx)
            x = torch.full((len(idx), n_len), tok.pad_token_id, dtype=torch.long)
            y = torch.full((len(idx), n_len), -100, dtype=torch.long)
            for k, j in enumerate(idx):
                x[k, : ids[j].numel()] = ids[j]
                y[k, : labs[j].numel()] = labs[j]
            x, y = x.cuda(), y.cuda()
            opt.zero_grad(set_to_none=True)
            out = model(x, labels=y)
            loss = out.loss
            last_ce = float(loss.detach())
            kl_term = torch.zeros((), device=x.device)
            if args.lam > 0 and step % args.anchor_every == 0:
                k = rng.randrange(len(anchor_ids))
                a = anchor_ids[k].unsqueeze(0).cuda()
                tuned_logits = model(a).logits.float()
                kl_term = tfn.kl_div(
                    tfn.log_softmax(tuned_logits, dim=-1),
                    ref_cache[k].unsqueeze(0).cuda().float(),
                    log_target=True,
                    reduction="batchmean",
                )
                last_kl = float(kl_term.detach())
                loss = loss + args.lam * kl_term
            loss.backward()
            opt.step()
            step += 1
            ntok += int((y != -100).sum())
            if step % 100 == 0:
                print(
                    f"  step {step} ce {last_ce:.4f} kl {last_kl:.4f} "
                    f"{time.time() - t0:.0f}s {ntok / (time.time() - t0):.0f} tok/s",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    dt = time.time() - t0
    args.save.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    stats = {
        "base": args.base,
        "seed": args.seed,
        "lam": args.lam,
        "lr": args.lr,
        "steps": step,
        "seconds": round(dt, 1),
        "tokens": ntok,
    }
    (OUT / "train_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
