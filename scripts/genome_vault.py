"""基因库：**参照基因组 #0001 的自验证** + 恢复后的验收电池。

设计见 `D:/Projects/Qiansui-Internal/GOAL-silicon-life.md` §40。

## 两条硬规矩（都来自实测）

1. **自验证**：参照基因组必须能从种子**重新物化**并与存档校验和一致
   —— 生物学做不到（碱基序列无需证明自己是正确展开），这是数字个体独有的。
2. **"备份成功"不是"文件写完了"**：是**恢复后能过六道闸**（§40.6）。
   本模块实现其中的**便宜闸**；贵的（解码 / 顺序 / 遗忘曲线）由调用方传入实测值。

## 为什么按「不可克隆」写

第三轴一旦为真，逐位恢复在原理上不可能。
**若协议假设逐位，它会在第三轴变真时静默失效。**
所以验收一律取更弱的那一档：`indistinguishable`。
"""

from __future__ import annotations

import hashlib
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

SPEC_VERSION = "g3-2026-11-17"

#: 冻结的全部常数（改任何一个 = 新物种）
CONSTANTS = {
    "N": 1024,
    "L": 32,
    "DEG": 8,
    "RAD": 8,
    "V": 64,
    "K_pools": 4,
    "beta": 0.0,
    "p_long": 0.5,
    "i_pace": 0.5,
    "p_pace": 0.02,
    "g_field": 1.0,
    "g_local": 2.0,
    "field_radius": 2,
    "amp_gate": 2.0,
    "leak": 0.95,
    "theta": 0.5,
    "refractory": 2.0,
    "struct_q": 0.2,
    "a_plus": 1.0,
    "a_minus": 0.6,
    "stdp_eta": 0.05,
    "f_target": 0.08,
}

#: 实测验收门槛（全部取自本项目的实验，不是拍的）
GATES = {
    "decode_linear": 0.533,  # M6t，64 维线性
    "order_acc": 0.698,  # M6ad2，时间顺序
    "firing_lo": 0.01,
    "firing_hi": 0.20,
    "lesion_frac": 0.5,
    "lesion_firing_min": 0.01,
    "tau_forget": 150.0,
    "tau_tol": 20.0,
    "plateau": 0.445,
    "plateau_tol": 0.05,
}


def _digest(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def materialize(seeds: dict[str, int], dev: str = "cpu") -> dict[str, torch.Tensor]:
    """从种子物化全部静态参数。**这是"展开"**。"""
    from scripts.m6_pools import build_pools, pool_id

    g = seeds
    w_in, pre, wrec, _pid = build_pools(
        g["net"], dev, CONSTANTS["K_pools"], CONSTANTS["beta"], p_long=CONSTANTS["p_long"]
    )
    gen = torch.Generator().manual_seed(g["net"] + 31337)
    pace = (torch.rand(CONSTANTS["N"], generator=gen) < CONSTANTS["p_pace"]).float().to(dev)
    mot = (
        torch.randn(8, CONSTANTS["N"], generator=torch.Generator().manual_seed(g["net"] + 4242))
        / CONSTANTS["N"] ** 0.5
    ).to(dev)
    return {
        "w_in": w_in,
        "pre": pre,
        "wrec": wrec,
        "pid": pool_id(dev, CONSTANTS["K_pools"]),
        "pace": pace,
        "wmot": mot,
    }


def freeze(seeds: dict[str, int], dev: str = "cpu") -> dict:
    """冻结一份基因组：种子 + 常数 + 物化后的校验和链。"""
    tensors = materialize(seeds, dev)
    return {
        "spec_version": SPEC_VERSION,
        "seeds": dict(seeds),
        "constants": dict(CONSTANTS),
        "checksum": _digest(tensors),
        "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
    }


def verify(genome: dict, dev: str = "cpu") -> tuple[bool, str]:
    """**种子-展开一致性自验**：从种子重新物化，与存档校验和比对。"""
    if genome.get("spec_version") != SPEC_VERSION:
        return False, f"spec 版本不符：{genome.get('spec_version')} != {SPEC_VERSION}"
    if genome.get("constants") != CONSTANTS:
        return False, "常数不符（这已经是另一个物种）"
    again = _digest(materialize(genome["seeds"], dev))
    if again != genome["checksum"]:
        return False, f"校验和不符：{again[:16]}… != {genome['checksum'][:16]}…"
    return True, "ok"


def restore_check(
    firing: float,
    decode_linear: float,
    order_acc: float,
    lesion_firing: float,
    tau_forget: float,
    plateau: float,
) -> tuple[bool, list[str]]:
    """**恢复后的验收电池**（§40.6）。返回 `(是否全过, 逐闸报告)`。"""
    g = GATES
    fails: list[str] = []
    if not (g["firing_lo"] <= firing <= g["firing_hi"]):
        fails.append(f"发放 {firing:.4f} 不在窗 [{g['firing_lo']}, {g['firing_hi']}]")
    if decode_linear < g["decode_linear"]:
        fails.append(f"解码 {decode_linear:.3f} < {g['decode_linear']}")
    if order_acc < g["order_acc"]:
        fails.append(f"顺序 {order_acc:.3f} < {g['order_acc']}")
    if lesion_firing < g["lesion_firing_min"]:
        fails.append(f"50% 损伤后发放 {lesion_firing:.4f} < {g['lesion_firing_min']}")
    if abs(tau_forget - g["tau_forget"]) > g["tau_tol"]:
        fails.append(f"τ {tau_forget:.0f} 偏离 {g['tau_forget']:.0f}±{g['tau_tol']:.0f}")
    if abs(plateau - g["plateau"]) > g["plateau_tol"]:
        fails.append(f"平台 {plateau:.3f} 偏离 {g['plateau']:.3f}±{g['plateau_tol']}")
    return (not fails), fails


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default='{"net": 0}')
    ap.add_argument("--freeze-to", type=_pathlib.Path, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = json.loads(args.seeds)

    genome = freeze(seeds, dev)
    ok, why = verify(genome, dev)
    print("参照基因组 #0001")
    print(f"  spec      {genome['spec_version']}")
    print(f"  seeds     {genome['seeds']}")
    print(f"  checksum  {genome['checksum'][:32]}…")
    print(f"  自验证    {'✓ ' + why if ok else '✗ ' + why}")
    if args.freeze_to:
        args.freeze_to.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_to.write_text(
            json.dumps(genome, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )
        print(f"  已冻结到  {args.freeze_to}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
