"""P3：用 Phase 1 当教师，生成 SIR→文本配对（合成数据，不碰用户正文）。

用法::

    python research/sel/gen_data.py --gguf D:\\AxoData\\models\\qwen2.5-0.5b-instruct-q4_k_m.gguf \
        --count 400 --seed 7 --out D:\\AxoData\\sel\\pairs\\v1

判据与数据口径见同目录 ``prereg_p3.md``（先落盘，再生成）。
输出**拒绝写进仓库根目录**：配对是教师产物，不属于源码。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
# Phase 1 的教师实现在 Axo 仓（本仓放研究脚本）；路径可用 AXO_SRC 覆盖。
_AXO_SRC = Path(os.environ.get("AXO_SRC", r"D:/Projects/BioSNN-MemoryTerminal/src"))
if _AXO_SRC.is_dir():
    sys.path.insert(0, str(_AXO_SRC))

from axo.expression.layer import ExpressionLayer  # noqa: E402
from axo.expression.sir import SIR, facts_preserved  # noqa: E402
from axo.llm.base import ChatBackend  # noqa: E402

PERSONA = "你是千岁（阿螈）。说话简短、直接，不用感叹号，不撒娇，不分点。"

#: 虚构话题：只用来组合出要点，不含任何用户的真实内容。
TOPICS: tuple[tuple[str, str], ...] = (
    ("Paper C 的基线实验", "准确率是 0.898"),
    ("本地语音组件", "已经能离线合成"),
    ("记忆库的巩固实验", "分簇回放提升了辨认率"),
    ("召回门槛的标定", "陌生话题不再被注入"),
    (
        "调度器的重试逻辑",
        "失败后会退避",
    ),
    ("日志里的连接错误", "是代理造成的"),
    ("模型的量化方案", "显存占用降到一半"),
    ("测试集的分层抽样", "三个种子结果接近"),
    ("文件工具的沙箱", "只允许读写交换目录"),
    ("提醒的触发时间", "默认 30 秒超时"),
    ("天气接口", "只需要城市名"),
    ("插件加载器", "会校验入口点"),
    ("对话区的渲染", "历史消息逐条兜底"),
    ("会话的预算", "每条消息按 5 token 计"),
    ("嵌入模型的维度", "是 512 维"),
    ("聚类阈值", "从 0.7 起测试"),
    ("版本号", "已经升到 0.6.0"),
    ("发布包的大小", "大约 215 MB"),
    ("下载地址", "在 releases 页面"),
    ("备份策略", "每天保留五份"),
    ("配置文件的格式", "是 JSON"),
    ("权限模型", "默认只读"),
    ("命令执行", "需要逐次确认"),
    ("文档的三语检查", "只检查结构"),
    ("提交前检查", "会跑格式与门槛"),
    ("回滚流程", "保留最近五个检查点"),
    ("温度参数", "固定为 0.7"),
    ("推理后端", "可以换成本机模型"),
    ("同步通道", "只同步记忆不同步人格"),
    ("可读目录", "一条都没有时工具不出现"),
)

_URL = "https://example.com/run/42"
_NUMBER = "0.898"


def build_sirs(count: int, seed: int) -> list[tuple[str, SIR]]:
    """生成 ``(sir_id, SIR)``。**结构优先**：5 意图 × 3 语气 × 2 长度循环取。"""
    # 修正之一（prereg_p3_amend1）：拒绝族 10%，其余四族各 22.5%。
    intents = ("answer",) * 9 + ("ask",) * 9 + ("emotion",) * 9 + ("action",) * 9 + ("refuse",) * 4
    tones = ("casual", "formal", "direct")
    lengths = ("short", "medium")
    out: list[tuple[str, SIR]] = []
    index = 0
    while len(out) < count:
        topic, detail = TOPICS[(index + seed) % len(TOPICS)]
        intent = intents[index % len(intents)]
        tone = tones[(index // len(intents)) % len(tones)]
        length = lengths[(index // (len(intents) * len(tones))) % len(lengths)]
        with_facts = index % 4 == 3
        fact = (_NUMBER if index % 8 == 3 else _URL) if with_facts else ""
        ordinal = f"第 {index + 1} 号"
        first = (
            f"{topic}（{ordinal}）：{detail}" if not with_facts else f"{topic}（{ordinal}）：{fact}"
        )
        points = (first, "下一步是再跑一轮验证") if length == "medium" else (first,)
        draft = "。".join(points) + "。"
        sir = SIR(
            intent=intent,
            key_points=points,
            tone=tone,
            length=length,
            facts=(fact,) if fact else (),
            draft=draft,
        )
        out.append((f"sir-{len(out) + 1:04d}", sir))
        index += 1
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _split(sir_id: str) -> str:
    bucket = int(hashlib.sha256(sir_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    return "valid" if bucket == 8 else "test"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--model", type=str, default="qwen2.5-0.5b")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    out_dir = args.out.resolve()
    if out_dir.is_relative_to(_ROOT):
        raise SystemExit(f"--out 不能写进仓库：{out_dir}。配对是教师产物，不属于源码。")
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = None
    if args.gguf is not None:
        from axo.llm.llama_server import BuiltinLlamaBackend, LlamaServerRuntime

        runtime = LlamaServerRuntime(
            model_path=args.gguf,
            context=4096,
            gpu_layers=0,
            log_path=out_dir / "llama.log",
        )
        base_url = runtime.start()
        backend: ChatBackend = BuiltinLlamaBackend(
            model=args.model, base_url=base_url, api_key="local", context_window=4096
        )
    else:
        if not args.base_url:
            raise SystemExit("要么给 --gguf，要么给 --base-url。")
        from axo.llm.openai_compat import OpenAICompatChatBackend

        backend = OpenAICompatChatBackend(
            model=args.model, base_url=args.base_url, api_key="local", context_window=4096
        )

    layer = ExpressionLayer(backend, PERSONA, max_tokens=160, temperature=0.7, think=None)
    pairs = []
    styled = 0
    try:
        for sir_id, sir in build_sirs(args.count, args.seed):
            outcome = layer.render(sir)
            if outcome.source == "styled":
                styled += 1
                if facts_preserved(outcome.text, sir.facts):
                    continue
                pairs.append(
                    {
                        "id": sir_id,
                        "split": _split(sir_id),
                        "sir": asdict(sir),
                        "text": outcome.text,
                        "route": outcome.route,
                        "teacher": args.model,
                    }
                )
    finally:
        if runtime is not None:
            runtime.stop()

    # 去重（按正文归一化），保留第一次出现
    seen: set[str] = set()
    unique = []
    for pair in pairs:
        key = "".join(pair["text"].split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)

    # 修正之一：拒绝族的合法重复不参与 D3；最多占最终配对的 5%。
    non_refuse = [pair for pair in unique if pair["sir"]["intent"] != "refuse"]
    refusals = [pair for pair in unique if pair["sir"]["intent"] == "refuse"]
    cap = int(len(non_refuse) * 0.05 / 0.95)
    unique = non_refuse + refusals[:cap]

    path = out_dir / "pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(pair, ensure_ascii=False) for pair in unique) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    counts = {
        split: sum(1 for p in unique if p["split"] == split) for split in ("train", "valid", "test")
    }
    non_refuse_unique = [p for p in unique if p["sir"]["intent"] != "refuse"]
    non_refuse_styled = sum(1 for p in pairs if p["sir"]["intent"] != "refuse")
    manifest = {
        "pairs": len(unique),
        "requested": args.count,
        "styled_ratio": round(styled / max(1, args.count), 3),
        "unique_ratio": round(len(unique) / max(1, len(pairs)), 3),
        "unique_ratio_non_refusal": round(len(non_refuse_unique) / max(1, non_refuse_styled), 3),
        "refusal_share": round(
            sum(1 for p in unique if p["sir"]["intent"] == "refuse") / max(1, len(unique)), 3
        ),
        "max_duplicate_share": round(
            max((sum(1 for q in unique if q["text"] == p["text"]) for p in unique), default=0)
            / max(1, len(unique)),
            4,
        ),
        "splits": counts,
        "seed": args.seed,
        "teacher": args.model,
        "sha256": _sha256(path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"pairs: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
