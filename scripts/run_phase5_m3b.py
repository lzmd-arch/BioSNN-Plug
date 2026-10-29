"""阶段五 M3b：真实 LLM 对话评测（判据见 sweep_results/phase5_m3b_prereg.md）。

两个臂跑同一批会话：

* **LLM 臂**：触发策略决定要不要调用 LLM；回答写入记忆，下一轮记忆问句取回；
* **纯 SNN 基线**：同一核心、不调用 LLM，只做记忆检索，检索不到就答 ``i don't know``。

两个臂都预置同一批已记住的知识（每 3 条知识里的第 1 条），否则基线结构性全错、提升比
恒为无穷大。⚠️ 即便如此，「任务完成率提升 ≥ 50%」这条判据在本合成集上仍然近乎平凡，
README 必须写明这一点。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import DialogueTrace, MockLLM, TriggerPolicy  # noqa: E402
from biosnn_mcp.dialogue import build_scenarios, run_session  # noqa: E402
from biosnn_mcp.llm_ollama import OllamaLLM  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_m3b"


def build_llm(kind: str, model: str, timeout: float = 120.0, *, think: bool = False):
    if kind == "mock":
        return MockLLM()
    if kind == "ollama":
        # 系统代理会把 loopback 送进代理，所以显式给一个空 ProxyHandler。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        return OllamaLLM(model=model, timeout=timeout, think=think, opener=opener)
    raise ValueError(f"未知 LLM 类型：{kind!r}")



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段五 M3b 真实 LLM 对话评测")
    parser.add_argument("--llm", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否让思考模型输出思考块（默认关闭；思考文字会污染关键短语判定）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个会话（0=全部，仅调试用）")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]
    policy = TriggerPolicy()
    llm = build_llm(args.llm, args.model, args.timeout, think=args.think)
    trace_path = args.out_dir / f"trace_{args.llm}.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    trace = DialogueTrace(trace_path)

    llm_rows = [
        run_session(
            scenario,
            llm=llm,
            policy=policy,
            use_llm=True,
            trace=trace,
            turn_offset=index * 100,
        )
        for index, scenario in enumerate(scenarios)
    ]
    baseline_rows = [
        run_session(
            scenario,
            llm=llm,
            policy=policy,
            use_llm=False,
            trace=trace,
            turn_offset=10000 + index * 100,
        )
        for index, scenario in enumerate(scenarios)
    ]
    first_calls = [row["first_call_turn"] for row in llm_rows if row["first_call_turn"] is not None]
    consolidated = [
        row["consolidation_rate"] for row in llm_rows if row["consolidation_rate"] is not None
    ]
    metrics = {
        "first_call_turn_max": max(first_calls) if first_calls else None,
        "trigger_accuracy": sum(row["trigger_accuracy"] for row in llm_rows) / len(llm_rows),
        "consolidation_rate": sum(consolidated) / len(consolidated) if consolidated else 0.0,
        "llm_completion_rate": sum(row["completion_rate"] for row in llm_rows) / len(llm_rows),
        "baseline_completion_rate": sum(row["completion_rate"] for row in baseline_rows)
        / len(baseline_rows),
    }
    metrics["completion_ratio"] = (
        metrics["llm_completion_rate"] / metrics["baseline_completion_rate"]
        if metrics["baseline_completion_rate"] > 0
        else None
    )
    baseline = metrics["baseline_completion_rate"]
    gate = {
        "first_call": metrics["first_call_turn_max"] is not None
        and metrics["first_call_turn_max"] <= 10,
        "trigger_accuracy": metrics["trigger_accuracy"] >= 0.70,
        "consolidation": metrics["consolidation_rate"] >= 0.80,
        # 基线为 0 时提升比无定义，按「未通过」记，不写成无穷大。
        "completion_uplift": baseline > 0 and metrics["llm_completion_rate"] >= 1.5 * baseline,
    }
    gate["m3b_ready"] = all(gate.values())
    summary = {
        "kind": f"real-LLM dialogue evaluation ({args.llm}, {args.model})",
        "metrics": metrics,
        "gate": gate,
        "sessions": {"llm": llm_rows, "baseline": baseline_rows},
    }
    (args.out_dir / f"summary_{args.llm}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
