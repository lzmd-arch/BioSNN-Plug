"""P4.0：教师到底能不能给出 logprobs（决定 KL 还是序列级 CE）。

两条路各打一次：

* **llama.cpp**（`--gguf`，本仓 1.5B 教师）：OpenAI 兼容 `/v1/chat/completions`，
  `logprobs: true, top_logprobs: 5`；
* **Ollama**（`--ollama-model`，可选对照）：原生 `/api/chat` 与 OpenAI 兼容各一次。

判据（跑之前写死）：

| 观测 | 结论 |
| :--- | :--- |
| 全词表 logits | 完整 KL（本路径几乎不可能） |
| top-k 且首个 token 的概率覆盖 ≥0.9 | 截断 KL |
| 有 logprobs 但覆盖 <0.9 / 只有训练 token 本身 | 序列级 CE（不许叫"自蒸馏"） |
| 没有 logprobs | 序列级 CE |

用法::

    python research/sel/probe_logprobs.py --gguf D:\\AxoData\\models\\qwen2.5-1.5b-instruct-q4_k_m.gguf
    python research/sel/probe_logprobs.py --ollama-model qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
_AXO_SRC = Path(os.environ.get("AXO_SRC", r"D:/Projects/BioSNN-MemoryTerminal/src"))
if _AXO_SRC.is_dir():
    sys.path.insert(0, str(_AXO_SRC))

PROMPT = [{"role": "user", "content": "用一句话说明今天适合做什么。"}]


def _probe_openai(base_url: str, model: str, *, api_key: str = "local") -> dict:
    body = {
        "model": model,
        "messages": PROMPT,
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers
        )
    report: dict = {"url": base_url, "status": response.status_code}
    try:
        data = response.json()
    except ValueError:
        report["error"] = response.text[:200]
        return report
    report["error_field"] = data.get("error") if isinstance(data, dict) else None
    try:
        content = data["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        report["logprobs"] = None
        return report
    report["logprobs"] = "content"
    first = content[0] if content else {}
    alternatives = first.get("top_logprobs") or []
    report["top_k"] = len(alternatives)
    report["token"] = first.get("token")
    report["prob"] = math.exp(first["logprob"]) if first.get("logprob") is not None else None
    report["coverage"] = round(
        sum(math.exp(item["logprob"]) for item in alternatives if item.get("logprob") is not None),
        4,
    )
    report["sample"] = [item.get("token") for item in alternatives]
    return report


def _probe_ollama(base_url: str, model: str) -> dict:
    body = {
        "model": model,
        "messages": PROMPT,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{base_url.rstrip('/')}/api/chat", json=body)
    report: dict = {"url": base_url + "/api/chat", "status": response.status_code}
    try:
        data = response.json()
    except ValueError:
        report["error"] = response.text[:200]
        return report
    logprobs = data.get("logprobs")
    report["logprobs"] = "present" if logprobs else None
    if logprobs:
        content = logprobs[0] if isinstance(logprobs, list) else logprobs
        report["sample"] = str(content)[:200]
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--llama-model", type=str, default="qwen2.5-1.5b")
    parser.add_argument("--ollama-model", type=str, default="")
    parser.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)

    reports = []
    runtime = None
    if args.gguf is not None:
        from axo.llm.llama_server import LlamaServerRuntime

        runtime = LlamaServerRuntime(
            model_path=args.gguf,
            context=2048,
            gpu_layers=0,
            log_path=Path(os.environ.get("TEMP", ".")) / "sel_probe_llama.log",
        )
        base_url = runtime.start()
        reports.append({"teacher": "llama.cpp", **_probe_openai(base_url, args.llama_model)})
        runtime.stop()
    if args.ollama_model:
        reports.append(
            {"teacher": "ollama/native", **_probe_ollama(args.ollama_url, args.ollama_model)}
        )
        reports.append(
            {
                "teacher": "ollama/openai",
                **_probe_openai(args.ollama_url + "/v1", args.ollama_model, api_key="ollama"),
            }
        )

    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    with_logprobs = [r for r in reports if r.get("logprobs")]
    if not with_logprobs:
        print("结论：没有 logprobs → 序列级 CE（如实写“监督微调”，不叫自蒸馏）")
        return 0
    for report in with_logprobs:
        coverage = report.get("coverage")
        if coverage is not None and coverage >= 0.9:
            print(f"结论（{report['teacher']}）：top-k 覆盖 {coverage} ≥0.9 → 截断 KL")
        elif coverage is not None:
            print(f"结论（{report['teacher']}）：top-k 覆盖 {coverage} <0.9 → 序列级 CE")
        else:
            print(f"结论（{report['teacher']}）：有 logprobs 但形状不完整 → 序列级 CE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
