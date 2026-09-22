#!/usr/bin/env python
"""依赖许可审计（计划书 §12.1）。

计划书点名要核查 eprop-PyTorch 与 Lava/Loihi 相关组件的许可证条款。本脚本做基线：
把全部依赖的许可证列出来，并在出现**强 copyleft** 时失败。

用法::

    uv run --with pip-licenses pip-licenses --format=json > /tmp/licenses.json
    uv run python scripts/check_licenses.py /tmp/licenses.json
    uv run python scripts/check_licenses.py /tmp/licenses.json --strict-unknown   # CI 用这个

## 「检测不到」与「显式声明」

按前缀比对只能认出**有名字**的许可证。一个依赖的 License 字段若是空的，它会落进
``unknown`` 桶——而 ``unknown`` 默认只打印不失败。于是「许可证不明」这件事可以一路
静默通过 CI。

这个洞真实存在过：SpikingJelly 用的是**启智开源许可证 1.0**（OIOSL），它既不以
``GPL`` 开头也不以 ``OSL`` 开头，且 wheel 元数据里 License 字段为空。计划书 §12.1
当时还以为它和本项目同为 Apache-2.0（见 ``docs/adr/ADR-0008``）。

``--strict-unknown`` 把这件事反过来：**未登记的 unknown 一律失败**。要放行就得进
``ACCEPTED_NON_STANDARD`` 白名单，并**写明理由**。白名单是漏洞，所以每条都必须可审计
——这是 §12.5 给 ``EXEMPT`` 表立的同一条规矩。白名单里登记了却没在依赖里出现的条目
也会被报出来，免得它悄悄腐烂成一句无人在意的旧话。

注意 ``--strict-unknown`` 只管 unknown，不管 caution：LGPL / MPL 这类弱 copyleft 在
Python 生态里太常见，把它们也判死会让审计变成噪声。caution 仍然是提醒。

## 为什么单独写成脚本，而不是在 CI 里内联 grep

因为它**必须能在本地跑**。第一版是 CI 里的一条
``grep -Ei '\\| *(A?GPL|SSPL|BUSL)' dependency-licenses.md``，它在首次 CI 运行时就
误报了：``pip-licenses --with-license-file`` 会把许可证**正文**整段灌进表格，而
PSF 许可证正文里有一句

    GPL-compatible licenses make it possible to combine Python with ...

按整行 grep 直接命中，于是 CI 红了一个根本没有强 copyleft 依赖的仓库。
纯文本 grep 分不清"许可证名称"与"许可证正文里提到 GPL"——这一条在本地跑一次就
能发现，而它在 CI 里烧掉了一整轮。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: 强 copyleft：出现即失败。这些许可证会传染到使用方，与 Apache-2.0 不兼容。
FORBIDDEN_PREFIXES = ("AGPL", "GPL", "SSPL", "BUSL", "CPAL", "OSL")

#: 弱 copyleft / 有条件许可证：只提醒，不拦。Python 生态里常见，且多数在"仅作为
#: 依赖使用"的场景下没有问题——但值得人看一眼。
CAUTION_PREFIXES = ("LGPL", "EUPL", "MPL", "CDDL", "EPL")

#: 许可证字段缺失或无法识别时的占位值。
UNKNOWN_MARKERS = ("", "UNKNOWN", "NONE")

#: 已知并接受的**非标准许可证**依赖：``包名 -> (许可证标识, 接受理由)``。
#:
#: 按**包名**而不是许可证名索引——因为需要写进这里的情形，恰恰是许可证字段为空、
#: 拿不到名字的那些。每条都必须写清理由与裁决记录的位置；没有理由的条目不该存在。
ACCEPTED_NON_STANDARD: dict[str, tuple[str, str]] = {
    "spikingjelly": (
        "Open-Intelligence Open Source License 1.0（启智开源许可证 1.0）",
        "非 Apache-2.0（计划书 §12.1 曾误记为同一许可证）。研究原型、非商业用途，"
        "商业使用须向 AITISA 披露的义务随分发传递。裁决见 docs/adr/ADR-0008。",
    ),
}


def classify(license_name: str) -> str:
    """返回 ``forbidden`` / ``caution`` / ``unknown`` / ``ok``。"""
    name = license_name.strip()
    if name.upper() in UNKNOWN_MARKERS:
        return "unknown"
    # 只看许可证名称的开头，不做子串匹配——正文里的"提到 GPL"不算。
    for prefix in FORBIDDEN_PREFIXES:
        if name.upper().startswith(prefix):
            return "forbidden"
    for prefix in CAUTION_PREFIXES:
        if name.upper().startswith(prefix):
            return "caution"
    return "ok"


def load_packages(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(
            f"{path} 的顶层结构应为 JSON 数组（pip-licenses --format=json），收到 {type(data).__name__}。"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("licenses_json", type=Path, help="pip-licenses --format=json 的输出")
    parser.add_argument("--strict", action="store_true", help="把 caution 与 unknown 也当作失败")
    parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="未登记在白名单里的 unknown 一律失败（CI 用这个；caution 仍只是提醒）",
    )
    args = parser.parse_args(argv)

    packages = load_packages(args.licenses_json)
    if not packages:
        print("依赖清单为空——审计没有实际检查任何东西，这本身就是一个问题。", file=sys.stderr)
        return 1

    forbidden, caution, unknown, accepted = [], [], [], []
    for pkg in sorted(packages, key=lambda p: p.get("Name", "")):
        name = pkg.get("Name", "?")
        verdict = classify(pkg.get("License") or "")
        row = (
            name,
            pkg.get("Version", "?"),
            (pkg.get("License") or "").strip() or "(空)",
        )
        # 顺序有意如此：白名单只救 unknown，救不了 forbidden / caution。
        # 一个上了白名单的包若哪天换成强 copyleft，这里仍然会拦住它。
        if verdict == "forbidden":
            forbidden.append(row)
        elif verdict == "caution":
            caution.append(row)
        elif name in ACCEPTED_NON_STANDARD:
            accepted.append(row)
        elif verdict == "unknown":
            unknown.append(row)

    print(
        f"审计 {len(packages)} 个依赖："
        f"强 copyleft {len(forbidden)}、需留意 {len(caution)}、"
        f"已登记的非标准许可 {len(accepted)}、未登记的未知 {len(unknown)}"
    )

    for label, rows in (
        ("强 copyleft", forbidden),
        ("需留意", caution),
        ("已登记的非标准许可证", accepted),
        ("未登记的未知许可证", unknown),
    ):
        if rows:
            print(f"\n[{label}]")
            for name, version, license_name in rows:
                print(f"  {name} {version}  ->  {license_name}")
                if label == "已登记的非标准许可证":
                    declared, reason = ACCEPTED_NON_STANDARD[name]
                    print(f"      登记为：{declared}")
                    print(f"      理由：{reason}")

    # 白名单里登记了、但依赖里没出现的条目——报出来，免得它腐烂成无人在意的旧话。
    installed = {pkg.get("Name", "") for pkg in packages}
    stale = sorted(set(ACCEPTED_NON_STANDARD) - installed)
    if stale:
        print("\n[白名单已失效]")
        for name in stale:
            print(f"  {name}：已登记但不在依赖中。请确认后从 ACCEPTED_NON_STANDARD 删除。")

    if forbidden:
        print(
            f"\n错误：{len(forbidden)} 个依赖使用强 copyleft 许可证，与骨架库的 Apache-2.0 不兼容。",
            file=sys.stderr,
        )
        return 1
    if args.strict_unknown and unknown:
        print(
            f"\n错误：{len(unknown)} 个依赖的许可证未登记且无法识别。"
            f"要么去掉该依赖，要么在 scripts/check_licenses.py 的 ACCEPTED_NON_STANDARD "
            f"里登记它并写明理由。",
            file=sys.stderr,
        )
        return 1
    if args.strict and (caution or unknown):
        print("\n--strict：caution / unknown 也视为失败。", file=sys.stderr)
        return 1

    print("\n未发现强 copyleft 依赖。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
