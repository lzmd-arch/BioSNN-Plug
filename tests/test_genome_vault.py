"""基因库守卫：**参照基因组必须能自证，且任何篡改都要被抓到。**

设计见目标文档 §40。这条测试钉的是"备份成功"的定义：
不是"文件写完了"，是"**从种子重新物化后校验和一致**"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch", reason="无 torch 的 CI 腿跳过")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.genome_vault import (
    CONSTANTS,
    GATES,
    SPEC_VERSION,
    freeze,
    materialize,
    restore_check,
    verify,
)


def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_freeze_verifies() -> None:
    """原样基因组必须自验证通过。"""
    g = freeze({"net": 0}, _dev())
    ok, why = verify(g, _dev())
    assert ok, why


def test_materialize_is_deterministic() -> None:
    """同种子两次物化必须逐位相同 —— 没有这条，自验证就没有意义。"""
    a = materialize({"net": 3}, _dev())
    b = materialize({"net": 3}, _dev())
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_tampered_seed_is_caught() -> None:
    g = freeze({"net": 0}, _dev())
    g["seeds"]["net"] = 1
    ok, _ = verify(g, _dev())
    assert not ok


def test_tampered_constant_is_caught() -> None:
    g = freeze({"net": 0}, _dev())
    g["constants"] = dict(g["constants"], amp_gate=9.9)
    ok, why = verify(g, _dev())
    assert not ok
    assert "物种" in why


def test_tampered_checksum_is_caught() -> None:
    g = freeze({"net": 0}, _dev())
    g["checksum"] = "0" * 64
    ok, _ = verify(g, _dev())
    assert not ok


def test_wrong_spec_version_is_caught() -> None:
    g = freeze({"net": 0}, _dev())
    g["spec_version"] = "g2-old"
    ok, _ = verify(g, _dev())
    assert not ok


def test_restore_check_passes_at_measured_values() -> None:
    """用本项目实测的六个值填进去，必须全过。"""
    ok, fails = restore_check(
        firing=0.15,
        decode_linear=0.533,
        order_acc=0.698,
        lesion_firing=0.037,
        tau_forget=150.0,
        plateau=0.445,
    )
    assert ok, fails


def test_restore_check_catches_each_gate() -> None:
    """六道闸各自都要能拦下来 —— 否则它是个永远通过的门。"""
    base = {
        "firing": 0.15,
        "decode_linear": 0.533,
        "order_acc": 0.698,
        "lesion_firing": 0.037,
        "tau_forget": 150.0,
        "plateau": 0.445,
    }
    bad = {
        "firing": 0.5,
        "decode_linear": 0.1,
        "order_acc": 0.5,
        "lesion_firing": 0.0,
        "tau_forget": 500.0,
        "plateau": 0.1,
    }
    for k, v in bad.items():
        ok, fails = restore_check(**dict(base, **{k: v}))
        assert not ok, f"闸 {k} 没有拦住 {v}"
        assert len(fails) == 1, fails


def test_gates_come_from_measurements() -> None:
    """门槛必须与实验报告的数值一致，不能漂。"""
    assert GATES["decode_linear"] == 0.533  # M6t
    assert GATES["order_acc"] == 0.698  # M6ad2
    assert GATES["tau_forget"] == 150.0  # M6ah
    assert CONSTANTS["amp_gate"] == 2.0  # M6az 可行带内
    assert SPEC_VERSION.startswith("g3-")
