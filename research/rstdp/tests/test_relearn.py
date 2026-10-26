"""重学实验的守卫：重力真的进到环境实例里，阈值/比值计算不含糊。"""

from __future__ import annotations

from research.rstdp.cartpole import make_cartpole
from research.rstdp.relearn import _first_reaching, _parse_seeds


def test_gravity_reaches_the_environment():
    env = make_cartpole(14.0)
    try:
        assert env.unwrapped.gravity == 14.0
    finally:
        env.close()


def test_default_gravity_is_untouched():
    env = make_cartpole()
    try:
        assert env.unwrapped.gravity == 9.8
    finally:
        env.close()


def test_first_reaching_uses_best_so_far():
    assert _first_reaching([10.0, 160.0, 20.0], 150.0, 10) == 20
    assert _first_reaching([10.0, 20.0], 150.0, 10) is None


def test_seed_parser():
    assert _parse_seeds("65-67") == [65, 66, 67]
    assert _parse_seeds("1,3") == [1, 3]
