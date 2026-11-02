"""隐藏群体执行层（H1/H2）：形状、确定性、以及两种读出各自的行为。"""

from __future__ import annotations

import torch

from research.rstdp_spiking.reservoir_actor import ReservoirActor


def _actor(seed: int = 0) -> ReservoirActor:
    return ReservoirActor(n_features=4, n_hidden=128, n_actions=2, steps=10, seed=seed)


def test_hidden_population_is_much_larger_than_actions():
    actor = _actor()
    assert actor.n_hidden == 128 and actor.n_actions == 2


def test_same_seed_gives_identical_hidden_spikes():
    a, b = _actor(0), _actor(0)
    features = torch.rand(4)
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    assert torch.equal(
        a.hidden_spikes(features, generator=gen_a), b.hidden_spikes(features, generator=gen_b)
    )


def test_counts_have_one_entry_per_hidden_neuron():
    actor = _actor()
    counts = actor.counts(
        actor.hidden_spikes(torch.rand(4), generator=torch.Generator().manual_seed(1))
    )
    assert counts.shape == (128,)


def test_ridge_readout_fits_and_changes_the_scores():
    torch.manual_seed(0)
    actor = _actor()
    counts = torch.rand(64, 128)
    actions = torch.randint(0, 2, (64,))
    before = actor.scores(counts[0])
    actor.fit_decoder(counts, actions, alpha=1e-3)
    after = actor.scores(counts[0])
    assert actor.decoder is not None
    assert not torch.allclose(before, after), "H2 拟合之后分数必须变化"


def test_local_reinforce_only_touches_the_chosen_column():
    actor = _actor()
    counts = torch.rand(128)
    actor.reinforce(counts, action=1, reward=1.0)
    assert actor.w_out[:, 1].abs().sum() > 0
    assert actor.w_out[:, 0].abs().sum() == 0


def test_hidden_weights_are_frozen():
    actor = _actor()
    before = actor.hidden.weight.clone()
    actor.reinforce(torch.rand(128), action=0, reward=1.0)
    actor.fit_decoder(torch.rand(32, 128), torch.zeros(32, dtype=torch.long))
    assert torch.equal(actor.hidden.weight, before), "隐藏层不得被任何一次更新改动"
