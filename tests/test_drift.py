import numpy as np

from overseerml.drift import decile_edges, psi


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=5_000), decile_edges(ref)) < 0.02


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(1.0, size=5_000), decile_edges(ref)) > 0.25


def test_psi_finite_when_a_bin_is_empty():
    ref = np.arange(1_000.0)
    assert np.isfinite(psi(ref, np.full(100, 5.0), decile_edges(ref)))


def test_decile_edges_dedupe_skewed_feature():
    # 95% zeros: most deciles coincide and must collapse to one edge.
    x = np.r_[np.zeros(950), np.arange(1.0, 51.0)]
    edges = decile_edges(x)
    assert len(edges) == len(np.unique(edges)) and edges[0] == 0
