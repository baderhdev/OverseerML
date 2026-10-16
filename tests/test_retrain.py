import numpy as np

from overseerml.retrain import MARGIN, decide


def test_promotes_only_at_or_above_margin():
    assert decide(0.50, 0.50 + MARGIN)[0]
    assert not decide(0.50, 0.50 + MARGIN / 2)[0]
    assert not decide(0.50, 0.40)[0]


def test_never_promotes_on_undefined_pr_auc():
    promoted, reason = decide(np.nan, 0.9)
    assert not promoted and "undefined" in reason
