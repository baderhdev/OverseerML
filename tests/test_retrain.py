import numpy as np
import pandas as pd

from overseerml.drift import arrived
from overseerml.retrain import MARGIN, decide, windows


def test_promotes_only_at_or_above_margin():
    assert decide(0.50, 0.50 + MARGIN)[0]
    assert not decide(0.50, 0.50 + MARGIN / 2)[0]
    assert not decide(0.50, 0.40)[0]


def test_never_promotes_on_undefined_pr_auc():
    promoted, reason = decide(np.nan, 0.9)
    assert not promoted and "undefined" in reason


def test_challenger_windows_do_not_overlap_comparison():
    # At step 46 labels exist up to 44: train 1-40, threshold 41-42, compare 43-44.
    assert windows(46) == (40, 42, 43)
    for step in range(35, 50):
        train_end, val_end, compare_from = windows(step)
        assert train_end < val_end < compare_from <= step - 2


def test_arrived_holds_only_labeled_steps():
    # At step 46 the latest three steps with labels are 42-44.
    steps = pd.Series(range(30, 50))
    assert steps[arrived(steps, 46)].tolist() == [42, 43, 44]
