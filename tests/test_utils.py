import numpy as np


from transieve.utils import (
    make_out_of_transit_mask,
)


def test_make_out_of_transit_mask_excludes_window():
    time = np.linspace(-1.0, 1.0, 101)
    mask = make_out_of_transit_mask(time, epoch=0.0, duration=0.2, margin=0.05)

    assert mask.dtype == bool
    assert mask.shape == time.shape

    inside = np.abs(time) <= (0.1 + 0.05)
    assert np.all(mask[inside] == 0)
    assert np.all(mask[~inside] == 1)
