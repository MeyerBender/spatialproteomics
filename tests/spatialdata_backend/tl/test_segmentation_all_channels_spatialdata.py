import numpy as np
import pytest

import spatialproteomics as sp
import spatialproteomics.tl.tool as tool
from spatialproteomics.constants import SDLayers


def _fake_segmentation(image, **kwargs):
    # one (empty) mask per channel, mimicking the segmentation of each channel separately
    return np.zeros(image.shape, dtype=int)


@pytest.mark.parametrize("method", ["stardist", "cellpose"])
def test_segmentation_without_channel_segments_all_channels(ds_image_spatialdata, monkeypatch, method):
    if method == "stardist":
        monkeypatch.setattr(tool, "_stardist", _fake_segmentation)
    else:
        monkeypatch.setattr(tool, "_cellpose", lambda image, **kwargs: (_fake_segmentation(image), None))

    channels = ds_image_spatialdata.images[SDLayers.IMAGE].coords["c"].values.tolist()
    assert len(channels) > 1

    sdata = getattr(sp.tl, method)(ds_image_spatialdata, channel=None, copy=True)

    for channel in channels:
        assert f"{SDLayers.SEGMENTATION}_{channel}" in sdata.labels
