import numpy as np
import pandas as pd

import spatialproteomics as sp
import spatialproteomics.tl.tool as tool
from spatialproteomics.constants import SDLayers


def test_spotiflow_without_channel_uses_all_channels(ds_segmentation, monkeypatch):
    sdata = ds_segmentation.pp.add_quantification().tl.convert_to_spatialdata()
    channels = sdata.images[SDLayers.IMAGE].coords["c"].values.tolist()
    # one spot per channel, placed inside a cell so that it is counted
    y, x = np.argwhere(sdata.labels[SDLayers.SEGMENTATION].values > 0)[0]

    def fake_spotiflow(img, channel_names, **kwargs):
        # the real function requires one channel name per channel of the image
        assert img.shape[0] == len(channel_names)
        n = len(channel_names)
        return pd.DataFrame({"y": [float(y)] * n, "x": [float(x)] * n, "channel": channel_names})

    monkeypatch.setattr(tool, "_spotiflow", fake_spotiflow)

    sdata = sp.tl.spotiflow(sdata, channel=None, copy=True)

    obs = sdata.tables[SDLayers.TABLE].obs
    for channel in channels:
        assert obs[f"{channel}_spots"].sum() == 1
