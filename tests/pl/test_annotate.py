import matplotlib.pyplot as plt

from spatialproteomics.constants import Dims


def test_annotate(ds_segmentation):
    fig, ax = plt.subplots()
    ds_segmentation.pl.annotate(ax=ax)
    assert len(ax.texts) == ds_segmentation.sizes[Dims.CELLS]
    plt.close(fig)


def test_annotate_bbox(ds_segmentation):
    # only the cells within the bounding box are annotated
    x, y = ds_segmentation.coords[Dims.X].values, ds_segmentation.coords[Dims.Y].values
    bbox = [x[0], x[0] + 40, y[0], y[0] + 40]
    expected = ds_segmentation.pp[bbox[0] : bbox[1], bbox[2] : bbox[3]].sizes[Dims.CELLS]
    assert 0 < expected < ds_segmentation.sizes[Dims.CELLS]

    fig, ax = plt.subplots()
    ds_segmentation.pl.annotate(bbox=bbox, ax=ax)
    assert len(ax.texts) == expected
    plt.close(fig)
