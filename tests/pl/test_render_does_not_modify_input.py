import matplotlib.pyplot as plt
import numpy as np

from spatialproteomics.constants import Attrs, Layers


def test_render_obs_does_not_modify_input_attrs(ds_segmentation):
    ds = ds_segmentation.pl.colorize()
    attrs_before = dict(ds[Layers.PLOT].attrs)

    rendered = ds.pl.render_obs("centroid-1")

    assert dict(ds[Layers.PLOT].attrs) == attrs_before
    assert Attrs.OBS_COLORS in rendered[Layers.PLOT].attrs


def test_render_segmentation_does_not_modify_input_attrs(ds_segmentation):
    # with multiple segmentation masks, the colors of the masks are stored in the attrs of the plot layer
    segmentation = ds_segmentation[Layers.SEGMENTATION].values
    n_channels = ds_segmentation.sizes["channels"]
    ds = ds_segmentation.pp.add_layer(np.stack([segmentation] * n_channels), key_added="_multi_segmentation")
    ds = ds.pl.colorize()
    attrs_before = dict(ds[Layers.PLOT].attrs)

    rendered = ds.pl.render_segmentation(layer_key="_multi_segmentation")

    assert dict(ds[Layers.PLOT].attrs) == attrs_before
    assert Attrs.SEGMENTATION_COLORS in rendered[Layers.PLOT].attrs


def test_scatter_labels_with_edges_does_not_modify_adjacency_matrix(ds_neighborhoods):
    ds = ds_neighborhoods
    if Layers.ADJACENCY_MATRIX not in ds:
        ds = ds.pp.drop_layers(Layers.NEIGHBORHOODS, suppress_warnings=True).nh.compute_neighborhoods_radius(radius=50)
    adjacency_matrix = ds[Layers.ADJACENCY_MATRIX].values
    # adjacency matrices computed with older versions contain self-loops
    adjacency_matrix[0, 0] = 1

    fig, ax = plt.subplots()
    ds.pl.scatter_labels(render_edges=True, legend=False, ax=ax)
    plt.close(fig)

    assert ds[Layers.ADJACENCY_MATRIX].values[0, 0] == 1
