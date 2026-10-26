import numpy as np

from spatialproteomics.constants import Layers


def test_render_neighborhoods_does_not_modify_accessor(ds_neighborhoods):
    # rendering on top of an existing plot must not change the object the accessor of the input refers to
    ds = ds_neighborhoods.pl.colorize()
    first = ds.pl.render_neighborhoods()

    assert Layers.PLOT in ds.pl._obj
    # calling the method again on the same object gives the same result
    second = ds.pl.render_neighborhoods()
    assert np.array_equal(first[Layers.PLOT].values, second[Layers.PLOT].values)
