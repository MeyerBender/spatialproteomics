import numpy as np
import pytest

from spatialproteomics.constants import Dims, Features, Labels, Layers, Props


def test_remove_label_type(ds_labels):
    ds = ds_labels.la.remove_label_type("B")

    assert Layers.LA_PROPERTIES in ds
    assert Dims.LABELS in ds.coords
    # the removed cells are unlabeled, hence an entry for label 0 is added
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 2, 3, 4]))
    assert 0 in ds.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values
    # making sure there was no in-place operation
    assert np.all(ds_labels.coords[Dims.LABELS].values == np.array([1, 2, 3, 4]))
    assert 0 not in ds_labels.coords[Dims.LABELS].values
    assert 0 not in ds_labels.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values

    with pytest.raises(ValueError, match="Cell type B not found"):
        ds.la.remove_label_type("B")


@pytest.mark.parametrize("cell_types", [[1, 2], ["B", "T"], [1, "T"]])
def test_remove_multiple_label_types(ds_labels, cell_types):
    ds = ds_labels.la.remove_label_type(cell_types)

    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 3, 4]))
    old_labels = ds_labels.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values
    new_labels = ds.pp.get_layer_as_df(celltypes_to_str=False)[Features.LABELS].values
    # cells of the removed types are unlabeled, all other cells keep their label
    assert np.all(new_labels[np.isin(old_labels, [1, 2])] == 0)
    assert np.all(new_labels[~np.isin(old_labels, [1, 2])] == old_labels[~np.isin(old_labels, [1, 2])])


def test_remove_label_type_adds_unlabeled_entry(ds_labels):
    ds = ds_labels.la.remove_label_type("B")
    props = ds.pp.get_layer_as_df(Layers.LA_PROPERTIES)
    assert props.loc[0, Props.NAME] == Labels.UNLABELED

    # looking up the names of the labels used to fail for the unlabeled cells
    df = ds.pp.get_layer_as_df()
    assert Labels.UNLABELED in df[Features.LABELS].values
    assert "B" not in df[Features.LABELS].values

    # removing another type keeps the single unlabeled entry
    ds = ds.la.remove_label_type("T")
    assert np.all(ds.coords[Dims.LABELS].values == np.array([0, 3, 4]))


def test_remove_label_type_then_predict_cell_types_argmax(ds_labels):
    ds = ds_labels.la.remove_label_type("B")
    # partial annotations with unlabeled cells are filled in by the argmax prediction
    ds = ds.la.predict_cell_types_argmax({"PAX5": "B", "CD3": "T"}, overwrite_existing_labels=False)
    assert Layers.LA_PROPERTIES in ds


def test_remove_label_type_with_existing_unlabeled_keeps_compositions(ds_labels):
    # if label 0 (unlabeled) already exists, the removed types are attributed to it in the neighborhood compositions
    df = ds_labels.pp.get_layer_as_df()
    df.loc[df.index[:5], Features.LABELS] = Labels.UNLABELED
    df["cell"] = df.index
    ds = ds_labels.pp.drop_layers(Layers.LA_PROPERTIES).la.add_labels_from_dataframe(df, label_col=Features.LABELS)
    assert 0 in ds.coords[Dims.LABELS].values
    ds = ds.nh.compute_neighborhoods_radius(radius=50)

    removed_id = ds.la._label_name_to_id("B")
    ds_removed = ds.la.remove_label_type("B")

    compositions = ds_removed[Layers.NEIGHBORHOODS]
    assert np.allclose(compositions.sum(Dims.LABELS).values, 1)
    expected = ds[Layers.NEIGHBORHOODS].sel({Dims.LABELS: [0, removed_id]}).sum(Dims.LABELS)
    assert np.allclose(compositions.sel({Dims.LABELS: 0}).values, expected.values)
    # the input is not modified
    assert removed_id in ds[Layers.NEIGHBORHOODS].coords[Dims.LABELS].values
    assert np.allclose(ds[Layers.NEIGHBORHOODS].sum(Dims.LABELS).values, 1)
