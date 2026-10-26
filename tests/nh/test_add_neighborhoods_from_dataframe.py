import numpy as np
import pandas as pd
import pytest


def test_add_neighborhoods_from_dataframe_negative_values(ds_labels):
    # a single negative neighborhood is enough to be rejected
    neighborhoods = np.ones(ds_labels.sizes["cells"], dtype=int)
    neighborhoods[0] = -1
    neighborhood_df = pd.DataFrame({"neighborhood": neighborhoods})

    with pytest.raises(AssertionError, match="Neighborhoods must be >= 0."):
        ds_labels.nh.add_neighborhoods_from_dataframe(neighborhood_df, neighborhood_col="neighborhood")
