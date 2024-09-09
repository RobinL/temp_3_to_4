from splink.datasets import splink_datasets
from splink.duckdb.linker import DuckDBLinker
import altair as alt
alt.renderers.enable('html')

import pandas as pd 
pd.options.display.max_rows = 1000
df = splink_datasets.historical_50k
df.head()

from splink.duckdb.blocking_rule_library import block_on

# Simple settings dictionary will be used for exploratory analysis
settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on(["first_name", "surname", "dob"]),
        block_on(["surname", "dob", "postcode_fake"]),
        block_on(["first_name", "dob", "occupation"]),
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
}
linker = DuckDBLinker(df, settings)

linker.debug_mode = False

linker.profile_columns(
    ["first_name", "surname", "substr(dob, 1,4)"], top_n=10, bottom_n=5
)

linker.cumulative_num_comparisons_from_blocking_rules_chart()

df_predict = linker.deterministic_link()
df_predict.as_pandas_dataframe().head()

clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, threshold_match_probability=1)

clusters.as_pandas_dataframe(limit=5)

linker.cluster_studio_dashboard(df_predict, clusters, "dashboards/50k_deterministic_cluster.html", sampling_method='by_cluster_size', overwrite=True)

from IPython.display import IFrame

IFrame(
    src="./dashboards/50k_deterministic_cluster.html", width="100%", height=1200
)  