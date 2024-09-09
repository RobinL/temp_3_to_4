# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

import pandas as pd

from splink import splink_datasets

pd.options.display.max_rows = 1000
df = splink_datasets.historical_50k
df.head()

from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

db_api = DuckDBAPI()
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=[
        block_on("first_name", "surname", "dob"),
        block_on("surname", "dob", "postcode_fake"),
        block_on("first_name", "dob", "occupation"),
    ],
    db_api=db_api,
    link_type="dedupe_only",
)

from splink import Linker, SettingsCreator

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname", "dob"),
        block_on("surname", "dob", "postcode_fake"),
        block_on("first_name", "dob", "occupation"),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=db_api)


df_predict = linker.inference.deterministic_link()
df_predict.as_pandas_dataframe().head()

clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=1
)

clusters.as_pandas_dataframe(limit=5)

linker.visualisations.cluster_studio_dashboard(
    df_predict,
    clusters,
    "dashboards/50k_deterministic_cluster.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

from IPython.display import IFrame

IFrame(src="./dashboards/50k_deterministic_cluster.html", width="100%", height=1200)