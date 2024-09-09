# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import Linker, DuckDBAPI, splink_datasets

import pandas as pd

pd.options.display.max_columns = 1000

db_api = DuckDBAPI()
df = splink_datasets.fake_1000

import json
import urllib

url = "https://raw.githubusercontent.com/moj-analytical-services/splink/847e32508b1a9cdd7bcd2ca6c0a74e547fb69865/docs/demos/demo_settings/saved_model_from_demo.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())


linker = Linker(df, settings, db_api=DuckDBAPI())

df_predictions = linker.inference.predict(threshold_match_probability=0.2)
df_predictions.as_pandas_dataframe(limit=5)

clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions, threshold_match_probability=0.5
)
clusters.as_pandas_dataframe(limit=10)

sql = f"""
select *
from {df_predictions.physical_name}
limit 2
"""
linker.misc.query_sql(sql)