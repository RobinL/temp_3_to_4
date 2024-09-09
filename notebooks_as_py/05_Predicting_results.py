from splink.duckdb.linker import DuckDBLinker
from splink.datasets import splink_datasets
import pandas as pd
pd.options.display.max_columns = 1000
df = splink_datasets.fake_1000

linker = DuckDBLinker(df)
linker.load_model("../demo_settings/saved_model_from_demo.json")

df_predictions = linker.predict(threshold_match_probability=0.2)
df_predictions.as_pandas_dataframe(limit=5)

clusters = linker.cluster_pairwise_predictions_at_threshold(df_predictions, threshold_match_probability=0.5)
clusters.as_pandas_dataframe(limit=10)

sql = f"""
select * 
from {df_predictions.physical_name}
limit 2
"""
linker.query_sql(sql)