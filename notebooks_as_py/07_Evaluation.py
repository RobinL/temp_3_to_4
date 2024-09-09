# Rerun our predictions to we're ready to view the charts
from splink.duckdb.linker import DuckDBLinker
from splink.datasets import splink_datasets

import altair as alt

df = splink_datasets.fake_1000
linker = DuckDBLinker(df)
linker.load_model("../demo_settings/saved_model_from_demo.json")
df_predictions = linker.predict(threshold_match_probability=0.2)

from splink.datasets import splink_dataset_labels

df_labels = splink_dataset_labels.fake_1000_labels
df_labels.head(5)
labels_table = linker.register_labels_table(df_labels)

linker.roc_chart_from_labels_table(labels_table)

linker.precision_recall_chart_from_labels_table(labels_table)

roc_table = linker.truth_space_table_from_labels_table(labels_table)
roc_table.as_pandas_dataframe(limit=5)