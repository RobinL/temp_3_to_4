# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

# Rerun our predictions to we're ready to view the charts
import pandas as pd

from splink import DuckDBAPI, Linker, splink_datasets

pd.options.display.max_columns = 1000

db_api = DuckDBAPI()
df = splink_datasets.fake_1000

import json
import urllib

from splink import block_on

url = "https://raw.githubusercontent.com/moj-analytical-services/splink/847e32508b1a9cdd7bcd2ca6c0a74e547fb69865/docs/demos/demo_settings/saved_model_from_demo.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())

# The data quality is very poor in this dataset, so we need looser blocking rules
# to achieve decent recall
settings["blocking_rules_to_generate_predictions"] = [
    block_on("first_name"),
    block_on("city"),
    block_on("email"),
    block_on("dob"),
]

linker = Linker(df, settings, db_api=DuckDBAPI())
df_predictions = linker.inference.predict(threshold_match_probability=0.01)

from splink.datasets import splink_dataset_labels

df_labels = splink_dataset_labels.fake_1000_labels
labels_table = linker.table_management.register_labels_table(df_labels)
df_labels.head(5)

splink_df = linker.evaluation.prediction_errors_from_labels_table(
    labels_table, include_false_negatives=True, include_false_positives=False
)
false_negatives = splink_df.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(false_negatives)

# Note I've picked a threshold match probability of 0.01 here because otherwise
# in this simple example there are no false positives
splink_df = linker.evaluation.prediction_errors_from_labels_table(
    labels_table, include_false_negatives=False, include_false_positives=True, threshold_match_probability=0.01
)
false_postives = splink_df.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(false_postives)

linker.evaluation.accuracy_analysis_from_labels_table(
    labels_table, output_type="threshold_selection", add_metrics=["f1"]
)

linker.evaluation.accuracy_analysis_from_labels_table(labels_table, output_type="roc")

roc_table = linker.evaluation.accuracy_analysis_from_labels_table(
    labels_table, output_type="table"
)
roc_table.as_pandas_dataframe(limit=5)

linker.evaluation.unlinkables_chart()