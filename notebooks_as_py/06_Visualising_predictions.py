# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

# Rerun our predictions to we're ready to view the charts
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

records_to_view = df_predictions.as_record_dict(limit=5)
linker.visualisations.waterfall_chart(records_to_view, filter_nulls=False)

linker.visualisations.comparison_viewer_dashboard(df_predictions, "scv.html", overwrite=True)

# You can view the scv.html file in your browser, or inline in a notbook as follows
from IPython.display import IFrame

IFrame(src="./scv.html", width="100%", height=1200)

df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predictions, threshold_match_probability=0.5
)

linker.visualisations.cluster_studio_dashboard(
    df_predictions,
    df_clusters,
    "cluster_studio.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

# You can view the scv.html file in your browser, or inline in a notbook as follows
from IPython.display import IFrame

IFrame(src="./cluster_studio.html", width="100%", height=1000)