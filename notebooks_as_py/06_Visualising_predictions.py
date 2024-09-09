# Rerun our predictions to we're ready to view the charts
from splink.duckdb.linker import DuckDBLinker
from splink.datasets import splink_datasets
import altair as alt

df = splink_datasets.fake_1000
linker = DuckDBLinker(df)
linker.load_model("../demo_settings/saved_model_from_demo.json")
df_predictions = linker.predict(threshold_match_probability=0.2)

records_to_view  = df_predictions.as_record_dict(limit=5)
linker.waterfall_chart(records_to_view, filter_nulls=False)


linker.comparison_viewer_dashboard(df_predictions, "scv.html", overwrite=True)

# You can view the scv.html file in your browser, or inline in a notbook as follows
from IPython.display import IFrame
IFrame(
    src="./scv.html", width="100%", height=1200
)  

df_clusters = linker.cluster_pairwise_predictions_at_threshold(df_predictions, threshold_match_probability=0.5)

linker.cluster_studio_dashboard(df_predictions, df_clusters, "cluster_studio.html", sampling_method="by_cluster_size", overwrite=True)

# You can view the scv.html file in your browser, or inline in a notbook as follows
from IPython.display import IFrame
IFrame(
    src="./cluster_studio.html", width="100%", height=1200
)