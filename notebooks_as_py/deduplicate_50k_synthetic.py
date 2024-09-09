from splink.athena.athena_linker import AthenaLinker
import altair as alt
alt.renderers.enable('mimetype')

import pandas as pd
pd.options.display.max_rows = 1000
df = pd.read_parquet("./data/historical_figures_with_errors_50k.parquet")

import boto3
my_session = boto3.Session(region_name="eu-west-1")

# Simple settings dictionary will be used for exploratory analysis
settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        "l.first_name = r.first_name and l.surname = r.surname",
        "l.surname = r.surname and l.dob = r.dob",
        "l.first_name = r.first_name and l.dob = r.dob",
        "l.postcode_fake = r.postcode_fake and l.first_name = r.first_name",
    ],
}

# Set the output bucket and the additional filepath to write outputs to
############################################
# EDIT THESE BEFORE ATTEMPTING TO RUN THIS #
############################################

bucket = "my_s3_bucket"
database = "my_athena_database"
filepath = "athena_testing"  # file path inside of your bucket
aws_filepath = f"s3://{bucket}/{filepath}"

# Sessions are generated with a unique ID...
linker = AthenaLinker(
    input_table_or_tables=df,
    boto3_session=my_session,
    # the bucket to store splink's parquet files
    output_bucket=bucket,
    # the database to store splink's outputs
    output_database=database,
    # folder to output data to
    output_filepath=filepath,  
    # table name within your database
    # if blank, it will default to __splink__input_table_randomid
    input_table_aliases="__splink__testings",
    settings_dict=settings,
)

linker.profile_columns(
    ["first_name", "postcode_fake", "substr(dob, 1,4)"], top_n=10, bottom_n=5
)

linker.cumulative_num_comparisons_from_blocking_rules_chart()

linker.drop_all_tables_created_by_splink(delete_s3_folders=True)

import splink.athena.athena_comparison_library as cl

settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        "l.first_name = r.first_name and l.surname = r.surname",
        "l.surname = r.surname and l.dob = r.dob",
        "l.first_name = r.first_name and l.dob = r.dob",
        "l.postcode_fake = r.postcode_fake and l.first_name = r.first_name",
    ],
    "comparisons": [
        cl.levenshtein_at_thresholds("first_name", [1,2], term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("surname", [1,2], term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("dob", [1,2], term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("postcode_fake", 2,term_frequency_adjustments=True),
        cl.exact_match("birth_place", term_frequency_adjustments=True),
        cl.exact_match("occupation",  term_frequency_adjustments=True),
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    "max_iterations": 10,
    "em_convergence": 0.01
}

# Write our dataframe to s3/our backing database
import awswrangler as wr
wr.s3.to_parquet(
    df,  # pandas dataframe
    path=f"{aws_filepath}/historical_figures_with_errors_50k",
    dataset=True,
    database=database,
    table="historical_figures_with_errors_50k",
    mode="overwrite",
    compression="snappy",
)

# Initialise our linker with historical_figures_with_errors_50k from our database
linker = AthenaLinker(
    input_table_or_tables="historical_figures_with_errors_50k",  
    settings_dict=settings,
    boto3_session=my_session,
    output_bucket=bucket,  # the bucket to store splink's parquet files 
    output_database=database,  # the database to store splink's outputs
    output_filepath=filepath  # folder to output data to
)

linker.estimate_probability_two_random_records_match(
    [
        "l.first_name = r.first_name and l.surname = r.surname and l.dob = r.dob",
        "substr(l.first_name,1,2) = substr(r.first_name,1,2) and l.surname = r.surname and substr(l.postcode_fake,1,2) = substr(r.postcode_fake,1,2)",
        "l.dob = r.dob and l.postcode_fake = r.postcode_fake",
    ],
    recall=0.6,
)

linker.estimate_u_using_random_sampling(max_pairs=5e6)

blocking_rule = "l.first_name = r.first_name and l.surname = r.surname"
training_session_names = linker.estimate_parameters_using_expectation_maximisation(blocking_rule)

blocking_rule = "l.dob = r.dob"
training_session_dob = linker.estimate_parameters_using_expectation_maximisation(blocking_rule)

linker.match_weights_chart()

linker.unlinkables_chart()

df_predict = linker.predict()
df_e = df_predict.as_pandas_dataframe(limit=5)
df_e

from splink.charts import waterfall_chart
records_to_plot = df_e.to_dict(orient="records")
linker.waterfall_chart(records_to_plot, filter_nulls=False)

clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, threshold_match_probability=0.95)

linker.cluster_studio_dashboard(df_predict, clusters, "dashboards/50k_cluster.html", sampling_method='by_cluster_size', overwrite=True)

from IPython.display import IFrame

IFrame(
    src="./dashboards/50k_cluster.html", width="100%", height=1200
)

linker.roc_chart_from_labels_column("cluster",match_weight_round_to_nearest=0.02)

records = linker.prediction_errors_from_labels_column(
    "cluster",
    threshold=0.999,
    include_false_negatives=False,
    include_false_positives=True,
).as_record_dict()
linker.waterfall_chart(records)

# Some of the false negatives will be because they weren't detected by the blocking rules
records = linker.prediction_errors_from_labels_column(
    "cluster",
    threshold=0.5,
    include_false_negatives=True,
    include_false_positives=False,
).as_record_dict(limit=50)

linker.waterfall_chart(records)

linker.drop_tables_in_current_splink_run(tables_to_exclude=df_predict)