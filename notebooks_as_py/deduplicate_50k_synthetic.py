import boto3

boto3_session = boto3.Session(region_name="eu-west-1")

# Set the output bucket and the additional filepath to write outputs to
############################################
# EDIT THESE BEFORE ATTEMPTING TO RUN THIS #
############################################

from splink.backends.athena import AthenaAPI


bucket = "MYTESTBUCKET"
database = "MYTESTDATABASE"
filepath = "MYTESTFILEPATH"  # file path inside of your bucket

aws_filepath = f"s3://{bucket}/{filepath}"
db_api = AthenaAPI(
    boto3_session,
    output_bucket=bucket,
    output_database=database,
    output_filepath=filepath,
)

import splink.comparison_library as cl
from splink import block_on

from splink import Linker, SettingsCreator, splink_datasets

df = splink_datasets.historical_50k

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname"),
        block_on("surname", "dob"),
    ],
    comparisons=[
        cl.ExactMatch("first_name").configure(term_frequency_adjustments=True),
        cl.LevenshteinAtThresholds("surname", [1, 3]),
        cl.LevenshteinAtThresholds("dob", [1, 2]),
        cl.LevenshteinAtThresholds("postcode_fake", [1, 2]),
        cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
        cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

from splink.exploratory import profile_columns

profile_columns(df, db_api, column_expressions=["first_name", "substr(surname,1,2)"])

from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)
from splink import block_on

cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    db_api=db_api,
    blocking_rules=[block_on("first_name", "surname"), block_on("surname", "dob")],
    link_type="dedupe_only",
)

import splink.comparison_library as cl


from splink import Linker, SettingsCreator

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "surname"),
        block_on("surname", "dob"),
    ],
    comparisons=[
        cl.ExactMatch("first_name").configure(term_frequency_adjustments=True),
        cl.LevenshteinAtThresholds("surname", [1, 3]),
        cl.LevenshteinAtThresholds("dob", [1, 2]),
        cl.LevenshteinAtThresholds("postcode_fake", [1, 2]),
        cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
        cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=db_api)

linker.training.estimate_probability_two_random_records_match(
    [
        block_on("first_name", "surname", "dob"),
        block_on("substr(first_name,1,2)", "surname", "substr(postcode_fake, 1,2)"),
        block_on("dob", "postcode_fake"),
    ],
    recall=0.6,
)

linker.training.estimate_u_using_random_sampling(max_pairs=5e6)

blocking_rule = block_on("first_name", "surname")
training_session_names = (
    linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule)
)

blocking_rule = block_on("dob")
training_session_dob = (
    linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule)
)

linker.visualisations.match_weights_chart()

linker.evaluation.unlinkables_chart()

df_predict = linker.inference.predict()
df_e = df_predict.as_pandas_dataframe(limit=5)
df_e

records_to_plot = df_e.to_dict(orient="records")
linker.visualisations.waterfall_chart(records_to_plot, filter_nulls=False)

clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=0.95
)

linker.visualisations.cluster_studio_dashboard(
    df_predict,
    clusters,
    "dashboards/50k_cluster.html",
    sampling_method="by_cluster_size",
    overwrite=True,
)

from IPython.display import IFrame

IFrame(src="./dashboards/50k_cluster.html", width="100%", height=1200)