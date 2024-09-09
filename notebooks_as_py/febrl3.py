# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink.datasets import splink_datasets

df = splink_datasets.febrl3

df = df.rename(columns=lambda x: x.strip())

df["cluster"] = df["rec_id"].apply(lambda x: "-".join(x.split("-")[:2]))

df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()

df.head(2)

df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()

df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()

from splink import DuckDBAPI, Linker, SettingsCreator

# TODO:  Allow missingness to be analysed without a linker
settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="dedupe_only",
)

linker = Linker(df, settings, db_api=DuckDBAPI())

from splink.exploratory import completeness_chart

completeness_chart(df, db_api=DuckDBAPI())

from splink.exploratory import profile_columns

profile_columns(df, db_api=DuckDBAPI(), column_expressions=["given_name", "surname"])

from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules = [
    block_on("soc_sec_id"),
    block_on("given_name"),
    block_on("surname"),
    block_on("date_of_birth"),
    block_on("postcode"),
]

db_api = DuckDBAPI()
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="dedupe_only",
    unique_id_column_name="rec_id",
)

import splink.comparison_library as cl

from splink import Linker

settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.NameComparison("given_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison(
            "date_of_birth",
            input_is_string=True,
            datetime_format="%Y%m%d",
        ),
        cl.DamerauLevenshteinAtThresholds("soc_sec_id", [2]),
        cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
        cl.ExactMatch("postcode").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI())

from splink import block_on

deterministic_rules = [
    block_on("soc_sec_id"),
    block_on("given_name", "surname", "date_of_birth"),
    "l.given_name = r.surname and l.surname = r.given_name and l.date_of_birth = r.date_of_birth",
]

linker.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.9
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

em_blocking_rule_1 = block_on("date_of_birth")
session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    em_blocking_rule_1
)

em_blocking_rule_2 = block_on("postcode")
session_postcode = linker.training.estimate_parameters_using_expectation_maximisation(
    em_blocking_rule_2
)

linker.visualisations.match_weights_chart()

results = linker.inference.predict(threshold_match_probability=0.2)

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster", match_weight_round_to_nearest=0.1, output_type="accuracy"
)

pred_errors_df = linker.evaluation.prediction_errors_from_labels_column(
    "cluster"
).as_pandas_dataframe()
len(pred_errors_df)
pred_errors_df.head()

records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster"
).as_record_dict(limit=10)
linker.visualisations.waterfall_chart(records)