from splink.datasets import splink_datasets
import altair as alt
alt.renderers.enable("html")

df = splink_datasets.fake_1000

df.head(2)

from splink.duckdb.linker import DuckDBLinker
from splink.duckdb.blocking_rule_library import block_on
import splink.duckdb.comparison_template_library as ctl
import splink.duckdb.comparison_library as cl

settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on("first_name"),
        block_on("surname"),
    ],
    "comparisons": [
        ctl.name_comparison("first_name"),
        ctl.name_comparison("surname"),
        ctl.date_comparison("dob", cast_strings_to_date=True),
        cl.exact_match("city", term_frequency_adjustments=True),
        ctl.email_comparison("email", include_username_fuzzy_level=False),
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
}

linker = DuckDBLinker(df, settings, set_up_basic_logging=False)
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)


linker.estimate_u_using_random_sampling(max_pairs=1e6, seed=5)

session_dob = linker.estimate_parameters_using_expectation_maximisation(block_on("dob"))
session_email = linker.estimate_parameters_using_expectation_maximisation(block_on("email"))

linker.truth_space_table_from_labels_column(
    "cluster", match_weight_round_to_nearest=0.1
).as_pandas_dataframe(limit=5)

linker.roc_chart_from_labels_column("cluster")

linker.precision_recall_chart_from_labels_column("cluster")

# Plot some false positives
linker.prediction_errors_from_labels_column(
    "cluster", include_false_negatives=True, include_false_positives=True
).as_pandas_dataframe(limit=5)

records = linker.prediction_errors_from_labels_column(
    "cluster", include_false_negatives=True, include_false_positives=True
).as_record_dict(limit=5)

linker.waterfall_chart(records)