import pandas as pd 
import altair as alt

from splink.datasets import splink_dataset_labels
pairwise_labels = splink_dataset_labels.fake_1000_labels

# Choose labels indicating a match
pairwise_labels = pairwise_labels[pairwise_labels["clerical_match_score"] == 1]
pairwise_labels

from splink.datasets import splink_datasets

df = splink_datasets.fake_1000
df.head(2)

from splink.duckdb.linker import DuckDBLinker
from splink.duckdb.blocking_rule_library import block_on
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl

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


linker.estimate_u_using_random_sampling(max_pairs=1e6)

# Register the pairwise labels table with the database, and then use it to estimate the m values
labels_df = linker.register_labels_table(pairwise_labels, overwrite=True)
linker.estimate_m_from_pairwise_labels(labels_df)


# If the labels table already existing in the dataset you could run
# linker.estimate_m_from_pairwise_labels("labels_tablename_here")


training_blocking_rule = block_on("first_name")
linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

linker.parameter_estimate_comparisons_chart()

linker.match_weights_chart()