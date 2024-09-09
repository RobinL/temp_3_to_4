# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink.datasets import splink_dataset_labels

pairwise_labels = splink_dataset_labels.fake_1000_labels

# Choose labels indicating a match
pairwise_labels = pairwise_labels[pairwise_labels["clerical_match_score"] == 1]
pairwise_labels

from splink import splink_datasets

df = splink_datasets.fake_1000
df.head(2)

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI(), set_up_basic_logging=False)
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email",
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Register the pairwise labels table with the database, and then use it to estimate the m values
labels_df = linker.table_management.register_labels_table(pairwise_labels, overwrite=True)
linker.training.estimate_m_from_pairwise_labels(labels_df)


# If the labels table already existing in the dataset you could run
# linker.training.estimate_m_from_pairwise_labels("labels_tablename_here")

training_blocking_rule = block_on("first_name")
linker.training.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

linker.visualisations.parameter_estimate_comparisons_chart()

linker.visualisations.match_weights_chart()