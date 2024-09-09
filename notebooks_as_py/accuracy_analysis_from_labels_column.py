# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import splink_datasets

df = splink_datasets.fake_1000
df.head(2)

from splink import SettingsCreator, Linker, block_on, DuckDBAPI

import splink.comparison_library as cl

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
        block_on("dob"),
        block_on("email"),
    ],
    comparisons=[
        cl.ForenameSurnameComparison("first_name", "surname"),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    retain_intermediate_calculation_columns=True,
)

db_api = DuckDBAPI()
linker = Linker(df, settings, db_api=db_api)
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email",
]

linker.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.7
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=5)

session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob"), estimate_without_term_frequencies=True
)
session_email = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("email"), estimate_without_term_frequencies=True
)
session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname"), estimate_without_term_frequencies=True
)

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster", output_type="table"
).as_pandas_dataframe(limit=5)

linker.evaluation.accuracy_analysis_from_labels_column("cluster", output_type="roc")

linker.evaluation.accuracy_analysis_from_labels_column(
    "cluster",
    output_type="threshold_selection",
    threshold_match_probability=0.5,
    add_metrics=["f1"],
)

# Plot some false positives
linker.evaluation.prediction_errors_from_labels_column(
    "cluster", include_false_negatives=True, include_false_positives=True
).as_pandas_dataframe(limit=5)

records = linker.evaluation.prediction_errors_from_labels_column(
    "cluster", include_false_negatives=True, include_false_positives=True
).as_record_dict(limit=5)

linker.visualisations.waterfall_chart(records)