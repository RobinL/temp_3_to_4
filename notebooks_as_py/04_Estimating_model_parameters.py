# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

# Begin by reading in the tutorial data again
from splink import splink_datasets

df = splink_datasets.fake_1000

import splink.comparison_library as cl

city_comparison = cl.LevenshteinAtThresholds("city", 2)
print(city_comparison.get_comparison("duckdb").human_readable_description)

email_comparison = cl.EmailComparison("email")
print(email_comparison.get_comparison("duckdb").human_readable_description)

from splink import Linker, SettingsCreator, block_on, DuckDBAPI

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.LevenshteinAtThresholds("dob", 1),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name", "city"),
        block_on("surname"),

    ],
    retain_intermediate_calculation_columns=True,
)

linker = Linker(df, settings, db_api=DuckDBAPI())

deterministic_rules = [
    block_on("first_name", "dob"),
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    block_on("email")
]

linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

training_blocking_rule = block_on("first_name", "surname")
training_session_fname_sname = (
    linker.training.estimate_parameters_using_expectation_maximisation(training_blocking_rule)
)

training_blocking_rule = block_on("dob")
training_session_dob = linker.training.estimate_parameters_using_expectation_maximisation(
    training_blocking_rule
)

linker.visualisations.match_weights_chart()

linker.visualisations.m_u_parameters_chart()

linker.visualisations.parameter_estimate_comparisons_chart()

settings = linker.misc.save_model_to_json(
    "../demo_settings/saved_model_from_demo.json", overwrite=True
)

linker.evaluation.unlinkables_chart()