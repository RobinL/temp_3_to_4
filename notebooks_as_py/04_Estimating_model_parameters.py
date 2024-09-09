# Begin by reading in the tutorial data again
from splink.duckdb.linker import DuckDBLinker
from splink.datasets import splink_datasets
import altair as alt
df = splink_datasets.fake_1000

import splink.duckdb.comparison_library as cl

email_comparison =  cl.levenshtein_at_thresholds("email", 2)
print(email_comparison.human_readable_description)

import splink.duckdb.comparison_template_library as ctl

first_name_comparison = ctl.name_comparison("first_name")
print(first_name_comparison.human_readable_description)


from splink.duckdb.blocking_rule_library import block_on

settings = {
    "link_type": "dedupe_only",
    "comparisons": [
        ctl.name_comparison("first_name"),
        ctl.name_comparison("surname"),
        ctl.date_comparison("dob", cast_strings_to_date=True),
        cl.exact_match("city", term_frequency_adjustments=True),
        ctl.email_comparison("email", include_username_fuzzy_level=False),
    ],
    "blocking_rules_to_generate_predictions": [
        block_on("first_name"),
        block_on("surname"),
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
}

linker = DuckDBLinker(df, settings)

deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

linker.estimate_u_using_random_sampling(max_pairs=1e6)

training_blocking_rule = block_on(["first_name", "surname"])
training_session_fname_sname = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

from numpy import fix


training_blocking_rule = block_on("dob")
training_session_dob = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

linker.match_weights_chart()

linker.m_u_parameters_chart()

settings = linker.save_model_to_json("../demo_settings/saved_model_from_demo.json", overwrite=True)

linker.unlinkables_chart()