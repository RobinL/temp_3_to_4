from splink.datasets import splink_datasets
df = splink_datasets.fake_1000

# Split a simple dataset into two, separate datasets which can be linked together.
df_l = df.sample(frac=0.5)
df_r = df.drop(df_l.index)

df_l.head(2)

from splink.duckdb.linker import DuckDBLinker
from splink.duckdb.blocking_rule_library import block_on
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl


settings = {
    "link_type": "link_only",
    "blocking_rules_to_generate_predictions": [
        block_on("first_name"),
        block_on("surname"),
    ],
    "comparisons": [
        ctl.name_comparison("first_name",),
        ctl.name_comparison("surname"),
        ctl.date_comparison("dob", cast_strings_to_date=True),
        cl.exact_match("city", term_frequency_adjustments=True),
        ctl.email_comparison("email", include_username_fuzzy_level=False),
    ],       
}

linker = DuckDBLinker([df_l, df_r], settings, input_table_aliases=["df_left", "df_right"])

linker.completeness_chart(cols=["first_name", "surname", "dob", "city", "email"])

deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)


linker.estimate_u_using_random_sampling(max_pairs=1e6, seed=1)

session_dob = linker.estimate_parameters_using_expectation_maximisation(block_on("dob"))
session_email = linker.estimate_parameters_using_expectation_maximisation(block_on("email"))
session_first_name = linker.estimate_parameters_using_expectation_maximisation(block_on("first_name"))

results = linker.predict(threshold_match_probability=0.9)

results.as_pandas_dataframe(limit=5)