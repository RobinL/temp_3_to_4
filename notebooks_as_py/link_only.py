# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import splink_datasets

df = splink_datasets.fake_1000

# Split a simple dataset into two, separate datasets which can be linked together.
df_l = df.sample(frac=0.5)
df_r = df.drop(df_l.index)

df_l.head(2)

import splink.comparison_library as cl

from splink import DuckDBAPI, Linker, SettingsCreator, block_on

settings = SettingsCreator(
    link_type="link_only",
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ],
    comparisons=[
        cl.NameComparison(
            "first_name",
        ),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
            invalid_dates_as_null=True,
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
)

linker = Linker(
    [df_l, df_r],
    settings,
    db_api=DuckDBAPI(),
    input_table_aliases=["df_left", "df_right"],
)

from splink.exploratory import completeness_chart

completeness_chart(
    [df_l, df_r],
    cols=["first_name", "surname", "dob", "city", "email"],
    db_api=DuckDBAPI(),
    table_names_for_chart=["df_left", "df_right"],
)


deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    block_on("email"),
]


linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=1)

session_dob = linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
session_email = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("email")
)
session_first_name = linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name")
)

results = linker.inference.predict(threshold_match_probability=0.9)

results.as_pandas_dataframe(limit=5)