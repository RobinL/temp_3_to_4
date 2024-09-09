# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import DuckDBAPI, block_on, splink_datasets

df = splink_datasets.fake_1000

from splink.blocking_analysis import count_comparisons_from_blocking_rule

db_api = DuckDBAPI()

br = block_on("substr(first_name, 1,1)", "surname")

counts = count_comparisons_from_blocking_rule(
    table_or_tables=df,
    blocking_rule=br,
    link_type="dedupe_only",
    db_api=db_api,
)

counts

br = "l.first_name = r.first_name and levenshtein(l.surname, r.surname) < 2"

counts = count_comparisons_from_blocking_rule(
    table_or_tables=df,
    blocking_rule= br,
    link_type="dedupe_only",
    db_api=db_api,
)
counts

from splink.blocking_analysis import n_largest_blocks

result = n_largest_blocks(    table_or_tables=df,
    blocking_rule= block_on("city", "first_name"),
    link_type="dedupe_only",
    db_api=db_api,
    n_largest=3
    )

result.as_pandas_dataframe()

from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules_for_analysis = [
    block_on("substr(first_name, 1,1)", "surname"),
    block_on("surname"),
    block_on("email"),
    block_on("city", "first_name"),
    "l.first_name = r.first_name and levenshtein(l.surname, r.surname) < 2",
]


cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=df,
    blocking_rules=blocking_rules_for_analysis,
    db_api=db_api,
    link_type="dedupe_only",
)

from splink.exploratory import profile_columns

profile_columns(df, column_expressions=["city || left(first_name,1)"], db_api=db_api)