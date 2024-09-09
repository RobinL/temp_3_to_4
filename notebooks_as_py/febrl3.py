import pandas as pd
import altair as alt
from splink.datasets import splink_datasets

df = splink_datasets.febrl3
df = df.rename(columns=lambda x: x.strip())

df["cluster"] = df["rec_id"].apply(lambda x: "-".join(x.split('-')[:2]))

# dob and ssn needs to be a string for fuzzy comparisons like levenshtein to be applied
df["date_of_birth"] = df["date_of_birth"].astype(str).str.strip()
df["date_of_birth"] = df["date_of_birth"].replace("", None)

df["soc_sec_id"] = df["soc_sec_id"].astype(str).str.strip()
df["soc_sec_id"] = df["soc_sec_id"].replace("", None)

df["postcode"] = df["postcode"].astype(str).str.strip()
df["postcode"] = df["postcode"].replace("", None)
df.head(2)


from splink.duckdb.linker import DuckDBLinker

settings = {
    "unique_id_column_name": "rec_id",
    "link_type": "dedupe_only",
}

linker = DuckDBLinker(df, settings)

linker.missingness_chart()

linker.profile_columns(list(df.columns))

from splink.duckdb.blocking_rule_library import block_on

blocking_rules = [
        block_on("soc_sec_id"),
        block_on("given_name"),
        block_on("surname"),
        block_on("date_of_birth"),
        block_on("postcode"),
]
linker.cumulative_num_comparisons_from_blocking_rules_chart(blocking_rules)

from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl


settings = {
    "unique_id_column_name": "rec_id",
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        ctl.name_comparison("given_name", term_frequency_adjustments=True),
        ctl.name_comparison("surname", term_frequency_adjustments=True),
        ctl.date_comparison("date_of_birth", 
                            damerau_levenshtein_thresholds=[],
                            cast_strings_to_date=True,
                            invalid_dates_as_null=True,
                            date_format="%Y%m%d"),
        cl.levenshtein_at_thresholds("soc_sec_id", [2]),
        cl.exact_match("street_number", term_frequency_adjustments=True),
        cl.exact_match("postcode", term_frequency_adjustments=True),
    ],
    "retain_intermediate_calculation_columns": True,
}

linker = DuckDBLinker(df, settings)

deterministic_rules = [
    "l.soc_sec_id = r.soc_sec_id",
    "l.given_name = r.given_name and l.surname = r.surname and l.date_of_birth = r.date_of_birth",
    "l.given_name = r.surname and l.surname = r.given_name and l.date_of_birth = r.date_of_birth"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.9)

linker.estimate_u_using_random_sampling(max_pairs=1e6)

em_blocking_rule_1 = block_on("substr(date_of_birth,1,3)")
em_blocking_rule_2 = block_on("substr(postcode,1,2)")
session_dob = linker.estimate_parameters_using_expectation_maximisation(em_blocking_rule_1)
session_postcode = linker.estimate_parameters_using_expectation_maximisation(em_blocking_rule_2)

linker.match_weights_chart()

results = linker.predict(threshold_match_probability=0.2)

linker.roc_chart_from_labels_column("cluster")

pred_errors_df = linker.prediction_errors_from_labels_column("cluster").as_pandas_dataframe()
len(pred_errors_df)
pred_errors_df.head()

records = linker.prediction_errors_from_labels_column("cluster").as_record_dict(limit=10)
linker.waterfall_chart(records)