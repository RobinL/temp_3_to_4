# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import splink_datasets

df_a = splink_datasets.febrl4a
df_b = splink_datasets.febrl4b


def prepare_data(data):
    data = data.rename(columns=lambda x: x.strip())
    data["cluster"] = data["rec_id"].apply(lambda x: "-".join(x.split("-")[:2]))
    data["date_of_birth"] = data["date_of_birth"].astype(str).str.strip()
    data["soc_sec_id"] = data["soc_sec_id"].astype(str).str.strip()
    data["postcode"] = data["postcode"].astype(str).str.strip()
    return data


dfs = [prepare_data(dataset) for dataset in [df_a, df_b]]

display(dfs[0].head(2))
display(dfs[1].head(2))

from splink import DuckDBAPI, Linker, SettingsCreator

basic_settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="link_only",
    # NB as we are linking one-one, we know the probability that a random pair will be a match
    # hence we could set:
    # "probability_two_random_records_match": 1/5000,
    # however we will not specify this here, as we will use this as a check that
    # our estimation procedure returns something sensible
)

linker = Linker(dfs, basic_settings, db_api=DuckDBAPI())

from splink.exploratory import completeness_chart

completeness_chart(dfs, db_api=DuckDBAPI())

from splink.exploratory import profile_columns

profile_columns(dfs, db_api=DuckDBAPI(), column_expressions=["given_name", "surname"])

from splink import DuckDBAPI, block_on
from splink.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_chart,
)

blocking_rules = [
    block_on("given_name", "surname"),
    # A blocking rule can also be an aribtrary SQL expression
    "l.given_name = r.surname and l.surname = r.given_name",
    block_on("date_of_birth"),
    block_on("soc_sec_id"),
    block_on("state", "address_1"),
    block_on("street_number", "address_1"),
    block_on("postcode"),
]


db_api = DuckDBAPI()
cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
    table_or_tables=dfs,
    blocking_rules=blocking_rules,
    db_api=db_api,
    link_type="link_only",
    unique_id_column_name="rec_id",
    source_dataset_column_name="source_dataset",
)

import splink.comparison_level_library as cll
import splink.comparison_library as cl


# the simple model only considers a few columns, and only two comparison levels for each
simple_model_settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="link_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.ExactMatch("given_name").configure(term_frequency_adjustments=True),
        cl.ExactMatch("surname").configure(term_frequency_adjustments=True),
        cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
    ],
    retain_intermediate_calculation_columns=True,
)

# the detailed model considers more columns, using the information we saw in the exploratory phase
# we also include further comparison levels to account for typos and other differences
detailed_model_settings = SettingsCreator(
    unique_id_column_name="rec_id",
    link_type="link_only",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=[
        cl.NameComparison("given_name").configure(term_frequency_adjustments=True),
        cl.NameComparison("surname").configure(term_frequency_adjustments=True),
        cl.DateOfBirthComparison(
            "date_of_birth",
            input_is_string=True,
            datetime_format="%Y%m%d",
            invalid_dates_as_null=True,
        ),
        cl.DamerauLevenshteinAtThresholds("soc_sec_id", [1, 2]),
        cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
        cl.DamerauLevenshteinAtThresholds("postcode", [1, 2]).configure(
            term_frequency_adjustments=True
        ),
        # we don't consider further location columns as they will be strongly correlated with postcode
    ],
    retain_intermediate_calculation_columns=True,
)


linker_simple = Linker(dfs, simple_model_settings, db_api=DuckDBAPI())
linker_detailed = Linker(dfs, detailed_model_settings, db_api=DuckDBAPI())

deterministic_rules = [
    block_on("soc_sec_id"),
    block_on("given_name", "surname", "date_of_birth"),
]

linker_detailed.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)

# We generally recommend setting max pairs higher (e.g. 1e7 or more)
# But this will run faster for the purpose of this demo
linker_detailed.training.estimate_u_using_random_sampling(max_pairs=1e6)

session_dob = (
    linker_detailed.training.estimate_parameters_using_expectation_maximisation(
        block_on("date_of_birth"), estimate_without_term_frequencies=True
    )
)
session_pc = (
    linker_detailed.training.estimate_parameters_using_expectation_maximisation(
        block_on("postcode"), estimate_without_term_frequencies=True
    )
)

session_dob.m_u_values_interactive_history_chart()

linker_detailed.visualisations.parameter_estimate_comparisons_chart()

linker_simple.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)
linker_simple.training.estimate_u_using_random_sampling(max_pairs=1e7)
session_ssid = (
    linker_simple.training.estimate_parameters_using_expectation_maximisation(
        block_on("given_name"), estimate_without_term_frequencies=True
    )
)
session_pc = linker_simple.training.estimate_parameters_using_expectation_maximisation(
    block_on("street_number"), estimate_without_term_frequencies=True
)
linker_simple.visualisations.parameter_estimate_comparisons_chart()

# import json
# we can have a look at the full settings if we wish, including the values of our estimated parameters:
# print(json.dumps(linker_detailed._settings_obj.as_dict(), indent=2))
# we can also get a handy summary of of the model in an easily readable format if we wish:
# print(linker_detailed._settings_obj.human_readable_description)
# (we suppress output here for brevity)

linker_simple.visualisations.match_weights_chart()

linker_detailed.visualisations.match_weights_chart()

# linker_simple.m_u_parameters_chart()
linker_detailed.visualisations.m_u_parameters_chart()

linker_simple.evaluation.unlinkables_chart()

linker_detailed.evaluation.unlinkables_chart()

predictions = linker_detailed.inference.predict(threshold_match_probability=0.2)
df_predictions = predictions.as_pandas_dataframe()
df_predictions.head(5)

linker_detailed.evaluation.accuracy_analysis_from_labels_column(
    "cluster", output_type="accuracy"
)

clusters = linker_detailed.clustering.cluster_pairwise_predictions_at_threshold(
    predictions, threshold_match_probability=0.99
)
df_clusters = clusters.as_pandas_dataframe().sort_values("cluster_id")
df_clusters.groupby("cluster_id").size().value_counts()

df_predictions["cluster_l"] = df_predictions["rec_id_l"].apply(
    lambda x: "-".join(x.split("-")[:2])
)
df_predictions["cluster_r"] = df_predictions["rec_id_r"].apply(
    lambda x: "-".join(x.split("-")[:2])
)
df_true_links = df_predictions[
    df_predictions["cluster_l"] == df_predictions["cluster_r"]
].sort_values("match_probability")

records_to_view = 3
linker_detailed.visualisations.waterfall_chart(
    df_true_links.head(records_to_view).to_dict(orient="records")
)

df_non_links = df_predictions[
    df_predictions["cluster_l"] != df_predictions["cluster_r"]
].sort_values("match_probability", ascending=False)
linker_detailed.visualisations.waterfall_chart(
    df_non_links.head(records_to_view).to_dict(orient="records")
)

# we need to append a full name column to our source data frames
# so that we can use it for term frequency adjustments
dfs[0]["full_name"] = dfs[0]["given_name"] + "_" + dfs[0]["surname"]
dfs[1]["full_name"] = dfs[1]["given_name"] + "_" + dfs[1]["surname"]


extended_model_settings = {
    "unique_id_column_name": "rec_id",
    "link_type": "link_only",
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        {
            "output_column_name": "Full name",
            "comparison_levels": [
                {
                    "sql_condition": "(given_name_l IS NULL OR given_name_r IS NULL) and (surname_l IS NULL OR surname_r IS NULL)",
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                # full name match
                cll.ExactMatchLevel("full_name", term_frequency_adjustments=True),
                # typos - keep levels across full name rather than scoring separately
                cll.JaroWinklerLevel("full_name", 0.9),
                cll.JaroWinklerLevel("full_name", 0.7),
                # name switched
                cll.ColumnsReversedLevel("given_name", "surname"),
                # name switched + typo
                {
                    "sql_condition": "jaro_winkler_similarity(given_name_l, surname_r) + jaro_winkler_similarity(surname_l, given_name_r) >= 1.8",
                    "label_for_charts": "switched + jaro_winkler_similarity >= 1.8",
                },
                {
                    "sql_condition": "jaro_winkler_similarity(given_name_l, surname_r) + jaro_winkler_similarity(surname_l, given_name_r) >= 1.4",
                    "label_for_charts": "switched + jaro_winkler_similarity >= 1.4",
                },
                # single name match
                cll.ExactMatchLevel("given_name", term_frequency_adjustments=True),
                cll.ExactMatchLevel("surname", term_frequency_adjustments=True),
                # single name cross-match
                {
                    "sql_condition": "given_name_l = surname_r OR surname_l = given_name_r",
                    "label_for_charts": "single name cross-matches",
                },  # single name typos
                cll.JaroWinklerLevel("given_name", 0.9),
                cll.JaroWinklerLevel("surname", 0.9),
                # the rest
                cll.ElseLevel(),
            ],
        },
        cl.DateOfBirthComparison(
            "date_of_birth",
            input_is_string=True,
            datetime_format="%Y%m%d",
            invalid_dates_as_null=True,
        ),
        {
            "output_column_name": "Social security ID",
            "comparison_levels": [
                cll.NullLevel("soc_sec_id"),
                cll.ExactMatchLevel("soc_sec_id", term_frequency_adjustments=True),
                cll.DamerauLevenshteinLevel("soc_sec_id", 1),
                cll.DamerauLevenshteinLevel("soc_sec_id", 2),
                cll.ElseLevel(),
            ],
        },
        {
            "output_column_name": "Street number",
            "comparison_levels": [
                cll.NullLevel("street_number"),
                cll.ExactMatchLevel("street_number", term_frequency_adjustments=True),
                cll.DamerauLevenshteinLevel("street_number", 1),
                cll.ElseLevel(),
            ],
        },
        {
            "output_column_name": "Postcode",
            "comparison_levels": [
                cll.NullLevel("postcode"),
                cll.ExactMatchLevel("postcode", term_frequency_adjustments=True),
                cll.DamerauLevenshteinLevel("postcode", 1),
                cll.DamerauLevenshteinLevel("postcode", 2),
                cll.ElseLevel(),
            ],
        },
        # we don't consider further location columns as they will be strongly correlated with postcode
    ],
    "retain_intermediate_calculation_columns": True,
}

# train
linker_advanced = Linker(dfs, extended_model_settings, db_api=DuckDBAPI())
linker_advanced.training.estimate_probability_two_random_records_match(
    deterministic_rules, recall=0.8
)
# We recommend increasing target rows to 1e8 improve accuracy for u
# values in full name comparison, as we have subdivided the data more finely

# Here, 1e7 for speed
linker_advanced.training.estimate_u_using_random_sampling(max_pairs=1e7)

session_dob = (
    linker_advanced.training.estimate_parameters_using_expectation_maximisation(
        "l.date_of_birth = r.date_of_birth", estimate_without_term_frequencies=True
    )
)

session_pc = (
    linker_advanced.training.estimate_parameters_using_expectation_maximisation(
        "l.postcode = r.postcode", estimate_without_term_frequencies=True
    )
)

linker_advanced.visualisations.parameter_estimate_comparisons_chart()

linker_advanced.visualisations.match_weights_chart()

predictions_adv = linker_advanced.inference.predict()
df_predictions_adv = predictions_adv.as_pandas_dataframe()
clusters_adv = linker_advanced.clustering.cluster_pairwise_predictions_at_threshold(
    predictions_adv, threshold_match_probability=0.99
)
df_clusters_adv = clusters_adv.as_pandas_dataframe().sort_values("cluster_id")
df_clusters_adv.groupby("cluster_id").size().value_counts()