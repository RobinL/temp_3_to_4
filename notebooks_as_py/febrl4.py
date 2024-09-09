import pandas as pd
import altair as alt
from splink.datasets import splink_datasets
from IPython.display import IFrame
alt.renderers.enable('html')

df_a = splink_datasets.febrl4a
df_b = splink_datasets.febrl4b

def prepare_data(data):
    data = data.rename(columns=lambda x: x.strip())
    data["cluster"] = data["rec_id"].apply(lambda x: "-".join(x.split('-')[:2]))
    data["date_of_birth"] = data["date_of_birth"].astype(str).str.strip()
    data["date_of_birth"] = data["date_of_birth"].replace("", None)

    data["soc_sec_id"] = data["soc_sec_id"].astype(str).str.strip()
    data["soc_sec_id"] = data["soc_sec_id"].replace("", None)

    data["postcode"] = data["postcode"].astype(str).str.strip()
    data["postcode"] = data["postcode"].replace("", None)
    return data
dfs = [
    prepare_data(dataset)
    for dataset in [df_a, df_b]
]

display(dfs[0].head(2))
display(dfs[1].head(2))

from splink.duckdb.linker import DuckDBLinker

basic_settings = {
    "unique_id_column_name": "rec_id",
    "link_type": "link_only",
    # NB as we are linking one-one, we know the probability that a random pair will be a match
    # hence we could set:
    # "probability_two_random_records_match": 1/5000,
    # however we will not specify this here, as we will use this as a check that
    # our estimation procedure returns something sensible
}

linker = DuckDBLinker(dfs, basic_settings)

linker.missingness_chart()

cols_to_profile = list(dfs[0].columns)
cols_to_profile = [col for col in cols_to_profile if col not in ("rec_id", "cluster")]
linker.profile_columns(cols_to_profile)

blocking_rules = [
    "l.given_name = r.given_name AND l.surname = r.surname",
    "l.date_of_birth = r.date_of_birth",
    "l.soc_sec_id = r.soc_sec_id",
    "l.state = r.state AND l.address_1 = r.address_1",
    "l.street_number = r.street_number AND l.address_1 = r.address_1",
    "l.postcode = r.postcode",
]
linker.cumulative_num_comparisons_from_blocking_rules_chart(blocking_rules)

import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl
import splink.duckdb.comparison_level_library as cll

# the simple model only considers a few columns, and only two comparison levels for each
simple_model_settings = {
    **basic_settings,
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        cl.exact_match("given_name", term_frequency_adjustments=True),
        cl.exact_match("surname", term_frequency_adjustments=True),
        cl.exact_match("street_number", term_frequency_adjustments=True),
    ],
    "retain_intermediate_calculation_columns": True,
}
# the detailed model considers more columns, using the information we saw in the exploratory phase
# we also include further comparison levels to account for typos and other differences
detailed_model_settings = {
    **basic_settings,
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        ctl.name_comparison("given_name", term_frequency_adjustments=True),
        ctl.name_comparison("surname", term_frequency_adjustments=True),
        ctl.date_comparison("date_of_birth", 
                            damerau_levenshtein_thresholds=[],
                            cast_strings_to_date=True,
                            invalid_dates_as_null=True,
                            date_format="%Y%m%d"),
        cl.damerau_levenshtein_at_thresholds("soc_sec_id", [1, 2]),
        cl.exact_match("street_number", term_frequency_adjustments=True),
        cl.damerau_levenshtein_at_thresholds("postcode", [1, 2], term_frequency_adjustments=True),
        # we don't consider further location columns as they will be strongly correlated with postcode
    ],
    "retain_intermediate_calculation_columns": True,
}


linker_simple = DuckDBLinker(dfs, simple_model_settings)
linker_detailed = DuckDBLinker(dfs, detailed_model_settings)

deterministic_rules = [
    "l.soc_sec_id = r.soc_sec_id",
    "l.given_name = r.given_name and l.surname = r.surname and l.date_of_birth = r.date_of_birth",
]

linker_detailed.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)

linker_detailed.estimate_u_using_random_sampling(max_pairs=1e7)

session_dob = linker_detailed.estimate_parameters_using_expectation_maximisation(
    "l.date_of_birth = r.date_of_birth"
)
session_pc = linker_detailed.estimate_parameters_using_expectation_maximisation(
    "l.postcode = r.postcode"
)

session_dob.m_u_values_interactive_history_chart()

linker_detailed.parameter_estimate_comparisons_chart()

linker_simple.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)
linker_simple.estimate_u_using_random_sampling(max_pairs=1e7)
session_ssid = linker_simple.estimate_parameters_using_expectation_maximisation(
    "l.given_name = r.given_name"
)
session_pc = linker_simple.estimate_parameters_using_expectation_maximisation(
    "l.street_number = r.street_number"
)
linker_simple.parameter_estimate_comparisons_chart()

# import json
# we can have a look at the full settings if we wish, including the values of our estimated parameters:
# print(json.dumps(linker_detailed._settings_obj.as_dict(), indent=2))
# we can also get a handy summary of of the model in an easily readable format if we wish:
# print(linker_detailed._settings_obj.human_readable_description)
# (we suppress output here for brevity)

linker_simple.match_weights_chart()

linker_detailed.match_weights_chart()

# linker_simple.m_u_parameters_chart()
linker_detailed.m_u_parameters_chart()

linker_simple.unlinkables_chart()

linker_detailed.unlinkables_chart()

predictions = linker_detailed.predict()
df_predictions = predictions.as_pandas_dataframe()
df_predictions.head(5)

# linker_detailed.roc_chart_from_labels_column("cluster")
linker_detailed.precision_recall_chart_from_labels_column("cluster")

clusters = linker_detailed.cluster_pairwise_predictions_at_threshold(predictions, threshold_match_probability=0.99)
df_clusters = clusters.as_pandas_dataframe().sort_values("cluster_id")
df_clusters.groupby("cluster_id").size().value_counts()

df_predictions["cluster_l"] = df_predictions["rec_id_l"].apply(lambda x: "-".join(x.split('-')[:2]))
df_predictions["cluster_r"] = df_predictions["rec_id_r"].apply(lambda x: "-".join(x.split('-')[:2]))
df_true_links = df_predictions[df_predictions["cluster_l"] == df_predictions["cluster_r"]].sort_values("match_probability")

records_to_view = 3
linker_detailed.waterfall_chart(df_true_links.head(records_to_view).to_dict(orient="records"))

df_non_links = df_predictions[df_predictions["cluster_l"] != df_predictions["cluster_r"]].sort_values("match_probability", ascending=False)
linker_detailed.waterfall_chart(df_non_links.head(records_to_view).to_dict(orient="records"))

# we need to append a full name column to our source data frames
# so that we can use it for term frequency adjustments
dfs[0]["full_name"] = dfs[0]["given_name"] + "_" + dfs[0]["surname"]
dfs[1]["full_name"] = dfs[1]["given_name"] + "_" + dfs[1]["surname"]


extended_model_settings = {
    **basic_settings,
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
                cll.exact_match_level("full_name", term_frequency_adjustments=True),
                # typos - keep levels across full name rather than scoring separately
                cll.jaro_winkler_level("full_name", 0.9),
                cll.jaro_winkler_level("full_name", 0.7),
                # name switched
                cll.columns_reversed_level("given_name", "surname"),
                # name switched + typo
                {
                    "sql_condition": "jaro_winkler_similarity(given_name_l, surname_r) + jaro_winkler_similarity(surname_l, given_name_r) >= 1.8",
                    "label_for_charts": "switched + jaro_winkler_similarity >= 1.8"
                },
                {
                    "sql_condition": "jaro_winkler_similarity(given_name_l, surname_r) + jaro_winkler_similarity(surname_l, given_name_r) >= 1.4",
                    "label_for_charts": "switched + jaro_winkler_similarity >= 1.4"
                },
                # single name match
                cll.exact_match_level("given_name", term_frequency_adjustments=True),
                cll.exact_match_level("surname", term_frequency_adjustments=True),
                # single name cross-match
                {
                    "sql_condition": "given_name_l = surname_r OR surname_l = given_name_r",
                    "label_for_charts": "single name cross-matches"
                },                # single name typos
                cll.jaro_winkler_level("given_name", 0.9),
                cll.jaro_winkler_level("surname", 0.9),
                # the rest
                cll.else_level()
            ]
        },
        ctl.date_comparison("date_of_birth", 
                            damerau_levenshtein_thresholds=[],
                            cast_strings_to_date=True,
                            invalid_dates_as_null=True,
                            date_format="%Y%m%d"),
        {
            "output_column_name": "Social security ID",
            "comparison_levels": [
                cll.null_level("soc_sec_id"),
                cll.exact_match_level("soc_sec_id", term_frequency_adjustments=True),
                cll.damerau_levenshtein_level("soc_sec_id", 1),
                cll.damerau_levenshtein_level("soc_sec_id", 2),
                cll.else_level()
            ]
        },
        {
            "output_column_name": "Street number",
            "comparison_levels": [
                cll.null_level("street_number"),
                cll.exact_match_level("street_number", term_frequency_adjustments=True),
                cll.damerau_levenshtein_level("street_number", 1),
                cll.else_level()
            ]
        },
        {
            "output_column_name": "Postcode",
            "comparison_levels": [
                cll.null_level("postcode"),
                cll.exact_match_level("postcode", term_frequency_adjustments=True),
                cll.damerau_levenshtein_level("postcode", 1),
                cll.damerau_levenshtein_level("postcode", 2),
                cll.else_level()
            ]
        },
        # we don't consider further location columns as they will be strongly correlated with postcode
    ],
    "retain_intermediate_calculation_columns": True,
}

# train
linker_advanced = DuckDBLinker(dfs, extended_model_settings)
linker_advanced.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)
# we increase target rows to improve accuracy for u values in full name comparison, as we have subdivided the data more finely
linker_advanced.estimate_u_using_random_sampling(max_pairs=1e8)
session_dob = linker_advanced.estimate_parameters_using_expectation_maximisation(
    "l.date_of_birth = r.date_of_birth"
)


session_pc = linker_advanced.estimate_parameters_using_expectation_maximisation(
    "l.postcode = r.postcode"
)

linker_advanced.parameter_estimate_comparisons_chart()

linker_advanced.match_weights_chart()

predictions_adv = linker_advanced.predict()
df_predictions_adv = predictions_adv.as_pandas_dataframe()
clusters_adv = linker_advanced.cluster_pairwise_predictions_at_threshold(predictions_adv, threshold_match_probability=0.99)
df_clusters_adv = clusters_adv.as_pandas_dataframe().sort_values("cluster_id")
df_clusters_adv.groupby("cluster_id").size().value_counts()