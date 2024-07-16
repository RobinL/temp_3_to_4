import splink.duckdb.comparison_library as cl
from splink.datasets import splink_datasets
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

df = splink_datasets.historical_50k

settings_dict = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on(["postcode_fake", "first_name"]),
        block_on(["first_name", "surname"]),
        block_on(["dob", "substr(postcode_fake,1,2)"]),
    ],
    "comparisons": [
        cl.exact_match(
            "first_name",
            term_frequency_adjustments=True,
        ),
        cl.jaro_winkler_at_thresholds(
            "surname",
            distance_threshold_or_thresholds=[0.9, 0.8],
        ),
        cl.levenshtein_at_thresholds("postcode_fake"),
        cl.levenshtein_at_thresholds("dob"),
    ],
}


linker = DuckDBLinker(df, settings_dict)

linker.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[
        block_on(["first_name", "surname"]),
        block_on(["dob", "postcode_fake"]),
    ],
    recall=0.7,
)

linker.estimate_u_using_random_sampling(max_pairs=1e6)

linker.estimate_parameters_using_expectation_maximisation(
    block_on(["first_name", "surname"])
)


df_predict = linker.predict()

linker.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=0.5
)
