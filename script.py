import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

df = splink_datasets.historical_50k

settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("postcode_fake", "first_name"),
        block_on("first_name", "surname"),
        block_on("dob", "substr(postcode_fake,1,2)"),
    ],
    comparisons=[
        cl.ExactMatch("first_name").configure(
            term_frequency_adjustments=True,
        ),
        cl.JaroWinklerAtThresholds("surname", score_threshold_or_thresholds=[0.9, 0.8]),
        cl.LevenshteinAtThresholds("postcode_fake"),
    ],
)

linker = Linker(df, settings, database_api=DuckDBAPI())

linker.training.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[
        block_on("first_name", "surname"),
        block_on("dob", "postcode_fake"),
    ],
    recall=0.7,
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")
)

linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))

df_predict = linker.inference.predict()

linker.clustering.cluster_pairwise_predictions_at_threshold(
    df_predict, threshold_match_probability=0.9
)
