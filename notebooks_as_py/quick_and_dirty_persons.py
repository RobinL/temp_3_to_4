from splink.datasets import splink_datasets
df = splink_datasets.historical_50k
df.head(5)

from splink.duckdb.linker import DuckDBLinker
from splink.duckdb.blocking_rule_library import block_on
import splink.duckdb.comparison_library as cl

settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on("full_name"),
        block_on(["substr(full_name,1,6)", "dob", "birth_place"]),
        block_on(["dob", "birth_place"]),
        block_on("postcode_fake"),
    ],
    "comparisons": [
        cl.jaro_at_thresholds("full_name", [0.9, 0.7], term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("dob", [1, 2]),
        cl.levenshtein_at_thresholds("postcode_fake", 2),
        cl.jaro_winkler_at_thresholds("birth_place", 0.9, term_frequency_adjustments=True),
        cl.exact_match("occupation",  term_frequency_adjustments=True),
    ],       
    
}

linker = DuckDBLinker(df, settings, set_up_basic_logging=False)
deterministic_rules = [
    "l.full_name = r.full_name",
    "l.postcode_fake = r.postcode_fake and l.dob = r.dob",
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.6)


linker.estimate_u_using_random_sampling(max_pairs=2e6)

results = linker.predict(threshold_match_probability=0.9)

results.as_pandas_dataframe(limit=5)