from splink.datasets import splink_datasets
from splink.duckdb.linker import DuckDBLinker
import altair as alt

df_origin = splink_datasets.transactions_origin
df_destination = splink_datasets.transactions_destination

display(df_origin.head(2))
display(df_destination.head(2))

# Simple settings just for exploratory analysis
settings = {"link_type": "link_only"}
linker = DuckDBLinker([df_origin, df_destination], settings,input_table_aliases=["__ori", "_dest"])
linker.profile_columns(["transaction_date", "memo", "round(amount/5, 0)*5"])

# Design blocking rules that allow for differences in transaction date and amounts
from splink.duckdb.blocking_rule_library import block_on, and_

blocking_rule_date_1 = """
    strftime(l.transaction_date, '%Y%m') = strftime(r.transaction_date, '%Y%m')
    and substr(l.memo, 1,3) = substr(r.memo,1,3)
    and l.amount/r.amount > 0.7   and l.amount/r.amount < 1.3
"""

# Offset by half a month to ensure we capture case when the dates are e.g. 31st Jan and 1st Feb
blocking_rule_date_2 = """
    strftime(l.transaction_date+15, '%Y%m') = strftime(r.transaction_date, '%Y%m')
    and substr(l.memo, 1,3) = substr(r.memo,1,3)
    and l.amount/r.amount > 0.7   and l.amount/r.amount < 1.3
"""

blocking_rule_memo = block_on("substr(memo,1,9)")

blocking_rule_amount_1 = """
round(l.amount/2,0)*2 = round(r.amount/2,0)*2 and yearweek(r.transaction_date) = yearweek(l.transaction_date)
"""

blocking_rule_amount_2 = """
round(l.amount/2,0)*2 = round((r.amount+1)/2,0)*2 and yearweek(r.transaction_date) = yearweek(l.transaction_date + 4)
"""

blocking_rule_cheat = block_on("unique_id")


linker.cumulative_num_comparisons_from_blocking_rules_chart(
    [
        blocking_rule_date_1,
        blocking_rule_date_2,
        blocking_rule_memo,
        blocking_rule_amount_1,
        blocking_rule_amount_2,
        blocking_rule_cheat,
    ]
)

# Full settings for linking model
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_level_library as cll

comparison_amount = {
    "output_column_name": "amount",
    "comparison_levels": [
        cll.null_level("amount"),
        cll.exact_match_level("amount"),
        cll.percentage_difference_level("amount",0.01),
        cll.percentage_difference_level("amount",0.03),
        cll.percentage_difference_level("amount",0.1),
        cll.percentage_difference_level("amount",0.3),
        cll.else_level()
    ],
    "comparison_description": "Amount percentage difference",
}

settings = {
    "link_type": "link_only",
    "probability_two_random_records_match": 1 / len(df_origin),
    "blocking_rules_to_generate_predictions": [
        blocking_rule_date_1,
        blocking_rule_date_2,
        blocking_rule_memo,
        blocking_rule_amount_1,
        blocking_rule_amount_2,
        blocking_rule_cheat
    ],
    "comparisons": [
        comparison_amount,
        cl.jaccard_at_thresholds(
            "memo", [0.9, 0.7]
        ),
        cl.datediff_at_thresholds("transaction_date", 
                                date_thresholds = [1, 4, 10, 30],
                                date_metrics = ["day", "day", "day", "day"],
                                include_exact_match_level=False
                                )
    ],
    "retain_intermediate_calculation_columns": True,
    "retain_matching_columns": True,
}

linker = DuckDBLinker([df_origin, df_destination], settings,input_table_aliases=["__ori", "_dest"])

linker.estimate_u_using_random_sampling(max_pairs=1e6)

linker.estimate_parameters_using_expectation_maximisation(block_on("memo"))

session = linker.estimate_parameters_using_expectation_maximisation(block_on("amount"))

linker.match_weights_chart()

df_predict = linker.predict(threshold_match_probability=0.001)

linker.comparison_viewer_dashboard(df_predict,"dashboards/comparison_viewer_transactions.html", overwrite=True)
from IPython.display import IFrame
IFrame(
    src="./dashboards/comparison_viewer_transactions.html", width="100%", height=1200
)

pred_errors =  linker.prediction_errors_from_labels_column("ground_truth", include_false_positives=True, include_false_negatives=False)
linker.waterfall_chart(pred_errors.as_record_dict(limit=5))

pred_errors =  linker.prediction_errors_from_labels_column("ground_truth", include_false_positives=False, include_false_negatives=True)
linker.waterfall_chart(pred_errors.as_record_dict(limit=5))