from splink.spark.jar_location import similarity_jar_location

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import types

conf = SparkConf()
# This parallelism setting is only suitable for a small toy example
conf.set("spark.driver.memory", "12g")
conf.set("spark.default.parallelism", "16")


# Add custom similarity functions, which are bundled with Splink
# documented here: https://github.com/moj-analytical-services/splink_scalaudfs
path = similarity_jar_location()
conf.set("spark.jars", path)

sc = SparkContext.getOrCreate(conf=conf)

spark = SparkSession(sc)
spark.sparkContext.setCheckpointDir("./tmp_checkpoints")

# Disable warnings for pyspark - you don't need to include this
import warnings
spark.sparkContext.setLogLevel("ERROR")
warnings.simplefilter("ignore", UserWarning)

from splink.datasets import splink_datasets
pandas_df = splink_datasets.fake_1000

df = spark.createDataFrame(pandas_df)

import splink.spark.comparison_library as cl
import splink.spark.comparison_template_library as ctl
from splink.spark.blocking_rule_library import block_on

settings = {
    "link_type": "dedupe_only",
    "comparisons": [
        ctl.name_comparison("first_name"),
        ctl.name_comparison("surname"),
        ctl.date_comparison("dob", cast_strings_to_date=True),
        cl.exact_match("city", term_frequency_adjustments=True),
        ctl.email_comparison("email", include_username_fuzzy_level=False),
    ],
    "blocking_rules_to_generate_predictions": [
        block_on("first_name"),
        "l.surname = r.surname",  # alternatively, you can write BRs in their SQL form
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    "em_convergence": 0.01
}

from splink.spark.linker import SparkLinker
linker = SparkLinker(df, settings, spark=spark)
deterministic_rules = [
    "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
    "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
    "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
    "l.email = r.email"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.6)


linker.estimate_u_using_random_sampling(max_pairs=5e5)

training_blocking_rule = "l.first_name = r.first_name and l.surname = r.surname"
training_session_fname_sname = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

training_blocking_rule = "l.dob = r.dob"
training_session_dob = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

results = linker.predict(threshold_match_probability=0.9)

results.as_pandas_dataframe(limit=5)