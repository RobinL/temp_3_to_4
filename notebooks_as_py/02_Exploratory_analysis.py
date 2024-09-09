from splink.datasets import splink_datasets
import altair as alt

df = splink_datasets.fake_1000
df.head(5)

# Initialise the linker, passing in the input dataset(s)
from splink.duckdb.linker import DuckDBLinker
linker = DuckDBLinker(df)

linker.missingness_chart()

linker.profile_columns(top_n=10, bottom_n=5)