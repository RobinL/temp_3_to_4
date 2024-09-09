# Uncomment and run this cell if you're running in Google Colab.
# !pip install splink

from splink import  splink_datasets

df = splink_datasets.fake_1000
df = df.drop(columns=["cluster"])
df.head(5)

from splink.exploratory import completeness_chart
from splink import DuckDBAPI
db_api = DuckDBAPI()
completeness_chart(df, db_api=db_api)

from splink.exploratory import profile_columns

profile_columns(df, db_api=DuckDBAPI(), top_n=10, bottom_n=5)