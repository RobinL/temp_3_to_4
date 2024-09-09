# Uncomment and run this cell if you're running in Google Colab.
# !pip install ipywidgets
# !pip install splink
# !jupyter nbextension enable --py widgetsnbextension

import urllib.request
import json
from pathlib import Path
from splink import Linker, DuckDBAPI, block_on, SettingsCreator, splink_datasets

df = splink_datasets.fake_1000

url = "https://raw.githubusercontent.com/moj-analytical-services/splink_demos/master/demo_settings/real_time_settings.json"

with urllib.request.urlopen(url) as u:
    settings = json.loads(u.read().decode())


linker = Linker(df, settings, db_api=DuckDBAPI())

linker.visualisations.waterfall_chart(
    linker.inference.predict().as_record_dict(limit=2)
)

record_1 = {
    "unique_id": 1,
    "first_name": "Lucas",
    "surname": "Smith",
    "dob": "1984-01-02",
    "city": "London",
    "email": "lucas.smith@hotmail.com",
}

record_2 = {
    "unique_id": 2,
    "first_name": "Lucas",
    "surname": "Smith",
    "dob": "1983-02-12",
    "city": "Machester",
    "email": "lucas.smith@hotmail.com",
}

linker._settings_obj._retain_intermediate_calculation_columns = True


# To `compare_two_records` the linker needs to compute term frequency tables
# If you have precomputed tables, you can linker.register_term_frequency_lookup()
linker.table_management.compute_tf_table("first_name")
linker.table_management.compute_tf_table("surname")
linker.table_management.compute_tf_table("dob")
linker.table_management.compute_tf_table("city")
linker.table_management.compute_tf_table("email")


df_two = linker.inference.compare_two_records(record_1, record_2)
df_two.as_pandas_dataframe()

import ipywidgets as widgets
from IPython.display import display


fields = ["unique_id", "first_name", "surname", "dob", "email", "city"]

left_text_boxes = []
right_text_boxes = []

inputs_to_interactive_output = {}

for f in fields:
    wl = widgets.Text(description=f, value=str(record_1[f]))
    left_text_boxes.append(wl)
    inputs_to_interactive_output[f"{f}_l"] = wl
    wr = widgets.Text(description=f, value=str(record_2[f]))
    right_text_boxes.append(wr)
    inputs_to_interactive_output[f"{f}_r"] = wr

b1 = widgets.VBox(left_text_boxes)
b2 = widgets.VBox(right_text_boxes)
ui = widgets.HBox([b1, b2])


def myfn(**kwargs):
    my_args = dict(kwargs)

    record_left = {}
    record_right = {}

    for key, value in my_args.items():
        if value == "":
            value = None
        if key.endswith("_l"):
            record_left[key[:-2]] = value
        elif key.endswith("_r"):
            record_right[key[:-2]] = value

    # Assuming 'linker' is defined earlier in your code
    linker._settings_obj._retain_intermediate_calculation_columns = True

    df_two = linker.inference.compare_two_records(record_left, record_right)

    recs = df_two.as_pandas_dataframe().to_dict(orient="records")

    display(linker.visualisations.waterfall_chart(recs, filter_nulls=False))


out = widgets.interactive_output(myfn, inputs_to_interactive_output)

display(ui, out)

record = {
    "unique_id": 123987,
    "first_name": "Robert",
    "surname": "Alan",
    "dob": "1971-05-24",
    "city": "London",
    "email": "robert255@smith.net",
}


df_inc = linker.inference.find_matches_to_new_records(
    [record], blocking_rules=[]
).as_pandas_dataframe()
df_inc.sort_values("match_weight", ascending=False)

@widgets.interact(
    first_name="Robert",
    surname="Alan",
    dob="1971-05-24",
    city="London",
    email="robert255@smith.net",
)
def interactive_link(first_name, surname, dob, city, email):
    record = {
        "unique_id": 123987,
        "first_name": first_name,
        "surname": surname,
        "dob": dob,
        "city": city,
        "email": email,
        "group": 0,
    }

    for key in record.keys():
        if type(record[key]) == str:
            if record[key].strip() == "":
                record[key] = None

    df_inc = linker.inference.find_matches_to_new_records(
        [record], blocking_rules=[f"(true)"]
    ).as_pandas_dataframe()
    df_inc = df_inc.sort_values("match_weight", ascending=False)
    recs = df_inc.to_dict(orient="records")

    display(linker.visualisations.waterfall_chart(recs, filter_nulls=False))

linker.visualisations.match_weights_chart()