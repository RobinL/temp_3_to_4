import json
from splink.datasets import splink_datasets
from splink.duckdb.linker import DuckDBLinker
import altair as alt
alt.renderers.enable('html')

with open("../../demo_settings/real_time_settings.json") as f:
    trained_settings = json.load(f)

df = splink_datasets.fake_1000

linker = DuckDBLinker(df, trained_settings)

linker.waterfall_chart(linker.predict().as_record_dict(limit=2))

from splink.term_frequencies import compute_term_frequencies_from_concat_with_tf
record_1  = {
     'unique_id':1,
     'first_name': "Lucas",
     'surname': "Smith",
     'dob': "1984-01-02",
     'city': "London",
     'email': "lucas.smith@hotmail.com"
}

record_2  = {
     'unique_id':2,
     'first_name': "Lucas",
     'surname': "Smith",
     'dob': "1983-02-12",
     'city': "Machester",
     'email': "lucas.smith@hotmail.com"
}

linker._settings_obj_._retain_intermediate_calculation_columns = True
linker._settings_obj_._retain_matching_columns = True

linker.compute_tf_table("first_name")
linker.compute_tf_table("surname")
linker.compute_tf_table("dob")
linker.compute_tf_table("city")
linker.compute_tf_table("email")


df_two = linker.compare_two_records(record_1, record_2)
df_two.as_pandas_dataframe()

import ipywidgets as widgets
fields = ["unique_id", "first_name","surname","dob","email","city"]

left_text_boxes = []
right_text_boxes = []

inputs_to_interactive_output = {}

for f in fields:
    wl = widgets.Text(description=f, value =str(record_1[f]))
    left_text_boxes.append(wl)
    inputs_to_interactive_output[f"{f}_l"] = wl
    wr = widgets.Text( description=f, value =str(record_2[f]))
    right_text_boxes.append(wr)
    inputs_to_interactive_output[f"{f}_r"] = wr


b1 = widgets.VBox(left_text_boxes)
b2 = widgets.VBox(right_text_boxes)
ui = widgets.HBox([b1,b2])

def myfn(**kwargs):
    my_args = dict(kwargs)
    
    record_left = {}
    record_right = {}
    
    for key, value in my_args.items():
        if value == '':
            value = None
        if key.endswith("_l"):
            record_left[key[:-2]] = value
        if key.endswith("_r"):
            record_right[key[:-2]] = value
            

    linker._settings_obj_._retain_intermediate_calculation_columns = True
    linker._settings_obj_._retain_matching_columns = True

    df_two = linker.compare_two_records(record_left, record_right)

    recs = df_two.as_pandas_dataframe().to_dict(orient="records")
    from splink.charts import waterfall_chart
    display(linker.waterfall_chart(recs, filter_nulls=False))


out = widgets.interactive_output(myfn, inputs_to_interactive_output)

display(ui,out)


record = {'unique_id': 123987,
 'first_name': "Robert",
 'surname': "Alan",
 'dob': "1971-05-24",
 'city': "London",
 'email': "robert255@smith.net"
}



df_inc = linker.find_matches_to_new_records([record], blocking_rules=[]).as_pandas_dataframe()
df_inc.sort_values("match_weight", ascending=False)

from splink.charts import waterfall_chart

@widgets.interact(first_name='Robert', surname="Alan", dob="1971-05-24", city="London", email="robert255@smith.net")
def interactive_link(first_name, surname, dob, city, email):    

    record = {'unique_id': 123987,
     'first_name': first_name,
     'surname': surname,
     'dob': dob,
     'city': city,
     'email': email,
     'group': 0}

    for key in record.keys():
        if type(record[key]) == str:
            if record[key].strip() == "":
                record[key] = None

    
    df_inc = linker.find_matches_to_new_records([record], blocking_rules=[f"(true)"]).as_pandas_dataframe()
    df_inc = df_inc.sort_values("match_weight", ascending=False)
    recs = df_inc.to_dict(orient="records")
    


    display(linker.waterfall_chart(recs, filter_nulls=False))


linker.match_weights_chart()