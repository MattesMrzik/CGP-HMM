#!/usr/bin/env python3
import json
import re
import os
import pandas as pd
# from My_internal_exon_model import My_Model
'''
if this is really fast,
ie prepare model is fast, then
it should also be fast to add description
ie read json file and export csv
then i dont need to write matrices with desciption in main_programm,
and i can just use this method here,
i could even pass the model

also add timer to this method
'''
# def add_state_and_emission_str_to_matrices(model : My_Model, path_to_json_file : str) -> None:
def add_description_to_A(model, path_to_json_file : str) -> None:
    assert path_to_json_file.endswith(".json"), "the path passed to add_state_and_emission_str_to_matrices() doesnt end in .json"
    df = pd.read_json(path_to_json_file)
    states_strs = [model.state_id_to_str(i) for i in range(model.number_of_states)]
    df.columns = states_strs
    df["from_state"] = states_strs
    df = df.set_index("from_state")
    out_file_path = re.sub(".json","_description.csv", path_to_json_file)
    df.to_csv(out_file_path, sep = ";", header = True, index = True)

def add_description_to_B(model, path_to_json_file : str) -> None:
    assert path_to_json_file.endswith(".json"), "the path passed to add_state_and_emission_str_to_matrices() doesnt end in .json"
    out_file_path = re.sub(".json","_with_description.csv", path_to_json_file)

    df = pd.read_json(path_to_json_file)
    states_strs = [model.state_id_to_str(i) for i in range(model.number_of_states)]
    df.columns = states_strs

    emissions_strs = [model.emission_id_to_str(i) for i in range(model.number_of_emissions)]
    emissions_strs[-3:] = ["dummy1", "dummy2", "dummy3"]
    df["emissions_strs"] = emissions_strs
    df = df.set_index("emissions_strs")
    df.to_csv(out_file_path, sep = ";", header = True, index = True)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from Config import Config
    config = Config()
    config.init_for_add_str_to_matrices()

    for dirpath, dirnames, filenames in os.walk(config.parent_input_dir):
        for file in filenames:
            if file == "A.json":
                add_description_to_A(config.model, os.path.join(dirpath, file))
            if file == "B.json":
                add_description_to_B(config.model, os.path.join(dirpath, file))

