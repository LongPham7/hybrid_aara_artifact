import json


def write_input_data_json(input_data, file_path):
    input_data_dictionary = {"input_data": input_data}
    with open(file_path, "w") as write_file:
        json.dump(input_data_dictionary, write_file)


def read_input_data_json(file_path):
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data["input_data"]


def read_analysis_time_json(file_path):
    with open(file_path, "r") as read_file:
        inference_result = json.load(read_file)
    return inference_result["analysis_time"]
