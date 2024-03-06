import json


def write_input_data_json(input_data, file_path):
    input_data_dictionary = {"input_data": input_data}
    with open(file_path, "w") as write_file:
        json.dump(input_data_dictionary, write_file)


def read_input_data_json(file_path):
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data["input_data"]


def write_execution_time_json(execution_time, config, file_path):
    data_analysis_mode = config["mode"]
    if data_analysis_mode == "opt":
        num_chains = 1
        num_samples_per_chain = 1
    elif data_analysis_mode == "bayeswc":
        num_chains = config["stan_params"]["num_chains"]
        num_samples_per_chain = config["stan_params"]["num_stan_samples"]
    else:
        num_chains = 1
        num_samples_per_chain = config["hmc_params"]["num_samples"]

    execution_time_dict = {
        "execution_time": execution_time,
        "num_chains": num_chains,
        "num_samples_per_chain": num_samples_per_chain
    }
    with open(file_path, "w") as write_file:
        json.dump(execution_time_dict, write_file)


def read_execution_time_json(file_path):
    with open(file_path, "r") as read_file:
        execution_time_dict = json.load(read_file)
    return execution_time_dict
