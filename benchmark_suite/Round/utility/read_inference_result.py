import json


def write_input_data_json(input_data, file_path):
    input_data_dictionary = {"input_data": input_data}
    with open(file_path, "w") as write_file:
        json.dump(input_data_dictionary, write_file)


def read_input_data_json(file_path):
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    return data["input_data"]


def read_inference_result_json(input_file, function_name):
    with open(input_file, "r") as read_file:
        inference_result_json = json.load(read_file)

    assert (inference_result_json["fid"] == function_name)
    inference_result_tc = inference_result_json["typing_context"]
    inference_result_rt = inference_result_json["return_type"]

    tc_const = inference_result_tc["[]"]
    tc_one = inference_result_tc["[::(*)]"]

    rt_const = inference_result_rt["[]"]
    rt_one = inference_result_rt["[::(*)]"]
    return tc_const, tc_one, rt_const, rt_one


# Split a list of posterior samples of cost bounds into two lists: (i) a list of
# input coefficients and (ii) a list of output coefficients


def decompose_posterior_distribution(posterior_distribution):
    list_tc_const, list_tc_one, list_rt_const, list_rt_one = posterior_distribution

    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_one = list_tc_one[i]
        list_input_coeffs.append((coeff_tc_const, coeff_tc_one))

        coeff_rt_const = list_rt_const[i]
        coeff_rt_one = list_rt_one[i]
        list_output_coeffs.append((coeff_rt_const, coeff_rt_one))
    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients


def decompose_inferred_coefficients(inferred_coefficients):
    tc_const, tc_one, rt_const, rt_one = inferred_coefficients
    input_coeffs = (tc_const, tc_one)
    output_coeffs = (rt_const, rt_one)
    return input_coeffs, output_coeffs
