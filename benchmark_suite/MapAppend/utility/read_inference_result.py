import json


# Read inference result from JSON files


def read_inference_result_json(input_file, function_name):
    with open(input_file, "r") as read_file:
        inference_result_json = json.load(read_file)

    assert (inference_result_json["fid"] == function_name)
    inference_result_tc = inference_result_json["typing_context"]
    inference_result_rt = inference_result_json["return_type"]

    tc_const = inference_result_tc["([], [])"]
    tc_first_input = inference_result_tc["([::(*)], [])"]
    tc_second_input = inference_result_tc["([], [::(*)])"]
    rt_const = inference_result_rt["[]"]
    rt_degree_one = inference_result_rt["[::(*)]"]
    return tc_const, tc_first_input, tc_second_input, rt_const, rt_degree_one


def read_inference_result_annotated_code_json(file_path):
    with open(file_path, "r") as read_file:
        lp_vars_distribution_json = json.load(read_file)

    tc_const = lp_vars_distribution_json["13"]
    tc_first = lp_vars_distribution_json["12"]
    tc_second = lp_vars_distribution_json["11"]

    rt_const = lp_vars_distribution_json["9"]
    rt_first = lp_vars_distribution_json["4"]
    rt_second = lp_vars_distribution_json["5"]

    return tc_const, tc_first, tc_second, rt_const, rt_first, rt_second


# Split a list of posterior samples of cost bounds into two lists: (i) a list of
# input coefficients and (ii) a list of output coefficients


def decompose_posterior_distribution(posterior_distribution):
    list_tc_const, list_tc_first_input, list_tc_second_input, list_rt_const, list_rt_degree_one = posterior_distribution
    list_input_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_first_input = list_tc_first_input[i]
        coeff_tc_second_input = list_tc_second_input[i]
        list_input_coeffs.append(
            (coeff_tc_const, coeff_tc_first_input, coeff_tc_second_input))

    list_output_coeffs = []
    for i, coeff_rt_const in enumerate(list_rt_const):
        coeff_rt_degree_one = list_rt_degree_one[i]
        list_output_coeffs.append((coeff_rt_const, coeff_rt_degree_one))

    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients


def decompose_inferred_coefficients(inferred_coefficients):
    tc_const, tc_first_input, tc_second_input, rt_const, rt_one = inferred_coefficients
    input_coeffs = (tc_const, tc_first_input, tc_second_input)
    output_coeffs = (rt_const, rt_one)
    return input_coeffs, output_coeffs


# Split a list of posterior samples of cost bounds for the annotated code


def decompose_posterior_distribution_annotated_code(posterior_distribution):
    list_tc_const, list_tc_first, list_tc_second, \
        list_rt_const, list_rt_first, list_rt_second = posterior_distribution

    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_first = list_tc_first[i]
        coeff_tc_second = list_tc_second[i]
        list_input_coeffs.append(
            (coeff_tc_const, coeff_tc_first, coeff_tc_second))

        coeff_rt_const = list_rt_const[i]
        coeff_rt_first = list_rt_first[i]
        coeff_rt_second = list_rt_second[i]
        list_output_coeffs.append(
            (coeff_rt_const, coeff_rt_first, coeff_rt_second))
    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients for the annotated code


def decompose_inferred_coefficients_annotated_code(inferred_coefficients):
    tc_const, tc_first, tc_second, rt_const, rt_first, rt_second = inferred_coefficients
    input_coeffs = (tc_const, tc_first, tc_second)
    output_coeffs = (rt_const, rt_first, rt_second)
    return input_coeffs, output_coeffs
