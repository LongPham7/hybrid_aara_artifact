import json


def read_inference_result_json(input_file, function_name):
    with open(input_file, "r") as read_file:
        inference_result_json = json.load(read_file)

    assert (inference_result_json["fid"] == function_name)
    inference_result_tc = inference_result_json["typing_context"]
    inference_result_rt = inference_result_json["return_type"]

    tc_const = inference_result_tc["[]"]
    tc_one = inference_result_tc["[::(*)]"]
    tc_two = inference_result_tc["[::(*); ::(*)]"]

    rt_const = inference_result_rt["[]"]
    rt_one = inference_result_rt["[::(*)]"]
    rt_two = inference_result_rt["[::(*); ::(*)]"]
    return tc_const, tc_one, tc_two, rt_const, rt_one, rt_two


def read_inference_result_annotated_code_json(file_path):
    with open(file_path, "r") as read_file:
        lp_vars_distribution_json = json.load(read_file)

    tc_const = lp_vars_distribution_json["261"]
    tc_one = lp_vars_distribution_json["260"]
    tc_two = lp_vars_distribution_json["259"]

    rt_const = lp_vars_distribution_json["257"]
    rt_one_first = lp_vars_distribution_json["255"]
    rt_one_second = lp_vars_distribution_json["207"]
    rt_two_first = lp_vars_distribution_json["6"]
    rt_two_second = lp_vars_distribution_json["6"]
    rt_two_first_and_second = lp_vars_distribution_json["169"]

    return tc_const, tc_one, tc_two, \
        rt_const, rt_one_first, rt_one_second, \
        rt_two_first, rt_two_second, rt_two_first_and_second


# Split a list of posterior samples of cost bounds into two lists: (i) a list of
# input coefficients and (ii) a list of output coefficients


def decompose_posterior_distribution(posterior_distribution):
    list_tc_const, list_tc_one, list_tc_two, list_rt_const, list_rt_one, list_rt_two = posterior_distribution

    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_one = list_tc_one[i]
        coeff_tc_two = list_tc_two[i]
        list_input_coeffs.append((coeff_tc_const, coeff_tc_one, coeff_tc_two))

        coeff_rt_const = list_rt_const[i]
        coeff_rt_one = list_rt_one[i]
        coeff_rt_two = list_rt_two[i]
        list_output_coeffs.append((coeff_rt_const, coeff_rt_one, coeff_rt_two))
    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients


def decompose_inferred_coefficients(inferred_coefficients):
    tc_const, tc_one, tc_two, rt_const, rt_one, rt_two = inferred_coefficients
    input_coeffs = (tc_const, tc_one, tc_two)
    output_coeffs = (rt_const, rt_one, rt_two)
    return input_coeffs, output_coeffs


# Split a list of posterior samples of cost bounds for the annotated code


def decompose_posterior_distribution_annotated_code(posterior_distribution):
    list_tc_const, list_tc_one, list_tc_two, \
        list_rt_const, list_rt_one_first, list_rt_one_second, \
        list_rt_two_first, list_rt_two_second, list_rt_two_first_and_second = posterior_distribution

    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_one = list_tc_one[i]
        coeff_tc_two = list_tc_two[i]

        list_input_coeffs.append((coeff_tc_const, coeff_tc_one, coeff_tc_two))

        coeff_rt_const = list_rt_const[i]
        coeff_rt_one_first = list_rt_one_first[i]
        coeff_rt_one_second = list_rt_one_second[i]
        coeff_rt_two_first = list_rt_two_first[i]
        coeff_rt_two_second = list_rt_two_second[i]
        coeff_rt_two_first_and_second = list_rt_two_first_and_second[i]
        list_output_coeffs.append((coeff_rt_const, coeff_rt_one_first, coeff_rt_one_second,
                                  coeff_rt_two_first, coeff_rt_two_second, coeff_rt_two_first_and_second))
    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients for the annotated code


def decompose_inferred_coefficients_annotated_code(inferred_coefficients):
    tc_const, tc_one, tc_two, \
        rt_const, rt_one_first, rt_one_second, \
        rt_two_first, rt_two_second, rt_two_first_and_second = inferred_coefficients
    input_coeffs = (tc_const, tc_one, tc_two)
    output_coeffs = (rt_const, rt_one_first, rt_one_second,
                     rt_two_first, rt_two_second, rt_two_first_and_second)
    return input_coeffs, output_coeffs
