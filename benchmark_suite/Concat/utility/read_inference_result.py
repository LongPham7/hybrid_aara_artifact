import json


def read_inference_result_json(input_file, function_name):
    with open(input_file, "r") as read_file:
        inference_result_json = json.load(read_file)

    assert (inference_result_json["fid"] == function_name)
    inference_result_tc = inference_result_json["typing_context"]
    inference_result_rt = inference_result_json["return_type"]

    tc_const = inference_result_tc["[]"]
    tc_outer = inference_result_tc["[::([])]"]
    tc_inner_sum = inference_result_tc["[::([::(*)])]"]
    tc_outer_squared = inference_result_tc["[::([]); ::([])]"]

    rt_const = inference_result_rt["[]"]
    rt_one = inference_result_rt["[::(*)]"]
    rt_two = inference_result_rt["[::(*); ::(*)]"]
    return tc_const, tc_outer, tc_inner_sum, tc_outer_squared, rt_const, rt_one, rt_two


def read_inference_result_annotated_code_json(input_file):
    with open(input_file, "r") as f:
        distribution_string = f.read()
    lp_vars_distribution_json = json.loads(distribution_string)

    tc_const = lp_vars_distribution_json["15"]
    tc_one_first_input = lp_vars_distribution_json["14"]
    tc_one_second_input = lp_vars_distribution_json["13"]
    tc_two_first_input = lp_vars_distribution_json["12"]
    tc_two_second_input = lp_vars_distribution_json["10"]
    tc_two_first_and_second = lp_vars_distribution_json["11"]

    rt_const = lp_vars_distribution_json["5"]
    rt_one = lp_vars_distribution_json["4"]
    rt_two = lp_vars_distribution_json["3"]
    return tc_const, tc_one_first_input, tc_one_second_input, tc_two_first_input, \
        tc_two_second_input, tc_two_first_and_second, rt_const, rt_one, rt_two


# Split a list of posterior samples of cost bounds into two lists: (i) a list of
# input coefficients and (ii) a list of output coefficients


def decompose_posterior_distribution(posterior_distribution):
    list_tc_const, list_tc_outer, list_tc_inner_sum, list_tc_outer_squared, \
        list_rt_const, list_rt_one, list_rt_two = posterior_distribution
    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_outer = list_tc_outer[i]
        coeff_tc_inner_sum = list_tc_inner_sum[i]
        coeff_tc_outer_squared = list_tc_outer_squared[i]
        list_input_coeffs.append(
            (coeff_tc_const, coeff_tc_outer, coeff_tc_inner_sum, coeff_tc_outer_squared))

        coeff_rt_const = list_rt_const[i]
        coeff_rt_one = list_rt_one[i]
        coeff_rt_two = list_rt_two[i]
        list_output_coeffs.append((coeff_rt_const, coeff_rt_one, coeff_rt_two))

    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients


def decompose_inferred_coefficients(inferred_coefficients):
    tc_const, tc_outer, tc_inner_sum, tc_outer_squared, \
        rt_const, rt_one, rt_two = inferred_coefficients
    input_coeffs = (tc_const, tc_outer, tc_inner_sum, tc_outer_squared)
    output_coeffs = (rt_const, rt_one, rt_two)
    return input_coeffs, output_coeffs


# Split a list of posterior samples of cost bounds for the annotated code


def decompose_posterior_distribution_annotated_code(posterior_distribution):
    list_tc_const, list_tc_one_first_input, list_tc_one_second_input, list_tc_two_first_input, \
        list_tc_two_second_input, list_tc_two_first_and_second, list_rt_const, list_rt_one, list_rt_two = posterior_distribution

    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_one_first_input = list_tc_one_first_input[i]
        coeff_tc_one_second_input = list_tc_one_second_input[i]
        coeff_tc_two_first_input = list_tc_two_first_input[i]
        coeff_tc_two_second_input = list_tc_two_second_input[i]
        coeff_tc_two_first_and_second = list_tc_two_first_and_second[i]
        list_input_coeffs.append(
            (coeff_tc_const, coeff_tc_one_first_input, coeff_tc_one_second_input, coeff_tc_two_first_input, coeff_tc_two_second_input, coeff_tc_two_first_and_second))

        coeff_rt_const = list_rt_const[i]
        coeff_rt_one = list_rt_one[i]
        coeff_rt_two = list_rt_two[i]
        list_output_coeffs.append((coeff_rt_const, coeff_rt_one, coeff_rt_two))
    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients for the annotated code


def decompose_inferred_coefficients_annotated_code(inferred_coefficients):
    coeff_tc_const, coeff_tc_one_first_input, coeff_tc_one_second_input, coeff_tc_two_first_input, \
        coeff_tc_two_second_input, coeff_tc_two_first_and_second, coeff_rt_const, coeff_rt_one, coeff_rt_two = inferred_coefficients
    input_coeffs = (coeff_tc_const, coeff_tc_one_first_input, coeff_tc_one_second_input,
                    coeff_tc_two_first_input, coeff_tc_two_second_input, coeff_tc_two_first_and_second)
    output_coeffs = (coeff_rt_const, coeff_rt_one, coeff_rt_two)
    return input_coeffs, output_coeffs
