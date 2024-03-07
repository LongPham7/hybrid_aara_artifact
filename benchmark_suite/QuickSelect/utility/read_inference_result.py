import json


def read_inference_result_json(input_file, function_name):
    with open(input_file, "r") as read_file:
        inference_result_json = json.load(read_file)

    assert (inference_result_json["fid"] == function_name)
    inference_result_tc = inference_result_json["typing_context"]
    inference_result_rt = inference_result_json["return_type"]

    tc_const = inference_result_tc["(*, [])"]
    tc_one = inference_result_tc["(*, [::(*)])"]
    tc_two = inference_result_tc["(*, [::(*); ::(*)])"]

    rt_const = inference_result_rt["*"]
    return tc_const, tc_one, tc_two, rt_const


def read_inference_result_annotated_code_json(input_file):
    with open(input_file, "r") as read_file:
        lp_vars_distribution_json = json.load(read_file)

    tc_const = lp_vars_distribution_json["124"] + \
        lp_vars_distribution_json["132"] + lp_vars_distribution_json["258"]
    tc_one = lp_vars_distribution_json["123"] + \
        lp_vars_distribution_json["131"] + lp_vars_distribution_json["257"]
    tc_two = lp_vars_distribution_json["122"] + \
        lp_vars_distribution_json["130"] + lp_vars_distribution_json["256"]

    rt_const = lp_vars_distribution_json["117"] + \
        lp_vars_distribution_json["1"] + lp_vars_distribution_json["251"]
    rt_one_first = lp_vars_distribution_json["115"] + \
        lp_vars_distribution_json["128"] + lp_vars_distribution_json["255"]
    rt_one_second = lp_vars_distribution_json["121"] + \
        lp_vars_distribution_json["129"] + lp_vars_distribution_json["245"]
    rt_two_first = lp_vars_distribution_json["2"] + \
        lp_vars_distribution_json["125"] + lp_vars_distribution_json["253"]
    rt_two_second = lp_vars_distribution_json["120"] + \
        lp_vars_distribution_json["127"] + lp_vars_distribution_json["2"]
    rt_two_first_and_second = lp_vars_distribution_json["119"] + \
        lp_vars_distribution_json["126"] + lp_vars_distribution_json["254"]
    return tc_const, tc_one, tc_two, rt_const, rt_one_first, rt_one_second, rt_two_first, rt_two_second, rt_two_first_and_second


# Split a list of posterior samples of cost bounds into two lists: (i) a list of
# input coefficients and (ii) a list of output coefficients


def decompose_posterior_distribution(posterior_distribution):
    list_tc_const, list_tc_one, list_tc_two, list_rt_const = posterior_distribution

    list_input_coeffs = []
    list_output_coeffs = []
    for i, coeff_tc_const in enumerate(list_tc_const):
        coeff_tc_one = list_tc_one[i]
        coeff_tc_two = list_tc_two[i]
        list_input_coeffs.append((coeff_tc_const, coeff_tc_one, coeff_tc_two))

        coeff_rt_const = list_rt_const[i]
        list_output_coeffs.append((coeff_rt_const,))
    return list_input_coeffs, list_output_coeffs


# Split inferred coefficients


def decompose_inferred_coefficients(inferred_coefficients):
    tc_const, tc_one, tc_two, rt_const = inferred_coefficients
    input_coeffs = (tc_const, tc_one, tc_two)
    output_coeffs = (rt_const,)
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
