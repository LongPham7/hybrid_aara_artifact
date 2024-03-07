import numpy as np
import csv
import math


def double_list(input_list):
    if len(input_list) == 0:
        return []
    else:
        head_element = input_list[0]
        recursive_result = double_list(input_list[1:])
        return [head_element, head_element] + recursive_result


def half_list(input_list):
    if len(input_list) <= 1:
        return []
    else:
        head_element = input_list[0]
        recursive_result = half_list(input_list[2:])
        return [head_element] + recursive_result


def round_list(input_list):
    if len(input_list) == 0:
        return []
    else:
        head_element = input_list[0]
        half_result = half_list(input_list[1:])
        recursive_result = round_list(half_result)
        double_result = double_list(recursive_result)
        return [head_element] + double_result

# Expand input data


def round_all_recursive_calls(input_list, data_collection_type):
    assert (data_collection_type ==
            "toplevel" or data_collection_type == "all_recursive_calls")

    if data_collection_type == "toplevel":
        return [input_list]

    if len(input_list) == 0:
        return [input_list]
    else:
        half_result = half_list(input_list[1:])
        cumulative_recursive_calls = round_all_recursive_calls(
            half_result, data_collection_type)
        return [input_list] + cumulative_recursive_calls

# Cost of running round


def round_cost(input_list, modulo=10, lower_cost=0.5):
    result = round_list(input_list)
    num_hits = 0
    num_misses = 0
    for x in result:
        if x % modulo == 0:
            num_hits += 1
        else:
            num_misses += 1
    return num_hits + lower_cost * num_misses


def create_runtime_cost_data(input_data):
    result = []
    for input_list in input_data:
        input_size = len(input_list)
        cost = round_cost(input_list)
        result.append((input_size, cost))
    return result

# Predicted cost


def get_predicted_cost(input_size, input_coeffs, output_coeffs):
    if len(input_coeffs) == 2:
        coeff_tc_zero, coeff_tc_one = input_coeffs[0], input_coeffs[1]
        input_potential = coeff_tc_zero + coeff_tc_one * input_size
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        output_size = input_size
        if len(output_coeffs) == 2:
            coeff_rt_zero, coeff_rt_one = output_coeffs[0], output_coeffs[1]
            output_potential = coeff_rt_zero + coeff_rt_one * output_size
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost


def get_ground_truth_cost(input_size):
    return pow(2, math.trunc(math.log2(input_size+1))) - 1


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost


def get_relative_gap_ground_truth(input_list, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_list)
    predicted_cost = get_predicted_cost(input_list, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
