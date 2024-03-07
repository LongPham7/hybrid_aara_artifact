import numpy as np
import math
import csv


def is_even_list(input_list):
    if len(input_list) % 2 == 0:
        return True
    else:
        return False


def split_list(input_list):
    if len(input_list) == 0:
        return []
    elif len(input_list) == 1:
        raise Exception("The given input list doesn't have an even length")
    else:
        head_element = input_list[0]
        recursive_result = split_list(input_list[2:])
        return [head_element] + recursive_result


def tail_list(input_list):
    if len(input_list) == 0:
        raise Exception("The given list is empty")
    else:
        return input_list[1:]


def even_split_odd_tail_result_and_cost(input_list, modulo=10, lower_cost=0.5):
    if len(input_list) == 0:
        return [], 0
    else:
        num_hits, num_misses = 0, 0
        for x in input_list:
            if x % modulo == 0:
                num_hits += 1
            else:
                num_misses += 1
        cost_current_iteration = num_hits + lower_cost * num_misses
        is_even = is_even_list(input_list)
        if is_even:
            split_result = split_list(input_list)
            recursive_result, cumulative_cost = even_split_odd_tail_result_and_cost(
                split_result)
            return recursive_result, cost_current_iteration + cumulative_cost
        else:
            tail_result = tail_list(input_list)
            recursive_result, cumulative_cost = even_split_odd_tail_result_and_cost(
                tail_result)
            return recursive_result, cost_current_iteration + cumulative_cost


def even_split_odd_tail_cost(input_list, modulo=10, lower_cost=0.5):
    _, cost = even_split_odd_tail_result_and_cost(
        input_list, modulo, lower_cost)
    return cost


def even_split_odd_tail_all_recursive_calls(input_list, data_collection_type):
    assert (data_collection_type ==
            "toplevel" or data_collection_type == "all_recursive_calls")

    if len(input_list) == 0 or data_collection_type == "toplevel":
        return [input_list]
    else:
        is_even = is_even_list(input_list)
        if is_even:
            split_result = split_list(input_list)
            cumulative_recursive_calls = even_split_odd_tail_all_recursive_calls(
                split_result, data_collection_type)
            return [input_list] + cumulative_recursive_calls
        else:
            tail_result = tail_list(input_list)
            cumulative_recursive_calls = even_split_odd_tail_all_recursive_calls(
                tail_result, data_collection_type)
            return [input_list] + cumulative_recursive_calls


def expand_input_data(input_data, data_collection_type):
    result = []
    for input_list in input_data:
        input_pair_expanded = even_split_odd_tail_all_recursive_calls(
            input_list, data_collection_type)
        result = result + input_pair_expanded
    return result


def create_runtime_cost_data(input_data):
    # By default, we only collect the top-level input data in the runtime cost
    # data
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="toplevel")
    result = []
    for input_list in input_data_expanded:
        input_size = len(input_list)
        cost = even_split_odd_tail_cost(input_list)
        result.append((input_size, cost))
    return result


# Predicted costs for even-split-odd-tail

def get_predicted_cost(input_size, input_coeffs, output_coeffs):
    if len(input_coeffs) == 2:
        coeff_tc_zero, coeff_tc_one = input_coeffs[0], input_coeffs[1]
        input_potential = coeff_tc_zero + coeff_tc_one * input_size
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        output_size = 0
        if len(output_coeffs) == 2:
            coeff_rt_zero, coeff_rt_one = output_coeffs[0], output_coeffs[1]
            output_potential = coeff_rt_zero + coeff_rt_one * output_size
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")

# Cost gap


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost


def get_ground_truth_cost(input_size):
    input_list = list(range(0, math.trunc(input_size)))
    _, worst_case_cost = even_split_odd_tail_result_and_cost(
        input_list, modulo=1)
    return worst_case_cost


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost


def get_relative_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
