import numpy as np
import csv

# Extend input data with recursive calls


def scan_and_swap(input_list, modulo=10, lower_cost=0.5):
    if len(input_list) <= 1:
        return input_list, False, 0
    else:
        first_element = input_list[0]
        second_element = input_list[1]
        if first_element % modulo == 0:
            cost_current_iteration = 1
        else:
            cost_current_iteration = lower_cost

        if first_element <= second_element:
            recursive_result, is_swapped, cumulative_cost = scan_and_swap(
                input_list[1:])
            return [first_element] + recursive_result, is_swapped, cumulative_cost + cost_current_iteration
        else:
            recursive_result, _, cumulative_cost = scan_and_swap(
                [first_element] + input_list[2:])
            return [second_element] + recursive_result, True, cumulative_cost + cost_current_iteration

# Extend input data with recursive calls


def bubble_sort_all_recursive_calls(input_list, data_collection_type):
    assert (data_collection_type ==
            "toplevel" or data_collection_type == "all_recursive_calls")
    if data_collection_type == "toplevel":
        return [input_list]

    input_list_scanned, is_swapped, scan_cost = scan_and_swap(input_list)
    if is_swapped:
        cumulative_recursive_calls = bubble_sort_all_recursive_calls(
            input_list_scanned, data_collection_type)
        return [input_list] + cumulative_recursive_calls
    else:
        return [input_list]


def expand_input_data(input_data, data_collection_type):
    result = []
    for input_list in input_data:
        input_list_expanded = bubble_sort_all_recursive_calls(
            input_list, data_collection_type)
        result = result + input_list_expanded
    return result

# Cost of running bubble sort


def bubble_sort_result_and_cost(input_list):
    input_list_scanned, is_swapped, scan_cost = scan_and_swap(input_list)
    if is_swapped:
        input_list_sorted, cumulative_cost = bubble_sort_result_and_cost(
            input_list_scanned)
        return input_list_sorted, scan_cost + cumulative_cost
    else:
        return input_list_scanned, scan_cost


def bubble_sort_cost(input_list):
    _, computational_cost = bubble_sort_result_and_cost(input_list)
    return computational_cost


def create_runtime_cost_data(input_data):
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="toplevel")
    result = []
    for input_list in input_data_expanded:
        input_size = len(input_list)
        cost = bubble_sort_cost(input_list)
        result.append((input_size, cost))
    return result

# Predicted cost for bubble sort


def get_predicted_cost(input_size, input_coeffs, output_coeffs):
    if len(input_coeffs) == 3:
        coeff_tc_zero, coeff_tc_one, coeff_tc_two = input_coeffs[0], input_coeffs[1], input_coeffs[2]
        input_potential = coeff_tc_zero + coeff_tc_one * input_size + \
            coeff_tc_two / 2 * input_size * (input_size - 1)
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        output_size = input_size
        if len(output_coeffs) == 3:
            coeff_rt_zero, coeff_rt_one, coeff_rt_two = output_coeffs[
                0], output_coeffs[1], output_coeffs[2]
            output_potential = coeff_rt_zero + coeff_rt_one * output_size + \
                coeff_rt_two / 2 * output_size * (output_size - 1)
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")

# Cost gap for bubble sort


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost

# Ground-truth worst-case cost of bubble sort


def get_ground_truth_cost(input_size):
    return input_size * (input_size - 1)

# Gap between the predicted cost and the ground-truth worst-case cost of bubble
# sort


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost

# Relative gap between the predicted cost and the ground-truth worst-case cost
# of bubble sort


def get_relative_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
