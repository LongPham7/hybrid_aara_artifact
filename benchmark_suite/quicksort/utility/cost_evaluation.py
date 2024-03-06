import numpy as np
import csv


def partition(pivot, input_list):
    if len(input_list) == 0:
        return [], []
    else:
        head_element = input_list[0]
        lower_list, upper_list = partition(pivot, input_list[1:])
        if head_element <= pivot:
            return [head_element] + lower_list, upper_list
        else:
            return lower_list, [head_element] + upper_list


def partition_cost(pivot, input_list, modulo=5, lower_cost=0.5):
    num_hits = 0
    num_misses = 0
    for x in input_list:
        if x % modulo == 0:
            num_hits += 1
        else:
            num_misses += 1
    cost = num_hits + lower_cost * num_misses
    return cost


def quicksort_result_and_cost(input_list):
    if len(input_list) == 0:
        return [], 0
    else:
        pivot = input_list[0]
        lower_list, upper_list = partition(pivot, input_list[1:])
        cost_of_partition = partition_cost(pivot, input_list[1:])
        lower_list_sorted, lower_list_cost = quicksort_result_and_cost(
            lower_list)
        upper_list_sorted, upper_list_cost = quicksort_result_and_cost(
            upper_list)
        return lower_list_sorted + [pivot] + upper_list_sorted, lower_list_cost + cost_of_partition + upper_list_cost


# Extend input data with recursive calls


def quicksort_all_recursive_calls(input_list, data_collection_type):
    assert (data_collection_type ==
            "toplevel" or data_collection_type == "all_recursive_calls")
    if len(input_list) == 0 or data_collection_type == "toplevel":
        return [input_list]
    else:
        pivot = input_list[0]
        lower_list, upper_list = partition(pivot, input_list[1:])
        lower_list_recursive_calls = quicksort_all_recursive_calls(
            lower_list, data_collection_type)
        upper_list_recursive_calls = quicksort_all_recursive_calls(
            upper_list, data_collection_type)
        return [input_list] + lower_list_recursive_calls + upper_list_recursive_calls


def expand_input_data(input_data, data_collection_type):
    result = []
    for input_list in input_data:
        input_list_expanded = quicksort_all_recursive_calls(
            input_list, data_collection_type)
        result = result + input_list_expanded
    return result

# Create runtime cost data


def quicksort_cost(input_list):
    _, computational_cost = quicksort_result_and_cost(input_list)
    return computational_cost


def create_runtime_cost_data(input_data):
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="toplevel")
    result = []
    for input_list in input_data_expanded:
        input_size = len(input_list)
        cost = quicksort_cost(input_list)
        result.append((input_size, cost))
    return result

# Predicted cost


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


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost


def get_ground_truth_cost(input_size):
    return input_size / 2 * (input_size - 1)


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost


def get_relative_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
