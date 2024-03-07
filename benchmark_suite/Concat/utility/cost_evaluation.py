import numpy as np
import csv


def get_inner_lists_total_length(nested_list):
    sizes_inner_lists = [len(inner_list) for inner_list in nested_list]
    return sum(sizes_inner_lists)


# Extend input data with recursive calls

def concat_all_recursive_calls(nested_list, data_collection_type):
    if data_collection_type == "toplevel":
        return [nested_list]
    else:
        num_inner_lists = len(nested_list)
        all_recursive_nested_lists = []
        for i in range(num_inner_lists):
            all_recursive_nested_lists.append(nested_list[i:])
        return all_recursive_nested_lists


def expand_input_data(input_data, data_collection_type):
    result = []
    for nested_list in input_data:
        input_pair_expanded = concat_all_recursive_calls(
            nested_list, data_collection_type)
        result = result + input_pair_expanded
    return result

# Cost of running concat


def concat_cost(nested_list, modulo=5, lower_cost=0.5):
    num_hits = 0
    num_misses = 0
    num_inner_lists = len(nested_list)
    for inner_list in nested_list:
        for j in inner_list:
            if j % modulo == 0:
                num_hits += 1
            else:
                num_misses += 1
    cost = num_hits + lower_cost * num_misses
    return cost


def create_runtime_cost_data(input_data):
    # By default, we include all recursive calls in the runtime cost data.
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="all_recursive_calls")
    result = []
    for nested_list in input_data_expanded:
        total_length = get_inner_lists_total_length(nested_list)
        outer_length = len(nested_list)
        input_size_pair = (total_length, outer_length)
        cost = concat_cost(nested_list)
        result.append((input_size_pair, cost))
    return result

# Predicted cost for concat


def get_predicted_cost(input_size_pair, input_coeffs, output_coeffs):
    inner_lists_total_length, outer_list_length = input_size_pair
    if len(input_coeffs) == 4:
        coeff_tc_constant, coeff_tc_outer = input_coeffs[0], input_coeffs[1]
        coeff_tc_sum_inner, coeff_tc_outer_squared = input_coeffs[2], input_coeffs[3]
        input_potential = coeff_tc_constant + coeff_tc_outer * outer_list_length \
            + coeff_tc_sum_inner * inner_lists_total_length + \
            coeff_tc_outer_squared / 2 * \
            outer_list_length * (outer_list_length - 1)
    else:
        raise Exception("The given input coefficients are not well-formed.")

    if output_coeffs is None:
        return input_potential
    else:
        output_size = inner_lists_total_length
        if len(output_coeffs) == 3:
            coeff_rt_const, coeff_rt_one, coeff_rt_two = output_coeffs[
                0], output_coeffs[1], output_coeffs[2]
            output_potential = coeff_rt_const + coeff_rt_one * output_size + \
                coeff_rt_two / 2 * output_size * (output_size - 1)
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed.")

# Cost gap for concat


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size_pair, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size_pair, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost

# Ground-truth worst-case cost of concat


def get_ground_truth_cost(input_size_pair):
    return input_size_pair[0]

# Gap between the predicted cost and the ground-truth worst-case cost of concat


def get_gap_ground_truth(input_size_pair, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size_pair)
    predicted_cost = get_predicted_cost(input_size_pair, input_coeffs, None)
    return predicted_cost - ground_truth_cost

# Relative gap between the predicted cost and the ground-truth worst-case cost
# of concat


def get_relative_gap_ground_truth(input_size_pair, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size_pair)
    predicted_cost = get_predicted_cost(input_size_pair, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
