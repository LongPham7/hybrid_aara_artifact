import numpy as np
import csv

# Cost of running append


def append_cost(input_pair, modulo=5, lower_cost=0.5):
    input1, _ = input_pair
    cumulative_cost = 0
    for x in input1:
        if x % 100 == 0:
            cumulative_cost += 1
        elif x % modulo == 1:
            cumulative_cost += 0.85
        elif x % modulo == 2:
            cumulative_cost += 0.65
        else:
            cumulative_cost += lower_cost
    return cumulative_cost


def create_runtime_cost_data(input_data):
    result = []
    for input_pair in input_data:
        input1, input2 = input_pair
        input_size_pair = (len(input1),  len(input2))
        cost = append_cost(input_pair)
        result.append((input_size_pair, cost))
    return result

# Predicted costs for append


def get_predicted_cost(input_size_pair, input_coeffs, output_coeffs):
    input_size1, input_size2 = input_size_pair
    if len(input_coeffs) == 3:
        coeff_tc_const, coeff_tc_one, coeff_tc_two = input_coeffs[
            0], input_coeffs[1], input_coeffs[2]
        input_potential = coeff_tc_const + coeff_tc_one * \
            input_size1 + coeff_tc_two * input_size2
    else:
        raise Exception("The given input coefficients are not well-formed.")

    if output_coeffs is None:
        return input_potential
    else:
        output_size = input_size1 + input_size2
        if len(output_coeffs) == 2:
            coeff_rt_const, coeff_rt_one = output_coeffs[0], output_coeffs[1]
            output_potential = coeff_rt_const + coeff_rt_one * output_size
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")

# Cost gap for append


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size_pair, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size_pair, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost


# Ground-truth worst-case cost of append


def get_ground_truth_cost(input_size_pair):
    input_size1, _ = input_size_pair
    return input_size1

# Gap between the predicted cost and the ground-truth worst-case cost of append


def get_gap_ground_truth(input_size_pair, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size_pair)
    predicted_cost = get_predicted_cost(input_size_pair, input_coeffs, None)
    return predicted_cost - ground_truth_cost


# Relative gap between the predicted cost and the ground-truth worst-case cost
# of append

def get_relative_gap_ground_truth(input_size_pair, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size_pair)
    predicted_cost = get_predicted_cost(input_size_pair, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
