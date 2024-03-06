import os
import sys
module_path = os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "benchmark_suite", "concat", "utility"))
if module_path not in sys.path:
    sys.path.append(module_path)

from cost_evaluation import expand_input_data

# List all calls to the annotated code


def concat_all_append_calls(nested_list):
    if len(nested_list) == 0:
        return []
    else:
        flattened_list = []
        for inner_list in nested_list[1:]:
            flattened_list += inner_list
        return [(nested_list[0], flattened_list)]

# Cost of running the annotated code


def cost_annotated_code(input_pair, modulo=5, lower_cost=0.5):
    input1, input2 = input_pair
    num_hits, num_misses = 0, 0
    for x in input1:
        if x % modulo == 0:
            num_hits += 1
        else:
            num_misses += 1
    return num_hits + lower_cost * num_misses


def create_runtime_cost_data_annotated_code(input_data):
    result = []
    # By default, we include all recursive calls in the runtime cost data.
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="all_recursive_calls")
    for nested_list in input_data_expanded:
        list_calls_annotated_code = concat_all_append_calls(nested_list)
        for input_pair in list_calls_annotated_code:
            input1, input2 = input_pair
            input_size_pair = (len(input1), len(input2))
            cost = cost_annotated_code(input_pair)
            result.append((input_size_pair, cost))
    return result

# Predicted cost for the annotated code


def get_predicted_cost_annotated_code(input_size, input_coeffs, output_coeffs):
    coeff_tc_const, coeff_tc_one_first_input, coeff_tc_one_second_input, coeff_tc_two_first_input, \
        coeff_tc_two_second_input, coeff_tc_two_first_and_second = input_coeffs

    input_size1, input_size2 = input_size
    output_size = input_size1 + input_size2

    if len(input_coeffs) == 6:
        input_potential = coeff_tc_const + coeff_tc_one_first_input * input_size1 \
            + coeff_tc_one_second_input * input_size2 \
            + coeff_tc_two_first_input / 2 * input_size1 * (input_size1 - 1) \
            + coeff_tc_two_second_input / 2 * input_size2 * (input_size2 - 1) + \
            + coeff_tc_two_first_and_second * input_size1 * input_size2
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 3:
            coeff_rt_const, coeff_rt_one, coeff_rt_two = output_coeffs
            output_potential = coeff_rt_const + coeff_rt_one * output_size \
                + coeff_rt_two / 2 * output_size * (output_size - 1)
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")


# Cost gap for the annotated code


def get_cost_gap_annotated_code(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost_annotated_code(
        input_size, input_coeffs, output_coeffs)

    if predicted_cost < actual_cost - 0.0001:
        input_size1, input_size2 = input_size[0], input_size[1]
        print("input1 size: {} input2 size: {} input_coeffs: {} output_coeffs: {}".format(
            input_size1, input_size2, input_coeffs, output_coeffs))

    return predicted_cost - actual_cost
