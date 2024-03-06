import os
import sys
module_path = os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "statistical_aara_test_suite", "z_algorithm", "utility"))
if module_path not in sys.path:
    sys.path.append(module_path)

from cost_evaluation import longest_common_prefix_result_and_cost


def longest_common_prefix_result_and_all_recursive_calls(xs1, xs2):
    if len(xs1) == 0 or len(xs2) == 0:
        return 0, [(xs1, xs2, 0)]
    else:
        if xs1[0] != xs2[0]:
            return 0, [(xs1, xs2, 0)]
        else:
            longest_common_prefix, recursive_calls = longest_common_prefix_result_and_all_recursive_calls(
                xs1[1:], xs2[1:])
            return 1 + longest_common_prefix, [(xs1, xs2, 1 + longest_common_prefix)] + recursive_calls


def longest_common_prefix_all_longest_common_prefix_calls(xs, data_collection_type):
    n = len(xs)
    z = [0] * n
    left = 0
    right = 0

    all_recursive_calls = []

    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])

        longest_common_prefix_length, recursive_calls = longest_common_prefix_result_and_all_recursive_calls(
            xs[z[i]:], xs[(i+z[i]):])
        if data_collection_type == "toplevel":
            all_recursive_calls.append(recursive_calls[0])
        elif data_collection_type == "all_recursive_calls":
            all_recursive_calls += recursive_calls
        else:
            raise Exception("The given data collection type is invalid")
        z[i] += longest_common_prefix_length

        if i + z[i] > right:
            left = i
            right = i + z[i]
    return all_recursive_calls


def get_all_longest_common_prefix_calls(input_data):
    input_data_longest_common_prefix = []
    for input_list in input_data:
        input_data_longest_common_prefix += longest_common_prefix_all_longest_common_prefix_calls(
            input_list, data_collection_type="toplevel")
    return input_data_longest_common_prefix


def get_cost_longest_common_prefix(longest_common_prefix_call):
    xs1, xs2, _ = longest_common_prefix_call
    _, cost = longest_common_prefix_result_and_cost(xs1, xs2)
    return cost


def create_runtime_cost_data_annotated_code(input_data):
    result = []
    input_data_longest_common_prefix = get_all_longest_common_prefix_calls(
        input_data)
    for longest_common_prefix_call in input_data_longest_common_prefix:
        input1, input2, _ = longest_common_prefix_call
        cost = get_cost_longest_common_prefix(longest_common_prefix_call)
        input_size = (len(input1), len(input2))
        result.append((input_size, cost))
    return result

# Predicted cost


def get_predicted_cost_annotated_code(input_size, input_coeffs, output_coeffs):
    input_size1, input_size2 = input_size

    if len(input_coeffs) == 3:
        coeff_tc_const, coeff_tc_one_first, coeff_tc_one_second = input_coeffs[
            0], input_coeffs[1], input_coeffs[2]
        input_potential = coeff_tc_const + coeff_tc_one_first * \
            input_size1 + coeff_tc_one_second * input_size2
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 1:
            output_potential = output_coeffs[0]
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")


def get_cost_gap_annotated_code(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost_annotated_code(
        input_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost
