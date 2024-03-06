import os
import sys
module_path = os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "benchmark_suite", "quicksort", "utility"))
if module_path not in sys.path:
    sys.path.append(module_path)

from cost_evaluation import partition, partition_cost


def quicksort_all_partition_calls(input_list):
    if len(input_list) == 0:
        return []
    else:
        head_element = input_list[0]
        lower_list, upper_list = partition(head_element, input_list[1:])
        list_partition_calls_lower_list = quicksort_all_partition_calls(
            lower_list)
        list_partition_calls_upper_list = quicksort_all_partition_calls(
            upper_list)
        return [(head_element, input_list[1:], lower_list, upper_list)] + list_partition_calls_lower_list + list_partition_calls_upper_list


def get_all_partition_calls(input_data):
    input_data_partition = []
    for input_list in input_data:
        input_data_partition += quicksort_all_partition_calls(input_list)
    return input_data_partition


def evaluate_cost_partition(partition_call, modulo=5, lower_cost=0.5):
    pivot, input_list, _, _ = partition_call
    return partition_cost(pivot, input_list, modulo, lower_cost)


def create_runtime_cost_data_annotated_code(input_data):
    result = []
    input_data_partition = get_all_partition_calls(input_data)
    for partition_call in input_data_partition:
        _, input, output1, output2 = partition_call
        cost = evaluate_cost_partition(partition_call)
        input_output_size = (len(input), len(output1), len(output2))
        result.append((input_output_size, cost))
    return result


def get_predicted_cost_annotated_code(input_output_size, input_coeffs, output_coeffs):
    input_size, output_size1, output_size2 = input_output_size

    if len(input_coeffs) == 3:
        coeff_tc_const, coeff_tc_one, coeff_tc_two = input_coeffs[
            0], input_coeffs[1], input_coeffs[2]
        input_potential = coeff_tc_const + coeff_tc_one * input_size + \
            coeff_tc_two / 2 * input_size * (input_size - 1)
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 6:
            coeff_rt_const, coeff_rt_one_first, coeff_rt_one_second = output_coeffs[
                0], output_coeffs[1], output_coeffs[2]
            coeff_rt_two_first, coeff_rt_two_second, coeff_rt_two_first_and_second = output_coeffs[
                3], output_coeffs[4], output_coeffs[5]
            output_potential = coeff_rt_const + coeff_rt_one_first * output_size1 + coeff_rt_one_second * output_size2 \
                + coeff_rt_two_first / 2 * output_size1 * (output_size1 - 1) + coeff_rt_two_second / 2 * output_size2 * (output_size2 - 1) \
                + coeff_rt_two_first_and_second * output_size1 * output_size2
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")


def get_cost_gap_annotated_code(cost_measurement, input_coeffs, output_coeffs):
    input_output_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost_annotated_code(
        input_output_size, input_coeffs, output_coeffs)

    if predicted_cost < actual_cost - 0.0001:
        print("input-output size: {} input_coeffs: {} output_coeffs: {}, predicted_cost = {}, actual_cost = {}".format(
            input_output_size, input_coeffs, output_coeffs, predicted_cost, actual_cost))

    return predicted_cost - actual_cost
