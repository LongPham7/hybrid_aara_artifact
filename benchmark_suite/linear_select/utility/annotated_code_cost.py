import os
import sys
module_path = os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "statistical_aara_test_suite", "linear_select", "utility"))
if module_path not in sys.path:
    sys.path.append(module_path)

from cost_evaluation import preprocess_list_acc, get_nth_element, partition_into_blocks_of_five, \
    median_of_list_five_or_fewer, collect_medians_and_leftover, partition, partition_cost

def linear_select_result_and_partition_calls(index, input_list):
    assert (0 <= index < len(input_list))
    if len(input_list) == 0:
        raise Exception("The input list must be non-empty")
    else:
        minima, input_list_trimmed = preprocess_list_acc([], input_list)
        mod_five = len(minima)
        if index < mod_five:
            result = get_nth_element(mod_five - index - 1, minima)
            return result, []
        else:
            index_trimmed = index - mod_five
            list_blocks_of_five = partition_into_blocks_of_five(
                input_list_trimmed)
            list_medians_and_leftover = [median_of_list_five_or_fewer(
                block_of_five) for block_of_five in list_blocks_of_five]
            list_medians, _ = collect_medians_and_leftover(
                list_medians_and_leftover)
            num_medians = len(list_medians)
            median_of_medians, partition_calls1 = linear_select_result_and_partition_calls(
                num_medians // 2, list_medians)
            lower_list, upper_list = partition(
                median_of_medians, input_list_trimmed)
            lower_list_length = len(lower_list)
            if index_trimmed == lower_list_length - 1:
                return median_of_medians, \
                    [(median_of_medians, input_list_trimmed,
                      lower_list, upper_list)] + partition_calls1
            elif index_trimmed < lower_list_length - 1:
                median, partition_calls2 = linear_select_result_and_partition_calls(
                    index_trimmed, lower_list)
                return median, \
                    [(median_of_medians, input_list_trimmed, lower_list,
                      upper_list)] + partition_calls1 + partition_calls2
            else:
                median, partition_calls2 = linear_select_result_and_partition_calls(
                    index_trimmed - lower_list_length, upper_list)
                return median, \
                    [(median_of_medians, input_list_trimmed, lower_list,
                      upper_list)] + partition_calls1 + partition_calls2


def get_all_partition_calls(input_data):
    input_data_partition = []
    for integer_list_pair in input_data:
        index, input_list = integer_list_pair
        _, partition_calls = linear_select_result_and_partition_calls(
            index, input_list)
        input_data_partition += partition_calls
    return input_data_partition


def evaluate_cost_partition(partition_call):
    pivot, input_list, _, _ = partition_call
    return partition_cost(pivot, input_list)


def create_runtime_cost_data_annotated_code(input_data):
    result = []
    input_data_partition = get_all_partition_calls(input_data)
    for partition_call in input_data_partition:
        _, input, output1, output2 = partition_call
        cost = evaluate_cost_partition(partition_call)
        input_output_size = (len(input), len(output1), len(output2))
        result.append((input_output_size, cost))
    return result

# Predicted cost


def get_predicted_cost_annotated_code(input_output_size, input_coeffs, output_coeffs):
    input_size, output_size1, output_size2 = input_output_size

    if len(input_coeffs) == 2:
        coeff_tc_const, coeff_tc_one = input_coeffs[0], input_coeffs[1]
        input_potential = coeff_tc_const + coeff_tc_one * input_size
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 3:
            coeff_rt_const, coeff_rt_one_first, coeff_rt_one_second = output_coeffs[
                0], output_coeffs[1], output_coeffs[2]
            output_potential = coeff_rt_const + coeff_rt_one_first * \
                output_size1 + coeff_rt_one_second * output_size2
            return input_potential - output_potential
        else:
            raise Exception(
                "The given output coefficients are not well-formed")


def get_cost_gap_annotated_code(cost_measurement, input_coeffs, output_coeffs):
    input_output_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost_annotated_code(
        input_output_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost
