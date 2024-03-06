import os
import sys
module_path = os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "benchmark_suite", "insertion_sort", "utility"))
if module_path not in sys.path:
    sys.path.append(module_path)

from cost_evaluation import insert, insertion_sort_result_and_cost

# Cost of running the annotated code


def cost_annotated_code(insert_call):
    element, input_list = insert_call
    _, cost = insert(element, input_list)
    return cost


def create_runtime_cost_data_annotated_code(input_data):
    result = []
    for input_list in input_data:
        if input_list == []:
            continue

        sorted_list, _ = insertion_sort_result_and_cost(input_list)
        insert_call = (sorted_list[0], sorted_list[1:])
        input_size = len(sorted_list[1:])
        cost = cost_annotated_code(insert_call)
        result.append((input_size, cost))
    return result

# Predicted cost for the annotated code


def get_predicted_cost_annotated_code(input_size, input_coeffs, output_coeffs):
    if len(input_coeffs) == 2:
        coeff_tc_const, coeff_tc_one = input_coeffs
        input_potential = coeff_tc_const + coeff_tc_one * input_size
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 2:
            output_size = input_size + 1
            coeff_rt_const, coeff_rt_one = output_coeffs
            output_potential = coeff_rt_const + coeff_rt_one * output_size
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
        print("input size: {} input_coeffs: {} output_coeffs: {}, actual_cost = {}".format(
            input_size, input_coeffs, output_coeffs, actual_cost))

    return predicted_cost - actual_cost
