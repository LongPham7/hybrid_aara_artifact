
def insert(element, input_list, modulo=5, lower_cost=0.5):
    cost_acc = 0
    for i in range(len(input_list)):
        if input_list[i] % 200 == 0:
            cost_acc += 1
        elif input_list[i] % modulo == 1:
            cost_acc += 0.85
        elif input_list[i] % modulo == 2:
            cost_acc += 0.65
        else:
            cost_acc += lower_cost

        if (element <= input_list[i]):
            return input_list[0:i] + [element] + input_list[i:], cost_acc
    return input_list + [element], cost_acc


def insertion_sort_result_and_cost(input_list):
    if len(input_list) == 0:
        return [], 0
    else:
        head_element = input_list[0]
        tail_sorted, tail_sort_cost = insertion_sort_result_and_cost(
            input_list[1:])
        input_list_sorted, insert_cost = insert(head_element, tail_sorted)
        return input_list_sorted, insert_cost + tail_sort_cost


# Extend input data with recursive calls

def insertion_sort_all_recursive_calls(input_list, data_collection_type):
    if data_collection_type == "toplevel":
        return [input_list]
    elif data_collection_type == "all_recursive_calls":
        return [input_list[i:] for i in range(len(input_list) + 1)]
    else:
        raise Exception("The given data collection type is invalid")


def expand_input_data(input_data, data_collection_type):
    result = []
    for input_list in input_data:
        input_list_expanded = insertion_sort_all_recursive_calls(
            input_list, data_collection_type)
        result = result + input_list_expanded
    return result

# Cost of running insertion sort twice


def insertion_sort_cost(input_list):
    _, computational_cost = insertion_sort_result_and_cost(input_list)
    return computational_cost


def create_runtime_cost_data(input_data):
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="toplevel")
    result = []
    for input_list in input_data_expanded:
        input_size = len(input_list)
        # This is the first call to insertion sort
        sorted_list, _ = insertion_sort_result_and_cost(input_list)
        # This is the second call to insertion sort
        _, cost = insertion_sort_result_and_cost(sorted_list)
        result.append((input_size, cost))
    return result

# Predicted cost


def get_predicted_cost(input_size, input_coeffs, output_coeffs):
    if len(input_coeffs) == 2:
        coeff_tc_zero, coeff_tc_one = input_coeffs[0], input_coeffs[1]
        input_potential = coeff_tc_zero + coeff_tc_one * input_size
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        output_size = input_size
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

# Ground-truth worst-case cost


def get_ground_truth_cost(input_size):
    return input_size

# Gap between the predicted cost and the ground-truth worst-case cost


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost

# Relative gap between the predicted cost and the ground-truth worst-case cost


def get_relative_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
