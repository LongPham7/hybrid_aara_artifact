
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


def partition_cost(pivot, input_list, modulo=10, lower_cost=0.5):
    num_hits = 0
    num_misses = 0
    for x in input_list:
        if x % modulo == 0:
            num_hits += 1
        else:
            num_misses += 1
    cost = num_hits + lower_cost * num_misses
    return cost


def quickselect_result_and_cost(index, input_list):
    if len(input_list) == 0:
        raise Exception("The input list is empty")
    elif len(input_list) == 1:
        if index == 0:
            return input_list[0], 0
        else:
            raise Exception("The index should be zero for a singleton list")
    else:
        head_element = input_list[0]
        lower_list, upper_list = partition(head_element, input_list[1:])
        cost_of_paritition = partition_cost(head_element, input_list[1:])
        if index < len(lower_list):
            result, recursive_cost = quickselect_result_and_cost(
                index, lower_list)
            return result, recursive_cost + cost_of_paritition
        elif index == len(lower_list):
            return head_element, cost_of_paritition
        else:
            result, recursive_cost = quickselect_result_and_cost(
                index - len(lower_list) - 1, upper_list)
            return result, cost_of_paritition + recursive_cost


# Extend input data with recursive calls


def quickselect_all_recursive_calls(integer_list_pair, data_collection_type):
    index, input_list = integer_list_pair
    assert (data_collection_type ==
            "toplevel" or data_collection_type == "all_recursive_calls")
    if data_collection_type == "toplevel":
        return [(index, input_list)]

    if len(input_list) == 0:
        raise Exception("The input list is empty")
    elif len(input_list) == 1:
        if index == 0:
            return [(index, input_list)]
        else:
            raise Exception("The index should be zero for a singleton list")
    else:
        head_element = input_list[0]
        lower_list, upper_list = partition(head_element, input_list[1:])
        if index < len(lower_list):
            cumulative_recursive_calls = quickselect_all_recursive_calls(
                index, lower_list, data_collection_type)
            return [(index, input_list)] + cumulative_recursive_calls
        elif index == len(lower_list):
            return [(index, input_list)]
        else:
            cumulative_recursive_calls = quickselect_all_recursive_calls(
                index - len(lower_list) - 1, upper_list, data_collection_type)
            return [(index, input_list)] + cumulative_recursive_calls


def expand_input_data(input_data, data_collection_type):
    result = []
    for integer_list_pair in input_data:
        input_list_expanded = quickselect_all_recursive_calls(
            integer_list_pair, data_collection_type)
        result = result + input_list_expanded
    return result

# Create runtime cost data


def quickselect_cost(integer_list_pair):
    index, input_list = integer_list_pair
    _, computational_cost = quickselect_result_and_cost(index, input_list)
    return computational_cost


def create_runtime_cost_data(input_data):
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="toplevel")
    result = []
    for integer_list_pair in input_data_expanded:
        _, input_list = integer_list_pair
        input_size = len(input_list)
        cost = quickselect_cost(integer_list_pair)
        result.append((input_size, cost))
    return result

# Predicted cost


def get_predicted_cost(input_size, input_coeffs, output_coeffs):
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
        if len(output_coeffs) == 1:
            output_potential = output_coeffs[0]
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
