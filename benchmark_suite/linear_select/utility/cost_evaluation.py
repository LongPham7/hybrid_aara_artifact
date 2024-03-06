
def insert(element, input_list):
    for i in range(len(input_list)):
        if (element <= input_list[i]):
            return input_list[0:i] + [element] + input_list[i:]
    return input_list + [element]


def insertion_sort(input_list):
    if len(input_list) == 0:
        return []
    else:
        head_element = input_list[0]
        tail_sorted = insertion_sort(input_list[1:])
        input_list_sorted = insert(head_element, tail_sorted)
        return input_list_sorted


def partition_into_blocks_of_five(input_list):
    if len(input_list) == 0:
        return []
    else:
        recursive_result = partition_into_blocks_of_five(input_list[5:])
        return [input_list[0:5]] + recursive_result


def median_of_list_five_or_fewer(input_list):
    assert (0 < len(input_list) <= 5)
    input_list_sorted = insertion_sort(input_list)
    index_of_median = (len(input_list) - 1) // 2
    list_leftover = input_list_sorted[:index_of_median] + \
        input_list_sorted[(index_of_median+1):]
    return input_list_sorted[index_of_median], list_leftover


def collect_medians_and_leftover(list_medians_and_leftover):
    list_medians = [median for median, _ in list_medians_and_leftover]
    list_leftover = [
        element for _, list_leftover in list_medians_and_leftover for element in list_leftover]
    return list_medians, list_leftover


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


def find_minimum_acc(acc, candidate, input_list):
    if len(input_list) == 0:
        return candidate, acc
    else:
        head_element = input_list[0]
        if head_element < candidate:
            return find_minimum_acc([candidate] + acc, head_element, input_list[1:])
        else:
            return find_minimum_acc([head_element] + acc, candidate, input_list[1:])


def find_minimum(input_list):
    return find_minimum_acc([], input_list[0], input_list[1:])


def preprocess_list_acc(minima_acc, input_list):
    if len(input_list) % 5 == 0:
        return minima_acc, input_list
    else:
        minimum, leftover = find_minimum(input_list)
        return preprocess_list_acc([minimum] + minima_acc, leftover)


def get_nth_element(index, input_list):
    return input_list[index]


def linear_select_result_and_cost(index, input_list):
    assert (0 <= index < len(input_list))
    if len(input_list) == 0:
        raise Exception("The input list must be non-empty")
    else:
        minima, input_list_trimmed = preprocess_list_acc([], input_list)
        mod_five = len(minima)
        if index < mod_five:
            result = get_nth_element(mod_five - index - 1, minima)
            return result, 0
        else:
            index_trimmed = index - mod_five
            list_blocks_of_five = partition_into_blocks_of_five(
                input_list_trimmed)
            list_medians_and_leftover = [median_of_list_five_or_fewer(
                block_of_five) for block_of_five in list_blocks_of_five]
            list_medians, _ = collect_medians_and_leftover(
                list_medians_and_leftover)
            num_medians = len(list_medians)
            median_of_medians, linear_select_cost1 = linear_select_result_and_cost(
                num_medians // 2, list_medians)
            lower_list, upper_list = partition(
                median_of_medians, input_list_trimmed)
            cost_of_partition = partition_cost(
                median_of_medians, input_list_trimmed)
            lower_list_length = len(lower_list)
            if index_trimmed == lower_list_length - 1:
                return median_of_medians, linear_select_cost1 + cost_of_partition
            elif index_trimmed < lower_list_length - 1:
                median, linear_select_cost2 = linear_select_result_and_cost(
                    index_trimmed, lower_list)
                return median, linear_select_cost1 + cost_of_partition + linear_select_cost2
            else:
                median, linear_select_cost2 = linear_select_result_and_cost(
                    index_trimmed - lower_list_length, upper_list)
                return median, linear_select_cost1 + cost_of_partition + linear_select_cost2


def linear_select_result_and_all_recursive_calls(index, input_list):
    assert (0 <= index < len(input_list))
    if len(input_list) == 0:
        raise Exception("The input list must be non-empty")
    else:
        minima, input_list_trimmed = preprocess_list_acc([], input_list)
        mod_five = len(minima)
        if index < mod_five:
            result = get_nth_element(mod_five - index - 1, minima)
            return result, [(index, input_list)]
        else:
            index_trimmed = index - mod_five
            list_blocks_of_five = partition_into_blocks_of_five(
                input_list_trimmed)
            list_medians_and_leftover = [median_of_list_five_or_fewer(
                block_of_five) for block_of_five in list_blocks_of_five]
            list_medians, _ = collect_medians_and_leftover(
                list_medians_and_leftover)
            num_medians = len(list_medians)
            median_of_medians, recursive_calls1 = linear_select_result_and_all_recursive_calls(
                num_medians // 2, list_medians)
            lower_list, upper_list = partition(
                median_of_medians, input_list_trimmed)
            lower_list_length = len(lower_list)
            if index_trimmed == lower_list_length - 1:
                return median_of_medians, [(index, input_list)] + recursive_calls1
            elif index_trimmed < lower_list_length - 1:
                median, recursive_calls2 = linear_select_result_and_all_recursive_calls(
                    index_trimmed, lower_list)
                return median, [(index, input_list)] + recursive_calls1 + recursive_calls2
            else:
                median, recursive_calls2 = linear_select_result_and_all_recursive_calls(
                    index_trimmed - lower_list_length, upper_list)
                return median, [(index, input_list)] + recursive_calls1 + recursive_calls2

# Extend input data with recursive calls


def linear_select_all_recursive_calls(integer_list_pair, data_collection_type):
    index, input_list = integer_list_pair
    if data_collection_type == "toplevel":
        return [(index, input_list)]
    elif data_collection_type == "all_recursive_calls":
        _, recursive_calls = linear_select_result_and_all_recursive_calls(
            index, input_list)
        return recursive_calls
    else:
        raise Exception("The given data collection type is invalid")


def expand_input_data(input_data, data_collection_type):
    result = []
    for integer_list_pair in input_data:
        input_list_expanded = linear_select_all_recursive_calls(
            integer_list_pair, data_collection_type)
        result = result + input_list_expanded
    return result

# Create runtime cost data


def linear_select_cost(integer_list_pair):
    index, input_list = integer_list_pair
    _, computational_cost = linear_select_result_and_cost(index, input_list)
    return computational_cost


def create_runtime_cost_data(input_data):
    input_data_expanded = expand_input_data(
        input_data, data_collection_type="toplevel")
    result = []
    for integer_list_pair in input_data_expanded:
        _, input_list = integer_list_pair
        input_size = len(input_list)
        cost = linear_select_cost(integer_list_pair)
        result.append((input_size, cost))
    return result

# Predicted cost


def get_predicted_cost(input_size, input_coeffs, output_coeffs):
    if len(input_coeffs) == 2:
        coeff_tc_const, coeff_tc_one = input_coeffs[0], input_coeffs[1]
        input_potential = coeff_tc_const + coeff_tc_one * input_size
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 1:
            coeff_rt_const = output_coeffs[0]
            output_potential = coeff_rt_const
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
    return 10 * input_size


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost


def get_relative_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
