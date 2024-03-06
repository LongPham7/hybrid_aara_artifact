
def incur_cost(element, modulo=5, lower_cost=0.5):
    if element % 100 == 0:
        return 1
    elif element % modulo == 1:
        return 0.85
    elif element % modulo == 2:
        return 0.65
    else:
        return lower_cost


def longest_common_prefix_result_and_cost(xs1, xs2, modulo=20, lower_cost=0.5):
    if len(xs1) == 0 or len(xs2) == 0:
        return 0, 0
    else:
        if xs1[0] != xs2[0]:
            return 0, 0
        else:
            recursive_result, cumulative_cost = longest_common_prefix_result_and_cost(
                xs1[1:], xs2[1:], modulo, lower_cost)
            cost_current_iteration = incur_cost(xs1[0] + xs2[0])
            return 1 + recursive_result, cost_current_iteration + cumulative_cost


def z_algorithm_result_and_cost(xs, modulo=20, lower_cost=0.5):
    n = len(xs)
    z = [0] * n
    left = 0
    right = 0

    cost = 0

    for i in range(1, n):
        cost += incur_cost(xs[i])

        if i < right:
            z[i] = min(right - i, z[i - left])

        longest_common_prefix_length, cost_longes_common_prefix = longest_common_prefix_result_and_cost(
            xs[z[i]:], xs[(i+z[i]):], modulo, lower_cost)
        z[i] += longest_common_prefix_length
        cost += cost_longes_common_prefix

        if i + z[i] > right:
            left = i
            right = i + z[i]
    return z, cost


# Extend input data with recursive calls


def z_algorithm_all_recursive_calls(input_list):
    return [input_list]


def expand_input_data(input_data):
    result = []
    for input_list in input_data:
        input_list_expanded = z_algorithm_all_recursive_calls(input_list)
        result = result + input_list_expanded
    return result


# Create runtime cost data


def z_algorithm_cost(input_list):
    _, computational_cost = z_algorithm_result_and_cost(input_list)
    return computational_cost


def create_runtime_cost_data(input_data):
    input_data_expanded = expand_input_data(input_data)
    result = []
    for input_list in input_data_expanded:
        input_size = len(input_list)
        cost = z_algorithm_cost(input_list)
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


def get_cost_gap(cost_measurement, input_coeffs, output_coeffs):
    input_size, actual_cost = cost_measurement
    predicted_cost = get_predicted_cost(
        input_size, input_coeffs, output_coeffs)
    return predicted_cost - actual_cost


def get_ground_truth_cost(input_size):
    return 2 * input_size


def get_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    return predicted_cost - ground_truth_cost


def get_relative_gap_ground_truth(input_size, input_coeffs):
    ground_truth_cost = get_ground_truth_cost(input_size)
    predicted_cost = get_predicted_cost(input_size, input_coeffs, None)
    gap = predicted_cost - ground_truth_cost
    return gap / ground_truth_cost
