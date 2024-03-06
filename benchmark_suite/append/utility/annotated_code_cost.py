# Cost of running the annotated code


def cost_annotated_code(stepping_function_call, modulo=5, lower_cost=0.5):
    head_element, _, _, _, _ = stepping_function_call
    if head_element % 100 == 0:
        return 1
    elif head_element % modulo == 1:
        return 0.85
    elif head_element % modulo == 2:
        return 0.65
    else:
        return lower_cost


def create_runtime_cost_data_annotated_code(input_data):
    result = []
    for input_pair in input_data:
        input1, input2 = input_pair
        if input1 == []:
            continue

        stepping_function_call = (
            input1[0], input1[1:], input2, input1[1:], input2)
        input_size = (1, len(input1)-1,  len(input2),
                      len(input1)-1, len(input2))
        cost = cost_annotated_code(stepping_function_call)
        result.append((input_size, cost))
    return result


# Predicted cost for the annotated code


def get_predicted_cost_annotated_code(input_size, input_coeffs, output_coeffs):
    _, input_size1, input_size2, output_size1, output_size2 = input_size

    if len(input_coeffs) == 3:
        coeff_tc_const, coeff_tc_first, coeff_tc_second = input_coeffs[
            0], input_coeffs[1], input_coeffs[2]
        input_potential = coeff_tc_const + coeff_tc_first * \
            input_size1 + coeff_tc_second * input_size2
    else:
        raise Exception("The given input coefficients are not well-formed")

    if output_coeffs is None:
        return input_potential
    else:
        if len(output_coeffs) == 3:
            coeff_rt_const, coeff_rt_first, coeff_rt_second = output_coeffs[
                0], output_coeffs[1], output_coeffs[2]
            output_potential = coeff_rt_const + coeff_rt_first * \
                output_size1 + coeff_rt_second * output_size2
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
        input_size1, input_size2 = input_size[1], input_size[2]
        print("input1 size: {} input2 size: {} input_coeffs: {} output_coeffs: {}".format(
            input_size1, input_size2, input_coeffs, output_coeffs))

    return predicted_cost - actual_cost
