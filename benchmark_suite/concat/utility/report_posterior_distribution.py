import numpy as np
import matplotlib.pyplot as plt

# Detect unsound coefficients for concat


def find_invalid_input_coeffs(posterior_distribution):
    list_tc_inner_sum = posterior_distribution[2]

    num_invalid_input_coeffs = 0
    for coeff_tc_inner_sum in list_tc_inner_sum:
        if coeff_tc_inner_sum < 1.0:
            num_invalid_input_coeffs += 1

    return num_invalid_input_coeffs, len(list_tc_inner_sum)


# Report statistics of the inferred coefficients for concat


def report_statistics(posterior_distribution):
    list_tc_const, list_tc_outer, list_tc_inner_sum, list_tc_outer_squared, \
        list_rt_const, list_rt_one, list_rt_two = posterior_distribution

    array_tc_const = np.array(list_tc_const)
    array_tc_outer = np.array(list_tc_outer)
    array_tc_inner_sum = np.array(list_tc_inner_sum)
    array_tc_outer_squared = np.array(list_tc_outer_squared)

    array_rt_const = np.array(list_rt_const)
    array_rt_one = np.array(list_rt_one)
    array_rt_two = np.array(list_rt_two)

    print("Statistics of concat's typing context")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_const), np.std(array_tc_const)))
    print("Degree one: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_outer), np.std(array_tc_outer)))
    print("Sum of all inner lists: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_inner_sum), np.std(array_tc_inner_sum)))
    print("Outer list squared: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_outer_squared), np.std(array_tc_outer_squared)))

    print("Statistics of concat's return type")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_const), np.std(array_rt_const)))
    print("Degree one: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_one), np.std(array_rt_one)))
    print("Degree two: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_two), np.std(array_rt_two)))

# Box plot of inferred coefficients


def box_plot_input_potential_coefficients(posterior_distribution):
    list_tc_const, list_tc_outer, list_tc_inner_sum, list_tc_outer_squared, \
        list_rt_const, list_rt_one, list_rt_two = posterior_distribution
    input_potential_coefficients = [
        list_tc_const, list_tc_outer, list_tc_inner_sum, list_tc_outer_squared]
    fig, ax = plt.subplots()
    ax.boxplot(input_potential_coefficients)
    ax.set_xticklabels(
        ["Constant", "Outer list", "Combined inner lists", "Outer list squared"])
    ax.set_title("Box plot of coefficients")
    fig.set_size_inches(7, 7)
    plt.show()

# Report statistics of inferred coefficients for append


def report_append_statistics(posterior_distribution):
    list_tc_const, list_tc_one_first_input, list_tc_one_second_input, list_tc_two_first_input, \
        list_tc_two_second_input, list_tc_two_first_and_second, list_rt_const, list_rt_one, list_rt_two = posterior_distribution

    array_tc_const = np.array(list_tc_const)
    array_tc_one_first_input = np.array(list_tc_one_first_input)
    array_tc_one_second_input = np.array(list_tc_one_second_input)
    array_tc_two_first_input = np.array(list_tc_two_first_input)
    array_tc_two_second_input = np.array(list_tc_two_second_input)
    array_tc_two_first_and_second = np.array(list_tc_two_first_and_second)

    array_rt_const = np.array(list_rt_const)
    array_rt_one = np.array(list_rt_one)
    array_rt_two = np.array(list_rt_two)

    print("Statistics of append's typing context")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_const), np.std(array_tc_const)))
    print("Degree one of the first input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_one_first_input), np.std(array_tc_one_first_input)))
    print("Degree one of the second input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_one_second_input), np.std(array_tc_one_second_input)))
    print("Degree two of the first input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_two_first_input), np.std(array_tc_two_first_input)))
    print("Degree two of the second input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_two_second_input), np.std(array_tc_two_second_input)))
    print("Degree two of the first and second inputs: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_two_first_and_second), np.std(array_tc_two_first_and_second)))

    print("Statistics of append's return type")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_const), np.std(array_rt_const)))
    print("Degree one: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_one), np.std(array_rt_one)))
    print("Degree two: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_two), np.std(array_rt_two)))
