import numpy as np
import matplotlib.pyplot as plt

# Detect unsound coefficients for append


def find_invalid_input_coeffs(posterior_distribution):
    list_tc_first_input = posterior_distribution[1]
    num_invalid_input_coeffs = 0
    for coeff_tc_first_input in list_tc_first_input:
        if coeff_tc_first_input < 1.0:
            num_invalid_input_coeffs += 1

    return num_invalid_input_coeffs, len(list_tc_first_input)


# Report statistics of the inferred coefficients for append


def report_statistics(posterior_distribution):
    list_tc_const, list_tc_first_input, list_tc_second_input, list_rt_const, list_rt_degree_one = posterior_distribution
    array_tc_const = np.array(list_tc_const)
    array_tc_first_input = np.array(list_tc_first_input)
    array_tc_second_input = np.array(list_tc_second_input)
    array_rt_const = np.array(list_rt_const)
    array_rt_degree_one = np.array(list_rt_degree_one)

    print("Statistics of inferred coefficients in the typing context")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_const), np.std(array_tc_const)))
    print("Degree one of the first input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_first_input), np.std(array_tc_first_input)))
    print("Degree one of the second input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_second_input), np.std(array_tc_second_input)))

    print("Statistics of inferred coefficients in the return type")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_const), np.std(array_rt_const)))
    print("Degree one: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_degree_one), np.std(array_rt_degree_one)))

# Box plot of inferred coefficients


def box_plot_input_potential_coefficients(posterior_distribution):
    list_tc_const, list_tc_first_input, list_tc_second_input, list_rt_const, list_rt_degree_one = posterior_distribution
    input_potential_coefficients = [
        list_tc_const, list_tc_first_input, list_tc_second_input]
    fig, ax = plt.subplots()
    ax.boxplot(input_potential_coefficients)
    ax.set_xticklabels(["Constant", "First input", "Second input"])
    ax.set_title("Box plot of coefficients")
    fig.set_size_inches(7, 7)
    plt.show()

# Report statistics of the inferred coefficients for the stepping function of append


def report_stepping_function_statistics(posterior_distribution):
    list_tc_const, list_tc_first, list_tc_second, \
        list_rt_const, list_rt_first, list_rt_second = posterior_distribution

    array_tc_const = np.array(list_tc_const)
    array_tc_first = np.array(list_tc_first)
    array_tc_second = np.array(list_tc_second)

    array_rt_const = np.array(list_rt_const)
    array_rt_first = np.array(list_rt_first)
    array_rt_second = np.array(list_rt_second)

    print("Statistics of stepping function's typing context")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_const), np.std(array_tc_const)))
    print("Degree one of the first input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_first), np.std(array_tc_first)))
    print("Degree one of the second input: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_second), np.std(array_tc_second)))

    print("Statistics of stepping function's return type")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_const), np.std(array_rt_const)))
    print("Degree one of the first output: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_first), np.std(array_rt_first)))
    print("Degree one of the second output: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_second), np.std(array_rt_second)))
