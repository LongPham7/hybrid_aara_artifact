import numpy as np
import matplotlib.pyplot as plt


def find_invalid_input_coeffs(posterior_distribution):
    list_tc_two = posterior_distribution[2]
    num_invalid_input_coeffs = 0
    for coeff_tc_two in list_tc_two:
        if coeff_tc_two < 1.0:
            num_invalid_input_coeffs += 1

    return num_invalid_input_coeffs, len(list_tc_two)


def report_statistics(posterior_distribution):
    list_tc_const, list_tc_one, list_tc_two, list_rt_const = posterior_distribution

    array_tc_const = np.array(list_tc_const)
    array_tc_one = np.array(list_tc_one)
    array_tc_two = np.array(list_tc_two)

    array_rt_const = np.array(list_rt_const)

    print("Statistics of concat's typing context")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_const), np.std(array_tc_const)))
    print("Degree one: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_one), np.std(array_tc_one)))
    print("Degree two: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_tc_two), np.std(array_tc_two)))

    print("Statistics of concat's return type")
    print("Degree zero: mean = {:.4f} std = {:.4f}".format(
        np.mean(array_rt_const), np.std(array_rt_const)))


def box_plot_input_potential_coefficients(posterior_distribution):
    list_tc_const, list_tc_one, list_tc_two, list_rt_const = posterior_distribution
    input_potential_coefficients = [list_tc_const, list_tc_one, list_tc_two]
    fig, ax = plt.subplots()
    ax.boxplot(input_potential_coefficients)
    ax.set_xticklabels(["Constant", "Degree one", "Degree two"])
    ax.set_title("Box plot of coefficients")
    fig.set_size_inches(7, 7)
    plt.show()
