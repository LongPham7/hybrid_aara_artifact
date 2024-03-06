import numpy as np
import matplotlib.pyplot as plt

# Calculate relative errors


def calculate_relative_error(input_size, list_input_coeffs, get_relative_gap_ground_truth):
    gaps_ground_truth = []
    for input_coeffs in list_input_coeffs:
        relative_gap = get_relative_gap_ground_truth(input_size, input_coeffs)
        gaps_ground_truth.append(relative_gap)
    return gaps_ground_truth


def calculate_relative_error_opt(input_size, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients):
    input_coeffs, _ = decompose_inferred_coefficients(
        inferred_coefficients)
    list_input_coeffs = [input_coeffs]
    return calculate_relative_error(input_size, list_input_coeffs, get_relative_gap_ground_truth)


def calculate_relative_error_bayesian(input_size, posterior_distribution, get_relative_gap_ground_truth, decompose_posterior_distribution):
    list_input_coeffs, _ = decompose_posterior_distribution(
        posterior_distribution)
    return calculate_relative_error(input_size, list_input_coeffs, get_relative_gap_ground_truth)

# Calculate relative errors for Opt using two-dimensional representative samples


def calculate_relative_error_opt_representative_samples_2D(inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients):
    input_size10 = (10, 10)
    input_size100 = (100, 100)
    input_size1000 = (1000, 1000)
    relative_error10 = calculate_relative_error_opt(
        input_size10, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    relative_error100 = calculate_relative_error_opt(
        input_size100, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    relative_error1000 = calculate_relative_error_opt(
        input_size1000, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    return relative_error10, relative_error100, relative_error1000

# Calculate relative errors for Opt using using two-dimensional representative
# samples (specifically for concat)


def calculate_relative_error_opt_representative_samples_2D_concat(inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients):
    input_size10 = (10 * 5, 10)
    input_size100 = (100 * 5, 100)
    input_size1000 = (1000 * 5, 1000)
    relative_error10 = calculate_relative_error_opt(
        input_size10, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    relative_error100 = calculate_relative_error_opt(
        input_size100, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    relative_error1000 = calculate_relative_error_opt(
        input_size1000, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    return relative_error10, relative_error100, relative_error1000

# Calculate relative errors for Opt using one-dimensional representative samples


def calculate_relative_error_opt_representative_samples(inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients):
    input_size10 = 10
    input_size100 = 100
    input_size1000 = 1000
    relative_error10 = calculate_relative_error_opt(
        input_size10, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    relative_error100 = calculate_relative_error_opt(
        input_size100, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    relative_error1000 = calculate_relative_error_opt(
        input_size1000, inferred_coefficients, get_relative_gap_ground_truth, decompose_inferred_coefficients)
    return relative_error10, relative_error100, relative_error1000

# Calculate relative errors for Bayesian resource analysis using two-dimensional
# representative samples


def calculate_relative_error_bayesian_representative_samples_2D(inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution):
    input_size10 = (10, 10)
    input_size100 = (100, 100)
    input_size1000 = (1000, 1000)
    relative_error10 = calculate_relative_error_bayesian(
        input_size10, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    relative_error100 = calculate_relative_error_bayesian(
        input_size100, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    relative_error1000 = calculate_relative_error_bayesian(
        input_size1000, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    return relative_error10, relative_error100, relative_error1000

# Calculate relative errors for Bayesian resource analysis using using
# two-dimensional representative samples (specifically for concat)


def calculate_relative_error_bayesian_representative_samples_2D_concat(inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution):
    input_size10 = (10 * 5, 10)
    input_size100 = (100 * 5, 100)
    input_size1000 = (1000 * 5, 1000)
    relative_error10 = calculate_relative_error_bayesian(
        input_size10, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    relative_error100 = calculate_relative_error_bayesian(
        input_size100, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    relative_error1000 = calculate_relative_error_bayesian(
        input_size1000, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    return relative_error10, relative_error100, relative_error1000

# Calculate relative errors for Bayesian resource analysis using one-dimensional
# representative samples


def calculate_relative_error_bayesian_representative_samples(inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution):
    input_size10 = 10
    input_size100 = 100
    input_size1000 = 1000
    relative_error10 = calculate_relative_error_bayesian(
        input_size10, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    relative_error100 = calculate_relative_error_bayesian(
        input_size100, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    relative_error1000 = calculate_relative_error_bayesian(
        input_size1000, inferred_coefficients, get_relative_gap_ground_truth, decompose_posterior_distribution)
    return relative_error10, relative_error100, relative_error1000

# Display relative errors


def display_relative_error(relative_error):
    relative_error10, relative_error100, relative_error1000 = relative_error
    print("Relative errors:")
    print("Input size = {}, 5th percentile = {}, median = {}, 95th percentile = {}".format(str(
        10), np.percentile(relative_error10, 5), np.median(relative_error10), np.percentile(relative_error10, 95)))
    print("Input size = {}, 5th percentile = {}, median = {}, 95th percentile = {}".format(str(100), np.percentile(
        relative_error100, 5), np.median(relative_error100), np.percentile(relative_error100, 95)))
    print("Input size = {}, 5th percentile = {}, median = {}, 95th percentile = {}".format(str(1000), np.percentile(
        relative_error1000, 5), np.median(relative_error1000), np.percentile(relative_error1000, 95)))

# Box plot of relative errors


def box_plot_relative_errors(three_lists_relative_errors):
    assert (len(three_lists_relative_errors) == 3)
    fig, ax = plt.subplots()
    ax.boxplot(three_lists_relative_errors)
    ax.set_xticklabels(["Opt", "BayesWC", "BayesPC"])
    ax.set_title("Relative errors from ground truth")
    fig.set_size_inches(7, 7)

    plt.show()
