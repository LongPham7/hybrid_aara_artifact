import numpy as np
import matplotlib.pyplot as plt
import os
import json

from benchmark_manipulation import get_module, list_benchmarks_data_driven_hybrid, \
    list_benchmarks_data_driven
from pathnames import bin_directory
from json_manipulation import read_input_data_json

# Compute a linear function


def compute_y_coordinate(start_point, end_point, x):
    start_x, start_y = start_point
    end_x, end_y = end_point
    gradient = (end_y - start_y) / (end_x - start_x)
    return gradient * (x - start_x) + start_y


def compute_line(start_point, end_point, xs):
    return [compute_y_coordinate(start_point, end_point, x) for x in xs]


# List all benchmarks and hybrid modes


list_analysis_info = []
for benchmark_name in list_benchmarks_data_driven_hybrid:
    analysis_info_data_driven = {"benchmark_name": benchmark_name,
                                 "hybrid_mode": "data_driven"}
    analysis_info_hybrid = {"benchmark_name": benchmark_name,
                            "hybrid_mode": "hybrid"}
    list_analysis_info.append(analysis_info_data_driven)
    list_analysis_info.append(analysis_info_hybrid)

for benchmark_name in list_benchmarks_data_driven:
    analysis_info_data_driven = {"benchmark_name": benchmark_name,
                                 "hybrid_mode": "data_driven"}
    list_analysis_info.append(analysis_info_data_driven)


# Get a sigma for coefficients


def get_coeff_sigma(analysis_info):
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = "bayespc"

    config_module = get_module("config", analysis_info)
    coeff_sigma = config_module.config_dict[hybrid_mode][data_analysis_mode]["coefficient_distribution"]
    return coeff_sigma


# Get the input coefficients inferred by Opt


def get_input_coeff(analysis_info):
    hybrid_mode = analysis_info["hybrid_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name
    analysis_info["data_analysis_mode"] = "opt"

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "inference_result.json")
    inference_result = read_inference_result_module.read_inference_result_json(
        output_file_path, function_name)

    # Get a function for decomposition
    decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
    return decompose_inference_result(inference_result)[0]


num_highest_degree_coeffs_dict = {
    "MapAppend": 2,
    "BubbleSort": 1,
    "Concat": 3,
    "EvenSplitOddTail": 1,
    "InsertionSort2": 1,
    "QuickSelect": 1,
    "MedinaOfMedians": 1,
    "ZAlgorithm": 1,
    "Round": 1,
    "QuickSort": 1}


def get_coeff_statistic(analysis_info):
    benchmark_name = analysis_info["benchmark_name"]
    coeff = get_input_coeff(analysis_info)
    num_highest_degree_coeffs = num_highest_degree_coeffs_dict[benchmark_name]
    max_coeff = max(coeff[-num_highest_degree_coeffs:])
    return max_coeff


def plot_coeff_sigma():
    coeff_statistics = []
    list_sigmas = []
    for analysis_info in list_analysis_info:
        benchmark_name = analysis_info["benchmark_name"]
        hybrid_mode = analysis_info["hybrid_mode"]

        coeff_statistic = get_coeff_statistic(analysis_info)
        coeff_statistics.append(coeff_statistic)

        sigma = get_coeff_sigma(analysis_info)
        list_sigmas.append(sigma)
        print("Benchmark {:19}: mode = {:11}, coeff statistic = {:.3f}, coeff sigma = {:.3f}".format(
            benchmark_name, hybrid_mode, coeff_statistic, sigma))

    fig, ax = plt.subplots()
    ax.scatter(coeff_statistics, list_sigmas, color='black')

    start_point = (0, 0.8)
    end_point = (6, 4)
    xs = np.linspace(0, 6, 10)
    ys = compute_line(start_point, end_point, xs)
    ax.plot(xs, ys)

    ax.set_xlabel("Coeff statistic")
    ax.set_ylabel("Coeff sigma")

    plt.show()


# Get a sigma for a Weibull cost model


def get_cost_model_sigma(analysis_info):
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = "bayespc"

    config_module = get_module("config", analysis_info)
    cost_model_sigma = config_module.config_dict[hybrid_mode][data_analysis_mode]["cost_model"][1]
    return cost_model_sigma


def get_cost_gap_statistic(analysis_info):
    # Statistic function
    def statistics(x): return np.percentile(x, q=90)

    # Project directory
    project_path = os.path.expanduser(
        os.path.join("/home", "hybrid_aara", "benchmark_suite"))

    # Retrieve the file storing cost gaps
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    cost_gaps_bin_path = os.path.join(
        project_path, "cost_gaps", benchmark_name, hybrid_mode, "opt")
    file_path = os.path.join(cost_gaps_bin_path, "cost_gaps.json")
    with open(file_path, "r") as read_file:
        cost_gap_dictionary = json.load(read_file)

    cost_gap_input_potential_only = cost_gap_dictionary["input_potential_only"]
    cost_gap_both_input_and_output = cost_gap_dictionary["both_input_and_output"]

    statistics_input_potential_only = statistics(cost_gap_input_potential_only)
    statistics_both_input_and_output = statistics(
        cost_gap_both_input_and_output)

    return statistics_input_potential_only, statistics_both_input_and_output


def plot_cost_model_sigma():
    list_statistics = []
    list_cost_sigmas = []
    for analysis_info in list_analysis_info:
        statistic = get_cost_gap_statistic(analysis_info)[1]
        list_statistics.append(statistic)

        benchmark_name = analysis_info["benchmark_name"]
        hybrid_mode = analysis_info["hybrid_mode"]
        cost_sigma = get_cost_model_sigma(analysis_info)
        list_cost_sigmas.append(cost_sigma)

        print("Benchmark {:19}: mode = {:11}, cost gap statistic = {:.3f}, cost gap sigma = {:.3f}".format(
            benchmark_name, hybrid_mode, statistic, cost_sigma))

    fig, ax = plt.subplots()
    ax.scatter(list_statistics, list_cost_sigmas, color='black')

    start_point = (0, 100)
    end_point = (188.7, 1200)
    xs = np.linspace(0, 188, 10)
    ys = compute_line(start_point, end_point, xs)
    ax.plot(xs, ys)

    ax.set_xlabel("Cost gap statistic")
    ax.set_ylabel("Cost model sigma")

    plt.show()


# Plot a histogram of cost gaps and the chosen sigma for a cost model


def plot_ax_cost_gap_sigma(ax, analysis_info):
    hybrid_mode = analysis_info["hybrid_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    cost_evaluation_module = get_module("cost_evaluation", analysis_info)
    if hybrid_mode == "hybrid":
        annotate_code_cost_module = get_module(
            "annotated_code_cost", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name
    analysis_info["data_analysis_mode"] = "opt"

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    if hybrid_mode == "data_driven":
        output_file_path = os.path.join(bin_path, "inference_result.json")
        inference_result = read_inference_result_module.read_inference_result_json(
            output_file_path, function_name)
    else:
        output_file_path = os.path.join(bin_path, "output_lp_vars.json")
        inference_result = read_inference_result_module.read_inference_result_annotated_code_json(
            output_file_path)

    # Retrieve the input data
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    input_data = read_input_data_json(input_data_file_path)
    if hybrid_mode == "data_driven":
        runtime_cost_data = cost_evaluation_module.create_runtime_cost_data(
            input_data)
    else:
        runtime_cost_data = annotate_code_cost_module.create_runtime_cost_data_annotated_code(
            input_data)

    # Get necessary functions and info for plotting
    if hybrid_mode == "data_driven":
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
        get_cost_gap = cost_evaluation_module.get_cost_gap
    else:
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients_annotated_code
        get_cost_gap = annotate_code_cost_module.get_cost_gap_annotated_code
    input_coeffs, output_coeffs = decompose_inference_result(inference_result)

    # Plot a histogram of cost gaps
    num_bins = 20
    cost_gaps_both_input_and_output = []
    for cost_measurement in runtime_cost_data:
        cost_gap = get_cost_gap(
            cost_measurement, input_coeffs, output_coeffs)
        cost_gaps_both_input_and_output.append(cost_gap)
    ax.hist(cost_gaps_both_input_and_output, bins=num_bins)

    # Draw a vertical line for the cost model sigma
    cost_model_sigma = get_cost_model_sigma(analysis_info)
    ax.axvline(x=cost_model_sigma / 8, color='red')

    # Draw a vertical line for the cost gap statistic
    statistic = get_cost_gap_statistic(analysis_info)[1]
    ax.axvline(x=statistic, color="green")


def plot_cost_gap_sigma():
    num_benchmarks = 10
    fig, ax = plt.subplots(
        2, num_benchmarks, tight_layout=True, figsize=(3 * num_benchmarks, 3 * 2))
    for i, analysis_info in enumerate(list_analysis_info):
        benchmark_name = analysis_info["benchmark_name"]
        if benchmark_name in benchmark_names_data_driven:
            row = 0
            col = i - 7
            ax[1, col].remove()
        else:
            row = i % 2
            col = i // 2
        if row == 0:
            ax[0, col].set_title(analysis_info["benchmark_name"])
        plot_ax_cost_gap_sigma(ax[row, col], analysis_info)
    plt.show()


# Suggest ideal values for coefficients' and cost models' sigmas


def suggest_ideal_sigmas():
    start_point_coeff = (0, 0.8)
    end_point_coeff = (6, 4)

    start_point_cost_gap = (0, 100)
    end_point_cost_gap = (188.7, 1200)

    for analysis_info in list_analysis_info:
        coeff_statistic = get_coeff_statistic(analysis_info)
        ideal_coeff_sigma = compute_y_coordinate(
            start_point_coeff, end_point_coeff, coeff_statistic)

        cost_gap_statistic = get_cost_gap_statistic(analysis_info)[1]
        ideal_cost_gap_sigma = compute_y_coordinate(
            start_point_cost_gap, end_point_cost_gap, cost_gap_statistic)

        benchmark_name = analysis_info["benchmark_name"]
        hybrid_mode = analysis_info["hybrid_mode"]
        print("Benchmark {:19}: mode = {:11}, ideal coeff_sigma = {:.3f}, ideal cost_gap_sigma = {:.3f}".format(
            benchmark_name, hybrid_mode, ideal_coeff_sigma, ideal_cost_gap_sigma))


if __name__ == "__main__":
    plot_coeff_sigma()
    plot_cost_model_sigma()
    # plot_cost_gap_sigma()
    # suggest_ideal_sigmas()
