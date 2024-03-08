import os
import sys
import numpy as np

from benchmark_manipulation import get_module, get_list_benchmark_hybrid_mode, \
    list_benchmarks_data_driven_hybrid, list_benchmarks_data_driven
from pathnames import bin_directory
from json_manipulation import read_input_data_json, read_execution_time_json
from visualization import plot_cost_gaps_opt, plot_cost_gaps_bayesian, \
    plot_gaps_ground_truth_opt, plot_gaps_ground_truth_bayesian, \
    plot_inferred_cost_bound, plot_inferred_cost_bound_3D, \
    plot_posterior_distribution_cost_bound, plot_median_cost_bound_3D
from visualization_relative_errors import plot_relative_errors_selected_benchmarks
from relative_errors import calculate_relative_error_opt_representative_samples, \
    calculate_relative_error_bayesian_representative_samples, \
    calculate_relative_error_opt_representative_samples_2D, \
    calculate_relative_error_bayesian_representative_samples_2D, \
    calculate_relative_error_opt_representative_samples_2D_concat, \
    calculate_relative_error_bayesian_representative_samples_2D_concat


# Plot the gaps between the predicted costs and observed costs


def analyze_inference_result_cost_gaps(analysis_info, plot_name, save, show):
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    cost_evaluation_module = get_module("cost_evaluation", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "inference_result.json")
    inference_result = read_inference_result_module.read_inference_result_json(
        output_file_path, function_name)

    # Retrieve the input data
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    input_data = read_input_data_json(input_data_file_path)
    runtime_cost_data = cost_evaluation_module.create_runtime_cost_data(
        input_data)

    # Get necessary functions and info for plotting
    if data_analysis_mode == "opt":
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
    else:
        decompose_inference_result = read_inference_result_module.decompose_posterior_distribution
    get_cost_gap = cost_evaluation_module.get_cost_gap

    if data_analysis_mode == "opt":
        plot_function = plot_cost_gaps_opt
    else:
        plot_function = plot_cost_gaps_bayesian

    # Plot a histogram of cost gaps
    plot_function(runtime_cost_data, inference_result,
                  decompose_inference_result, get_cost_gap, analysis_info,
                  plot_name, save, show)


def analyze_inference_result_cost_gaps_annotated_code(analysis_info, plot_name, save, show):
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Get modules
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    if hybrid_mode == "hybrid":
        annotated_code_cost_module = get_module(
            "annotated_code_cost", analysis_info)

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "output_lp_vars.json")
    if not os.path.exists(output_file_path):
        # If the file output_lp_vars.json does not exist, we do nothing.
        return None
    inference_result_annotated_code = read_inference_result_module.read_inference_result_annotated_code_json(
        output_file_path)

    # Retrieve the input data
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    input_data = read_input_data_json(input_data_file_path)
    runtime_cost_data_annotated_code = annotated_code_cost_module.create_runtime_cost_data_annotated_code(
        input_data)

    # Get necessary functions and info for plotting
    if data_analysis_mode == "opt":
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients_annotated_code
    else:
        decompose_inference_result = read_inference_result_module.decompose_posterior_distribution_annotated_code
    get_cost_gap_annotated_code = annotated_code_cost_module.get_cost_gap_annotated_code

    if data_analysis_mode == "opt":
        plot_function = plot_cost_gaps_opt
    else:
        plot_function = plot_cost_gaps_bayesian

    # Plot a histogram of cost gaps for the code fragment Raml.stat(e)
    plot_function(runtime_cost_data_annotated_code, inference_result_annotated_code,
                  decompose_inference_result, get_cost_gap_annotated_code,
                  analysis_info, plot_name, save, show)


# Plot the gaps between the predicted costs and true worst-case costs


def analyze_inference_result_cost_gaps_ground_truth(analysis_info, plot_name, save, show):
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    cost_evaluation_module = get_module("cost_evaluation", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "inference_result.json")
    inference_result = read_inference_result_module.read_inference_result_json(
        output_file_path, function_name)

    # Retrieve the input data
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    input_data = read_input_data_json(input_data_file_path)
    runtime_cost_data = cost_evaluation_module.create_runtime_cost_data(
        input_data)

    # Get necessary functions and info for plotting
    if data_analysis_mode == "opt":
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
    else:
        decompose_inference_result = read_inference_result_module.decompose_posterior_distribution
    get_gap_ground_truth = cost_evaluation_module.get_gap_ground_truth

    if data_analysis_mode == "opt":
        plot_function = plot_gaps_ground_truth_opt
    else:
        plot_function = plot_gaps_ground_truth_bayesian

    # Plot a histogram of the gaps between the predicted costs and true
    # worst-case costs
    plot_function(runtime_cost_data, inference_result,
                  decompose_inference_result, get_gap_ground_truth, analysis_info,
                  plot_name, save, show)


# Plot the inferred cost bound(s)


cost_bound_plot_lim_dict = {
    "MapAppend": 70,
    "BubbleSort": 8000,
    "Concat": 3000,
    "EvenSplitOddTail": 400,
    "InsertionSort2": 400,
    "MedianOfMedians": 1500,
    "QuickSelect": 2000,
    "QuickSort": 50000,
    "Round": 400,
    "ZAlgorithm": 300
}


def analyze_inference_result_cost_bound(analysis_info, plot_name, save, show, axis_label):
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    cost_evaluation_module = get_module("cost_evaluation", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "inference_result.json")
    inference_result = read_inference_result_module.read_inference_result_json(
        output_file_path, function_name)

    # Retrieve the input data
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    input_data = read_input_data_json(input_data_file_path)
    runtime_cost_data = cost_evaluation_module.create_runtime_cost_data(
        input_data)

    # Get necessary functions and info for plotting
    if data_analysis_mode == "opt":
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
    else:
        decompose_inference_result = read_inference_result_module.decompose_posterior_distribution
    get_ground_truth_cost = cost_evaluation_module.get_ground_truth_cost
    get_predicted_cost = cost_evaluation_module.get_predicted_cost
    lim = cost_bound_plot_lim_dict[benchmark_name]

    if benchmark_name == "MapAppend" or benchmark_name == "Concat":
        if data_analysis_mode == "opt":
            plot_function = plot_inferred_cost_bound_3D
        else:
            plot_function = plot_median_cost_bound_3D
    else:
        if data_analysis_mode == "opt":
            plot_function = plot_inferred_cost_bound
        else:
            plot_function = plot_posterior_distribution_cost_bound

    # Plot the inferred cost bound(s)
    plot_function(runtime_cost_data, inference_result, decompose_inference_result,
                  get_ground_truth_cost, get_predicted_cost, lim, analysis_info,
                  plot_name, save, show, axis_label)


# Analyze the inference result of a single experiment by plotting the cost gaps,
# the gaps between the predicted costs and true worst-case costs, and the
# inferred cost bounds(s)


def analyze_inference_result_single_experiment(analysis_info):
    save = True
    show = False

    # If we choose not to show the plots, we will print out messages on the
    # terminal to indicate which benchmark we are currently processing.
    if not show:
        benchmark_name = analysis_info["benchmark_name"]
        hybrid_mode = analysis_info["hybrid_mode"]
        data_analysis_mode = analysis_info["data_analysis_mode"]
        print("Plot inference results: benchmark = {}, hybrid mode = {}, data analysis mode = {}".format(
            benchmark_name, hybrid_mode, data_analysis_mode))

    analyze_inference_result_cost_gaps(analysis_info, "cost_gaps", save, show)
    analyze_inference_result_cost_gaps_ground_truth(
        analysis_info, "cost_gaps_ground_truth", save, show)
    analyze_inference_result_cost_bound(
        analysis_info, "inferred_cost_bound", save, show, axis_label=True)
    analyze_inference_result_cost_bound(
        analysis_info, "inferred_cost_bound_no_axis_labels", save, show, axis_label=False)
    analyze_inference_result_cost_gaps_annotated_code(
        analysis_info, "cost_gaps_annotated_code", save, show)


def analyze_inference_result_benchmark_hybrid_mode(benchmark_name, hybrid_mode):
    analysis_info_opt = {"benchmark_name": benchmark_name,
                         "hybrid_mode": hybrid_mode,
                         "data_analysis_mode": "opt"}
    analysis_info_bayeswc = {"benchmark_name": benchmark_name,
                             "hybrid_mode": hybrid_mode,
                             "data_analysis_mode": "bayeswc"}
    analysis_info_bayespc = {"benchmark_name": benchmark_name,
                             "hybrid_mode": hybrid_mode,
                             "data_analysis_mode": "bayespc"}

    analyze_inference_result_single_experiment(analysis_info_opt)
    analyze_inference_result_single_experiment(analysis_info_bayeswc)
    analyze_inference_result_single_experiment(analysis_info_bayespc)


def analyze_inference_result_all_benchmarks():
    list_benchmark_hybrid_mode = get_list_benchmark_hybrid_mode()

    for benchmark_hybrid_mode in list_benchmark_hybrid_mode:
        benchmark_name, hybrid_mode = benchmark_hybrid_mode
        analyze_inference_result_benchmark_hybrid_mode(
            benchmark_name, hybrid_mode)


# Calculate relative errors


def analyze_relative_errors(analysis_info):
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    cost_evaluation_module = get_module("cost_evaluation", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "inference_result.json")
    inference_result = read_inference_result_module.read_inference_result_json(
        output_file_path, function_name)

    # Get necessary functions and info for plotting
    if data_analysis_mode == "opt":
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
    else:
        decompose_inference_result = read_inference_result_module.decompose_posterior_distribution
    get_relative_gap_ground_truth = cost_evaluation_module.get_relative_gap_ground_truth

    if benchmark_name == "MapAppend":
        if data_analysis_mode == "opt":
            calculate_relative_error_function = calculate_relative_error_opt_representative_samples_2D
        else:
            calculate_relative_error_function = calculate_relative_error_bayesian_representative_samples_2D
    elif benchmark_name == "Concat":
        if data_analysis_mode == "opt":
            calculate_relative_error_function = calculate_relative_error_opt_representative_samples_2D_concat
        else:
            calculate_relative_error_function = calculate_relative_error_bayesian_representative_samples_2D_concat
    else:
        if data_analysis_mode == "opt":
            calculate_relative_error_function = calculate_relative_error_opt_representative_samples
        else:
            calculate_relative_error_function = calculate_relative_error_bayesian_representative_samples

    # Calculate relative errors
    relative_error = calculate_relative_error_function(
        inference_result, get_relative_gap_ground_truth, decompose_inference_result)

    def create_summary(relative_error):
        return {
            "lower": np.percentile(relative_error, 5),
            "median": np.percentile(relative_error, 50),
            "upper": np.percentile(relative_error, 95),
        }

    relative_error10, relative_error100, relative_error1000 = relative_error
    summary = {"size10": create_summary(relative_error10),
               "size100": create_summary(relative_error100),
               "size1000": create_summary(relative_error1000)}
    return summary


# Calculate a desired quantity (e.g., relative errors and proportions of sound
# cost bounds) of all benchmarks


def calculate_desired_quantity_all_benchmarks(calculate_quantity):

    def calculate_desired_quantity_benchmark_hybrid_mode(benchmark_name, hybrid_mode):
        analysis_info_opt = {"benchmark_name": benchmark_name,
                             "hybrid_mode": hybrid_mode,
                             "data_analysis_mode": "opt"}
        analysis_info_bayeswc = {"benchmark_name": benchmark_name,
                                 "hybrid_mode": hybrid_mode,
                                 "data_analysis_mode": "bayeswc"}
        analysis_info_bayespc = {"benchmark_name": benchmark_name,
                                 "hybrid_mode": hybrid_mode,
                                 "data_analysis_mode": "bayespc"}

        quantity_opt = calculate_quantity(analysis_info_opt)
        quantity_bayeswc = calculate_quantity(analysis_info_bayeswc)
        quantity_bayespc = calculate_quantity(analysis_info_bayespc)
        quantity_dict = {
            "opt": quantity_opt,
            "bayeswc": quantity_bayeswc,
            "bayespc": quantity_bayespc}
        return quantity_dict

    list_benchmark_hybrid_mode = get_list_benchmark_hybrid_mode()

    quantity_benchmark_dict = {}
    for benchmark_hybrid_mode in list_benchmark_hybrid_mode:
        benchmark_name, hybrid_mode = benchmark_hybrid_mode
        quantity_dict = calculate_desired_quantity_benchmark_hybrid_mode(
            benchmark_name, hybrid_mode)

        if benchmark_name not in quantity_benchmark_dict:
            quantity_benchmark_dict[benchmark_name] = {
                hybrid_mode: quantity_dict}
        else:
            quantity_benchmark_dict[benchmark_name][hybrid_mode] = quantity_dict

    return quantity_benchmark_dict


# Get a string of the data analysis mode


def get_string_data_analysis_mode(data_analysis_mode):
    if data_analysis_mode == "opt":
        return "Opt"
    elif data_analysis_mode == "bayeswc":
        return "BayesWC"
    else:
        return "BayesPC"


# Display a table of relative errors


def display_relative_errors_benchmark(benchmark_name, relative_error):

    def get_size_string(benchmark_name, size):
        if benchmark_name == "MapAppend":
            return "({}, {})".format(size, size)
        elif benchmark_name == "Concat":
            return "({}, {})".format(size * 5, size)
        else:
            return str(size)

    def print_relative_error_fixed_size_data_driven_data_analysis_mode(size, size_string, data_analysis_mode, relative_error_dict):
        size_key = "size{}".format(size)
        relative_error = relative_error_dict[data_analysis_mode][size_key]
        string_data_analysis_mode = get_string_data_analysis_mode(
            data_analysis_mode)
        print("{:12} {:7} {:8.2f} {:8.2f} {:8.2f}".format(
            size_string, string_data_analysis_mode, relative_error["lower"], relative_error["median"], relative_error["upper"]))

    def print_relative_error_fixed_size_data_driven(benchmark_name, size, relative_error):
        size_string = get_size_string(benchmark_name, size)
        print_relative_error_fixed_size_data_driven_data_analysis_mode(
            size, size_string, "opt", relative_error)
        print_relative_error_fixed_size_data_driven_data_analysis_mode(
            size, "", "bayeswc", relative_error)
        print_relative_error_fixed_size_data_driven_data_analysis_mode(
            size, "", "bayespc", relative_error)

    def print_relative_error_fixed_size_data_driven_data_hybrid_analysis_mode(size, size_string, data_analysis_mode, relative_error_dict_data_driven, relative_error_dict_hybrid):
        size_key = "size{}".format(size)
        relative_error_data_driven = relative_error_dict_data_driven[data_analysis_mode][size_key]
        relative_error_hybrid = relative_error_dict_hybrid[data_analysis_mode][size_key]
        string_data_analysis_mode = get_string_data_analysis_mode(
            data_analysis_mode)
        print("{:12} {:7} {:8.2f} {:8.2f} {:8.2f} ".format(
            size_string, string_data_analysis_mode, relative_error_data_driven["lower"], relative_error_data_driven["median"], relative_error_data_driven["upper"]), end="")
        print("{:8.2f} {:8.2f} {:8.2f}".format(
            relative_error_hybrid["lower"], relative_error_hybrid["median"], relative_error_hybrid["upper"]))

    def print_relative_error_fixed_size_data_driven_hybrid(benchmark_name, size, relative_error_data_driven, relative_error_hybrid):
        size_string = get_size_string(benchmark_name, size)
        print_relative_error_fixed_size_data_driven_data_hybrid_analysis_mode(
            size, size_string, "opt", relative_error_data_driven, relative_error_hybrid)
        print_relative_error_fixed_size_data_driven_data_hybrid_analysis_mode(
            size, "", "bayeswc", relative_error_data_driven, relative_error_hybrid)
        print_relative_error_fixed_size_data_driven_data_hybrid_analysis_mode(
            size, "", "bayespc", relative_error_data_driven, relative_error_hybrid)

    relative_error_data_driven = relative_error["data_driven"]
    if "hybrid" in relative_error:
        print("{:12} {:7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "Input Size", "Method", "5th", "50th", "95th", "5th", "50th", "95th"))
        relative_error_hybrid = relative_error["hybrid"]

        print_relative_error_fixed_size_data_driven_hybrid(
            benchmark_name, 10, relative_error_data_driven, relative_error_hybrid)
        print_relative_error_fixed_size_data_driven_hybrid(
            benchmark_name, 100, relative_error_data_driven, relative_error_hybrid)
        print_relative_error_fixed_size_data_driven_hybrid(
            benchmark_name, 1000, relative_error_data_driven, relative_error_hybrid)
    else:
        print("{:12} {:7} {:>8} {:>8} {:>8}".format(
            "Input Size", "Method", "5th", "50th", "95th"))
        print_relative_error_fixed_size_data_driven(
            benchmark_name, 10, relative_error_data_driven)
        print_relative_error_fixed_size_data_driven(
            benchmark_name, 100, relative_error_data_driven)
        print_relative_error_fixed_size_data_driven(
            benchmark_name, 1000, relative_error_data_driven)


def display_relative_errors_all_benchmarks(relative_error):
    for benchmark_name, relative_error in relative_error.items():
        print("Benchmark: {}".format(benchmark_name))
        display_relative_errors_benchmark(benchmark_name, relative_error)


# Calculate the proportions of sound cost bounds in a posterior distribution


def get_proportion_sound_cost_bounds(analysis_info):
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    report_posterior_distribution_module = get_module(
        "report_posterior_distribution", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Retrieve the inference result
    bin_path = bin_directory(analysis_info)
    output_file_path = os.path.join(bin_path, "inference_result.json")
    inference_result = read_inference_result_module.read_inference_result_json(
        output_file_path, function_name)

    # If the data analysis mode is Opt, we change the format of the inference
    # result such that it can be processed by the function
    # find_invalid_input_coeffs in the module
    # report_posterior_distribution_module.
    if data_analysis_mode == "opt":
        inference_result = [[coefficient] for coefficient in inference_result]

    # Calculate the proportion of sound cost bounds
    find_invalid_input_coeffs = report_posterior_distribution_module.find_invalid_input_coeffs
    num_unsound_coeffs, total_num_coeffs = find_invalid_input_coeffs(
        inference_result)

    # benchmark_name = analysis_info["benchmark_name"]
    # data_analysis_mode = analysis_info["data_analysis_mode"]
    # print("{}, {}, {}: total number = {}".format(
    #     benchmark_name, hybrid_mode, data_analysis_mode, total_num_coeffs))

    return 1 - num_unsound_coeffs / total_num_coeffs


# Display a table of proportions of sound cost bounds


def display_proportion_sound_cost_bounds(proportion_sound_cost_bonds_all_benchmarks):

    column_benchmark_name = "Benchmark"
    column_data_analysis_mode = "Analysis"
    column_proportion_data_driven = "Data-driven"
    column_proportion_hybrid = "Hybrid"

    list_all_benchmark_names = list_benchmarks_data_driven_hybrid + \
        list_benchmarks_data_driven
    width_benchmark_name = max([len(column_benchmark_name)] + [len(name)
                                                               for name in list_all_benchmark_names])
    width_data_analysis_mode = max([len(column_data_analysis_mode)] + [len(name)
                                                                       for name in ["Opt", "BayesWC", "BayesPC"]])
    width_proportion = max(len(column_proportion_data_driven),
                           len((column_proportion_hybrid)))
    precision_proportion = 1

    def print_proportion_data_driven_data_analysis_mode(benchmark_name, data_analysis_mode, proportion):
        string_data_analysis_mode = get_string_data_analysis_mode(
            data_analysis_mode)
        print("{:{width_benchmark_name}} {:{width_data_analysis_mode}} {:{width_proportion}.{precision_proportion}f}%".format(
            benchmark_name,
            string_data_analysis_mode,
            100 * proportion[data_analysis_mode],
            width_benchmark_name=width_benchmark_name,
            width_data_analysis_mode=width_data_analysis_mode,
            width_proportion=width_proportion-1,
            precision_proportion=precision_proportion))

    def print_proportion_data_driven_hybrid_data_analysis_mode(benchmark_name, data_analysis_mode, proportion_data_driven, proportion_hybrid):
        string_data_analysis_mode = get_string_data_analysis_mode(
            data_analysis_mode)
        print("{:{width_benchmark_name}} {:{width_data_analysis_mode}} {:{width_proportion}.{precision_proportion}f}% {:{width_proportion}.{precision_proportion}f}%".format(
            benchmark_name,
            string_data_analysis_mode,
            100 * proportion_data_driven[data_analysis_mode],
            100 * proportion_hybrid[data_analysis_mode],
            width_benchmark_name=width_benchmark_name,
            width_data_analysis_mode=width_data_analysis_mode,
            width_proportion=width_proportion-1,
            precision_proportion=precision_proportion))

    print("{:{width_benchmark_name}} {:{width_data_analysis_mode}} {:{width_proportion}} {:>{width_proportion}}".format(
          column_benchmark_name, column_data_analysis_mode,
          column_proportion_data_driven, column_proportion_hybrid,
          width_benchmark_name=width_benchmark_name,
          width_data_analysis_mode=width_data_analysis_mode,
          width_proportion=width_proportion))

    for benchmark_name, proportion_sound_cost_bonds in proportion_sound_cost_bonds_all_benchmarks.items():
        if "hybrid" in proportion_sound_cost_bonds:
            proportion_data_driven = proportion_sound_cost_bonds["data_driven"]
            proportion_hybrid = proportion_sound_cost_bonds["hybrid"]
            print_proportion_data_driven_hybrid_data_analysis_mode(
                benchmark_name, "opt", proportion_data_driven, proportion_hybrid)
            print_proportion_data_driven_hybrid_data_analysis_mode(
                "", "bayeswc", proportion_data_driven, proportion_hybrid)
            print_proportion_data_driven_hybrid_data_analysis_mode(
                "", "bayespc", proportion_data_driven, proportion_hybrid)
        else:
            proportion = proportion_sound_cost_bonds["data_driven"]
            print_proportion_data_driven_data_analysis_mode(
                benchmark_name, "opt", proportion)
            print_proportion_data_driven_data_analysis_mode(
                "", "bayeswc", proportion)
            print_proportion_data_driven_data_analysis_mode(
                "", "bayespc", proportion)


# Get the execution time of AARA


def get_execution_time(analysis_info):
    # Retrieve the execution time
    bin_path = bin_directory(analysis_info)
    execution_time_file_path = os.path.join(bin_path, "execution_time.json")
    execution_time_dict = read_execution_time_json(execution_time_file_path)
    return execution_time_dict


# Display a table of execution time of AARA


def display_execution_time(execution_time_all_benchmarks):

    column_benchmark_name = "Benchmark"
    column_data_analysis_mode = "Analysis"
    column_time = "Total time"
    column_num_chains = "#Chains"
    column_num_iterations = "#Iterations"

    list_all_benchmark_names = list_benchmarks_data_driven_hybrid + \
        list_benchmarks_data_driven
    width_benchmark_name = max([len(column_benchmark_name)] + [len(name)
                                                               for name in list_all_benchmark_names])
    width_data_analysis_mode = max([len(column_data_analysis_mode)] + [len(name)
                                                                       for name in ["Opt", "BayesWC", "BayesPC"]])
    width_time = len(column_time)
    width_num_chains = len(column_num_chains)
    width_num_iterations = len(column_num_iterations)
    precision_time = 2

    def get_corrected_execution_time(data_analysis_mode, execution_time_dict):
        execution_time = execution_time_dict[data_analysis_mode]["execution_time"]
        num_chains = execution_time_dict[data_analysis_mode]["num_chains"]
        num_samples_per_chain_original = execution_time_dict[
            data_analysis_mode]["num_samples_per_chain"]

        if data_analysis_mode == "bayeswc":
            # For BayesWC, we use Stan, and the number of iterations supplied to
            # Stan when we invoke it is only the number of iterations "after"
            # warmup. So we need to add the number of warmup iterations in order
            # to obtain the total number of iterations (per chain). Pystan
            # (i.e., the Python-Stan binding) calls the function lookup_default
            # in httpstan:
            # https://github.com/stan-dev/httpstan/blob/09c0cf229fd3276babe262355155b95ed02d336f/httpstan/services/arguments.py#L35.
            # In turn, httpStan in turn calls CmdStan, which stores the default
            # value of 1000 for the number of warmup iterations:
            # https://github.com/stan-dev/cmdstan/blob/c8f2e95e56d33c99910d4f8163a3da0390c6e2ad/src/cmdstan/arguments/arg_sample.hpp#L21.
            num_samples_per_chain = num_samples_per_chain_original + 1000
        else:
            num_samples_per_chain = num_samples_per_chain_original
        return execution_time, num_chains, num_samples_per_chain

    def print_execution_time_data_driven(benchmark_name, data_analysis_mode, execution_time_dict):
        string_data_analysis_mode = get_string_data_analysis_mode(
            data_analysis_mode)
        execution_time, num_chains, num_samples_per_chain = get_corrected_execution_time(
            data_analysis_mode, execution_time_dict)
        print("{:{width_benchmark_name}} {:{width_data_analysis_mode}} {:{width_time}.{precision_time}f}s {:{width_num_chains}} {:{width_num_iterations}}".format(
            benchmark_name,
            string_data_analysis_mode,
            execution_time,
            num_chains,
            num_samples_per_chain,
            width_benchmark_name=width_benchmark_name,
            width_data_analysis_mode=width_data_analysis_mode,
            width_time=width_time-1,
            precision_time=precision_time,
            width_num_chains=width_num_chains,
            width_num_iterations=width_num_iterations
        ))

    def print_execution_time_data_driven_hybrid(benchmark_name, data_analysis_mode, execution_time_data_driven, execution_time_hybrid):
        string_data_analysis_mode = get_string_data_analysis_mode(
            data_analysis_mode)
        execution_time_data_driven, num_chains_data_driven, num_samples_per_chain_data_driven = get_corrected_execution_time(
            data_analysis_mode, execution_time_data_driven)
        execution_time_hybrid, num_chains_hybrid, num_samples_per_chain_hybrid = get_corrected_execution_time(
            data_analysis_mode, execution_time_hybrid)
        print("{:{width_benchmark_name}} {:{width_data_analysis_mode}} {:{width_time}.{precision_time}f}s {:{width_num_chains}} {:{width_num_iterations}} {:{width_time}.{precision_time}f}s {:{width_num_chains}} {:{width_num_iterations}}".format(
            benchmark_name,
            string_data_analysis_mode,
            execution_time_data_driven,
            num_chains_data_driven,
            num_samples_per_chain_data_driven,
            execution_time_hybrid,
            num_chains_hybrid,
            num_samples_per_chain_hybrid,
            width_benchmark_name=width_benchmark_name,
            width_data_analysis_mode=width_data_analysis_mode,
            width_time=width_time-1,
            precision_time=precision_time,
            width_num_chains=width_num_chains,
            width_num_iterations=width_num_iterations))

    print("{:{width_benchmark_name}} {:{width_data_analysis_mode}} {:{width_time}} {:{width_num_chains}} {:{width_num_iterations}} {:{width_time}} {:{width_num_chains}} {:{width_num_iterations}}".format(
        column_benchmark_name, column_data_analysis_mode, column_time,
        column_num_chains, column_num_iterations, column_time,
        column_num_chains, column_num_iterations,
        width_benchmark_name=width_benchmark_name,
        width_data_analysis_mode=width_data_analysis_mode,
        width_time=width_time,
        width_num_chains=width_num_chains,
        width_num_iterations=width_num_iterations))

    for benchmark_name, execution_time_dict in execution_time_all_benchmarks.items():
        if "hybrid" in execution_time_dict:
            execution_time_data_driven = execution_time_dict["data_driven"]
            execution_time_hybrid = execution_time_dict["hybrid"]
            print_execution_time_data_driven_hybrid(
                benchmark_name, "opt", execution_time_data_driven, execution_time_hybrid)
            print_execution_time_data_driven_hybrid(
                "", "bayeswc", execution_time_data_driven, execution_time_hybrid)
            print_execution_time_data_driven_hybrid(
                "", "bayespc", execution_time_data_driven, execution_time_hybrid)
        else:
            execution_time = execution_time_dict["data_driven"]
            print_execution_time_data_driven(
                benchmark_name, "opt", execution_time)
            print_execution_time_data_driven("", "bayeswc", execution_time)
            print_execution_time_data_driven("", "bayespc", execution_time)


if __name__ == "__main__":
    script_arg = sys.argv[1]

    if script_arg == "plot":
        analyze_inference_result_all_benchmarks()
    elif script_arg == "soundness":
        print("Proportions of sound cost bounds of benchmarks")
        proportion_sound_cost_bounds_dict = calculate_desired_quantity_all_benchmarks(
            get_proportion_sound_cost_bounds)
        display_proportion_sound_cost_bounds(proportion_sound_cost_bounds_dict)
    elif script_arg == "plot_relative_errors":
        relative_error = calculate_desired_quantity_all_benchmarks(
            analyze_relative_errors)
        plot_relative_errors_selected_benchmarks(relative_error)
    elif script_arg == "relative_errors":
        print("Relative errors of benchmarks")
        relative_error = calculate_desired_quantity_all_benchmarks(
            analyze_relative_errors)
        display_relative_errors_all_benchmarks(relative_error)
    else:
        print("Analysis time of Hybrid AARA")
        execution_time_dict = calculate_desired_quantity_all_benchmarks(
            get_execution_time)
        display_execution_time(execution_time_dict)
