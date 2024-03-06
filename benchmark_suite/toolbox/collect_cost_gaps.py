import os
import json

from benchmark_manipulation import get_module, get_list_benchmark_hybrid_mode
from pathnames import bin_directory
from json_manipulation import read_input_data_json


def record_cost_gaps(analysis_info, runtime_cost_data, inference_result,
                     get_cost_gap, decompose_inference_result):
    hybrid_mode = analysis_info["hybrid_mode"]

    # Calculate cost gaps
    input_coeffs, output_coeffs = decompose_inference_result(inference_result)
    cost_gaps_input_potential_only = []
    for cost_measurement in runtime_cost_data:
        cost_gap = get_cost_gap(cost_measurement, input_coeffs, None)
        assert (cost_gap >= -0.0001)
        cost_gaps_input_potential_only.append(cost_gap)

    cost_gaps_both_input_and_output = []
    for cost_measurement in runtime_cost_data:
        cost_gap = get_cost_gap(
            cost_measurement, input_coeffs, output_coeffs)
        assert (cost_gap >= -0.0001)
        cost_gaps_both_input_and_output.append(cost_gap)

    # Project directory
    project_path = os.path.expanduser(
        os.path.join("/home", "hybrid_aara", "statistical_aara_test_suite"))

    # Create a bin directory if necessary
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]
    cost_gaps_bin_path = os.path.join(
        project_path, "cost_gaps", benchmark_name, hybrid_mode, data_analysis_mode)
    if not os.path.exists(cost_gaps_bin_path):
        os.makedirs(cost_gaps_bin_path)

    # Record the cost gaps in a JSON file
    file_path = os.path.join(cost_gaps_bin_path, "cost_gaps.json")
    with open(file_path, "w") as write_file:
        json.dump({"input_potential_only": cost_gaps_input_potential_only,
                   "both_input_and_output": cost_gaps_both_input_and_output}, write_file)


def record_cost_gap_benchmark(analysis_info):
    hybrid_mode = analysis_info["hybrid_mode"]

    # Get modules
    config_module = get_module("config", analysis_info)
    read_inference_result_module = get_module(
        "read_inference_result", analysis_info)
    cost_evaluation_module = get_module("cost_evaluation", analysis_info)
    if hybrid_mode == "hybrid":
        annotated_code_cost_module = get_module(
            "annotated_code_cost", analysis_info)

    # Get the function name
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Retrieve the input data
    bin_path = bin_directory(analysis_info)
    input_data_file_path = os.path.join(bin_path, "input_data.json")
    input_data = read_input_data_json(input_data_file_path)
    if hybrid_mode == "data_driven":
        runtime_cost_data = cost_evaluation_module.create_runtime_cost_data(
            input_data)
    else:
        runtime_cost_data = annotated_code_cost_module.create_runtime_cost_data_annotated_code(
            input_data)

    # Retrieve the inference result
    if hybrid_mode == "data_driven":
        output_file_path = os.path.join(bin_path, "inference_result.json")
        function_name = analysis_info["function_name"]
        inference_result = read_inference_result_module.read_inference_result_json(
            output_file_path, function_name)
        get_cost_gap = cost_evaluation_module.get_cost_gap
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients
    else:
        output_lp_vars_file_path = os.path.join(
            bin_path, "output_lp_vars.json")
        inference_result = read_inference_result_module.read_inference_result_annotated_code_json(
            output_lp_vars_file_path)
        get_cost_gap = annotated_code_cost_module.get_cost_gap_annotated_code
        decompose_inference_result = read_inference_result_module.decompose_inferred_coefficients_annotated_code

    record_cost_gaps(analysis_info, runtime_cost_data, inference_result,
                     get_cost_gap, decompose_inference_result)


def record_cost_gap_benchmark_hybrid_mode(benchmark_name, hybrid_mode):
    analysis_info_opt = {"benchmark_name": benchmark_name,
                         "hybrid_mode": hybrid_mode,
                         "data_analysis_mode": "opt"}
    record_cost_gap_benchmark(analysis_info_opt)


def record_cost_gap_all_benchmarks():
    list_benchmark_hybrid_mode = get_list_benchmark_hybrid_mode()

    for benchmark_hybrid_mode in list_benchmark_hybrid_mode:
        benchmark_name, hybrid_mode = benchmark_hybrid_mode
        record_cost_gap_benchmark_hybrid_mode(benchmark_name, hybrid_mode)


if __name__ == "__main__":
    record_cost_gap_all_benchmarks()
