import os
import sys
import importlib.util


# Get a module inside each benchmark's utility directory


def get_module(module_name, analysis_info):
    benchmark_name = analysis_info["benchmark_name"]

    # Benchmark directory
    benchmark_directory = os.path.expanduser(os.path.join(
        "/home", "hybrid_aara", "benchmark_suite", benchmark_name))
    utility_directory = os.path.join(benchmark_directory, "utility")

    # Import the config module
    module_path = os.path.join(utility_directory, "{}.py".format(module_name))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# List all combinations of benchmarks names and hybrid modes


list_benchmarks_data_driven_hybrid = [
    "MapAppend", "Concat", "InsertionSort2", "MedianOfMedians", "QuickSelect", "QuickSort", "ZAlgorithm"]

list_benchmarks_data_driven = [
    "BubbleSort", "Round", "EvenSplitOddTail"]


def get_list_benchmark_hybrid_mode():

    list_benchmark_hybrid_mode = []
    for benchmark_name in list_benchmarks_data_driven_hybrid:
        list_benchmark_hybrid_mode.append((benchmark_name, "data_driven"))
        list_benchmark_hybrid_mode.append((benchmark_name, "hybrid"))

    for benchmark_name in list_benchmarks_data_driven:
        list_benchmark_hybrid_mode.append((benchmark_name, "data_driven"))

    return list_benchmark_hybrid_mode
