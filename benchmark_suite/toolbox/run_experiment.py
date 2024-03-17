import multiprocessing
import sys
from joblib import Parallel, delayed

from benchmark_manipulation import get_module, get_list_benchmark_hybrid_mode, \
    list_benchmarks_data_driven_hybrid
from aara import run_aara


# Run AARA to generate inference results


def run_single_experiment(analysis_info):
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Import modules
    config_module = get_module("config", analysis_info)
    ocaml_code_module = get_module("ocaml_code", analysis_info)

    # Add a function name to analysis_info
    function_name = config_module.function_name_dict[hybrid_mode]
    analysis_info["function_name"] = function_name

    # Get the degree
    degree = config_module.degree

    # Get input data
    if (benchmark_name == "QuickSelect" or benchmark_name == "QuickSort") and hybrid_mode == "hybrid" and data_analysis_mode == "bayespc":
        input_data = config_module.input_data_extracted
    else:
        input_data = config_module.input_data

    # Get a config
    config = config_module.config_dict[hybrid_mode][data_analysis_mode]

    # Get OCaml code
    ocaml_code = ocaml_code_module.create_ocaml_code(input_data, analysis_info)

    # Perform AARA
    run_aara(ocaml_code, input_data, degree, config, analysis_info)


def run_experiment_benchmark_hybrid_mode_data_analysis_mode(benchmark_name, hybrid_mode, data_analysis_mode):
    if data_analysis_mode == "opt":
        analysis_info = {"benchmark_name": benchmark_name,
                         "hybrid_mode": hybrid_mode,
                         "data_analysis_mode": "opt"}
    elif data_analysis_mode == "bayeswc":
        analysis_info = {"benchmark_name": benchmark_name,
                         "hybrid_mode": hybrid_mode,
                         "data_analysis_mode": "bayeswc"}
    else:
        analysis_info = {"benchmark_name": benchmark_name,
                         "hybrid_mode": hybrid_mode,
                         "data_analysis_mode": "bayespc"}
    run_single_experiment(analysis_info)


def run_experiment_benchmark_hybrid_mode(benchmark_name, hybrid_mode):
    run_experiment_benchmark_hybrid_mode_data_analysis_mode(
        benchmark_name, hybrid_mode, "opt")
    run_experiment_benchmark_hybrid_mode_data_analysis_mode(
        benchmark_name, hybrid_mode, "bayeswc")
    run_experiment_benchmark_hybrid_mode_data_analysis_mode(
        benchmark_name, hybrid_mode, "bayespc")


def run_all_benchmarks():
    list_benchmark_hybrid_mode = get_list_benchmark_hybrid_mode()

    # Sequential computation
    # for benchmark_hybrid_mode in list_benchmark_hybrid_mode:
    #     benchmark_name, hybrid_mode = benchmark_hybrid_mode
    #     run_experiment_benchmark_hybrid_mode(benchmark_name, hybrid_mode)

    # Parallel computation
    # n_jobs = multiprocessing.cpu_count()
    n_jobs = 1
    Parallel(n_jobs=n_jobs)(delayed(run_experiment_benchmark_hybrid_mode)(
        benchmark_name, hybrid_mode) for benchmark_name, hybrid_mode in list_benchmark_hybrid_mode)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        run_all_benchmarks()
    else:
        if len(sys.argv) == 3:
            # The user provides a benchmark name and hybrid mode
            benchmark_name = sys.argv[1]
            hybrid_mode = sys.argv[2]
            run_experiment_benchmark_hybrid_mode(benchmark_name, hybrid_mode)
        elif len(sys.argv) == 4:
            # The user provides a benchmark name, hybrid mode, and data-analysis
            # mode
            benchmark_name = sys.argv[1]
            hybrid_mode = sys.argv[2]
            data_analysis_mode = sys.argv[3]
            run_experiment_benchmark_hybrid_mode_data_analysis_mode(
                benchmark_name, hybrid_mode, data_analysis_mode)
        else:
            # The user only provides a benchmark name, but not a hybrid mode
            benchmark_name = sys.argv[1]
            run_experiment_benchmark_hybrid_mode(benchmark_name, "data_driven")
            if benchmark_name in list_benchmarks_data_driven_hybrid:
                run_experiment_benchmark_hybrid_mode(benchmark_name, "hybrid")
