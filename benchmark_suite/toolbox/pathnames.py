import os

# Figure out the bin directory of a benchmark


def bin_directory(analysis_info):
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Project directory
    project_path = os.path.expanduser(
        os.path.join("/home", "hybrid_aara", "benchmark_suite"))

    # Create a bin directory if necessary
    bin_path = os.path.join(
        project_path, benchmark_name, "bin", hybrid_mode, data_analysis_mode)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    return bin_path

# Figure out the image directory of a benchmark


def image_directory(analysis_info):
    benchmark_name = analysis_info["benchmark_name"]
    hybrid_mode = analysis_info["hybrid_mode"]
    data_analysis_mode = analysis_info["data_analysis_mode"]

    # Project directory
    project_path = os.path.expanduser(
        os.path.join("/home", "hybrid_aara", "benchmark_suite"))

    # Create a bin directory if necessary
    bin_path = os.path.join(
        project_path, benchmark_name, "images", hybrid_mode, data_analysis_mode)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    return bin_path
