import os
import shutil

from benchmark_manipulation import list_benchmarks_data_driven_hybrid, list_benchmarks_data_driven


for benchmark_name in list_benchmarks_data_driven_hybrid:
    # Source directory
    source_image_directory = os.path.expanduser(os.path.join(
        "/home", "hybrid_aara", "statistical_aara_test_suite", benchmark_name, "images"))

    # Destination directory
    destination_image_directory = os.path.expanduser(os.path.join(
        "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", benchmark_name))

    for hybrid_mode in ["data_driven", "hybrid"]:
        for data_analysis_mode in ["opt", "bayeswc", "bayespc"]:
            for plot_name in ["inferred_cost_bound", "inferred_cost_bound_no_axis_labels"]:
                data_driven_directory = os.path.join(
                    source_image_directory, hybrid_mode)
                source_file = os.path.join(data_driven_directory,
                                           data_analysis_mode, "{}.pdf".format(plot_name))
                destination_file = os.path.join(destination_image_directory, hybrid_mode,
                                                data_analysis_mode, "{}.pdf".format(plot_name))
                shutil.copyfile(source_file, destination_file)

for benchmark_name in list_benchmarks_data_driven:
    # Source directory
    source_image_directory = os.path.expanduser(os.path.join(
        "/home", "hybrid_aara", "statistical_aara_test_suite", benchmark_name, "images"))

    # Destination directory
    destination_image_directory = os.path.expanduser(os.path.join(
        "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", benchmark_name))

    for data_analysis_mode in ["opt", "bayeswc", "bayespc"]:
        for plot_name in ["inferred_cost_bound", "inferred_cost_bound_no_axis_labels"]:
            # Data-driven analysis
            data_driven_directory = os.path.join(
                source_image_directory, "data_driven")
            source_file = os.path.join(data_driven_directory,
                                       data_analysis_mode, "{}.pdf".format(plot_name))

            destination_file = os.path.join(destination_image_directory, "data_driven",
                                            data_analysis_mode, "{}.pdf".format(plot_name))
            shutil.copyfile(source_file, destination_file)

# Relative errors
source_file = os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "statistical_aara_test_suite", "images", "relative_errors.pdf"))
destination_file = os.path.expanduser(os.path.join(
    "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", "gaps.pdf"))
shutil.copyfile(source_file, destination_file)
