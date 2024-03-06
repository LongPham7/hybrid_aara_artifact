import os
import subprocess

from benchmark_manipulation import list_benchmarks_data_driven_hybrid, list_benchmarks_data_driven

for benchmark_name in list_benchmarks_data_driven_hybrid:
  # Directory in Dropbox storing plots
    dropbox_image_directory = os.path.expanduser(os.path.join(
        "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", benchmark_name))

    for hybrid_mode in ["data_driven", "hybrid"]:
        for data_analysis_mode in ["opt", "bayeswc", "bayespc"]:
            for plot_name in ["inferred_cost_bound", "inferred_cost_bound_no_axis_labels"]:
                # Data-driven analysis
                input_file = os.path.join(dropbox_image_directory, hybrid_mode,
                                          data_analysis_mode, "{}.pdf".format(plot_name))
                output_file = os.path.join(dropbox_image_directory, hybrid_mode,
                                           data_analysis_mode, "{}_crop.pdf".format(plot_name))
                subprocess.run(["pdfcrop", input_file, output_file])

for benchmark_name in list_benchmarks_data_driven:
    # Directory in Dropbox storing plots
    dropbox_image_directory = os.path.expanduser(os.path.join(
        "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", benchmark_name))

    for data_analysis_mode in ["opt", "bayeswc", "bayespc"]:
        for plot_name in ["inferred_cost_bound", "inferred_cost_bound_no_axis_labels"]:
            # Data-driven analysis
            input_file = os.path.join(dropbox_image_directory, "data_driven",
                                      data_analysis_mode, "{}.pdf".format(plot_name))
            output_file = os.path.join(dropbox_image_directory, "data_driven",
                                       data_analysis_mode, "{}_crop.pdf".format(plot_name))
            subprocess.run(["pdfcrop", input_file, output_file])

# Relative errors
input_file = os.path.expanduser(os.path.join(
    "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", "gaps.pdf"))
output_file = os.path.expanduser(os.path.join(
    "~", "Dropbox", "CMU research", "My publications", "bayesian_aara", "pldi_2024", "images", "gaps-crop.pdf"))
subprocess.run(["pdfcrop", input_file, output_file])
