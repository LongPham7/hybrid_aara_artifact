import numpy as np
import matplotlib.pyplot as plt
import os
from pathnames import image_directory

# Figure size for plots of inferred cost bounds with axis labels and ticks
FIGSIZE = (3.5, 3)

# Figure size for plots of inferred cost bounds without axis labels and ticks
FIGSIZE_SMALL = (2, 2)

# Originally, we also included the line \usepackage[varqu]{zi4} under
# 'text.latex.preamble' in order to load the LaTex package zi4. However, it
# causes an error in rendering some of the plots.
# plt.rcParams.update({
#     "font.family": "serif",
#     "text.usetex": True,
#     "text.latex.preamble": r"""
#         \usepackage[tt=false]{libertine}
#         \usepackage[libertine]{newtxmath}"""
# })


# Save a plot


def save_plot(plot_name, analysis_info):
    image_directory_path = image_directory(analysis_info)
    image_path = os.path.join(image_directory_path, "{}.pdf".format(plot_name))
    plt.savefig(image_path, format="pdf", dpi=300, bbox_inches='tight')


# Plot cost gaps


def plot_cost_gaps(runtime_cost_data, list_input_coeffs, list_output_coeffs,
                   get_cost_gap, analysis_info, plot_name, save, show):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    num_bins = 20
    cost_gaps_input_potential_only = []
    for input_coeffs in list_input_coeffs:
        for cost_measurement in runtime_cost_data:
            cost_gap = get_cost_gap(cost_measurement, input_coeffs, None)
            cost_gaps_input_potential_only.append(cost_gap)
    ax1.hist(cost_gaps_input_potential_only, bins=num_bins)
    ax1.set_xlabel("Cost gap")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Cost gaps with only input potential")

    cost_gaps_both_input_and_output = []
    for input_coeffs, output_coeffs in zip(list_input_coeffs, list_output_coeffs):
        for cost_measurement in runtime_cost_data:
            cost_gap = get_cost_gap(
                cost_measurement, input_coeffs, output_coeffs)
            cost_gaps_both_input_and_output.append(cost_gap)
    ax2.hist(cost_gaps_both_input_and_output, bins=num_bins)
    ax2.set_xlabel("Cost gap")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Cost gaps with both input and output potential")

    # We may want to prevent scientific notation (e.g., 1.2e10) in the axis
    # labels. Otherwise, the rendering engine for Tex crashes.
    # ax1.ticklabel_format(useOffset=False, style='plain')
    # ax2.ticklabel_format(useOffset=False, style='plain')

    if save:
        save_plot(plot_name, analysis_info)
    if show:
        plt.show()


def plot_cost_gaps_opt(runtime_cost_data, inferred_coefficients,
                       decompose_inferred_coefficients, get_cost_gap,
                       analysis_info, plot_name, save=True, show=True):
    input_coeffs, output_coeffs = decompose_inferred_coefficients(
        inferred_coefficients)
    plot_cost_gaps(
        runtime_cost_data, [input_coeffs], [output_coeffs], get_cost_gap,
        analysis_info, plot_name, save, show)


def plot_cost_gaps_bayesian(runtime_cost_data, posterior_distribution,
                            decompose_posterior_distribution, get_cost_gap,
                            analysis_info, plot_name, save=True, show=True):
    list_input_coeffs, list_output_coeffs = decompose_posterior_distribution(
        posterior_distribution)
    plot_cost_gaps(
        runtime_cost_data, list_input_coeffs, list_output_coeffs, get_cost_gap,
        analysis_info, plot_name, save, show)


# Plot the gaps between the predicted costs and true worst-case costs


def plot_gaps_ground_truth(runtime_cost_data, list_input_coeffs, get_gap_ground_truth,
                           analysis_info, plot_name, save, show):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    num_bins = 50
    gaps_ground_truth = []
    for input_coeffs in list_input_coeffs:
        for cost_measurement in runtime_cost_data:
            input_size, _ = cost_measurement
            cost_gap = get_gap_ground_truth(input_size, input_coeffs)
            gaps_ground_truth.append(cost_gap)

    ax.hist(gaps_ground_truth, bins=num_bins)
    ax.set_xlabel("Cost gap")
    ax.set_ylabel("Frequency")
    ax.set_yticks([])
    ax.set_title("Distance from the ground truth")

    # We may want to prevent scientific notation (e.g., 1.2e10) in the axis
    # labels. Otherwise, the rendering engine for Tex crashes.
    # ax.ticklabel_format(useOffset=False, style='plain')

    if save:
        save_plot(plot_name, analysis_info)
    if show:
        plt.show()


def plot_gaps_ground_truth_opt(runtime_cost_data, inferred_coefficients,
                               decompose_inferred_coefficients, get_gap_ground_truth,
                               analysis_info, plot_name, save=True, show=True):
    input_coeffs, _ = decompose_inferred_coefficients(inferred_coefficients)
    plot_gaps_ground_truth(
        runtime_cost_data, [input_coeffs], get_gap_ground_truth, analysis_info,
        plot_name, save, show)


def plot_gaps_ground_truth_bayesian(runtime_cost_data, posterior_distribution,
                                    decompose_posterior_distribution, get_gap_ground_truth,
                                    analysis_info, plot_name, save=True, show=True):
    list_input_coeffs, _ = decompose_posterior_distribution(
        posterior_distribution)
    plot_gaps_ground_truth(
        runtime_cost_data, list_input_coeffs, get_gap_ground_truth, analysis_info,
        plot_name, save, show)


# Plot two-dimensional runtime cost data


def calculate_max_sizes(runtime_cost_data):
    input_sizes1 = [input_size1 for ((input_size1, _), _) in runtime_cost_data]
    input_sizes2 = [input_size2 for ((_, input_size2), _) in runtime_cost_data]
    max_input_size1 = np.max(input_sizes1)
    max_input_size2 = np.max(input_sizes2)
    return max_input_size1, max_input_size2


def plot_runtime_cost_data_3D(ax, runtime_cost_data):
    input_sizes1 = [input_size1 for ((input_size1, _), _) in runtime_cost_data]
    input_sizes2 = [input_size2 for ((_, input_size2), _) in runtime_cost_data]
    costs = [cost for (_, cost) in runtime_cost_data]
    ax.scatter(input_sizes1, input_sizes2, costs,
               color='k', marker='.', label="Runtime Data")


# Plot a two-dimensional true worst-case cost bound


def plot_true_cost_bound_3D(ax, runtime_cost_data, get_ground_truth_cost):
    max_input_size1, max_input_size2 = calculate_max_sizes(runtime_cost_data)
    x_range = max_input_size1 * 1.5
    y_range = max_input_size2 * 1.5
    xs = np.linspace(0, x_range + 1, 100)
    ys = np.linspace(0, y_range + 1, 100)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    true_cost_bound = get_ground_truth_cost((X, Y))
    ax.plot_surface(X, Y, true_cost_bound, color='r', alpha=1/5)


# Determine the figure size of inferred cost bounds based on whether the axis
# labels and ticks will be shown


def get_figure_size(axis_label):
    if axis_label:
        return FIGSIZE
    else:
        return FIGSIZE_SMALL


# Plot a two-dimensional cost bound inferred by Opt


def plot_inferred_cost_bound_3D(runtime_cost_data, inferred_coefficients,
                                decompose_inferred_coefficients, get_ground_truth_cost,
                                get_predicted_cost, zlim, analysis_info,
                                plot_name, save=True, show=True, axis_label=True):
    figsize = get_figure_size(axis_label)
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    max_input_size1, max_input_size2 = calculate_max_sizes(runtime_cost_data)
    x_range = max_input_size1 * 1.5
    y_range = max_input_size2 * 1.5
    xs = np.linspace(0, x_range + 1, 100)
    ys = np.linspace(0, y_range + 1, 100)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Inferred cost bound
    input_coeffs, _ = decompose_inferred_coefficients(
        inferred_coefficients)
    predicted_cost_bounds = get_predicted_cost(
        (X, Y), input_coeffs, None)
    ax.plot_surface(X, Y, predicted_cost_bounds, color='b', alpha=1/5)

    # Runtime cost data
    plot_runtime_cost_data_3D(ax, runtime_cost_data)

    # True worst-case bound
    plot_true_cost_bound_3D(ax, runtime_cost_data, get_ground_truth_cost)

    ax.set_xlim(0, x_range+1)
    ax.set_ylim(0, y_range+1)
    ax.set_zlim(0, zlim)

    if axis_label:
        benchmark_name = analysis_info["benchmark_name"]
        if benchmark_name == "MapAppend":
            ax.set_xlabel("Input Size 1")
            ax.set_ylabel("Input Size 2")
        else:
            ax.set_xlabel("Total Size")
            ax.set_ylabel("Outer List Size")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if save:
        save_plot(plot_name, analysis_info)
    if show:
        plt.show()


# Plot a two-dimensional median cost bound inferred by Bayesian analysis


def plot_median_cost_bound_3D(runtime_cost_data, posterior_distribution,
                              decompose_posterior_distribution, get_ground_truth_cost,
                              get_predicted_cost, zlim, analysis_info,
                              plot_name, save=True, show=True, axis_label=True):
    figsize = get_figure_size(axis_label)
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    max_input_size1, max_input_size2 = calculate_max_sizes(runtime_cost_data)
    x_range = max_input_size1 * 1.5
    y_range = max_input_size2 * 1.5
    xs = np.linspace(0, x_range + 1, 100)
    ys = np.linspace(0, y_range + 1, 100)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Calculate the median cost bound and plot it
    list_input_coeffs, _ = decompose_posterior_distribution(
        posterior_distribution)
    list_predicted_cost_bounds = []
    for input_coeffs in list_input_coeffs:
        list_predicted_cost_bounds.append(
            get_predicted_cost((X, Y), input_coeffs, None))
    median_predicted_cost_bounds = np.median(
        list_predicted_cost_bounds, axis=0)
    ax.plot_surface(X, Y, median_predicted_cost_bounds, color='b', alpha=1/5)

    # Runtime cost data
    plot_runtime_cost_data_3D(ax, runtime_cost_data)

    # True worst-case bound
    plot_true_cost_bound_3D(ax, runtime_cost_data, get_ground_truth_cost)

    ax.set_xlim(0, x_range+1)
    ax.set_ylim(0, y_range+1)
    ax.set_zlim(0, zlim)

    if axis_label:
        benchmark_name = analysis_info["benchmark_name"]
        if benchmark_name == "MapAppend":
            ax.set_xlabel("Input Size 1")
            ax.set_ylabel("Input Size 2")
        else:
            ax.set_xlabel("Total Size")
            ax.set_ylabel("Outer List Size")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if save:
        save_plot(plot_name, analysis_info)
    if show:
        plt.show()


# Plot one-dimensional runtime cost data


def calculate_max_size(runtime_cost_data):
    input_sizes = [input_size for (input_size, _) in runtime_cost_data]
    max_input_size = np.max(input_sizes)
    return max_input_size


def plot_runtime_cost_data(ax, runtime_cost_data):
    # Runtime data
    input_sizes = [input_size for (input_size, _) in runtime_cost_data]
    costs = [cost for (_, cost) in runtime_cost_data]
    ax.scatter(input_sizes, costs, color='k', marker='.', label="Runtime Data")


# Plot a one-dimensional true worst-case cost bound


def plot_true_cost_bound(ax, runtime_cost_data, get_ground_truth_cost):
    # True worst-case bound
    max_input_size = calculate_max_size(runtime_cost_data)
    x_range = max_input_size * 1.5
    xs = np.linspace(0, x_range + 1, 100)
    ground_truth_costs = [get_ground_truth_cost(n) for n in xs]
    ax.plot(xs, ground_truth_costs, color='r',
            linestyle="solid", label="True Bound")


def plot_cost_bound(ax, runtime_cost_data, input_coeffs, output_coeffs,
                    get_ground_truth_cost, get_predicted_cost):
    # Make sure that the order of rendering the runtime cost data, true
    # worst-case bound, and inferred cost bound is consistent in both the
    # functions plot_cost_bound (for Opt) and plot_list_cost_bounds (for BayesWC
    # and BayesPC).

    # Runtime cost data
    plot_runtime_cost_data(ax, runtime_cost_data)

    # True worst-case bound
    plot_true_cost_bound(ax, runtime_cost_data, get_ground_truth_cost)

    max_input_size = calculate_max_size(runtime_cost_data)
    x_range = max_input_size * 1.5
    xs = np.linspace(0, x_range + 1, 100)

    # Inferred cost bound
    predicted_cost_bounds = get_predicted_cost(xs, input_coeffs, output_coeffs)
    ax.plot(xs, predicted_cost_bounds, color='b', label="Inferred Bound")

    ax.set_xlim(0, x_range+1)


# Plot the cost bound inferred by Opt


def plot_inferred_cost_bound(runtime_cost_data, inferred_coefficients,
                             decompose_inferred_coefficients, get_ground_truth_cost,
                             get_predicted_cost, ylim, analysis_info,
                             plot_name, save=True, show=True, axis_label=True):
    figsize = get_figure_size(axis_label)
    _, ax = plt.subplots(figsize=figsize, tight_layout=True)

    input_coeffs, _ = decompose_inferred_coefficients(inferred_coefficients)
    plot_cost_bound(ax, runtime_cost_data, input_coeffs, None,
                    get_ground_truth_cost, get_predicted_cost)
    ax.set_ylim(0, ylim)

    # Use a log scale for the y axis in quicksort
    benchmark_name = analysis_info["benchmark_name"]
    if benchmark_name == "QuickSort":
        ax.set_ylim(1, ylim)
        ax.set_yscale('log')

    # Axis labels and ticks
    if axis_label:
        ax.set_xlabel("Input Size")
        ax.set_ylabel("Cost")
    else:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    # Legend
    if axis_label:
        if benchmark_name == "QuickSort":
            ax.legend(loc='lower right', framealpha=1, fancybox=False)
        else:
            ax.legend(loc='upper left', framealpha=1, fancybox=False)

    if save:
        save_plot(plot_name, analysis_info)
    if show:
        plt.show()


# Plot the posterior distribution of cost bounds inferred by Bayesian resource
# analysis


def plot_list_cost_bounds(ax, runtime_cost_data, list_input_coeffs, list_output_coeffs,
                          get_ground_truth_cost, get_predicted_cost):
    max_input_size = calculate_max_size(runtime_cost_data)
    x_range = max_input_size * 1.5
    xs = np.linspace(0, x_range + 1, 100)

    # Runtime cost data
    plot_runtime_cost_data(ax, runtime_cost_data)

    # True worst-case bound
    plot_true_cost_bound(ax, runtime_cost_data, get_ground_truth_cost)

    # Inferred cost bounds
    list_ten_percentiles = []
    list_fifty_percentiles = []
    list_ninety_percentiles = []
    for x in xs:
        list_predicted_cost_bounds = []
        for i in range(0, len(list_input_coeffs)):
            input_coeffs = list_input_coeffs[i]
            if list_output_coeffs is None:
                output_coeffs = None
            else:
                output_coeffs = list_output_coeffs[i]
            predicted_cost_bound = get_predicted_cost(
                x, input_coeffs, output_coeffs)
            list_predicted_cost_bounds.append(predicted_cost_bound)

        ten_percentile = np.percentile(list_predicted_cost_bounds, 10)
        fifty_percentile = np.percentile(list_predicted_cost_bounds, 50)
        ninety_percentile = np.percentile(list_predicted_cost_bounds, 90)
        list_ten_percentiles.append(ten_percentile)
        list_fifty_percentiles.append(fifty_percentile)
        list_ninety_percentiles.append(ninety_percentile)

    ax.plot(xs, list_fifty_percentiles, color='b',
            label="Inferred Bounds")
    ax.fill_between(xs, list_ten_percentiles,
                    list_ninety_percentiles, alpha=0.2)

    ax.set_xlim(0, x_range+1)


def plot_posterior_distribution_cost_bound(runtime_cost_data, posterior_distribution,
                                           decompose_posterior_distribution, get_ground_truth_cost,
                                           get_predicted_cost, ylim, analysis_info,
                                           plot_name, save=True, show=True, axis_label=True):
    figsize = get_figure_size(axis_label)
    _, ax = plt.subplots(figsize=figsize, tight_layout=True)

    list_input_coeffs, _ = decompose_posterior_distribution(
        posterior_distribution)
    plot_list_cost_bounds(ax, runtime_cost_data, list_input_coeffs,
                          None, get_ground_truth_cost, get_predicted_cost)
    ax.set_ylim(0, ylim)

    # Use a log scale for the y axis in quicksort
    benchmark_name = analysis_info["benchmark_name"]
    if benchmark_name == "QuickSort":
        ax.set_ylim(1, ylim)
        ax.set_yscale('log')

    # Axis labels and ticks
    if axis_label:
        ax.set_xlabel("Input Size")
        ax.set_ylabel("Cost")
    else:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    # Legend
    # if benchmark_name == "quicksort":
    #     ax.legend(loc='lower right', framealpha=1, fancybox=False)
    # else:
    #     ax.legend(loc='upper left', framealpha=1, fancybox=False)

    if save:
        save_plot(plot_name, analysis_info)
    if show:
        plt.show()
