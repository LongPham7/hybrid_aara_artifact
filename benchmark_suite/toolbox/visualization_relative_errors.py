import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import itertools


# plt.rcParams.update({
#     "font.family": "serif",
#     "text.usetex": True,
#     'text.latex.preamble': r"""
#         \usepackage[tt=false]{libertine}
#         \usepackage[varqu]{zi4}
#         \usepackage[libertine]{newtxmath}"""
# })

c = ['tab:red', 'tab:blue', 'y']
m = 'o'


def plot_relative_errors_benchmark(ax, relative_error_benchmark):
    ax.axhline(0, color='k')
    ymin = -1
    ax.fill_between(x=[0, 10], y1=0, y2=ymin, color='black', alpha=.05)

    input_sizes = [10, 100, 1000]

    for i, size in enumerate(input_sizes):
        size_string = "size{}".format(size)

        relative_error_opt = relative_error_benchmark["data_driven"]["opt"][size_string]["lower"]
        ax.scatter([i+.15], relative_error_opt, marker=m, color=c[0])

        relative_error_bayeswc = list(
            relative_error_benchmark["data_driven"]["bayeswc"][size_string].values())
        ax.plot([i+.30]*3, relative_error_bayeswc, marker=m, color=c[1])

        relative_error_bayespc = list(
            relative_error_benchmark["data_driven"]["bayespc"][size_string].values())
        ax.plot([i+.45]*3, relative_error_bayespc, marker=m, color=c[2])

        if "hybrid" not in relative_error_benchmark:
            continue

        relative_error_opt = relative_error_benchmark["hybrid"]["opt"][size_string]["lower"]
        ax.scatter([i+.65], relative_error_opt, facecolor='none')

        relative_error_bayeswc = list(
            relative_error_benchmark["hybrid"]["bayeswc"][size_string].values())
        ax.plot([i+.80]*3, relative_error_bayeswc,
                c=c[1], marker=m, mfc='none', mec=c[1])

        relative_error_bayespc = list(
            relative_error_benchmark["hybrid"]["bayespc"][size_string].values())
        ax.plot([i+.95]*3, relative_error_bayespc,
                c=c[2], marker=m, mfc='none', mec=c[2])

    ax.set_xlabel('Input Size')
    ax.set_xlim([0, 3.1])
    ax.set_xticks(
        [0, 0.5, 1, 1.5, 2, 2.5],
        labels=list(itertools.chain.from_iterable([['', i] for i in input_sizes])))
    xticks = ax.xaxis.get_major_ticks()
    for i in [1, 3, 5]:
        xticks[i].tick1line.set_visible(False)
    xticks[2].tick1line.set_markeredgewidth(2)
    xticks[4].tick1line.set_markeredgewidth(2)
    ax.set_ylim([ymin, None])


def plot_relative_errors_selected_benchmarks(relative_error):
    gs_kw = dict(width_ratios=[2, 2, 2, 1, 1])
    fig, axes = plt.subplots(
        tight_layout=True, gridspec_kw=gs_kw, figsize=(10, 3), ncols=5)

    # gs = gridspec.GridSpec(2, 4, height_ratios=[10,1])
    # axes = [
    #     fig.add_subplot(gs[0, 0]),
    #     fig.add_subplot(gs[0, 1]),
    #     fig.add_subplot(gs[0, 2]),
    #     fig.add_subplot(gs[0, 3]),
    #     fig.add_subplot(gs[1, :]),
    # ]

    plot_relative_errors_benchmark(axes[0], relative_error['QuickSort'])
    plot_relative_errors_benchmark(axes[1], relative_error['QuickSelect'])
    plot_relative_errors_benchmark(axes[2], relative_error['MedianOfMedians'])
    plot_relative_errors_benchmark(axes[3], relative_error['Round'])
    plot_relative_errors_benchmark(
        axes[4], relative_error['EvenSplitOddTail'])

    axes[0].set_title(r'\texttt{QuickSort}')
    axes[1].set_title(r'\texttt{QuickSelect}')
    axes[2].set_title(r'\texttt{MedianOfMedians}')
    axes[3].set_title(r'\texttt{Round}')
    axes[4].set_title(r'\texttt{EvenOddTail}')

    axes[0].set_ylabel('Relative Gap in Inferred Cost Bound')
    axes[0].set_yscale('symlog', base=2)
    axes[1].set_yscale('symlog', base=2)
    axes[2].set_yscale('symlog', base=2)
    axes[3].set_yscale('symlog', base=2)
    axes[4].set_yscale('symlog', base=2)

    yticks = [-1, 0] + [2**i for i in range(12)]
    axes[0].set_yticks(yticks, labels=map(str, yticks))
    yticks = [-1, 0] + [2**i for i in range(9)]
    axes[1].set_yticks(yticks, labels=map(str, yticks))
    yticks = [-1, 0] + [2**i for i in range(3)]
    axes[2].set_yticks(yticks, labels=map(str, yticks))
    yticks = [-1, 0] + [2**i for i in range(4)]
    axes[3].set_yticks(yticks, labels=map(str, yticks))
    yticks = [-1, 0] + [2**i for i in range(4)]
    axes[4].set_yticks(yticks, labels=map(str, yticks))

    axes[3].text(x=.98, y=.15, s='Unsound\nRegion', va='top',
                 ha='right', transform=axes[3].transAxes)
    axes[3].text(x=.98, y=.95, s='Sound\nRegion', va='top',
                 ha='right', transform=axes[3].transAxes)

    axes[1].plot([], [], c=c[0], marker=m, mec=c[0],
                 mfc=c[0], label=r'\textsc{OPT} Data-Driven')
    axes[1].plot([], [], c=c[1], marker=m, mec=c[1], mfc=c[1],
                 label=r'\textsc{BayesWC} Data-Driven')
    axes[1].plot([], [], c=c[2], marker=m, mec=c[2], mfc=c[2],
                 label=r'\textsc{BayesPC} Data-Driven')
    axes[1].plot([], [], c=c[0], marker=m, mec=c[0],
                 mfc='none', label=r'\textsc{OPT} Hybrid')
    axes[1].plot([], [], c=c[1], marker=m, mec=c[1],
                 mfc='none', label=r'\textsc{BayesWC} Hybrid')
    axes[1].plot([], [], c=c[2], marker=m, mec=c[2],
                 mfc='none', label=r'\textsc{BayesPC} Hybrid')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(
        0.5, .95), ncols=6, fancybox=False, framealpha=1)

    # Save the plot
    image_directory_path = os.path.expanduser(os.path.join(
        "/home", "hybrid_aara", "benchmark_suite", "images"))
    if not os.path.exists(image_directory_path):
        os.makedirs(image_directory_path)
    image_path = os.path.join(image_directory_path, "relative_errors.pdf")
    plt.savefig(image_path, format="pdf", dpi=300, bbox_inches='tight')
