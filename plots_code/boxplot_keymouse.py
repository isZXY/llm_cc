import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from const import *
from matplotlib.legend_handler import HandlerLine2D

if __name__ == '__main__':
    csv = pd.read_csv('../data/boxplot/keymouse_boxplot.csv')
    gameid_list = [400083, 600027, 400154, 600015, 400015, 699997, 600018, 600016, 400016]
    grouped = csv[csv['gameid'].isin(gameid_list)].groupby('gameid')

    data = [group_data['mousee'].tolist() for group_name, group_data in grouped]
    medians = [np.median(group_data['mousee']) for group_name, group_data in grouped]
    order = np.argsort(medians)[::-1]

    fig, ax = plt.subplots(figsize=(4, 2.7))

    plt.rcParams['font.sans-serif'] = ['Arial']

    labels = [GAME_DICT.get(str(group_name)) for group_name in grouped.groups.keys()]
    color = [GAME_TRADITIONAL_CATEGORY_COLOR[GAME_TRADITIONAL_CATEGORY[str(group_name)]] for group_name in
             grouped.groups.keys()]
    # GAME_TRADITIONAL_CATEGORY_COLOR[GAME_TRADITIONAL_CATEGORY[str(label)]

    medianprops = dict(linestyle='-', linewidth=2, color='red')
    meanprops = dict(marker='^', markeredgecolor='black', markerfacecolor='green', markersize=7)

    data = [data[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [color[i] for i in order]

    bplot = ax.boxplot(data, positions=range(len(data)), widths=0.6, showfliers=False, showmeans=True,
                       patch_artist=True, medianprops=medianprops, meanprops=meanprops, labels=labels)
    # ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=90)
    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # plt.xlim(9, 30)
    # plt.ylim(0, 10)
    # plt.yticks(np.arange(0, 3, 3))
    # plt.xticks(np.arange(9, 31, 3))

    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    mean_legend = plt.Line2D([0], [0], marker='^', color='green', label='Mean',
                             markeredgecolor='black', markersize=7, linestyle='none')
    median_legend = plt.Line2D([0], [0], color='red', label='Median', linewidth=1, linestyle='-')
    iqr_legend = plt.Line2D([0], [0], marker='s', color='black', label='25%-75%', markerfacecolor='none', markersize=10,
                            linestyle='none')
    ten_ninety_legend = plt.Line2D([0], [0], marker='|', color='black', label='10%-90%', linewidth=2, linestyle='none')

    ax.legend(handles=[mean_legend, median_legend, iqr_legend, ten_ninety_legend], loc='upper left')
    ax.set_xticklabels(labels, rotation=30)

    plt.savefig('../results/mousee_boxplot.pdf')
    # plt.title("纵坐标：((t2.avg_intra_ratio / 100 * t2.frame_total) / t2.frame_total)*100")
    plt.show()
