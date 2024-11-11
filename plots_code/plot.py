import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import sys

model_dict = {
    'bba': (1, 'blue', 's'),       # 蓝色, 符号 's'
    'udr_1': (2, 'red', 'o'),      # 红色, 符号 'o'
    'udr_2': (3, 'magenta', '<'),  # 品红色, 符号 '<'
    'udr_3': (4, 'green', '>'),    # 绿色, 符号 '>'
    'udr_real': (5, 'cyan', '^'),  # 青色, 符号 '^'
    'genet': (6, 'yellow', 'd'),   # 黄色, 符号 'd'
    'mpc': (7, 'purple', 'x'),     # 紫色, 符号 'x'
    'llmcc': (8, 'aqua', '*'),     # 水绿色, 符号 '*'
    'mixed': (9, 'orange', '+')    # 橙色, 符号 '+'
}

def get_ticks(num, cnt):
    length = max(num) - min(num)
    if length < cnt:
        return 1
    return round(length / cnt)

def paint(filename, outputfile, title):
    data = pd.read_csv(filename)

    time_stamps = list(data["time_stamp"])
    bit_rates = list(data["bit_rate"])
    models = rewards = list(data["model"])
    rewards = list(data["reward"])
    culmulative_rewards = list(data["reward"])
    for i in range(1, len(culmulative_rewards)):
        culmulative_rewards[i] += culmulative_rewards[i - 1]

    xtick = get_ticks(time_stamps, 20)

    plt.figure(figsize=(30, 18))

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.subplot(2, 2, 1)
    ytick = get_ticks(bit_rates, 20)
    plt.gca().yaxis.set_major_locator(MultipleLocator(ytick))
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))
    plt.grid()
    #plt.title('Bit Rate')
    plt.xlabel('Time Stamp')
    plt.ylabel('Bit Rate')
    plt.plot(time_stamps, bit_rates, marker='o', markersize=6, linewidth=1, linestyle='--')

    plt.subplot(2, 2, 2)
    ytick = get_ticks(rewards, 20)
    plt.gca().yaxis.set_major_locator(MultipleLocator(ytick))
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))
    plt.grid()
    #plt.title('Reward')
    plt.xlabel('Time Stamp')
    plt.ylabel('Reward')
    plt.plot(time_stamps, rewards, linewidth=2.5)

    plt.subplot(2, 2, 3)
    ytick = get_ticks(culmulative_rewards, 20)
    plt.gca().yaxis.set_major_locator(MultipleLocator(ytick))
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))
    plt.grid()
    #plt.title('Culmulative Rewards')
    plt.xlabel('Time Stamp')
    plt.ylabel('Culmulative Rewards')
    plt.plot(time_stamps, culmulative_rewards, linewidth=2.5)

    plt.subplot(2, 2, 4)
    plt.xlabel('Time Stamp')
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))
    plt.grid()
    plt.ylabel('Model')
    for i in range(len(models)):
        (num, col, mark) = model_dict[models[i]]
        plt.scatter(time_stamps[i], num, 100, col, mark, label=models[i])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.suptitle(title, fontsize=48)

    plt.savefig(outputfile)
    plt.show()

# plt.style.use('seaborn-white')

dir = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts/results_start_timestamp_initialized/fcc-test_video1/trace_num_100_fixed_True/llmcc'

outputdir = './plots/llmcc'

for root, dirs, files in os.walk(dir):
    for file in files:
        file_path = os.path.join(root, file)
        out_path = os.path.join(outputdir, file[:-4]+'.png')
        print(file)
        paint(file_path, out_path, file[:-4])
        # sys.exit()