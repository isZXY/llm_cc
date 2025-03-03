import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import sys

from const import *

# 提取文件名中的数字编号
def extract_number(file_name):
    match = re.search(r'test_(\d+)_', file_name)
    return int(match.group(1)) if match else None

def get_ticks(num, cnt):
    length = max(num) - min(num)
    if length < cnt:
        return 1
    return round(length / cnt)

dir_pre = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts_llmcc_Feb/results_start_timestamp_initialized/fcc-test_video1/trace_num_100_fixed_True'

output_dir = '/data3/wuduo/xuanyu/llmcc/plots_code/plots/llmcc'

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = []

def readcsv(file, model):
    all = pd.read_csv(file)
    time_stamps = list(all["time_stamp"])
    culmulative_rewards = list(all["reward"])
    
    # 计算累积奖励
    for i in range(1, len(culmulative_rewards)):
        culmulative_rewards[i] += culmulative_rewards[i - 1]
    
    # 提取文件名中的数字编号
    file_number = extract_number(file)
    
    return (time_stamps, culmulative_rewards, model, file_number)

def get_data(model):
    now = []
    dir = os.path.join(dir_pre, model)
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            now.append(readcsv(file_path, model))
    
    # 按文件编号排序
    now = sorted(now, key=lambda x: x[3])
    data.append(now)

get_data('bba')
get_data('genet')
get_data('mpc')
get_data('udr_1')
get_data('udr_2')
get_data('udr_3')
# get_data('mixed_high_freq')
# get_data('mixed_low_freq')
get_data('llmcc')

# 根据文件编号保存图像
for i in range(100):
    print(i)
    plt.cla()
    plt.figure(figsize=(30, 18))
    plt.xlabel('Time Stamp')
    plt.ylabel('Culmulative Rewards')
    
    # 使用文件编号作为文件名
    filename = f"test_{i}_plot.png"
    outputfile = os.path.join(output_dir, filename)
    
    tot_time = []
    tot_re = []
    for item in data:
        (time_stamps, culmulative_rewards, model, file_number) = item[i]
        tot_time += time_stamps
        tot_re += culmulative_rewards
    
    xtick = get_ticks(tot_time, 30)
    ytick = get_ticks(tot_re, 30)
    
    plt.gca().yaxis.set_major_locator(MultipleLocator(ytick))
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))
    plt.grid()
    
    for item in data:
        (time_stamps, culmulative_rewards, model, file_number) = item[i]
        (col, mak) = model_dict[model]
        plt.plot(time_stamps, culmulative_rewards, linewidth=2.5, c=col, marker=mak, label=model)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.suptitle(filename, fontsize=48)
    plt.savefig(outputfile)
