import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import sys
import numpy as np
from const import *

algorithms = ['genet', 'udr_1', 'llmcc', 'udr_2', 'udr_3', 'mpc', 'bba']


# 提取文件名中的数字编号
def extract_number(file_name):
    match = re.search(r'test_(\d+)_', file_name)
    return int(match.group(1)) if match else None


def get_ticks(num, cnt):
    length = max(num) - min(num)
    if length < cnt:
        return 1
    return round(length / cnt)


dir_pre = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts_llmcc_OK/results/fcc-test_video1/trace_num_100_fixed_True'
output_dir = '/data3/wuduo/xuanyu/llmcc/plots_code/plots/llmcc_rank'

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
get_data('llmcc')

# 根据文件编号保存图像


# 创建一个字典，key 是算法名称，value 是一个 list，存储它在所有 trace 中的排名
algorithm_rankings = {algorithm: [] for algorithm in algorithms}

for i in range(100):
    print(i)
    
    reward_model_pairs = []
    for item in data:
        (time_stamps, culmulative_rewards, model, file_number) = item[i]
        culmulative_rewards = culmulative_rewards[-1]
        reward_model_pairs.append((culmulative_rewards, model))

    reward_model_pairs.sort(key=lambda x: x[0], reverse=True)

    sorted_models = [pair[1] for pair in reward_model_pairs]

    ## print(sorted_models)

    # 更新每个算法的排名信息
    for rank, model in enumerate(sorted_models):
        # 排名从 0 开始，存储每个算法在当前 trace 中的排名（rank + 1）
        algorithm_rankings[model].append(rank + 1)
    
    # ['genet', 'udr_1', 'llmcc', 'udr_2', 'udr_3', 'mpc', 'bba']
    # 获取所有的
    # dict：，key是算法名，value是一个list，list不断append他在一个trace下的排名

# 创建一个字典，初始化频次为0
rank_frequencies = {algorithm: [0] * 7 for algorithm in algorithm_rankings}

# 统计每个算法的排名频次
for algorithm, ranks in algorithm_rankings.items():
    for rank in ranks:
        rank_frequencies[algorithm][rank - 1] += 1  # 累加排名频次

# 计算百分比
rank_percentages = np.zeros((len(algorithm_rankings), 6))

for i, (algorithm, ranks) in enumerate(rank_frequencies.items()):
    total_ranks = sum(ranks)
    for j in range(6):
        rank_percentages[i, j] = (ranks[j] / total_ranks) * 100

# 输出结果
print(rank_percentages)

# 排名标签
ranks = ['1st', '2nd', '3rd', '4th', '5th', '6th']

# 创建堆叠柱状图
fig, ax = plt.subplots(figsize=(10, 6))

# 使用堆叠柱状图绘制
bars = ax.bar(algorithms, rank_percentages[:, 0], color='lightblue', label=ranks[0])

# 堆叠其余的排名部分
for i in range(1, rank_percentages.shape[1]):
    bars = ax.bar(algorithms, rank_percentages[:, i], bottom=np.sum(rank_percentages[:, :i], axis=1), 
                  label=ranks[i], color=plt.cm.get_cmap('Set3')(i / rank_percentages.shape[1]))

# 添加标题和标签
ax.set_title('Ranking Percentages of Algorithms')
ax.set_xlabel('Algorithm')
ax.set_ylabel('Percentage (%)')

# 添加图例
ax.legend(title='Ranking', loc='upper left')


plt.savefig("cum_reward_rank.pdf")