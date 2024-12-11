import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from collections import defaultdict


def readcsv(file):
    all = pd.read_csv(file)
    time_stamps = list(all["time_stamp"])
    culmulative_rewards = list(all["reward"])
    for i in range(1, len(culmulative_rewards)):
        culmulative_rewards[i] += culmulative_rewards[i - 1]
    return (time_stamps, culmulative_rewards)


# 获取文件夹中所有 CSV 文件的路径
def get_all_csv_files(base_folder):
    csv_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# 从文件名中提取网站信息
def classify_website(file_name):
    # 定义网站分类关键词
    websites = ['ebay', 'facebook', 'amazon', 'msn', 'yahoo', 'google']
    
    # 遍历关键词，如果文件名中包含任何一个关键词，就归为该类
    for website in websites:
        if website in file_name.lower():  # 使用小写以确保不区分大小写
            return website
    return None  # 如果没有匹配的分类，返回 None
# 处理每个算法文件夹中的 CSV 文件
def process_algorithm_folder(algorithm_folder,model):
    website_rewards = {}  # 用于存储每个网站的累计 reward
    csv_files = get_all_csv_files(algorithm_folder)
    
    for csv_file in csv_files:
        # 读取 CSV 文件
        time_stamps, culmulative_rewards = readcsv(csv_file) # culmulative_rewards[-1]
            
        # 提取网站信息
        website = classify_website(csv_file)
        if website:
            if website not in website_rewards:
                website_rewards[website] = []
            website_rewards[website].append(culmulative_rewards[-1])
    
    return website_rewards

# 打印每个算法和网站的累计 reward 范围
def get_reward_ranges(base_folder):
    # 获取所有算法文件夹
    algorithms = os.listdir(base_folder)
    record = []
    for algorithm in algorithms:
        algorithm_folder = os.path.join(base_folder, '{}/seed_100003'.format(algorithm))
        if os.path.isdir(algorithm_folder):
            website_rewards = process_algorithm_folder(algorithm_folder,algorithm)

            for website, rewards in website_rewards.items():
                record.append((algorithm,website,rewards))


    return record

def boxplot_senario(record):
    # 构造一个字典，按 site 和 algorithm 分类存储 rewards
    site_algorithm_rewards = defaultdict(lambda: defaultdict(list))

    # 将数据存入字典
    for algorithm, website, rewards in record:
        site_algorithm_rewards[website][algorithm].extend(rewards)

    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 为了让不同算法在同一站点下归类显示，需要将数据按站点和算法整理成适当的格式
    positions = []  # 存储箱线图的位置
    labels = []  # 存储每个箱线图的标签
    data_to_plot = []  # 存储每个箱线图的实际数据

    # 设置不同 site 之间的间距
    site_gap = 2  # 设置不同站点之间的间隔，增大该值会增加间隔

    # 处理每个 site
    site_idx = 0
    for website, algorithms in site_algorithm_rewards.items():
        positions_for_site = []
        for idx, (algorithm, rewards) in enumerate(algorithms.items()):
            # 每个算法的箱线图要偏移一定的距离
            positions_for_site.append(site_idx + idx * 0.2)  # 0.2 是每个算法柱子的宽度
            data_to_plot.append(rewards)
            labels.append(f'{algorithm} ({website})')

        positions.extend(positions_for_site)
        site_idx += site_gap  # 每个站点之间的间隔

    # 绘制箱线图
    ax.boxplot(data_to_plot, positions=positions, widths=0.15,showfliers=False)

    # 设置 x 轴标签
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)

    # 设置 y 轴标签
    ax.set_ylabel('Rewards')
    # ax.set_ylim(-150, 120)  # 这里设置了 Y 轴的范围，从 0 到 50，你可以根据你的数据调整此值

    # 设置标题
    ax.set_title('Boxplot of Rewards for Different Algorithms on Different Sites')

    # 显示图表
    plt.tight_layout()


    # 保存图片并显示
    plt.savefig('/data3/wuduo/xuanyu/llmcc/plots_code/plots/rewards_boxplot_senario.pdf')
    plt.show()


def box_algo(record):

    # 构造一个字典，按算法和站点分类存储 rewards
    algorithm_site_rewards = defaultdict(lambda: defaultdict(list))

    # 将数据存入字典
    for algorithm, website, rewards in record:
        algorithm_site_rewards[algorithm][website].extend(rewards)

    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 为了让不同站点在同一个算法下归类显示，需要将数据按算法和站点整理成适当的格式
    positions = []  # 存储箱线图的位置
    labels = []  # 存储每个箱线图的标签
    data_to_plot = []  # 存储每个箱线图的实际数据

    # 设置不同算法之间的间距
    algo_gap = 2  # 设置不同算法之间的间隔，增大该值会增加间隔

    # 处理每个算法
    algo_idx = 0
    for algorithm, websites in algorithm_site_rewards.items():
        positions_for_algorithm = []
        for idx, (website, rewards) in enumerate(websites.items()):
            # 每个站点的箱线图要偏移一定的距离
            positions_for_algorithm.append(algo_idx + idx * 0.2)  # 0.2 是每个站点箱线图的宽度
            data_to_plot.append(rewards)
            labels.append(f'{website} ({algorithm})')

        positions.extend(positions_for_algorithm)
        algo_idx += algo_gap  # 每个算法之间的间隔

    # 绘制箱线图
    ax.boxplot(data_to_plot, positions=positions, widths=0.15,showfliers=False)

    # 设置 x 轴标签
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)

    # 设置 y 轴标签
    ax.set_ylabel('Rewards')

    # 设置标题
    ax.set_title('Boxplot of Rewards for Different Sites on Different Algorithms')
    
    # 设置 Y 轴范围
    # ax.set_ylim(-150, 120)  # 设置了 Y 轴的范围，可以根据数据调整

    # 显示图表
    plt.tight_layout()

    # 保存图片并显示
    plt.savefig('/data3/wuduo/xuanyu/llmcc/plots_code/plots/rewards_boxplot_algo.pdf')
    plt.show()



if __name__ == '__main__':
    # 设置你的根文件夹路径（包含所有算法文件夹）
    base_folder = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts_without_sft/results_start_timestamp_initialized/fcc-test_video1/trace_num_100_fixed_True'

    # 执行并打印结果
    record= get_reward_ranges(base_folder)
    box_algo(record)
    boxplot_senario(record)
