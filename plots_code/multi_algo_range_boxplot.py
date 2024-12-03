import os
import pandas as pd
import re

# 获取文件夹中所有 CSV 文件的路径
def get_all_csv_files(base_folder):
    csv_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# 从文件名中提取网站信息
def extract_website(file_name):
    # 假设网站信息位于文件名中，例如："http---www.facebook.com"
    match = re.search(r'http---([a-zA-Z0-9.-]+)', file_name)
    if match:
        return match.group(1)
    return None

# 处理每个算法文件夹中的 CSV 文件
def process_algorithm_folder(algorithm_folder):
    website_rewards = {}  # 用于存储每个网站的累计 reward
    csv_files = get_all_csv_files(algorithm_folder)
    
    for csv_file in csv_files:
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 确保文件中包含'reward'列
        if 'reward' in df.columns:
            # 提取 'reward' 列的累计值
            cumulative_reward = df['reward'].sum()
            
            # 提取网站信息
            website = extract_website(csv_file)
            if website:
                if website not in website_rewards:
                    website_rewards[website] = []
                website_rewards[website].append(cumulative_reward)
    
    return website_rewards

# 打印每个算法和网站的累计 reward 范围
def print_reward_ranges(base_folder):
    # 获取所有算法文件夹
    algorithms = os.listdir(base_folder)
    
    for algorithm in algorithms:
        algorithm_folder = os.path.join(base_folder, '{}/seed_100003'.format(algorithm))
        if os.path.isdir(algorithm_folder):
            website_rewards = process_algorithm_folder(algorithm_folder)
            print(f"Algorithm: {algorithm}")
            for website, rewards in website_rewards.items():
                reward_range = (min(rewards), max(rewards))
                print(f"  Website: {website}, Reward Range: {reward_range}")

# 设置你的根文件夹路径（包含所有算法文件夹）
base_folder = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts_without_sft/results_start_timestamp_initialized/fcc-test_video1/trace_num_100_fixed_True'

# 执行并打印结果
print_reward_ranges(base_folder)
