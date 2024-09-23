# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
from exp_pool import DatasetPool
import csv
CSV_HEADER = [
    'Schemes', 'RTT(ms)', 'queue', 'loss','BW (Mbps)', 'Avg_Reward', 'sr_max','sr_avg','sr_min','rtt_max','rtt_avg','rtt_min','rttvar_max','rttvar_avg','rttvar_min','loss_max','loss_avg','loss_min'
]


def standard_prompt_filled(sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min):
    prompt = (
        f"<|Task description|>You are tasked with selecting the most appropriate congestion control algorithm based on the provided network statistics and scenario. Your goal is to select the algorithm that best suits the given network status.\n"
        f"<|Network stat|>Sending Rate(Mbps): max {sr_max}, average {sr_avg}, min {sr_min}; RTT(ms): max {rtt_max}, average {rtt_avg}, min {rtt_min}; RTT Variance(ms): max {rttvar_max}, average {rttvar_avg}, min {rttvar_min}; Packet loss(%): max {loss_max}, average {loss_avg}, min {loss_min};\n"
    )
    return prompt

def extract_csv_info(csv_path):
    df = pd.read_csv(csv_path)          
    # 取出 rtt_var_l.get_avg() 列
    avg_reward = df['reward'].mean()

    sr_max = df['thr_s.get_max()'].max() *100
    sr_avg = df['thr_s.get_avg()'].mean()*100
    sr_min = df['thr_s.get_min()'].min()*100

    rtt_max = df['rtt_s.get_max()'].max()*100
    rtt_avg = df['rtt_s.get_avg()'].mean()*100
    rtt_min = df['rtt_s.get_min()'].min()*100

    rttvar_max = df['rtt_var_s.get_max()'].max()
    rttvar_avg = df['rtt_var_m.get_avg()'].mean()
    rttvar_min = df['rtt_var_s.get_min()'].min()

    loss_max = df['lost_s.get_max()'].max() 
    loss_avg = df['lost_s.get_avg()'].mean()
    loss_min = df['lost_s.get_min()'].min()

    return avg_reward,sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min


if __name__ == '__main__':
    # init dataset pool
    dataset_pool = DatasetPool()

    # 找到所有相同启动配置的不同算法,提取统计信息
    extracted_info_csv_path = '/data3/wuduo/xuanyu/llmcc/datasets/CC/extracted_info.csv'
    with open(extracted_info_csv_path, mode='w') as file:  # 使用 'wb' 创建文件
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)
            
    directory = "/data3/wuduo/xuanyu/llmcc/datasets/CC/csv"
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            logfile_path = os.path.join(root, file)
            # 提取文件名中的信息，写入csv
            set_env = file.split("_")
            schemes,rtt,queue,loss,bw = set_env[0],int(set_env[2])*2,int(set_env[3]),float(set_env[4]),set_env[1]
            # 计算csv中的信息，写入csv
            avg_reward,sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min = extract_csv_info(logfile_path)
            row = [schemes,rtt,queue,loss,bw,avg_reward,sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min]
            with open(extracted_info_csv_path, mode='a') as file:  # 使用 'ab' 代替 'a'
                writer = csv.writer(file)
                writer.writerow(row)

    
    
    # 计算出最高Reward的算法，作为标签
    # 对数据分组
    # 读取 CSV 文件
    df = pd.read_csv(extracted_info_csv_path)

    # 按照 A, B, C 列分组
    grouped = df.groupby(['RTT(ms)','queue','loss','BW (Mbps)'])

    # 找到 D 列中有多个不同值的组
    result = grouped.filter(lambda x: x['Schemes'].nunique() > 1)

    # 输出结果
    for name, group in grouped:
        # 获取该分组 D 列的最大值
        max_in_group = group['Avg_Reward'].max()
        max_row = group[group['Avg_Reward'] == max_in_group]

        best_scheme = max_row['Schemes'].values[0]
        # 提取数据，合成Prompt
        for index, row in group.iterrows():
            prompt_ = standard_prompt_filled(row['sr_max'],row['sr_avg'],row['sr_min'],row['rtt_max'],row['rtt_avg'],row['rtt_min'],row['rttvar_max'],row['rttvar_avg'],row['rttvar_min'],row['loss_max'],row['loss_avg'],row['loss_min'])
            dataset_pool.add(prompt_,best_scheme)

    # 组成pair，作为数据对并导出
    dataset_pool_output = '/data3/wuduo/xuanyu/llmcc/datasets/CC/dataset_pool.pkl'
    pickle.dump(dataset_pool, open(dataset_pool_output, 'wb'))
