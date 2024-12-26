# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import torch
import csv
import math

CSV_HEADER = ['time_stamp', 'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time', 'smoothness', 'model', 'reward','bw_change','bandwidth_utilization','bitrate_smoothness','rebuf_time_ratio','next_video_chunk_sizes','video_chunk_remain']

data = []
base_dir_withoutchange = "/data3/wuduo/xuanyu/llmcc/datasets/ABR/artifacts_random_trace"
base_dir_withchange = "/data3/wuduo/xuanyu/llmcc/datasets/ABR/artifacts_random_trace"


class _DatasetPool:
    '''
    The experience pool used to save & load data.

    Data Structure:
        Pairs of [Prompts, Probed Time Series, Labels].
        Prompts: string, Natural Language that describe the tasks.
        Probed Time Series: ndarray, The Multi-dimension time series probed during startup.
        Labels: string, The Best Algos's name. "Best" is evaluated during dataset collection.
    '''
    def __init__(self):
        
        # self.prompts = [] 
        self.states = [] # probed ts组成
        self.actions = [] # 对应选择的label
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        # self.prompts.append(prompt)
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def extend(self, states, actions, rewards, dones):
        # self.prompts.extend(prompts)
        self.states.extend(states)  # sometime state is also called obs (observation)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)


    def __len__(self):
        return len(self.states)

def standard_prompt_filled():
    prompt = (
        f"<|Task description|>You are tasked with selecting the most appropriate Adaptive Bitrate Control algorithm based on the provided network statistics and scenario. Your goal is to select the algorithm that best suits the given network status.\n"
        # f"<|Network stat|>Sending Rate(Mbps): max {sr_max}, average {sr_avg}, min {sr_min}; RTT(ms): max {rtt_max}, average {rtt_avg}, min {rtt_min}; RTT Variance(ms): max {rttvar_max}, average {rttvar_avg}, min {rttvar_min}; Packet loss(%): max {loss_max}, average {loss_avg}, min {loss_min};\n"
    )
    return prompt


def handle_csv_withoutchange(file, model,decision_interval_per_record = 5):
    # time_stamp,bit_rate,buffer_size,rebuffer_time,chunk_size,download_time,smoothness,model,reward,bw_change,bandwidth_utilization,bitrate_smoothness,rebuf_time_ratio
    all = pd.read_csv(file)
    
    total_rows = len(all)
    num_segments = math.floor(total_rows / decision_interval_per_record) # 多余的忽略
    
    # done record
    dones = [False]* (num_segments-1)
    dones.append(True)
    
    # action
    actions = [model] * num_segments



    S_INFO = 5
    S_LEN = decision_interval_per_record  # take how many frames in the past
    states = []
    rewards = []
    # 每5条一个记录，计算出分割的点。
    for i in range(num_segments):
        start_row = i * decision_interval_per_record
        end_row = start_row + decision_interval_per_record  # 结束行索引
        segment = all.iloc[start_row:end_row]  # 切片
        

        # reward 
        reward_sum = segment['reward'].sum()
        rewards.append(reward_sum)


        # state
        state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
        target_columns = [
            'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time',
            'smoothness', 'bw_change', 'bandwidth_utilization', 'bitrate_smoothness', 'rebuf_time_ratio'
        ]



        VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
        BUFFER_NORM_FACTOR = 10.0
        CHUNK_TIL_VIDEO_END_CAP = 48.0
        BITRATE_LEVELS = 6
        M_IN_K = 1000.0

        bit_rate = segment['bit_rate'].fillna(0).to_numpy()
        buffer_size = segment['buffer_size'].fillna(0).to_numpy()
        chunk_size = segment['chunk_size'].fillna(0).to_numpy()
        delay = segment['download_time'].fillna(0).to_numpy()
        # next_video_chunk_sizes = segment
        video_chunk_remain = segment['video_chunk_remain'].fillna(0).to_numpy()

        
        # 对 bit_rate 执行归一化处理
        state[0] = bit_rate / np.max(VIDEO_BIT_RATE)  # 对应 bit_rate 的归一化处理

        # 对 buffer_size 进行规范化
        state[1] = buffer_size / BUFFER_NORM_FACTOR  # 对应 buffer_size 的规范化处理

        # 对 chunk_size 和 delay 计算比值
        state[2] = (chunk_size / delay) / M_IN_K  # 对应 chunk_size / delay 的计算

        # 对 delay 进行规范化
        state[3] = (delay / M_IN_K) / BUFFER_NORM_FACTOR  # 对应 delay 的规范化处理
        # state[4]= np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[4] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        # segment_array = segment[target_columns].fillna(0).to_numpy()
        # # 转换为 PyTorch Tensor
        # tensor = torch.tensor(segment_array, dtype=torch.float32)
        states.append(state) # tensor shape: (5,10) (decision_interval_per_record , features)
    
    

    return states,actions,rewards,dones


def handle_csv_withchange(file,decision_interval_per_record = 5):
    # time_stamp,bit_rate,buffer_size,rebuffer_time,chunk_size,download_time,smoothness,model,reward,bw_change,bandwidth_utilization,bitrate_smoothness,rebuf_time_ratio
    all = pd.read_csv(file)
    
    total_rows = len(all)
    num_segments = math.floor(total_rows / decision_interval_per_record) # 多余的忽略
    
    # done record
    dones = [False]* (num_segments-1)
    dones.append(True)
    
    # action
    actions = []



    S_INFO = 5
    S_LEN = decision_interval_per_record  # take how many frames in the past
    states = []
    rewards = []
    # 每5条一个记录，计算出分割的点。
    for i in range(num_segments):
        start_row = i * decision_interval_per_record
        end_row = start_row + decision_interval_per_record  # 结束行索引
        segment = all.iloc[start_row:end_row]  # 切片
        

        # reward 
        reward_sum = segment['reward'].sum()
        rewards.append(reward_sum)
      

        # state
        state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
        target_columns = [
            'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time',
            'smoothness', 'bw_change', 'bandwidth_utilization', 'bitrate_smoothness', 'rebuf_time_ratio'
        ]

        # action
        action = segment['model'].value_counts().idxmax()
        actions.append(action)


        VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
        BUFFER_NORM_FACTOR = 10.0
        CHUNK_TIL_VIDEO_END_CAP = 48.0
        BITRATE_LEVELS = 6
        M_IN_K = 1000.0

        bit_rate = segment['bit_rate'].fillna(0).to_numpy()
        buffer_size = segment['buffer_size'].fillna(0).to_numpy()
        chunk_size = segment['chunk_size'].fillna(0).to_numpy()
        delay = segment['download_time'].fillna(0).to_numpy()
        # next_video_chunk_sizes = segment
        video_chunk_remain = segment['video_chunk_remain'].fillna(0).to_numpy()

        
        # 对 bit_rate 执行归一化处理
        state[0] = bit_rate / np.max(VIDEO_BIT_RATE)  # 对应 bit_rate 的归一化处理

        # 对 buffer_size 进行规范化
        state[1] = buffer_size / BUFFER_NORM_FACTOR  # 对应 buffer_size 的规范化处理

        # 对 chunk_size 和 delay 计算比值
        state[2] = (chunk_size / delay) / M_IN_K  # 对应 chunk_size / delay 的计算

        # 对 delay 进行规范化
        state[3] = (delay / M_IN_K) / BUFFER_NORM_FACTOR  # 对应 delay 的规范化处理
        # state[4]= np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[4] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        # segment_array = segment[target_columns].fillna(0).to_numpy()
        # # 转换为 PyTorch Tensor
        # tensor = torch.tensor(segment_array, dtype=torch.float32)
        states.append(state) # tensor shape: (5,10) (decision_interval_per_record , features)
    
    

    return states,actions,rewards,dones


def cal_maximum_cum_reward(file):
    all = pd.read_csv(file)
    reward_sum = all['reward'].sum()
    return reward_sum


if __name__ == '__main__':
    # set dataset path & extracted_info
    algorithm_names = ['bba', 'genet', 'mpc', 'udr_1', 'udr_2', 'udr_3', 'udr_real']

    # init dataset pool class
    dataset_pool = _DatasetPool()

    # 准备所有的state,action,reward 存储进入数据集

    ## withoutchange的
    for algorithm in algorithm_names: 
        print("without + {}".format(algorithm))
        folder_path = os.path.join(base_dir_withoutchange, algorithm+'/seed_100003')
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):  # 筛选出CSV文件
                file_path = os.path.join(folder_path, file)
                states, actions, rewards, dones = handle_csv_withoutchange(file_path,algorithm)
                dataset_pool.extend(states, actions, rewards, dones)

    trace_reward_list = []

    ## 获取trace对应的最高reward
    for i in range(1,1001):
        folder_path = os.path.join(base_dir_withchange, str(i),'fcc-test_video1/trace_num_100_fixed_True/genet/seed_100003')
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):  # 筛选出CSV文件
                file_path = os.path.join(folder_path, file)
                trace_reward_list.append(cal_maximum_cum_reward(file_path))

    # 计算9%的百分位值
    percentile_90 = np.percentile(trace_reward_list, 90)
    print(f"The 90th percentile value is: {percentile_90}")


    ## withchange的
    for i in range(1,1001):
        print("with + {}".format(i))
        folder_path = os.path.join(base_dir_withchange, str(i),'fcc-test_video1/trace_num_100_fixed_True/genet/seed_100003')
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):  # 筛选出CSV文件
                file_path = os.path.join(folder_path, file)
                states, actions, rewards, dones = handle_csv_withchange(file_path)
                if cal_maximum_cum_reward(file_path) >= percentile_90:
                    dataset_pool.extend(states, actions, rewards, dones)

    # 组成pair，作为数据对并导出
    dataset_pool_output = '/data3/wuduo/xuanyu/llmcc/datasets/ABR/dataset_pool_ABR.pkl'

    pickle.dump(dataset_pool, open(dataset_pool_output, 'wb'))