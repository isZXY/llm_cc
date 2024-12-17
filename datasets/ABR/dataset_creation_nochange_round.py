# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import torch
import csv
import math

CSV_HEADER = [
    'time_stamp', 'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time', 'smoothness', 'model', 'reward','bw_change','bandwidth_utilization','bitrate_smoothness','rebuf_time_ratio'
]

data = []
base_dir = "/data3/wuduo/xuanyu/llmcc/datasets/ABR/trace_num_100_fixed_True"




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


def handle_csv(file, model,chunk_size = 15):
    # time_stamp,bit_rate,buffer_size,rebuffer_time,chunk_size,download_time,smoothness,model,reward,bw_change,bandwidth_utilization,bitrate_smoothness,rebuf_time_ratio
    all = pd.read_csv(file)
    
    total_rows = len(all)
    num_segments = math.floor(total_rows / chunk_size) # 多余的忽略
    
    # done record
    dones = [False]* (num_segments-1)
    dones.append(True)
    
    # action
    actions = [model] * num_segments

    states = []
    returns = []
    
    # 每15条一个记录，计算出分割的点。
    for i in range(num_segments):
        start_row = i * chunk_size
        end_row = start_row + chunk_size  # 结束行索引
        segment = all.iloc[start_row:end_row]  # 切片
        

        # return 
        reward_sum = segment['reward'].sum()
        returns.append(reward_sum)


        # state
        target_columns = [
            'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time',
            'smoothness', 'bw_change', 'bandwidth_utilization', 'bitrate_smoothness', 'rebuf_time_ratio'
        ]

        segment_array = segment[target_columns].fillna(0).to_numpy()
        # 转换为 PyTorch Tensor
        tensor = torch.tensor(segment_array, dtype=torch.float32)
        states.append(tensor)
    
    

    return states,actions,returns,dones





if __name__ == '__main__':
    # set dataset path & extracted_info
    algorithm_names = ['bba', 'genet', 'mpc', 'udr_1', 'udr_2', 'udr_3']

    # init dataset pool class
    dataset_pool = _DatasetPool()

    # 准备所有的state,action,return 存储进入数据集
    for algorithm_folder in os.listdir(base_dir): 
        folder_path = os.path.join(base_dir, algorithm_folder+'/seed_100003')
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):  # 筛选出CSV文件
                file_path = os.path.join(folder_path, file)
                states, actions, returns, dones = handle_csv(file_path,algorithm_folder)
                dataset_pool.extend(states, actions, returns, dones)



    # 组成pair，作为数据对并导出
    dataset_pool_output = '/data3/wuduo/xuanyu/llmcc/datasets/ABR/dataset_pool_ABR.pkl'
    pickle.dump(dataset_pool, open(dataset_pool_output, 'wb'))
