# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import torch
import csv

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
        self.prompts = []
        self.probed_ts = []
        self.labels = []

    def add(self, prompts,probed_ts,label):
        self.prompts.append(prompts)
        self.probed_ts.append(probed_ts)
        self.labels.append(label)


    def __len__(self):
        return len(self.labels)



CSV_HEADER = [
    'time_stamp', 'bit_rate', 'buffer_size', 'rebuffer_time', 'chunk_size', 'download_time', 'smoothness', 'model', 'reward','bw_change','bandwidth_utilization','bitrate_smoothness','rebuf_time_ratio'
]

data = []
base_dir = "/data3/wuduo/xuanyu/llmcc/datasets/ABR/dataset_pool_ABR.pkl"

def readcsv(file, model):
    # time_stamp,bit_rate,buffer_size,rebuffer_time,chunk_size,download_time,smoothness,model,reward,bw_change,bandwidth_utilization,bitrate_smoothness,rebuf_time_ratio

    all = pd.read_csv(file)
    time_stamps = list(all["time_stamp"])
    bw_change = list(all["bw_change"])
    bandwidth_utilization = list(all["bandwidth_utilization"])
    bitrate_smoothness = list(all["bitrate_smoothness"])
    rebuf_time_ratio = list(all["rebuf_time_ratio"])
    culmulative_rewards = list(all["reward"])
    for i in range(1, len(culmulative_rewards)):
        culmulative_rewards[i] += culmulative_rewards[i - 1]
    return (time_stamps, culmulative_rewards, model,bw_change,bandwidth_utilization,bitrate_smoothness,rebuf_time_ratio)


def get_data(model):
    now = []
    dir = os.path.join(base_dir, model)
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            now.append(readcsv(file_path, model))

    now = sorted(now, key = lambda x: min(x[0]))
    data.append(now)




def standard_prompt_filled(sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min):
    prompt = (
        f"<|Task description|>You are tasked with selecting the most appropriate Adaptive Bitrate Control algorithm based on the provided network statistics and scenario. Your goal is to select the algorithm that best suits the given network status.\n"
        # f"<|Network stat|>Sending Rate(Mbps): max {sr_max}, average {sr_avg}, min {sr_min}; RTT(ms): max {rtt_max}, average {rtt_avg}, min {rtt_min}; RTT Variance(ms): max {rttvar_max}, average {rttvar_avg}, min {rttvar_min}; Packet loss(%): max {loss_max}, average {loss_avg}, min {loss_min};\n"
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


def extract_ts(csv_path):
    pass


def calculate_reward(csv_path):
    """
    计算一个csv文件中的累积reward。
    假设reward在csv文件中有一列名为'reward'。
    """
    df = pd.read_csv(csv_path)
    return df['reward'].sum()  # 假设'reward'列包含奖励值

if __name__ == '__main__':
    # set dataset path & extracted_info
    algorithm_names = ['bba', 'genet', 'mpc', 'udr_1', 'udr_2', 'udr_3']




    # init dataset pool class
    dataset_pool = _DatasetPool()

    # get data pair
    get_data('bba')
    get_data('genet')
    get_data('mpc')
    get_data('udr_1')
    get_data('udr_2')
    get_data('udr_3') # now data has sorted five algos eles


    # 提取相同trace都没个算法的最终reward，最高值既label；ts采用剩下的维度
    # 数据结构： [五个算法] 里面是100个子列表（对应trace），每个列表对应的是一个temp
    for trace_index in range(100):  # 假设有100个trace
        best_algorithm = None
        best_reward = -float('inf')  # 初始最差reward

        for algorithm_index, algorithm_data in enumerate(data):
            # 获取当前算法的trace数据
            trace_data = algorithm_data[trace_index]  # 每个trace是一个元组

            reward = trace_data[1][-1]

            # 更新最优算法和最优reward
            if reward > best_reward:
                best_reward = reward
                best_algorithm = algorithm_names[algorithm_index]

        for algorithm_index, algorithm_data in enumerate(data):
            trace_data = algorithm_data[trace_index]  # 每个trace是一个元组
            
            # 假设你已经有了 trace_data 结构，它是一个包含7个元素的元组
            # 这里的 trace_data 是一个示例
            # 提取第0, 3, 4, 5, 6个元素
            lists = [trace_data[i][:50] for i in [0, 3, 4, 5, 6]]  # 切片到50个元素

            # 将这些列表堆叠成一个 (5, 列表长度) 的 ndarray
            ndarray = np.vstack(lists).T

            # 转换为 PyTorch tensor
            tensor = torch.tensor(ndarray)

            dataset_pool.add(f"<|Task description|>You are tasked with selecting the most appropriate Adaptive Bitrate Control algorithm based on the provided network statistics and scenario. Your goal is to select the algorithm that best suits the given network status.\n",tensor,best_algorithm)

    #     # 提取数据，合成Prompt
    #     for index, row in group.iterrows():

    #         # 找到csv。合成ts数据
    #         if name[2] == 0.0:
    #             tt = '0'
    #         else:
    #             tt = name[2]
    #         coordinated_file_path = os.path.join(raw_file_directory,"{}_{}_{}_{}_{}_cwnd.csv".format(row['Schemes'],name[3],name[0],name[1],tt))
    #         try:
    #             df_ts = pd.read_csv(coordinated_file_path)
    #         except FileNotFoundError:
    #             print("{}_{}_{}_{}_{}_cwnd.csv not found".format(row['Schemes'],name[3],name[0],name[1],tt))
    #             continue


    #         top_50_rows = df_ts.head(50)
    #         data_array = top_50_rows.values
    #         ts_tensor_data = torch.tensor(data_array, dtype=torch.float32) 
    #         if ts_tensor_data.shape != (50, 77):
    #             print("{}_{}_{}_{}_{}_cwnd.csv shape error".format(row['Schemes'],name[3],name[0],name[1],tt))
    #             continue

    #         prompt_ = standard_prompt_filled(row['sr_max'],row['sr_avg'],row['sr_min'],row['rtt_max'],row['rtt_avg'],row['rtt_min'],row['rttvar_max'],row['rttvar_avg'],row['rttvar_min'],row['loss_max'],row['loss_avg'],row['loss_min'])
    #         dataset_pool.add(prompt_,ts_tensor_data,best_scheme)

            

    # # 组成pair，作为数据对并导出
    dataset_pool_output = '/data3/wuduo/xuanyu/llmcc/datasets/ABR/dataset_pool_ABR.pkl'
    pickle.dump(dataset_pool, open(dataset_pool_output, 'wb'))
