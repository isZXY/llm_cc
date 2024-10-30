# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import csv
import torch

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
    'time_stamp','bit_rate','buffer_size','rebuffer_time','chunk_size','download_time','smoothness','model','reward'
]


def standard_prompt_filled():
    prompt = (
        f"<|Task description|>You are tasked with selecting the most appropriate adaptive bitrate algorithm based on the provided network statistics and scenario. Your goal is to select the algorithm that best suits the given network status. The given algorithms are 'genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real', 'mpc', 'bba','mixed'\n"
        f"<|CSV Head Explanation|>Each feature in the following time series are: time_stamp,bit_rate,buffer_size,rebuffer_time,chunk_size,download_time,smoothness,model,reward\n"
        # f"<|Network stat|>bit_rate(Kbps): max {sr_max}, average {sr_avg}, min {sr_min}; buffer_size(s): max {rtt_max}, average {rtt_avg}, min {rtt_min}; RTT rebuffer_time(ms): max {rttvar_max}, average {rttvar_avg}, min {rttvar_min}; chunk_size: max {loss_max}, average {loss_avg}, min {loss_min};\n"
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

if __name__ == '__main__':
    # set dataset path & extracted_info
    # raw_file_directory = "/data3/wuduo/xuanyu/llmcc/datasets/CC/csv"
    extracted_info_csv_path = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts/results/fcc-test_video1/trace_num_100_fixed_True'


    # init dataset pool class
    dataset_pool = _DatasetPool()

    # 遍历文件，读取csv，组合
    for algorithm_name in os.listdir(extracted_info_csv_path):
        algorithm_folder = os.path.join(extracted_info_csv_path, algorithm_name+'/seed_100003')
        
        # 检查是否是文件夹
        if os.path.isdir(algorithm_folder):
            # 遍历子文件夹中的CSV文件
            for file_name in os.listdir(algorithm_folder):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(algorithm_folder, file_name)
                    
                    # 读取CSV文件
                    df = pd.read_csv(file_path)
                    # 选择数值列
                    df = df.select_dtypes(include='number')
                    df = df.drop(df.columns[-2], axis=1)

                    top_30_rows = df.head(30).to_numpy()
                    ts_tensor_data = torch.tensor(top_30_rows, dtype=torch.float32) 
                    # 提取对应的算法名
                    # 这里你可以根据具体需求提取数据
                    prompt_ = standard_prompt_filled()
                    best_scheme = algorithm_name
                    dataset_pool.add(prompt_,ts_tensor_data,best_scheme)
                    # print(f'算法名: {best_scheme}, 文件名: {file_name}')
                    # print(df)  # 打印读取的数据
    dataset_pool_output = '/data3/wuduo/xuanyu/llmcc/environments/adaptive_bitrate_streaming/artifacts/dataset_pool.pkl'
    pickle.dump(dataset_pool, open(dataset_pool_output, 'wb'))

    # # extract info csv
    # if not os.path.exists(extracted_info_csv_path):
    #     with open(extracted_info_csv_path, mode='w') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(CSV_HEADER)
                
    #     for root, dirs, files in os.walk(raw_file_directory):
    #         for file in files:
    #             logfile_path = os.path.join(root, file)
    #             set_env = file.split("_")
    #             schemes,rtt,queue,loss,bw = set_env[0],int(set_env[2])*2,int(set_env[3]),float(set_env[4]),set_env[1]
    #             # 计算csv中的信息，写入csv
    #             avg_reward,sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min = extract_csv_info(logfile_path)
    #             row = [schemes,rtt,queue,loss,bw,avg_reward,sr_max,sr_avg,sr_min,rtt_max,rtt_avg,rtt_min,rttvar_max,rttvar_avg,rttvar_min,loss_max,loss_avg,loss_min]
    #             with open(extracted_info_csv_path, mode='a') as file:  # 使用 'ab' 代替 'a'
    #                 writer = csv.writer(file)
    #                 writer.writerow(row)
    # df = pd.read_csv(extracted_info_csv_path)
    

    # 


    # # Group to get label
    # grouped = df.groupby(['RTT(ms)','queue','loss','BW (Mbps)'])

    # # 找到 D 列中有多个不同值的组
    # result = grouped.filter(lambda x: x['Schemes'].nunique() > 1)

    # # 输出结果
    # for name, group in grouped: # (20, 1000, 0.02, 'wired12')
    #     # 获取该分组 D 列的最大值
    #     max_in_group = group['Avg_Reward'].max()
    #     max_row = group[group['Avg_Reward'] == max_in_group]

    #     best_scheme = max_row['Schemes'].values[0]


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
    # dataset_pool_output = '/data3/wuduo/xuanyu/llmcc/datasets/CC/dataset_pool.pkl'
    # pickle.dump(dataset_pool, open(dataset_pool_output, 'wb'))
