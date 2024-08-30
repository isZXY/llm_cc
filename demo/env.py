import os
import time
import random

# from -> to paras
initial_bandwidth = 1  # Mbps
target_bandwidth = 9     # Mbps
initial_loss = 0.05      # %
target_loss = 1         # %
initial_rtt = 800        # ms 
target_rtt =  100          # ms 
duration = 30           # s 仿真时间


def generate_random_fluctuation(value, fluctuation_percentage):
    '''生成随机浮动值'''
    fluctuation = value * fluctuation_percentage
    return value + random.uniform(-fluctuation, fluctuation)


def generate_mahimahi_command(bandwidth, loss, rtt):
    '''Mahimahi命令生成'''
    return f"mm-delay {rtt} mm-loss uplink {loss} mm-link --uplink-queue=100 {bandwidth}Mbit/s --downlink-queue=100"

# 主要仿真循环
for t in range(duration):
    # 根据时间计算目标值
    current_bandwidth = initial_bandwidth - (initial_bandwidth - target_bandwidth) * (t / duration)
    current_loss = initial_loss + (target_loss - initial_loss) * (t / duration)
    current_rtt = initial_rtt + (target_rtt - initial_rtt) * (t / duration)
    
    
    # 引入随机波动
    current_bandwidth = generate_random_fluctuation(current_bandwidth, 0.2)  # 10%的随机波动
    current_loss = generate_random_fluctuation(current_loss, 0.2)            # 20%的随机波动
    current_rtt = generate_random_fluctuation(current_rtt, 0.2)              # 10%的随机波动
    print(current_bandwidth,current_loss,current_rtt)


    # 生成对应的Mahimahi命令
    command = generate_mahimahi_command(current_bandwidth, current_loss, current_rtt)
    # print(f"Time: {t}s, Command: {command}")
    
    # 执行命令
    # os.system(command)
    
    # 每秒执行一次调整
    time.sleep(1)
