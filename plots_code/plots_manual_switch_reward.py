import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import sys
from const import *

#  给定几个csv文件，画出几根曲线；
#  设置画布大小；
#  横轴是timestamp，纵轴是reward；
#  颜色配色和花纹根据算法来确定。


switch_plot_dict ={
    'arbitary_switch_true_trace27.csv': ('#FF8C00', '.'),
    'no_switch_true_trace27.csv': ('#4B0082', 'o'),
    'manual_switch_true_trace27.csv': ('#CD853F', '.'),

}

csv_file_list = [
    'no_switch_true_trace27.csv',
    'arbitary_switch_true_trace27.csv',
    'manual_switch_true_trace27.csv',
    
]



def find_change_indices(lst):
    # 找出变化的索引
    change_indices = [i for i in range(1, len(lst)) if lst[i] != lst[i-1]]
    
    if not change_indices:  # 如果没有变化
        return [-1], lst[0], lst[0]  # 返回 -1 和相同元素
    
    # 如果有变化，返回变化索引和变化前后的元素
    first_change_index = change_indices[0]
    return change_indices, lst[first_change_index - 1], lst[first_change_index]

def readcsv(path,file):
    all = pd.read_csv(path)
    time_stamps = list(all["time_stamp"])
    culmulative_rewards = list(all["reward"])
    model = list(all['model'])

    # 计算累积奖励
    for i in range(1, len(culmulative_rewards)):
        culmulative_rewards[i] += culmulative_rewards[i - 1]
    
    return (file, time_stamps, culmulative_rewards, model)


def get_data(path,file):
    now = []
    now.append(readcsv(path,file))
    
    data.append(now)


def get_ticks(num, cnt):
    length = max(num) - min(num)
    if length < cnt:
        return 1
    return round(length / cnt)


dir_pre = '/data3/wuduo/xuanyu/llmcc/重要实验结果备份/任务0:切换有效的实验验证'
output_dir = '/data3/wuduo/xuanyu/llmcc/plots_code/plots/manual_switch'

if __name__ == "__main__":
    # 设置输入输出文件路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"manual_switch_validation.pdf"
    outputfile = os.path.join(output_dir, filename)

    # 获取作图数据
    data = []
    for file in csv_file_list:
        file_path = os.path.join(dir_pre,file)
        get_data(file_path,file)

    # 设置画图属性
    plt.cla()
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(3, 2.2))
    # plt.title('Cumulative Reward of Cases')
        

    # 设置tick（坐标轴）
    tot_time = []
    tot_re = []
    for csv in data:
        (file, time_stamps, culmulative_rewards, model) = csv[0]
        tot_time += time_stamps
        tot_re += culmulative_rewards
    
    xtick = get_ticks(tot_time, 5)
    ytick = get_ticks(tot_re, 5)
    plt.gca().yaxis.set_major_locator(MultipleLocator(ytick))
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))


    # 作图
    model_plotted = set()
    for i,csv in enumerate(data):
        (file, time_stamps, culmulative_rewards, model) = csv[0]

        plt.plot(
            time_stamps,
            culmulative_rewards,
            linewidth=1,  # 线条宽度
            color=switch_plot_dict[file][0],
            marker=switch_plot_dict[file][1],
            markersize=2,  # 标记大小稍微减小
            linestyle='-',  # 使用实线
            label="{} switch".format(file.split('_')[0]),
            alpha=1
        )
        

    # 加标注

    plt.annotate(
                text="genet",  # 显示的文字
                xy=(time_stamps[0], culmulative_rewards[0]),  # 箭头指向的位置
                xytext = (time_stamps[0], culmulative_rewards[0] - 8),  # 文字向下偏移
                arrowprops=dict(
                    arrowstyle="fancy",   # 箭头样式
                    color=switch_plot_dict['no_switch_true_trace27.csv'][0],      # 箭头颜色
                    lw=1,              # 箭头线宽
                    mutation_scale=10, # 控制箭头大小
                ),  # 箭头样式
                fontsize=3,  # 字体大小
                color=switch_plot_dict['no_switch_true_trace27.csv'][0],  # 文字颜色
                horizontalalignment='center',  # 文字居中
                verticalalignment='bottom'  # 文字底部对齐
            )

    for i, csv in enumerate(data):
        (file, time_stamps, culmulative_rewards, model) = csv[0]
        indice, last_algo, current_algo = find_change_indices(model)
        indice = indice[0]
        
        if indice != -1:
            # 判断奇偶索引，决定箭头的方向
            if i % 2 == 0:
                # 偶数索引，箭头自下而上
                xytext = (time_stamps[indice], culmulative_rewards[indice] + 8)  # 文字向上偏移
                arrowprops = dict(
                    arrowstyle="fancy",   # 箭头样式
                    color=switch_plot_dict[file][0],      # 箭头颜色
                    lw=1,              # 箭头线宽
                    mutation_scale=10, # 控制箭头大小
                )
            else:
                # 奇数索引，箭头自上而下
                xytext = (time_stamps[indice], culmulative_rewards[indice] - 8)  # 文字向下偏移
                arrowprops = dict(
                    arrowstyle="fancy",   # 箭头样式
                    color=switch_plot_dict[file][0],      # 箭头颜色
                    lw=1,              # 箭头线宽
                    mutation_scale=10, # 控制箭头大小
                )
            
            plt.annotate(
                text="→{}".format(current_algo),  # 显示的文字
                xy=(time_stamps[indice], culmulative_rewards[indice]),  # 箭头指向的位置
                xytext=xytext,  # 文字的位置
                arrowprops=arrowprops,  # 箭头样式
                fontsize=3,  # 字体大小
                color=switch_plot_dict[file][0],  # 文字颜色
                horizontalalignment='center',  # 文字居中
                verticalalignment='bottom'  # 文字底部对齐
            )




    # 设置X Y标签
    plt.xlabel('Time Elapsed(s)')
    plt.ylabel('Cumulative Rewards')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    

    # 设置图注
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize='x-small',loc = 'lower left')
    plt.tight_layout()

    plt.savefig(outputfile)
