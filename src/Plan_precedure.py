import json
import torch
import time
import os
from llmcc_model import Model
from planner import Planner
import shutil


# 定义动作标签到索引的映射
label_to_index = {
    "genet": 0,
    "udr_1": 1,
    "udr_2": 2,
    "udr_3": 3,
    "udr_real": 4,
    "mpc": 5,
    "bba": 6
}


# 反向映射（可选）
index_to_label = {v: k for k, v in label_to_index.items()}


def Plan_precedure(planner):
    count =0
    while True:     
        if os.path.exists('network_data.pt'):
            count += 1
            print("got new network data the {} time".format(count))
            with open('network_data.pt', 'r') as f:
                 # 从 .pt 文件加载字典
                tensor_dict = torch.load(filename)
                
                # 提取 Tensor
                states = tensor_dict['state']
                # print(states)
                timestep = tensor_dict['timestep']
                # print(timestep)
                
                algo_predictions = planner.plan(states,timestep)

                algo_predictions_labels = index_to_label[algo_predictions.item()]
                

                data = {
                    'algorithm': algo_predictions_labels  # 保存标签字符串
                } 

                print("change selection @ {}".format(algo_predictions_labels))
                with open('algo_selection_data.json', 'w') as ff:
                    json.dump(data, ff)
                     
            os.remove('network_data.pt')

        time.sleep(1)  # 等待下一次检查




if __name__ == "__main__":
    os.chdir('/data3/wuduo/xuanyu/llmcc/swap')

    # reset swap folder files
    folder_path = '/data3/wuduo/xuanyu/llmcc/swap'

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

            
    device = torch.device("cuda:2")
    plm_path = "../llama-7b"

    model = Model(plm_path, device)

     # 3. Set train parameters··
    checkpoint_save_path = "/data3/wuduo/xuanyu/llmcc/checkpoints_Feb08_1824/checkpoint-151662-eval-epoch20"

    planner = Planner(model, checkpoint_save_path, 1, device)

    print("start listening...")
    Plan_precedure(planner)