import json
import torch
import time
import os
from llmcc_model import Model
from planner import Planner
import shutil


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
                timestep = tensor_dict['timestep']
                
                algo_predictions = planner.plan(states,timestep)
                data = {
                        'algorithm': algo_predictions
                    }
                print("change selection for reason {}".format(algo_predictions))
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

            
    device = torch.device("cuda:4")
    plm_path = "../llama-7b"

    model = Model(plm_path, device)

     # 3. Set train parameters··
    checkpoint_save_path = "../checkpoints"

    planner = Planner(model, checkpoint_save_path, 1, device)

    print("start listening...")
    Plan_precedure(planner)