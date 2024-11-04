import json
import torch
import time
import os
from llmcc_model import Model
from planner import Planner


def Plan_precedure(planner):
    while True:     
        if os.path.exists('network_data.json'):
            with open('network_data.json', 'r') as f:
                data = json.load(f)
                prompt = data['prompt']
                
                ts = torch.tensor(data['ts'])
                algo_token_id, predictions = planner.plan(prompt,ts)
                data = {
                        'algorithm': algo_token_id
                    }
                print("change selection for reason {}".format(predictions))
                with open('algo_selection_data.json', 'w') as ff:

                     json.dump(data, ff)
            os.remove('network_data.json')

        time.sleep(1)  # 等待下一次检查




if __name__ == "__main__":
    os.chdir('/data3/wuduo/xuanyu/llmcc/swap')

    device = torch.device("cpu")
    plm_path = "../llama-7b"

    model = Model(plm_path, device)

     # 3. Set train parameters
    checkpoint_save_path = "../checkpoints"

    planner = Planner(model, checkpoint_save_path, device)

    print("start listening...")
    Plan_precedure(planner)