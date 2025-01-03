from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

label_to_index = {
    "genet": 0,
    "udr_1": 1,
    "udr_2": 2,
    "udr_3": 3,
    "udr_real": 4,
    "mpc": 5,
    "bba": 6,
    "mixed": 7
}


# 反向映射（可选）
index_to_label = {v: k for k, v in label_to_index.items()}


def process_batch(batch, device='cpu'):
    """
    Process batch of data.
    """
    states, actions, returns, timesteps = batch
    # now states shape:  (1, 4,5) (features, decision_interval_per_record)
    states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)  # (1,8,4,5)

    actions = [label_to_index[action[0]] for action in actions]  # 转换为索引 (1,8,1)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
    actions = actions.unsqueeze(-1).to(device)  
    labels = actions.clone().to(device)   # 离散动作的标签，用于分类损失

    returns = torch.tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1).to(device) 

    # 时间步处理
    timesteps = torch.tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)
    timesteps = timesteps.unsqueeze(-1).to(device) 
    return states, actions, returns, timesteps, labels


class Planner:
    def __init__(self, model, checkpoint_save_path, device):
        self.device = device
        self.model = model
        self.checkpoint_save_path = checkpoint_save_path


    # def tokens_to_text(self, tokens):
    #     words = [self.model.tokenizer.decode(token) for token in tokens]
        
    #     text = " ".join(words).replace(" ##", "")  
    #     return text.strip()


    def inference_on_dataset(self, test_loader):
        self.model.eval()
        true_predictions = 0

        total_predictions = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                
                states, actions, returns, timesteps, labels = process_batch(batch,self.device)
                logits = self.model(states, actions, returns, timesteps, labels)
                logits = logits.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)

                labels = labels.squeeze(-1)  # 形状变为 (8,) 真实标签
                predicted_action_indices = torch.argmax(logits, dim=-1)  # 形状为 (1, 8)
                predicted_actions = [index_to_label[idx.item()] for idx in predicted_action_indices[0]]

                actions_labels  = [index_to_label[idx.item()] for idx in labels[0]]
                print('predicted:',predicted_actions, 'label:',actions_labels)

                total_predictions +=1

                if predicted_actions[-1] == actions_labels[-1]:
                    true_predictions +=1

        print("acc = {}%".format(true_predictions/total_predictions *100))


    def plan(self,prompt,ts):
        '''
        prompt: need to be a 1-dim tuple
        ts: need to be a tensor in (1,30,7), 30 is alternative
        '''
        self.model.eval()
        predictions = []

        with torch.no_grad():
            algo_token_id, generated_sequence = self.model(prompt, ts)

            predictions.append(self.tokens_to_text(generated_sequence))
            algo_token_id = self.tokens_to_text([algo_token_id])
            return algo_token_id, predictions
