from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datetime import datetime
import pytz

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


def process_batch(batch, batch_size,device='cpu'):
    """
    Process batch of data.
    """
    states, actions, returns, timesteps = batch 
    # len([16,5,5]) =8 --->>  (16,8,5,5) (batch_size, window_length, features, decision_interval)
    # states = torch.cat(states, dim=0).unsqueeze(1).float().to(device)  # (16,8,5,5)
    states = torch.stack(states, dim=1).float().to(device)

    

    
    # actions = [label_to_index[action[0]] for action in actions]  # 转换为索引  ->tensor形状(16,8,1)
    # actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
    # actions = actions.unsqueeze(-1).to(device)  
    try:
        actions = torch.stack([
            torch.tensor([label_to_index[action] for action in action_tuple], dtype=torch.long).view(batch_size, 1)
            for action_tuple in actions
        ], dim=1).float().to(device)
    except Exception as e:
        print(f"Error processing 'actions': {e}, skipping this batch.")
        return None  # Return None or any signal indicating an error in 'actions'

    labels = actions.clone().to(dtype=torch.int64).to(device)   # 离散动作的标签，用于分类损失

    returns = torch.stack(returns, dim=1).unsqueeze(-1).float().to(device)    # returns = torch.tensor(returns, dtype=torch.float32, device=device).reshape(16, -1, 1).to(device) 

    # 时间步处理

    # timesteps = torch.tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)
    # timesteps = timesteps.unsqueeze(-1).to(device) 
    timesteps = torch.stack(timesteps, dim=1).unsqueeze(-1).to(dtype=torch.int64).to(device)

    return states, actions, returns, timesteps, labels

class Planner:
    def __init__(self, model, checkpoint_save_path,batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.checkpoint_save_path = checkpoint_save_path
        self.loss_fcn = nn.CrossEntropyLoss()

        tz = pytz.timezone('Asia/Shanghai')  # 设置时区为北京时间
        current_time = datetime.now(tz).strftime("%b%d_%H%M")

        # 创建一个以当前时间命名的文件夹
        log_dir = os.path.join('logs_test', current_time)
        self.boardwriter = SummaryWriter(log_dir=log_dir)

    # def tokens_to_text(self, tokens):
    #     words = [self.model.tokenizer.decode(token) for token in tokens]
        
    #     text = " ".join(words).replace(" ##", "")  
    #     return text.strip()


    def inference_on_dataset(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            test_loss = 0.0
            for i, batch in tqdm(enumerate(test_loader)):
                states, actions, returns, timesteps, labels = process_batch(batch,self.batch_size,self.device)

                # 这里喂数据的逻辑是有一些问题的，要按照inference的逻辑来喂
                # 单次应该给一个state，一个target return 和timestep 让大模型推理出一个actions
                
                logits = self.model(states, actions, returns, timesteps, labels)

                predicted = torch.argmax(logits, dim=-1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                logits = logits.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)
                labels = labels.squeeze(-1)  # 形状变为 (8,) 真实标签

                loss = self.loss_fcn(logits, labels)
                test_loss += loss.item()

                self.boardwriter.add_scalar('Loss/Test', test_loss / len(test_loader),1)
                predicted_action_indices = torch.argmax(logits, dim=-1)  # 形状为 (1, 8)
                predicted_actions = [index_to_label[idx.item()] for idx in predicted_action_indices[0]]

                actions_labels  = [index_to_label[idx.item()] for idx in labels[0]]
                print('predicted:',predicted_actions, 'label:',actions_labels)

        accuracy = correct / total
        self.boardwriter.add_scalar('Epoch Accuracy/Test', accuracy, 1)

        print("acc = {}%".format(accuracy))


    def simu_real_env_on_dataset(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            test_loss = 0.0
            for i, batch in tqdm(enumerate(test_loader)):
                states, actions, returns, timesteps, labels = process_batch(batch,self.batch_size,self.device)

                # 这里喂数据的逻辑是有一些问题的，要按照inference的逻辑来喂
                # 单次应该给一个state，一个target return 和timestep 让大模型推理出一个actions
                # self, state, target_return, timestep, **kwargs
                # target_return = exp_dataset_info.max_return * args.target_return_scale
                target_return = 1
                for i in range(states.shape[1]):
                    action_pred = self.model.sample(states[:,i,:,:],target_return,timesteps[:,i,:])
                    
                    predicted = torch.argmax(action_pred, dim=-1)
                    correct += (predicted == labels[:,i,:]).sum().item()
                    total += labels[:,i,:].size(0)

                    logits = action_pred.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)

                    # loss = self.loss_fcn(logits, labels)
                    # test_loss += loss.item()

                    # self.boardwriter.add_scalar('Loss/Test', test_loss / len(test_loader),1)
                    # predicted_action_indices = torch.argmax(logits, dim=-1)  # 形状为 (1, 8)
                    # predicted_actions = [index_to_label[idx.item()] for idx in predicted_action_indices[0]]

                    # actions_labels  = [index_to_label[idx.item()] for idx in labels[0]]
                    print('predicted:',index_to_label[predicted.item()], 'label:',index_to_label[labels[:,i,:].item()])

            accuracy = correct / total
            self.boardwriter.add_scalar('Epoch Accuracy/Test', accuracy, 1)

            print("acc = {}%".format(accuracy))

    def plan(self,state_data,timesteps):
        '''
        prompt: need to be a 1-dim tuple
        ts: need to be a tensor in (1,30,7), 30 is alternative
        '''
        self.model.eval()
        predictions = []
        
        target_return = 1
        with torch.no_grad():
            # state_data should be a shape of [1,5,5]
            action_pred = self.model.sample(state_data,target_return,timesteps)
            predicted = torch.argmax(action_pred, dim=-1)
            return predicted