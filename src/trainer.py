from torch import nn
import torch.optim as optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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



class Trainer:
    def __init__(self, model, boardwriter, train_loader, val_loader, learning_rate, train_epochs, checkpoint_save_path, device):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.checkpoint_save_path = checkpoint_save_path

        self.boardwriter = boardwriter
        trained_parameters = []
        for p in self.model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
        self.optimizer = optim.Adam(trained_parameters, lr=learning_rate)
        self.loss_fcn = nn.CrossEntropyLoss()
        self.train_steps = len(train_loader)

        self.global_step = 0

    def train(self):
        for epoch in range(self.train_epochs):
            for i, batch in tqdm(enumerate(self.train_loader)):
                states, actions, returns, timesteps, labels = process_batch(batch,self.device)
                
                self.global_step += 1
                self.model.train()

                self.optimizer.zero_grad()

                logits = self.model(states, actions, returns, timesteps, labels)
                
                predicted_action_indices = torch.argmax(logits, dim=-1)  # 形状为 (1, 8)
                predicted_actions = [index_to_label[idx.item()] for idx in predicted_action_indices[0]]
                
                logits = logits.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)

                # logits = logits.squeeze(0)[-1,:]  # 形状变为 (8, 8)，去掉 batch size 维度
                labels = labels.squeeze(-1)  # 形状变为 (8,) 真实标签
                actions_labels  = [index_to_label[idx.item()] for idx in labels[0]]
                # print('choose' + index_to_label[predict_result])
                
                loss = self.loss_fcn(logits, labels)

                print('predicted:',predicted_actions, 'label:',actions_labels,'loss:',loss.item())
                self.boardwriter.add_scalar(
                    'iter Loss/train', loss.item(), self.global_step)

                loss.backward()
                self.optimizer.step()

                if self.global_step % 500 == 0:
                    self.model.save_model(os.path.join(
                        self.checkpoint_save_path, 'checkpoint-{}'.format(self.global_step)))

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, batch in tqdm(enumerate(self.val_loader)):
                    states, actions, returns, timesteps, labels = process_batch(batch,self.device)
                    logits = self.model(states, actions, returns, timesteps, labels)
                    logits = logits.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)

                    labels = labels.squeeze(-1)  # 形状变为 (8,) 真实标签

                    loss = self.loss_fcn(logits, labels)
                    val_loss += loss.item()
                self.boardwriter.add_scalar(
                    'Epoch Loss/Validation', val_loss / len(self.val_loader), epoch)

            self.model.save_model(os.path.join(
                self.checkpoint_save_path, 'checkpoint-{}-eval-epoch{}'.format(self.global_step, epoch)))
