from torch import nn
import torch.optim as optim
import torch
import os
import time


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

    # returns = torch.tensor(returns, dtype=torch.float32, device=device).reshape(16, -1, 1).to(device) 
    returns = torch.stack(returns, dim=1).unsqueeze(-1).float().to(device)
    # 时间步处理

    # timesteps = torch.tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)
    # timesteps = timesteps.unsqueeze(-1).to(device) 
    timesteps = torch.stack(timesteps, dim=1).unsqueeze(-1).to(dtype=torch.int64).to(device)

    return states, actions, returns, timesteps, labels




class Trainer:
    def __init__(self, model, boardwriter, train_loader, val_loader, learning_rate, train_epochs, checkpoint_save_path,batch_size, device):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

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
            start_time = time.time()
            for i, batch in tqdm(enumerate(self.train_loader)):
                # states, actions, returns, timesteps, labels = process_batch(batch,self.batch_size,self.device)
                processed_batch = process_batch(batch, self.batch_size, self.device)



                # 如果处理失败（返回 None），记录当前的批次序号，并跳过该批次
                if processed_batch is None:
                    print(f"Skipping batch {i} due to error in 'actions'.")
                    self.boardwriter.add_scalar('Error Batch/Skipped', i, self.global_step)
                    continue  # 跳过当前批次

                # 如果处理成功，继续训练过程
                states, actions, returns, timesteps, labels = processed_batch
                    
                self.global_step += 1
                self.model.train()

                self.optimizer.zero_grad()

                logits = self.model(states, actions, returns, timesteps, labels)
                
                predicted_action_indices = torch.argmax(logits, dim=-1)  # logits 形状为 (batch_size, 8, 7) -> predicted_action_indices (16,8)
                predicted_actions = [index_to_label[idx.item()] for idx in predicted_action_indices[0]]
                
                logits = logits.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)

                # logits = logits.squeeze(0)[-1,:]  # 形状变为 (8, 8)，去掉 batch size 维度
                labels = labels.squeeze(-1)  # 形状变为 (8,) 真实标签
                actions_labels  = [index_to_label[idx.item()] for idx in labels[0]]
                # print('choose' + index_to_label[predict_result])
                
                loss = self.loss_fcn(logits, labels)

                print('predicted:',predicted_actions, 'label:',actions_labels,'loss:',loss.item())

                if self.global_step % 50 == 0: 
                    self.boardwriter.add_scalar('iter Loss/train', loss.item(), self.global_step) # 训练集损失
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.boardwriter.add_scalar('LearningRate', current_lr, self.global_step) # 学习率

                    # 计算梯度范数
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.boardwriter.add_scalar('GradientNorm', grad_norm, self.global_step)




                loss.backward()
                self.optimizer.step()

                if self.global_step % 500 == 0:
                    self.boardwriter.add_text('Prediction/Train', f'Predicted: {predicted_actions}, Label: {actions_labels}', self.global_step)

                    self.model.save_model(os.path.join(
                        self.checkpoint_save_path, 'checkpoint-{}'.format(self.global_step)))

            # Validate
            self.model.eval()



            # 验证精度计算
            correct = 0
            total = 0
            with torch.no_grad():
                val_loss = 0.0
                for i, batch in tqdm(enumerate(self.val_loader)):
                    states, actions, returns, timesteps, labels = process_batch(batch,self.batch_size,self.device)
                    logits = self.model(states, actions, returns, timesteps, labels)

                    predicted = torch.argmax(logits, dim=-1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    logits = logits.permute(0, 2, 1)  # 调整形状为 (batch_size, num_classes, sequence_length)
                    labels = labels.squeeze(-1)  # 形状变为 (8,) 真实标签

                    loss = self.loss_fcn(logits, labels)
                    val_loss += loss.item()
                self.boardwriter.add_scalar('Epoch Loss/Validation', val_loss / len(self.val_loader), epoch)


                accuracy = correct / total
                self.boardwriter.add_scalar('Epoch Accuracy/Validation', accuracy, epoch)

            self.model.save_model(os.path.join(
                self.checkpoint_save_path, 'checkpoint-{}-eval-epoch{}'.format(self.global_step, epoch)))

        epoch_time = time.time() - start_time
        self.boardwriter.add_scalar('Epoch Time/Training', epoch_time, epoch)
