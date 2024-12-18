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

    states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)

    actions = [label_to_index[action[0]] for action in actions]  # 转换为索引
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
    actions = actions.unsqueeze(-1)
    labels = actions.clone()  # 离散动作的标签，用于分类损失

    returns = torch.tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)

    # 时间步处理
    timesteps = torch.tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)
    timesteps = timesteps.unsqueeze(-1)
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
                states, actions, returns, timesteps, labels = process_batch(batch)
                
                pass

                # 1.1 embed action, return, timestep
                action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
                returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
                time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

                # 1.2 time embeddings are treated similar to positional embeddings
                action_embeddings = action_embeddings + time_embeddings
                returns_embeddings = returns_embeddings + time_embeddings



                # Step 2: process states, turn them into embeddings.


                
                # self.global_step += 1
                # self.model.train()

                # self.optimizer.zero_grad()

                # total_loss = self.model(batch_prompt, batch_ts, batch_label, mode="train")

                # self.boardwriter.add_scalar(
                #     'iter Loss/train', total_loss.item(), self.global_step)

                # loss.backward()
                # self.optimizer.step()

                # if self.global_step % 500 == 0:
                #     self.model.save_model(os.path.join(
                #         self.checkpoint_save_path, 'checkpoint-{}'.format(self.global_step)))

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_prompt, batch_ts, batch_label in tqdm(self.val_loader):
                    logits = self.model(batch_prompt, batch_ts).squeeze()
                    batch_label = batch_label.to(self.device)
                    loss = self.loss_fcn(logits, batch_label)
                    val_loss += loss.item()
                self.boardwriter.add_scalar(
                    'Epoch Loss/Validation', val_loss / len(self.val_loader), epoch)

            self.model.save_model(os.path.join(
                self.checkpoint_save_path, 'checkpoint-{}-eval-epoch{}'.format(self.global_step, epoch)))
