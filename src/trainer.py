from torch import nn
import torch.optim as optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
            for i, (batch_prompt, batch_ts, batch_label) in tqdm(enumerate(self.train_loader)):
                self.global_step += 1
                self.model.train()

                self.optimizer.zero_grad()

                total_loss = self.model(batch_prompt, batch_ts, batch_label, mode="train")

                self.boardwriter.add_scalar(
                    'iter Loss/train', total_loss.item(), self.global_step)

                loss.backward()
                self.optimizer.step()

                if self.global_step % 500 == 0:
                    self.model.save_model(os.path.join(
                        self.checkpoint_save_path, 'checkpoint-{}'.format(self.global_step)))

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
