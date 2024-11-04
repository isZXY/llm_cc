from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Planner:
    def __init__(self, model, test_loader, checkpoint_save_path, device):
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.checkpoint_save_path = checkpoint_save_path


    def tokens_to_text(self, tokens):
        words = [self.model.tokenizer.decode(token) for token in tokens]
        
        text = " ".join(words).replace(" ##", "")  
        return text.strip()


    # def train(self):
    #     for epoch in range(self.train_epochs):
    #         for i, (batch_prompt, batch_ts, batch_label) in tqdm(enumerate(self.train_loader)):
    #             self.global_step += 1
    #             self.model.train()

    #             self.optimizer.zero_grad()

    #             logits = self.model(batch_prompt, batch_ts).squeeze()
    #             batch_label = batch_label.to(self.device)
    #             loss = self.loss_fcn(logits, batch_label)
    #             self.boardwriter.add_scalar(
    #                 'iter Loss/train', loss.item(), self.global_step)

    #             loss.backward()
    #             self.optimizer.step()

    #             if self.global_step % 500 == 0:
    #                 self.model.save_model(os.path.join(
    #                     self.checkpoint_save_path, 'checkpoint-{}'.format(self.global_step)))

    #         # Validate
    #         self.model.eval()
    #         with torch.no_grad():
    #             val_loss = 0.0
    #             for batch_prompt, batch_ts, batch_label in tqdm(self.val_loader):
    #                 logits = self.model(batch_prompt, batch_ts).squeeze()
    #                 batch_label = batch_label.to(self.device)
    #                 loss = self.loss_fcn(logits, batch_label)
    #                 val_loss += loss.item()
    #             self.boardwriter.add_scalar(
    #                 'Epoch Loss/Validation', val_loss / len(self.val_loader), epoch)

    #         self.model.save_model(os.path.join(
    #             self.checkpoint_save_path, 'checkpoint-{}-eval-epoch{}'.format(self.global_step, epoch)))
    
    def plan(self):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_prompt, batch_ts, batch_label in tqdm(self.test_loader):
                algo_token_id, generated_sequence = self.model(batch_prompt, batch_ts)

                predictions.append(self.tokens_to_text(generated_sequence))
                algo_token_id = self.tokens_to_text([algo_token_id])
                print("selected algorithm id: {}".format(algo_token_id))
                print(predictions)

