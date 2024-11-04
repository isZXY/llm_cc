from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Planner:
    def __init__(self, model, checkpoint_save_path, device):
        self.device = device
        self.model = model
        self.checkpoint_save_path = checkpoint_save_path


    def tokens_to_text(self, tokens):
        words = [self.model.tokenizer.decode(token) for token in tokens]
        
        text = " ".join(words).replace(" ##", "")  
        return text.strip()
    

    def inference_on_dataset(self, test_loader):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_prompt, batch_ts, batch_label in tqdm(test_loader):
                algo_token_id, generated_sequence = self.model(batch_prompt, batch_ts)

                predictions.append(self.tokens_to_text(generated_sequence))
                algo_token_id = self.tokens_to_text([algo_token_id])
                print("selected algorithm id: {}".format(algo_token_id))
                print(predictions)

        
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
