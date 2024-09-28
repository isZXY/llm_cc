import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.optim as optim

from dataset_loader import DatasetLoader
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel
from llmcc_model import Model

from tqdm import tqdm

if __name__ == "__main__":
    # 0. set device, model path,init Tensorboard
    device = torch.device("cuda:2")
    plm_path = "../llama-7b"
    boardwriter = SummaryWriter(log_dir='logs')
    num_classes = 7

    # 1. load dataset
    train_loader,val_loader,test_loader = DatasetLoader('../datasets/dataset_pool.pkl')

    # 2. Load Model
    model = Model(plm_path,device,num_classes)



    # 3. Set train paras
    learning_rate = .01
    train_epochs = 1
    # checkpoint_save_path = ()
    # train_steps = len(train_loader)
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    optimizer = optim.Adam(trained_parameters, lr=learning_rate) # 优化器
    loss_fcn = nn.CrossEntropyLoss()

    # 迭代加载数据
    for epoch in range(train_epochs):
        tot_iter_count = 0
        train_loss = []

        for i, (batch_prompt, batch_ts, batch_label) in tqdm(enumerate(train_loader)):
            tot_iter_count+=1
            # model_optim.zero_grad()
            optimizer.zero_grad()
            
            logits = model(batch_prompt, batch_ts)
            
            loss = loss_fcn(logits, batch_label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            pass