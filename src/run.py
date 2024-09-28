import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from dataset_loader import DatasetLoader
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel


from tqdm import tqdm

if __name__ == "__main__":
    # 0. set device, model path,init Tensorboard
    device = torch.device("cuda:2")
    plm_path = "../llama-7b"
    boardwriter = SummaryWriter(log_dir='logs')

    # 1. load dataset
    train_loader,val_loader,test_loader = DatasetLoader('../datasets/dataset_pool.pkl')

    # 2. Load Model,Tokenizer
    pretrained_llm = PretrainedLanguageModel(model_name='llama',model_path=plm_path,device=device)

    # 3. Tokenize &  Embedding
    embedder = DatasetEmbedding(pretrained_llm)

    # 4. train(forward & backward)
    # checkpoint_save_path = ()
    # train_steps = len(train_loader)
    # 
    # for p in model.parameters():
    #     if p.requires_grad is True:
    #         trained_parameters.append(p)

    # model_optim = optim.Adam(trained_parameters, lr=args.learning_rate) # 优化器

    # 迭代加载数据

    criterion = nn.CrossEntropyLoss()
    train_epochs = 1
    for epoch in range(train_epochs):
        tot_iter_count = 0
        train_loss = []

        for i, (batch_prompt, batch_ts, batch_label) in tqdm(enumerate(train_loader)):
            tot_iter_count+=1
            # model_optim.zero_grad()
            
            
            
            pass