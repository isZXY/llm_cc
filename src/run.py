import torch
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import DatasetLoader
from llmcc_model import Model
from trainer import Trainer

if __name__ == "__main__":

    '''
    还差:ts input的对齐
    '''

    # 0. set device, model path,init Tensorboard
    device = torch.device("cuda:2")
    plm_path = "../llama-7b"
    boardwriter = SummaryWriter(log_dir='logs')
    num_classes = 7
    
    # 1. load dataset
    train_loader,val_loader,test_loader = DatasetLoader('../datasets/dataset_pool.pkl')

    # 2. Load Model
    model = Model(plm_path,device,num_classes)
    
    
    # 3. Set train parameters
    learning_rate = .01
    train_epochs = 1
    checkpoint_save_path = "../checkpoints"

    # set Trainer and train
    trainer = Trainer(model,boardwriter,train_loader,val_loader,learning_rate,train_epochs,checkpoint_save_path)
    
    trainer.train()


