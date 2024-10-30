import os
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import DatasetLoader
from llmcc_model import Model
from trainer import Trainer

from dataset_pool import _DatasetPool


if __name__ == "__main__":

    '''
    还差:eval评估
    '''
    os.chdir('/data3/wuduo/xuanyu/llmcc/src')

    # 0. set device, model path,init Tensorboard
    device = torch.device("cpu")
    plm_path = "../llama-7b"
    boardwriter = SummaryWriter(log_dir='logs')
    num_classes = 7

    # 1. load dataset
    dataset_loader = DatasetLoader(
        '../datasets/dataset_pool_abr.pkl', batch_size=2, train_prop=0.6, val_prop=0.2, test_prop=0.2)
    train_loader, val_loader, test_loader = dataset_loader.load_dataset()

    # 2. Load Model
    model = Model(plm_path, device, num_classes)

    # 3. Set train parameters
    learning_rate = .01
    train_epochs = 5
    checkpoint_save_path = "../checkpoints"

    # set Trainer and train
    trainer = Trainer(model, boardwriter, train_loader, val_loader,
                      learning_rate, train_epochs, checkpoint_save_path,device)

    trainer.train()
