import os
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import DatasetLoader
from llmcc_model import Model
from planner import Planner

from dataset_pool import _DatasetPool
from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":
    '''
    还差:eval评估
    '''
    os.chdir('/data3/wuduo/xuanyu/llmcc/src')

    # 0. set device, model path, init Tensorboard
    device = torch.device("cpu")
    plm_path = "../llama-7b"

    batch_size = 1
    # 1. load dataset
    dataset_loader = DatasetLoader('ABR',
        '../datasets/ABR/dataset_pool_ABR.pkl', batch_size=batch_size, train_prop=0.6, val_prop=0.2, test_prop=0.2)
    train_loader, val_loader, test_loader = dataset_loader.load_dataset()

    # 2. Load Model
    model = Model(plm_path, device)
    checkpoint_save_path = "/data3/wuduo/xuanyu/llmcc/checkpoints/checkpoint-36110-eval-epoch4"
    model.load_model(checkpoint_save_path)
    



    # set Planner and plan
    planner = Planner(model, checkpoint_save_path,batch_size,device)

    # planner.inference_on_dataset(test_loader)

    planner.simu_real_env_on_dataset(test_loader)