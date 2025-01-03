import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pytz

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
    device = torch.device("cuda:7")
   # device = torch.device("cpu")
    plm_path = "../llama-7b"


    tz = pytz.timezone('Asia/Shanghai')  # 设置时区为北京时间
    current_time = datetime.now(tz).strftime("%b%d_%H%M")

    # 创建一个以当前时间命名的文件夹
    log_dir = os.path.join('logs', current_time)
    boardwriter = SummaryWriter(log_dir=log_dir)

    batch_size = 1
    # 1. load dataset
    dataset_loader = DatasetLoader('ABR',
        '../datasets/ABR/dataset_pool_ABR.pkl', batch_size=batch_size, train_prop=0.6, val_prop=0.2, test_prop=0.2)
    train_loader, val_loader, test_loader = dataset_loader.load_dataset()

    # 2. Load Model
    model = Model(plm_path, device)

    # 3. Set train parameters
    learning_rate = .01
    train_epochs = 30
    checkpoint_save_path = "../checkpoints_{}".format(current_time)

    # set Trainer and train
    trainer = Trainer(model, boardwriter, train_loader, val_loader,
                      learning_rate, train_epochs, checkpoint_save_path,batch_size,device)
    
    trainer.train()
