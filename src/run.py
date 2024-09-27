import torch
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import DatasetLoader
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel

if __name__ == "__main__":
    # 0. set device, model path,init Tensorboard
    device = torch.device("cuda:2")
    plm_path = "../llama-7b"
    boardwriter = SummaryWriter(log_dir='logs')

    # 1. load dataset
    dataset_loader = DatasetLoader('../datasets/dataset_pool.pkl')
    dataset_train,dataset_val,dataset_test = dataset_loader.load_dataset()

    # 2. Load Model,Tokenizer
    pretrained_llm = PretrainedLanguageModel(model_name='llama',model_path=plm_path,device=device)

    # 3. Tokenize &  Embedding
    embedder = DatasetEmbedding(pretrained_llm)

    # 4. train(forward & backward)
