import torch.nn as nn
class TimeLLM_Encode(nn.Module):
    '''用于编码时序数据
    input: 一个时序数据文件
    output: embedding后要交给大模型的输入
    '''
    def __init__(self, file_path):

        pass
    def forward(self,x_enc):
        # 1. Input Embedding
        ## 1.1 normalize
        x_enc = self.normalize_layers(x_enc, 'norm') # RevIN