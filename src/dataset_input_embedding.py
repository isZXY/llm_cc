import torch
from torch import nn

class _TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(_TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x # (B * N, P, d_model)


class _ReplicationPad1d(nn.Module):
    '''
     ReplicationPad1d 层的作用是对输入张量进行一种特殊的“填充”操作，即复制输入张量的最后一列（在时间维度上）若干次，并将这些复制的列拼接到原始输入的末尾
    '''
    def __init__(self, padding) -> None:
        super(_ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        # input (batch, 特征维度，时序长度)(B,N,T)
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])# (B,N) ->(B,N,1) ->(B, N, stride)
        output = torch.cat([input, replicate_padding], dim=-1) #(B,N,T+stride)
        return output


class _PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout): # patch模型的隐藏层维度,patch 长度 Lp, 步幅
        super(_PatchEmbedding, self).__init__()

        # Patching
        self.patch_len = patch_len # patch 长度
        self.stride = stride # S,sliding stride
        self.padding_patch_layer = _ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = _TokenEmbedding(patch_len, d_model) # (Lp--linear--> dm)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # x:(batch, 特征维度，时序长度)(B,N,T)
        n_vars = x.shape[1] # 特征维度N
        x = self.padding_patch_layer(x) # (B,N,T) -> (B,N,T+stride)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # padding展开，长度是patch_len,步长是stride # (B, N, num_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # (B*N, num_patches P, patch_len Lp)
        # Input encoding
        x = self.value_embedding(x) # 用的是卷积而非线性层 -> Embedding(B * N, P, d_model dm)
        return self.dropout(x), n_vars
 

class DatasetEmbedding:
    '''
    Tokenize dataset for use.
    '''    
    def __init__(self,plm,device):
        # set training device
        self.device = device
        # load tokenizer & plm
        self.tokenizer = plm.tokenizer()
        self.llm_model = plm.llm_model()
        
        # create mapping from llm vocab to maintained prototype.
        self.llm_embeddings_weight = self.llm_model.get_input_embeddings().weight # (vocab_size, embedding_dim)
        self.vocab_size = self.llm_embeddings_weight.shape[0] # llm vocab size
        self.maintained_prototype_token_size = 1000
        self.vocab_mapping_to_prototype_layer = nn.Linear(self.vocab_size, self.maintained_prototype_token_size)

        # Patch Embedding 
        self.d_model = 16 # patch模型的隐藏层维度
        self.patch_embedding = _PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)


    def tokenize_dataset(self,dataset):
        
        pass

    def __natural_language_embedding(self,prompt):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # prompt tokenize
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))  #将 prompt Embedding为高维向量

        return prompt_embeddings
    
    def __time_series_embedding(self,ts):
        prototype_embeddings = self.mapping_layer(self.llm_embeddings_weight.permute(1, 0)).permute(1, 0) 
        


        B, T, N = ts.size()  #B:batch size, T time steps，N feature dimension
        x_enc = x_enc.permute(0, 2, 1).contiguous() #(B,N,T)
        enc_out, n_vars = self.patch_embedding(ts.to(torch.float32))