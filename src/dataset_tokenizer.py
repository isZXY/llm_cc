import torch
from torch import nn


class _PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout): # patch模型的隐藏层维度,patch 长度 Lp, 步幅
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len # patch 长度
        self.stride = stride # S,sliding stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model) # (Lp--linear--> dm)

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
    def __init__(self,plm):
        # load tokenizer & plm
        self.tokenizer = plm.tokenizer()
        self.llm_model = plm.llm_model()
        # Patch Embedding Load
        self.patch_embedding = PatchEmbedding( 
        configs.d_model, self.patch_len, self.stride, configs.dropout)

    def tokenize_dataset(self,dataset):
        pass

    def __natural_language_tokenize(self,prompt):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # 将prompt tokenize
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  #将 prompt Embedding为高维向量

        return prompt_ids # :TODO prompt embedding 
    
    def __time_series_tokenize(self,ts):
        B, T, N = ts.size()  #B:batch size, T time steps，N feature dimension
        x_enc = x_enc.permute(0, 2, 1).contiguous() #(B,N,T)
        enc_out, n_vars = self.patch_embedding(ts.to(torch.float32))