import torch
from torch import nn, Tensor
from math import sqrt


class _ReprogrammingLayer(nn.Module):
    def __init__(self, device, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        # self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        # patch模型的隐藏层维度 16; n_heads num of heads 8; d_ff前馈神经网络的维度 32 # d_llm LLM model dimension 7b 4096
        super(_ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads).to(device)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads).to(device)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads).to(device)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm).to(device)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target_embedding <-> enc_out((B * N,num_patches P, d_model dm)),source_embedding <-> source_embeddings,value_embedding <-> source_embeddings)
        B, L, _ = target_embedding.shape  # (B * N,num_patches P, d_model dm)
        S, _ = source_embedding.shape
        H = self.n_heads
        
        target_embedding = self.query_projection(
            target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(
            target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum(
            "blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum(
            "bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(_TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False).to(device)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x  # (B * N, P, d_model)


class _ReplicationPad1d(nn.Module):
    '''
     ReplicationPad1d 层的作用是对输入张量进行一种特殊的“填充”操作，即复制输入张量的最后一列（在时间维度上）若干次，并将这些复制的列拼接到原始输入的末尾
    '''

    def __init__(self, padding) -> None:
        super(_ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        # input (batch, 特征维度，时序长度)(B,N,T)
        # (B,N) ->(B,N,1) ->(B, N, stride)
        replicate_padding = input[:, :, -
                                  1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding],
                           dim=-1)  # (B,N,T+stride)
        return output


class _PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, device):  # patch模型的隐藏层维度,patch 长度 Lp, 步幅
        super(_PatchEmbedding, self).__init__()

        # Patching
        self.patch_len = patch_len  # patch 长度
        self.stride = stride  # S,sliding stride
        self.padding_patch_layer = _ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = _TokenEmbedding(
            patch_len, d_model, device)  # (Lp--linear--> dm)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # x:(batch, 特征维度，时序长度)(B,N,T)
        n_vars = x.shape[1]  # 特征维度N
        x = self.padding_patch_layer(x)  # (B,N,T) -> (B,N,T+stride)
        # padding展开，长度是patch_len,步长是stride # (B, N, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (B*N, num_patches P, patch_len Lp)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        # 用的是卷积而非线性层 -> Embedding(B * N, P, d_model dm)
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class _Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(_Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class StateEmbedding(nn.Module):
    '''
    Tokenize dataset for use.
    '''

    def __init__(self, tokenizer, llm_model, device):

        super(StateEmbedding, self).__init__()

        # set training device
        self.device = device

        # load tokenizer & plm
        self.tokenizer = tokenizer
        self.llm_model = llm_model

        # create mapping from llm vocab to maintained prototype.
        # (vocab_size, embedding_dim)
        self.llm_embeddings_weight = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.llm_embeddings_weight.shape[0]  # llm vocab size
        self.maintained_prototype_token_size = 1000
        self.vocab_mapping_to_prototype_layer = nn.Linear(
            self.vocab_size, self.maintained_prototype_token_size).to(self.device)

        # Patch Embedding
        self.d_model = 16  # patch模型的隐藏层维度
        self.patch_len = 2  # patch 长度
        self.stride = 1  # 步幅
        self.dropout_rate = .1  # dropout rate
        self.patch_embedding = _PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.dropout_rate, self.device)

        # Normalize layer
        self.feature_dimension = 5
        self.normalize_layers = _Normalize(
            self.feature_dimension, affine=False)

        # Reprogramming layer
        self.n_heads = 5
        self.d_ff = 32  # dimension of fcn
        self.d_llm = 4096  # dimension of llm model,LLama7b:4096; GPT2-small:768; BERT-base:768
        self.reprogramming_layer = _ReprogrammingLayer(
            self.device, self.d_model, self.n_heads, self.d_ff, self.d_llm)

    def forward(self, state_ts, prompt=None):
        if prompt== None:
            dataset_concat_embedding = self.state_embedding(state_ts)
        else:
            dataset_concat_embedding = self.state_embedding(state_ts,prompt)
        return dataset_concat_embedding

    def state_embedding(self,state_ts,prompt =None):
        # state: input(1,8,4,5) (batch_size,episode,features,decision_interval)-> (1,8,4096) (1, seq_len, embed_size)
        splits = torch.split(state_ts, 1, dim=2)
        squeezed_splits = [split.squeeze(dim=2) for split in splits]
        ts_embeddings = []
        for i, state in enumerate(squeezed_splits):
            ts_embedding = self.__time_series_embedding(state)
            ts_embeddings.append(ts_embedding)

        # stacked_state = torch.cat(ts_embeddings, dim=1)

        dataset_concat_embedding = ts_embeddings
        # extract dataset to single data
        if prompt is not None:
            prompt_embeddings = self.__natural_language_embedding(prompt)

            dataset_concat_embedding = torch.cat(
                [prompt_embeddings, ts_embeddings], dim=1)

        return dataset_concat_embedding

    def __natural_language_embedding(self, prompt):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                    truncation=True, max_length=2048).input_ids  # prompt tokenize
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt_ids.to(self.device))  # 将 prompt Embedding为高维向量

        return prompt_embeddings

    def __time_series_embedding(self, ts):
        # -- Input Embedding:
        # 1-> Normalize
        # 2-> divide ts into patches,length = Lp, num = P, Stride = S
        # 3-> get Xp.shape = (P * Lp) ,Embedding as Xp_Embedding.shape = (P * dm) Use Linear Layer
        ts = self.normalize_layers(ts, 'norm')  # Normalize
        B, T, N = ts.size()  # B:batch size, T time steps，N feature dimension
        ts = ts.permute(0, 2, 1).contiguous()  # (B, N, T)
        # (B,N,T) -> (B * N, num_patches P, d_model dm)
        embedded_ts_patch, n_vars = self.patch_embedding(ts.to(torch.float32))

        # -- Patch Reprogramming:
        prototype_embeddings = self.vocab_mapping_to_prototype_layer(
            self.llm_embeddings_weight.permute(1, 0)).permute(1, 0)
        reprogramming_enc_out = self.reprogramming_layer(
            embedded_ts_patch, prototype_embeddings, prototype_embeddings)

        bs = ts.shape[0]
        llm_embed_size = reprogramming_enc_out.shape[-1]
        reprogramming_enc_out = reprogramming_enc_out.reshape(
            bs, -1, llm_embed_size)

        return reprogramming_enc_out


class ActionEmbedding(nn.Module):
    def __init__(self,plm_embed_size,device):
        super(ActionEmbedding, self).__init__()

        self.embed_action = nn.Linear(1, plm_embed_size).to(device)

    def forward(self,actions):
        actions = actions.float() 
        return self.embed_action(actions)

class ReturnEmbedding(nn.Module):
    def __init__(self,plm_embed_size,device):
        super(ReturnEmbedding, self).__init__()
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)

    def forward(self,returns):
        return self.embed_return(returns)

class TimeEmbedding(nn.Module):
    def __init__(self,plm_embed_size,device,max_ep_len):
        super(TimeEmbedding, self).__init__()
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)

    def forward(self,timesteps):
        x = self.embed_timestep(timesteps).squeeze(dim=2)

        return x