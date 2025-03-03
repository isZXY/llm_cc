import torch
from torch import nn, Tensor
from math import sqrt


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
        # self.llm_embeddings_weight = self.llm_model.get_input_embeddings().weight
        # self.vocab_size = self.llm_embeddings_weight.shape[0]  # llm vocab size
        # self.maintained_prototype_token_size = 1000
        # self.vocab_mapping_to_prototype_layer = nn.Linear(
        #     self.vocab_size, self.maintained_prototype_token_size).to(self.device)

        # # Patch Embedding
        # self.d_model = 16  # patch模型的隐藏层维度
        # self.patch_len = 2  # patch 长度
        # self.stride = 1  # 步幅
        # self.dropout_rate = .1  # dropout rate
        # self.patch_embedding = _PatchEmbedding(
        #     self.d_model, self.patch_len, self.stride, self.dropout_rate, self.device)

        # # Normalize layer
        # self.feature_dimension = 5
        # self.normalize_layers = _Normalize(
        #     self.feature_dimension, affine=False)

        # Reprogramming layer
        self.n_heads = 5
        self.d_ff = 32  # dimension of fcn
        self.d_llm = 4096  # dimension of llm model,LLama7b:4096; GPT2-small:768; BERT-base:768
        # self.reprogramming_layer = _ReprogrammingLayer(
        #     self.device, self.d_model, self.n_heads, self.d_ff, self.d_llm)

        plm_embed_size = 4096
        conv_size = 4
        state_feature_dim = 256
        self.embed_state1 = nn.Linear(5, plm_embed_size).to(device)
        self.embed_state2 = nn.Linear(5, plm_embed_size).to(device)    
        self.embed_state3 = nn.Linear(5, plm_embed_size).to(device)    
        self.embed_state4 = nn.Linear(5, plm_embed_size).to(device)    
        self.embed_state5 = nn.Linear(5, plm_embed_size).to(device)
        # self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    

    def forward(self, state_ts, time_embeddings):
        dataset_concat_embedding = self.state_embedding(state_ts,time_embeddings)

        return dataset_concat_embedding

    def state_embedding(self,state_ts,time_embeddings):
        # state: input(1,8,5,5) (batch_size,episode,features,decision_interval)-> (1,8,4096) (1, seq_len, embed_size)
        splits = torch.split(state_ts, 1, dim=2) # 5个(1,8,1,5)
        squeezed_splits = [split.squeeze(dim=2) for split in splits] # 五个　（1，8，5）

        states_embedding_list = []


        state1 = self.embed_state1(squeezed_splits[0]).to(self.device) + time_embeddings
        state2 = self.embed_state2(squeezed_splits[1]).to(self.device) + time_embeddings
        state3 = self.embed_state3(squeezed_splits[2]).to(self.device) + time_embeddings
        state4 = self.embed_state4(squeezed_splits[3]).to(self.device) + time_embeddings
        state5 = self.embed_state5(squeezed_splits[4]).to(self.device) + time_embeddings


        states_embedding_list.append(state1)
        states_embedding_list.append(state2)
        states_embedding_list.append(state3)
        states_embedding_list.append(state4)
        states_embedding_list.append(state5)

        return states_embedding_list
    
        # ts_embeddings = []
        # for i, state in enumerate(squeezed_splits):
        #     ts_embedding = self.__time_series_embedding(state)
        #     ## ** use for test**
        #     # state_temp = state[:,-1,:].unsqueeze(dim=1)
        #     # ts_embedding = self.__time_series_embedding(state_temp)
        #     ## ** use for test**
        #     ts_embeddings.append(ts_embedding)

        # # stacked_state = torch.cat(ts_embeddings, dim=1)

        # dataset_concat_embedding = ts_embeddings
        # # extract dataset to single data
        # if prompt is not None:
        #     prompt_embeddings = self.__natural_language_embedding(prompt)

        #     dataset_concat_embedding = torch.cat(
        #         [prompt_embeddings, ts_embeddings], dim=1)

        # return dataset_concat_embedding



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