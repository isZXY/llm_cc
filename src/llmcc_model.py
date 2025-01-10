import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_input_embedding import StateEmbedding,ActionEmbedding,ReturnEmbedding,TimeEmbedding
from pretrained_model import PretrainedLanguageModel
from peft import get_peft_model, LoraConfig
from token_config import TokenConfig
import numpy as np
from collections import deque
import random

class _CategoryProjection(nn.Module):
    def __init__(self, hidden_size, num_classes,device):
        super(_CategoryProjection, self).__init__()
        self.classification_head = nn.Linear(
            hidden_size, num_classes).to(device)  # 线性层映射到分类标签
        # self.softmax = nn.Softmax(dim=-1)  # 用于生成分类概率

    def forward(self, llm_last_hidden_state):
        # (batch_size, hidden_size)
        cls_token_hidden_state = llm_last_hidden_state[:, -1:, :]
        logits = self.classification_head(
            cls_token_hidden_state)  # (batch_size, num_classes)
        return logits


class _ExplainableTokenGenerationHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, custom_token_size,eos_token_id,tokenizer,device,max_length=50):
        '''
        :para hidden_size: llm_last_hidden_state size
        vocab_size: llm vocab size
        custom_token_size : custom_token_num
        '''
        super(_ExplainableTokenGenerationHead, self).__init__()

        # Token Selector, Selector the specific and precise token head 
        self.token_selector = nn.Linear(hidden_size, custom_token_size).to(device)
        self.tokenizer = tokenizer
        # Explainable NLP Head
        self.language_gen_head = nn.Linear(hidden_size, vocab_size).to(device)
        
        # limitation of max length
        self.max_length = max_length

        self.eos_token_id = eos_token_id  # 添加这一行来初始化 eos_token_id

        self.vocab_size = vocab_size

        self.device = device


    def forward(self, llm_last_hidden_state, labels=None, explanation_labels=None):
        cls_token_hidden_state = llm_last_hidden_state[:, -1, :]  # (batch_size, hidden_size) (2,4096)
        
        # Token Selector, Selector the specific and precise token head 
        token_logits = self.token_selector(cls_token_hidden_state)  # (batch_size, custom_token_size) (2,8)
        selected_token_id = torch.argmax(token_logits, dim=-1)  # (batch_size,)


        # Explainable NLP Head 
        generated_tokens = []

        # 初始化输入状态
        input_state = llm_last_hidden_state

        for _ in range(self.max_length):
            # Step 1: 计算 logits
            logits = self.language_gen_head(input_state)  # (batch_size, seq_len, vocab_size)

            # 使用 softmax 转换 logits 为概率分布
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)  # 只取最后一个时间步的 logits
            # 选择下一个 token
            next_token = torch.multinomial(probabilities, num_samples=1)  # (batch_size, 1)
            generated_tokens.append(next_token)

            # Step 2: 更新输入状态以进行自回归生成
            next_token_one_hot = torch.zeros((next_token.size(0), 1, self.vocab_size), device=self.device)  # (batch_size, 1, vocab_size)
            next_token_one_hot.scatter_(2, next_token.unsqueeze(2), 1)  # 将选择的 token 转换为 one-hot 编码

            # 这里我们需要直接使用 next_token 更新 input_state
            input_state = torch.cat([input_state, next_token_one_hot], dim=1)  # (batch_size, seq_len + 1, hidden_size)

            # EOS early quit, 假设 eos_token_id 是你的结束符 token ID
            if torch.all(next_token == self.eos_token_id):
                break


        # Step 3: 将生成的 tokens 转换为文本
        generated_sentence = torch.cat(generated_tokens, dim=1)  # (batch_size, sentence_length)
        generated_sentence_text = [self.tokenizer.decode(token_id.item()) for token_id in generated_sentence.flatten()]

        # 打印生成的句子
        print("Generated sentence:", ' '.join(generated_sentence_text))
                
        return selected_token_id, generated_sentence


class Model(nn.Module):
    def __init__(self, plm_path, device):
        super(Model, self).__init__()
        self.device = device

        # Load Model
        model = PretrainedLanguageModel(
            model_name='llama', model_path=plm_path, device=device)
        self.plm_embed_size = 4096
        self.tokenizer, self.llm_model = model.get_model()
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Record size attribute of the llm
        self.vocab_size = model.get_vocab_size()
        self.custom_token_size = model.get_custom_token_size()

        # set PEFT LoRA
        self.llm_model = self.__set_peft_model(rank=32)

        # set input embedding layer
        self.action_embedding = ActionEmbedding(self.plm_embed_size,self.device)
        self.return_embedding = ReturnEmbedding(self.plm_embed_size,self.device)
        self.time_embedding = TimeEmbedding(self.plm_embed_size,self.device,8)

        self.state_embedding_layer = StateEmbedding(
            self.tokenizer, self.llm_model, self.device)
        
        self.embed_ln = nn.LayerNorm(self.plm_embed_size).to(device)
        self.action_head = nn.Linear(self.plm_embed_size,7).to(device)


        # ## set output project layer
        # # self.output_projection = _CategoryProjection(
        # #     self.llm_model.config.hidden_size, num_classes,self.device)
        # self.output_projection = _ExplainableTokenGenerationHead(self.llm_model.config.hidden_size,self.vocab_size,self.custom_token_size,self.tokenizer.eos_token_id,self.tokenizer,device,max_length=50)
        
        # last hidden state -> logits (vocabulary)
        self.output_projection = nn.Linear(self.llm_model.config.hidden_size,self.vocab_size).to(device)

        # Set modules except llm:TODO
        self.modules_except_llm = nn.ModuleList([
            self.state_embedding_layer.vocab_mapping_to_prototype_layer, self.state_embedding_layer.patch_embedding, self.state_embedding_layer.normalize_layers, self.state_embedding_layer.reprogramming_layer, self.output_projection, self.action_head, self.embed_ln,self.action_embedding.embed_action, self.return_embedding.embed_return,self.time_embedding.embed_timestep
        ])

        self.custom_token_indices = model.get_custom_token_indices()

        # generation max length
        self.generation_max_length = 50

        
        # the following are used for evaluation
        self.dt_win_length = 8
        self.states_dq = deque([torch.zeros((1, 0, self.plm_embed_size), device=device)], maxlen=self.dt_win_length)
        self.returns_dq = deque([torch.zeros((1, 0, self.plm_embed_size), device=device)], maxlen=self.dt_win_length)
        self.actions_dq = deque([torch.zeros((1, 0, self.plm_embed_size), device=device)], maxlen=self.dt_win_length)
        
    def __set_peft_model(self, rank):
        # 定义 LoRA 的配置
        peft_config = LoraConfig(
            r=16,  # rank 参数，决定矩阵分解的秩
            lora_alpha=32,  # lora 的 scaling 参数
            lora_dropout=0.05  # dropout 概率
        )

        peft_model = get_peft_model(self.llm_model, peft_config)
        return peft_model

    def save_model(self, checkpoint_path):
        # save lora weights
        self.llm_model.save_pretrained(checkpoint_path)
        # save other modules except plm
        torch.save(self.modules_except_llm.state_dict(), os.path.join(
            checkpoint_path, 'modules_except_plm.pth'))

    def load_model(self, checkpoint_path):
        self.llm_model.load_adapter(checkpoint_path, adapter_name='default')
        self.modules_except_llm.load_state_dict(torch.load(
            os.path.join(checkpoint_path, 'modules_except_plm.pth')))

    def forward(self,states, actions, returns, timesteps, labels,attention_mask=None):
                # 1.1 embed action, return, timestep
                action_embeddings = self.action_embedding(actions)  # shape: (1, seq_len, embed_size) (1,8,4096)
                returns_embeddings = self.return_embedding(returns)  # shape: (1, seq_len, embed_size)(1,8,4096)
                time_embeddings = self.time_embedding(timesteps)  # shape: (1, seq_len, embed_size)(1,8,1,4096)
                

                # 1.2 time embeddings are treated similar to positional embeddings
                action_embeddings = action_embeddings + time_embeddings
                returns_embeddings = returns_embeddings + time_embeddings


                # Step 2: process states, turn them into embeddings.
                states_embedding_list = self.state_embedding_layer(states)

                
                # Step 3: stack returns, states, actions embeddings.
                # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
                # which works nice in an autoregressive sense since states predict actions
                stacked_inputs = []
                action_embed_positions = np.zeros(returns_embeddings.shape[1])  # record the positions of action embeddings
                
                for i in range(returns_embeddings.shape[1]):
                    stacked_input = torch.cat((returns_embeddings[:, i:i + 1], states_embedding_list[0][:, i:i + 5], states_embedding_list[1][:, i:i + 5], states_embedding_list[2][:, i:i + 5],states_embedding_list[3][:, i:i + 5],states_embedding_list[4][:, i:i + 5],action_embeddings[:, i:i + 1]), dim=1)
                    stacked_inputs.append(stacked_input)
                    action_embed_positions[i] = (i + 1) * (2 + 5*5)
                stacked_inputs = torch.cat(stacked_inputs, dim=1)
                stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
                stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
                
                # Step 4: feed stacked embeddings into the plm
                # 4.1 create attention mask
                if attention_mask is None:
                    # 1 if can be attended to, 0 if not
                    attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

                last_hidden_state = self.llm_model(
                    inputs_embeds=stacked_inputs_ln,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    )

                logits = last_hidden_state['last_hidden_state']
                logits_used = logits[:, action_embed_positions - 2]
                action_pred = self.action_head(logits_used)

                return action_pred
                # algo_logits = self.get_logits(last_hidden_state[:, -1, :])
                # masked_logits = torch.full_like(algo_logits, float('-inf'))
                # masked_logits[:, self.custom_token_indices] = algo_logits[:, self.custom_token_indices]
                # probabilities = F.softmax(masked_logits, dim=-1)
                # # Sample next token for each item in the batch (dim=0 for batch size)
                # next_token = torch.multinomial(probabilities, 1)[:, 0].tolist()  # shape: (batch_size,)
                # print(next_token)


                # return stacked_inputs_ln, attention_mask
    
    
                # # we feed in the input embeddings (not word indices as in NLP) to the model
                # transformer_outputs = self.plm(
                #     inputs_embeds=stacked_inputs_ln,
                #     attention_mask=attention_mask,
                #     output_hidden_states=True,
                #     stop_layer_idx=self.which_layer,
                # )
                # logits = transformer_outputs['last_hidden_state']
                # if self.residual:
                #     logits = logits + stacked_inputs_ln  # residual add

                # # Step 5: predict actions
                # # we need to locate the logits corresponding to the state embeddings
                # # simply using `action_embed_positions[i] - 2` will do.
                # logits_used = logits[:, action_embed_positions - 2]
                # action_pred = self.action_head(logits_used)

                # return action_pred
                     

    # def first_forward(self, batch_prompt, batch_ts):
    #     '''
    #     The first input need a special embedding precedure
    #     ''' 

    #     batch_ts = batch_ts.float().to(self.device) 
    #     input_embedding = self.input_embedding_layer(batch_prompt, batch_ts) 

    #     # (batch size, sequence length, hidden size)
    #     llm_last_hidden_state = self.llm_model(
    #         inputs_embeds=input_embedding).last_hidden_state
        
    #     # selected_token, generated_sentence = self.output_projection(llm_last_hidden_state)
    #     return llm_last_hidden_state

    # def forward(self, batch_prompt, batch_ts, batch_label=None, mode="inference"):
    #     """
    #     Forward pass for the model. If mode is 'train', compute the loss and return it.
    #     Otherwise, return the generated sequence.

    #     :param batch_prompt: The input prompt to the model.
    #     :param batch_ts: The time-series data (if any).
    #     :param batch_label: The ground-truth labels for loss computation (only used in train mode).
    #     :param mode: Mode of operation - 'train' or 'inference'.
    #     :return: Depending on the mode, returns either the loss or the generated sequence.
    #     """
    #     # store generated sequence
    #     generated_sequence = []
    #     total_loss = 0

    #     # finish first forward          
    #     last_hidden_state = self.first_forward(batch_prompt, batch_ts)

    #     # The first token should be a chosen algo
    #     algo_logits = self.get_logits(last_hidden_state[:, -1, :])
    #     masked_logits = torch.full_like(algo_logits, float('-inf'))
    #     masked_logits[:, self.custom_token_indices] = algo_logits[:, self.custom_token_indices]
    #     probabilities = F.softmax(masked_logits, dim=-1)
    #     # Sample next token for each item in the batch (dim=0 for batch size)
    #     next_token = torch.multinomial(probabilities, 1)[:, 0].tolist()  # shape: (batch_size,)
    #     generated_sequence.extend(next_token)
    #     algo_token_id = next_token

        

    #     for t in range(50):
    #         next_embedding = self.llm_model.get_input_embeddings()(torch.tensor(next_token).to(self.device))
    #         # next_embedding shape needs to match last_hidden_state
    #         next_embedding = next_embedding.unsqueeze(dim=1)  # shape: (batch_size, 1, hidden_size)

    #         input_tensor = torch.cat([last_hidden_state, next_embedding], dim=1)

    #         last_hidden_state = self.llm_model(inputs_embeds=input_tensor).last_hidden_state
        
    #         # set a token in max probabilities
    #         logits = self.get_logits(last_hidden_state[:, -1, :])

    #         probabilities = F.softmax(logits, dim=-1) # (batch_size, vocab_size)
    #         next_token = torch.multinomial(probabilities, 1)[:, 0].tolist()  # shape: (batch_size,)
    #         generated_sequence.extend(next_token)

    #         # if EOS
    #         if next_token == self.tokenizer.eos_token_id:
    #             break
            

    #         # Compute the loss for the current token in training mode
    #         if mode == "train" and batch_label is not None:
    #             token_label = batch_label[:, t+1]  # Get the label for the current token (shifted by 1)
    #             token_loss = F.cross_entropy(logits, token_label)  # Cross-entropy loss
    #             total_loss += token_loss

    #     # sequence_loss = self.loss_fcn(logits.view(-1, self.vocab_size), batch_label[:, 1:].view(-1))  # 后续 tokens

    #     # In 'train' mode, return the total loss
    #     if mode == "train":
    #         return total_loss
    #     else:
    #         # In inference mode, return the generated sequence
    #         return algo_token_id, generated_sequence
    

    def sample(self, state, target_return, timestep):
        """
        Sample action function, used for evaluation/testing.
        """
        # Step 1: stack previous state, action, return features in the dequeue
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_action_embeddings = self.actions_dq[i]
            prev_stacked_inputs.append(torch.cat((prev_return_embeddings, prev_state_embeddings, prev_action_embeddings), dim=1))
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)


        return_embeddings = self.return_embedding(target_return)
        time_embeddings = self.time_embedding(timestep)

        return_embeddings = return_embeddings + time_embeddings

        # Step 4: process state
        state = state.unsqueeze(dim=1)
        state = state.to(self.device)
        states_embedding_list = self.state_embedding_layer(state)

        state_embeddings = torch.cat(states_embedding_list, dim=1)
        # Step 5: stack return, stage and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)  # mind the order
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)  # mind the order
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        attention_mask=None
        # 1 if can be attended to, 0 if not
        if attention_mask is None:
        # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        last_hidden_state = self.llm_model(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = last_hidden_state['last_hidden_state']

        # Step 6: predict the algo for next interval
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        # action_pred = action_pred.reshape(-1)
        # bitrate, _ = self._sample(action_pred)

        predicted = torch.argmax(action_pred, dim=-1)
        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = predicted
        action_embeddings = self.action_embedding(action_tensor) + time_embeddings
        
        # update deques
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings) 
        self.actions_dq.append(action_embeddings)

        return action_pred

    # def _sample(self, logits):
    #     pi = F.softmax(logits, 0).cpu().numpy()
    #     idx = random.choices(np.arange(pi.size), pi)[0]
    #     lgprob = np.log(pi[idx])
    #     return idx, lgprob

    def get_logits(self,last_hidden_state):
        logits = self.output_projection(last_hidden_state)
        return logits