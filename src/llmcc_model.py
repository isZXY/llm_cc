import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel
from peft import get_peft_model, LoraConfig
from token_config import TokenConfig

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
        self.tokenizer, self.llm_model = model.get_model()
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Record size attribute of the llm
        self.vocab_size = model.get_vocab_size()
        self.custom_token_size = model.get_custom_token_size()

        # set PEFT LoRA
        self.llm_model = self.__set_peft_model(rank=32)

        # set first-time input embedding layer
        self.input_embedding_layer = DatasetEmbedding(
            self.tokenizer, self.llm_model, self.device)

        # ## set output project layer
        # # self.output_projection = _CategoryProjection(
        # #     self.llm_model.config.hidden_size, num_classes,self.device)
        # self.output_projection = _ExplainableTokenGenerationHead(self.llm_model.config.hidden_size,self.vocab_size,self.custom_token_size,self.tokenizer.eos_token_id,self.tokenizer,device,max_length=50)
        
        # last hidden state -> logits (vocabulary)
        self.output_projection = nn.Linear(self.llm_model.config.hidden_size,self.vocab_size)

        # Set modules except llm
        self.modules_except_llm = nn.ModuleList([
            self.input_embedding_layer.vocab_mapping_to_prototype_layer, self.input_embedding_layer.patch_embedding, self.input_embedding_layer.normalize_layers, self.input_embedding_layer.reprogramming_layer, self.output_projection
        ])

        self.custom_token_indices = model.get_custom_token_indices()


        # generation max length
        self.generation_max_length = 50
        
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
        self.llm_model.load_adapter(checkpoint_path, adapter_name='deafult')
        self.modules_except_llm.load_state_dict(torch.load(
            os.path.join(checkpoint_path, 'modules_except_plm.pth')))


    def first_forward(self, batch_prompt, batch_ts):
        '''
        The first input need a special embedding precedure
        ''' 

        batch_ts = batch_ts.float().to(self.device) 
        input_embedding = self.input_embedding_layer(batch_prompt, batch_ts) 

        # (batch size, sequence length, hidden size)
        llm_last_hidden_state = self.llm_model(
            inputs_embeds=input_embedding).last_hidden_state
        
        # selected_token, generated_sentence = self.output_projection(llm_last_hidden_state)
        return llm_last_hidden_state

    def forward(self, batch_prompt, batch_ts):

        # store generated sequence
        generated_sequence = []

        # finish first forward          
        last_hidden_state = self.first_forward(batch_prompt, batch_ts)

        # The first token should be a chosen algo
        logits = self.get_logits(last_hidden_state[:, -1, :])
        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits[:, self.custom_token_indices] = logits[:, self.custom_token_indices]  
        probabilities = F.softmax(masked_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1).item()
        generated_sequence.append(next_token)
        algo_token_id = next_token

        for _ in range(50):


            next_embedding = self.llm_model.get_input_embeddings()(torch.tensor(next_token).to(self.device)).unsqueeze(dim=0)

            input_tensor = torch.cat([last_hidden_state, next_embedding.unsqueeze(dim=0)], dim=1)

            last_hidden_state = self.llm_model(inputs_embeds=input_tensor).last_hidden_state
        

            # set a token in max probabilities
            logits = self.get_logits(last_hidden_state[:, -1, :])

            probabilities = F.softmax(logits, dim=-1) # (batch_size, vocab_size)
            next_token = torch.multinomial(probabilities, 1).item()
            generated_sequence.append(next_token)

            # if EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            



        return algo_token_id, generated_sequence
    

    def get_logits(self,last_hidden_state):
        logits = self.output_projection(last_hidden_state)
        return logits