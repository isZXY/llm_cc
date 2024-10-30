import os
import torch
import torch.nn as nn
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel
from peft import get_peft_model, LoraConfig


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
    def __init__(self, hidden_size, vocab_size, custom_token_size,device,max_length=50):
        '''
        :para hidden_size: llm_last_hidden_state size
        vocab_size: llm vocab size
        custom_token_size : custom_token_num
        '''
        super(_ExplainableTokenGenerationHead, self).__init__()

        # Token Selector, Selector the specific and precise token head 
        self.token_selector = nn.Linear(hidden_size, custom_token_size).to(device)
        
        # Explainable NLP Head
        self.language_gen_head = nn.Linear(hidden_size, vocab_size).to(device)
        
        # limitation of max length
        self.max_length = max_length

        self.device = device


    def forward(self, llm_last_hidden_state, labels=None, explanation_labels=None):
        cls_token_hidden_state = llm_last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
        # Token Selector, Selector the specific and precise token head 
        token_logits = self.token_selector(cls_token_hidden_state)  # (batch_size, custom_token_size)
        selected_token = torch.argmax(token_logits, dim=-1)  # (batch_size,)


        # Explainable NLP Head 
        generated_tokens = []
        input_state = llm_last_hidden_state
        
        for _ in range(self.max_length):
            logits = self.language_gen_head(input_state)  # (batch_size, seq_len, vocab_size)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (batch_size, 1)
            generated_tokens.append(next_token)
            
            # update input state to autoregressive
            input_state = torch.cat([input_state, logits[:, -1:, :]], dim=1)
            
            # EOS early quit
            if torch.all(next_token == self.eos_token_id):
                break

        generated_sentence = torch.cat(generated_tokens, dim=1)  # (batch_size, sentence_length)
        
        return selected_token, generated_sentence


class Model(nn.Module):
    def __init__(self, plm_path, device, num_classes):
        super(Model, self).__init__()
        self.device = device

        # Load Model
        model = PretrainedLanguageModel(
            model_name='llama', model_path=plm_path, device=device)
        self.tokenizer, self.llm_model = model.get_model()
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # set PEFT LoRA
        self.llm_model = self.__set_peft_model(rank=32)

        self.input_embedding_layer = DatasetEmbedding(
            self.tokenizer, self.llm_model, self.device)
        # self.output_projection = _CategoryProjection(
        #     self.llm_model.config.hidden_size, num_classes,self.device)

        self.output_projection = _ExplainableTokenGenerationHead(self.llm_model.config.hidden_size,)

        self.modules_except_llm = nn.ModuleList([
            self.input_embedding_layer.vocab_mapping_to_prototype_layer, self.input_embedding_layer.patch_embedding, self.input_embedding_layer.normalize_layers, self.input_embedding_layer.reprogramming_layer, self.output_projection
        ])

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

    def forward(self, batch_prompt, batch_ts):
        batch_ts = batch_ts.float().to(self.device)
        input_embedding = self.input_embedding_layer(batch_prompt, batch_ts)
        # (batch size, sequence length, hidden size)
        llm_last_hidden_state = self.llm_model(
            inputs_embeds=input_embedding).last_hidden_state
        output_category = self.output_projection(llm_last_hidden_state)
        return output_category
