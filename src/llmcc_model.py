import torch.nn as nn
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel
from peft import get_peft_model, LoraConfig, TaskType


class _CategoryProjection(nn.Module):
    def __init__(self,hidden_size,num_classes):
        super(_CategoryProjection, self).__init__()
        self.classification_head = nn.Linear(hidden_size, num_classes)  # 线性层映射到分类标签
        # self.softmax = nn.Softmax(dim=-1)  # 用于生成分类概率

    def forward(self,llm_last_hidden_state):
        cls_token_hidden_state = llm_last_hidden_state[:, -1:, :]  # (batch_size, hidden_size)
        logits = self.classification_head(cls_token_hidden_state)  # (batch_size, num_classes)
        return logits


class Model(nn.Module):
    def __init__(self,plm_path,device,num_classes):
        super(Model, self).__init__()
        self.device = device

        # Load Model
        self.tokenizer, self.llm_model = PretrainedLanguageModel(model_name='llama',model_path=plm_path,device=device)
        for param in self.llm_model.parameters():
            param.requires_grad = False
        # set PEFT LoRA
        self.llm_model = self.__set_peft_model(rank=32)
        
        self.input_embedding_layer = DatasetEmbedding(self.tokenizer, self.llm_model,self.device)
        self.output_projection = _CategoryProjection(self.llm_model.config.hidden_size,num_classes)

    def __set_peft_model(self,rank):
        # 定义 LoRA 的配置
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLASSIFICATION 
            r=16,  # rank 参数，决定矩阵分解的秩
            lora_alpha=32,  # lora 的 scaling 参数
            lora_dropout=0.05  # dropout 概率
        )

        peft_model = get_peft_model(self.llm_model, peft_config)
        return peft_model


    def forward(self, batch_prompt, batch_ts):
        batch_ts.float().to(self.device)
        input_embedding = self.input_embedding_layer(batch_prompt,batch_ts)
        llm_last_hidden_state = self.llm_model(input_embedding).last_hidden_state # (batch size, sequence length, hidden size)
        output_category = self.output_projection(llm_last_hidden_state)
        return output_category
