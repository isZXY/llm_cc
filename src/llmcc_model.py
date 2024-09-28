import torch.nn as nn
from dataset_input_embedding import DatasetEmbedding
from pretrained_model import PretrainedLanguageModel

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
        self.tokenizer, self.llm_model = PretrainedLanguageModel(model_name='llama',model_path=plm_path,device=device)
        self.device = device
        for param in self.llm_model.parameters():
            param.requires_grad = False # 冻结大模型的参数
        self.input_embedding_layer = DatasetEmbedding(self.tokenizer, self.llm_model,self.device)
        self.output_projection = _CategoryProjection(self.llm_model.config.hidden_size,num_classes)

    def forward(self, batch_prompt, batch_ts):
        input_embedding = self.input_embedding_layer(batch_prompt,batch_ts)
        llm_last_hidden_state = self.llm_model(input_embedding).last_hidden_state # (batch size, sequence length, hidden size)
        output_category = self.output_projection(llm_last_hidden_state)
        return output_category
