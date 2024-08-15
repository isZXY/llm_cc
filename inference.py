from transformers import AutoModelForSequenceClassification,LlamaForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from torch import nn
from datasets import load_dataset
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer
from exp_pool import ExperiencePoolClassification
import pickle
import evaluate
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Optional
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model,TaskType,PeftModel
import torch
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. load dataset
## 1.1 load from exp_pool
exp_pool = pickle.load(open('./exp_pool_sage_classification.pkl', 'rb'))
features = [feature.tolist() for feature in exp_pool.features]
features_as_strings = [' '.join(map(str, feature)) for feature in features]
id2label = {0: "vegas", 1: "htcp",2:"westwood",3:"bbr",4:"cubic",5:"reno",6:"bic"}
label2id = {"vegas": 0, "htcp": 1,"westwood":2,"bbr":3,"cubic":4,"reno":5,"bic":6}
labels_as_ids = [label2id[label] for label in exp_pool.labels]
result_dict = {
    "features":features_as_strings,
    "label": labels_as_ids,
}
dataset= Dataset.from_dict(result_dict)
dataset = dataset.train_test_split(test_size=0.2, seed=42)


## 1.2 tokenize
tokenizer = LlamaTokenizer.from_pretrained('./llama-7b')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_dataset(dataset):
    return tokenizer(dataset["features"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_dataset, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print('load dataset&tokenize done')

# 2. load evaluate func
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
print('load evaluate func done')


# 1. 先加载完整的 LLaMA 模型
model = LlamaForSequenceClassification.from_pretrained("./llama-7b",num_labels=7, id2label=id2label, label2id=label2id)

if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))
# Adjust model embeddings if tokenizer size changed
model.config.pad_token_id = tokenizer.pad_token_id


lora_model = PeftModel.from_pretrained(model, "/home/wuduo/xuanyu/llmcc/output_dir")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)
print('load lora done')

# 设置推理的 Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,  # 根据你的GPU内存调整批次大小
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
) 
print('load trainer done')
# 使用模型进行预测
predictions = trainer.predict(dataset['test'])

# 预测结果
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)
print("Predicted labels:", preds)
# 将预测的标签ID翻译为标签名
predicted_labels = [id2label[pred.item()] for pred in preds]
true_labels = [id2label[true_label] for true_label in dataset['test']['label']]
df = pd.DataFrame({
    'True Label': true_labels,
    'Predicted Label': predicted_labels
})

# 保存为CSV文件
df.to_csv("predictions_vs_true_labels_output_dir.csv", index=False)
# 如果你需要计算准确率或其他指标
# 你可以使用之前定义的 compute_metrics 函数
metrics = compute_metrics((predictions.predictions, dataset['test']['label']))
print("Metrics:", metrics)
