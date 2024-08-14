from transformers import AutoModelForSequenceClassification,LlamaForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from peft import PeftModel
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
from peft import LoraConfig, get_peft_model,TaskType
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0. set model info
model_name = "./llama-7b"

# 1. load dataset
## 1.1 load from exp_pool
exp_pool = pickle.load(open('./exp_pool_sage_classification.pkl', 'rb'))
features = [feature.tolist() for feature in exp_pool.features]
features_as_strings = [' '.join(map(str, feature)) for feature in features]
id2label = {0: "vegas", 1: "htcp",2:"westwood",3:"bbr",4:"cubic",5:"reno",6:"bic"}
label2id = {"vegas": 0, "htcp": 1,"westwood":2,"bbr":3,"cubic":4,"reno":5,"bic":6}
labels_as_ids = [label2id[label] for label in  exp_pool.labels]
result_dict = {
    "features":features_as_strings,
    "label": labels_as_ids,
}
dataset = Dataset.from_dict(result_dict)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

## 1.2 tokenize
tokenizer = LlamaTokenizer.from_pretrained(model_name)
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

# 3. Load model from checkpoint for inference
model = LlamaForSequenceClassification.from_pretrained("llama-7b", num_labels=7, id2label=id2label, label2id=label2id)
# Adjust model embeddings if tokenizer size changed
if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))

# Load the LoRA weights
model = PeftModel.from_pretrained(model, "path/to/save/folder/checkpoint-32210")
model.to(device)

# 3. Predict
test_dataset = dataset["test"]
inputs = tokenizer(test_dataset["features"], return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    print(predictions)

# 4. Compute accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_dataset["label"], predictions)
print("Test Accuracy:", accuracy)