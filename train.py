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
from peft import LoraConfig, get_peft_model,TaskType
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0. set model info
model_name = "/home/wuduo/xuanyu/llmcc/llama-7b"
# model_name = "distilbert/distilbert-base-uncased"

# 1. load dataset
## 1.1 load from exp_pool
exp_pool = pickle.load(open('/home/wuduo/xuanyu/llmcc/exp_pool_sage_classification.pkl', 'rb'))
features = [feature.tolist() for feature in exp_pool.features]
features_as_strings = [' '.join(map(str, feature)) for feature in features]
id2label = {0: "vegas", 1: "htcp",2:"westwood",3:"bbr",4:"cubic",5:"reno",6:"bic"}
label2id = {"vegas": 0, "htcp": 1,"westwood":2,"bbr":3,"cubic":4,"reno":5,"bic":6}
labels_as_ids = [label2id[label] for label in  exp_pool.labels]
result_dict = {
    "features":features_as_strings,
    "label": labels_as_ids,
}
dataset= Dataset.from_dict(result_dict)
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

# 3. train
## 3.1 load model 
model = LlamaForSequenceClassification.from_pretrained(model_name,num_labels=7, id2label=id2label, label2id=label2id)
# Adjust model embeddings if tokenizer size changed
if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))


## 3.1.1 lora config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    inference_mode=False,
    task_type=TaskType.SEQ_CLS,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.to(device)

## 3.2 set training arguments
training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
print('set training arguments done')

## 3.3 set trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
) 
print('set trainer class done')

# 4. start training
trainer.train()

model.save_pretrained("output_dir")
# 5. predict
predictions = trainer.predict(dataset["test"])
preds = np.argmax(predictions.predictions, axis=1)

accuracy_score = compute_metrics((predictions.predictions, dataset["test"]["label"]))
print("Test Accuracy:", accuracy_score)

print("Detailed Test Results:")
print(predictions.metrics)