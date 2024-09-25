import torch
from torch import nn
import numpy as np
import pickle
import os
from exp_pool import DatasetPool
from torch.utils.tensorboard import SummaryWriter

import evaluate
from datasets import Dataset
from transformers import AutoModelForSequenceClassification,LlamaForSequenceClassification,LlamaForCausalLM
from transformers import AutoTokenizer,LlamaTokenizer
from transformers import Trainer,TrainingArguments
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType



# 0. set device, model path,init Tensorboard
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda")
model_name = "./llama-7b"
boardwriter = SummaryWriter(log_dir='logs')
# model_name = "distilbert/distilbert-base-uncased"

# 1. load dataset
## 1.1 load from exp_pool, transfer label index, split data(train,validation,test)
exp_pool = pickle.load(open('./datasets/dataset_pool.pkl', 'rb'))
id2label = {0: "vegas", 1: "htcp",2:"westwood",3:"bbr",4:"cubic",5:"reno",6:"bic"}
label2id = {"vegas": 0, "htcp": 1,"westwood":2,"bbr":3,"cubic":4,"reno":5,"bic":6}
labels_as_ids = [label2id[label] for label in exp_pool.labels]
result_dict = {
    "features":exp_pool.prompts,
    "label": labels_as_ids,
}
dataset= Dataset.from_dict(result_dict)
train_val_test_split = dataset.train_test_split(test_size=0.4, seed=42)  # 60% train, 40% (val+test)
valid_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42)  # 50% val, 50% test
dataset_train = train_val_test_split['train']
dataset_val = valid_test_split['train']
dataset_test = valid_test_split['test']


## 1.2 tokenize dataset , add padding token, collate padding
tokenizer = LlamaTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 定义 tokenize 函数
def tokenize_dataset(dataset):
    return tokenizer(dataset["features"], padding="max_length", truncation=True, max_length=1024)

# dataset = dataset.map(tokenize_dataset, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# print('load dataset&tokenize done')
# 对每个数据集进行 tokenization
train_dataset = dataset_train.map(tokenize_dataset, batched=True)
val_dataset = dataset_val.map(tokenize_dataset, batched=True)
test_dataset = dataset_test.map(tokenize_dataset, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# train_dataset = dataset['train']
# vald_dataset = dataset['test']
print('load dataset&tokenize done')

# 2. set training Arug: load evaluate func, def loss func
## 2.1 evaluate func
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
print('load evaluate func done')

## 2.2 loss func
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        boardwriter.add_scalar('Loss/train', loss.item(), self.state.global_step)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             print("{}, gradient: {}".format(name, param.grad.mean()))
        #         else:
        #             print("{} has not gradient".format(name))

        return loss.detach()

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")
    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     boardwriter.add_scalar('Loss/train', loss.item(), self.state.global_step)
    #     return (loss, outputs) if return_outputs else loss 

# 3. model loading
## 3.1 load model 
model = LlamaForSequenceClassification.from_pretrained(model_name,num_labels=7, id2label=id2label, label2id=label2id)
# Adjust model embeddings if tokenizer size changed
if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))


## 3.2 lora config
lora_config = LoraConfig(
    r=8, # rank
    lora_alpha=32, # 控制低秩适应矩阵的影响力
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    inference_mode=False,
    task_type=TaskType.SEQ_CLS,
)

## 3.3 Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.to(device)

# 4. train
## 4.1 set training arguments
training_args = TrainingArguments(
    output_dir="./train_checkpoints/",
    save_steps=5000, 
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=30,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    
)
print('set training arguments done')

## 4.2 set trainer class
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
) 
print('set trainer class done')

## 4.3 start training
trainer.train(resume_from_checkpoint=False)


## 4.4 save trained models
model.save_pretrained("output_dir")


# 5. test - predict
predictions = trainer.predict(dataset["test"])
preds = np.argmax(predictions.predictions, axis=1)

accuracy_score = compute_metrics((predictions.predictions, dataset_test["label"]))
print("Test Accuracy:", accuracy_score)

print("Detailed Test Results:")
print(predictions.metrics)