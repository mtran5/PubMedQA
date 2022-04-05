from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os
import pandas as pd
import torch
from utils import preprocess_function

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

bio = pd.read_pickle("BioASQ_train.pkl")
bio = Dataset.from_pandas(bio)

config = AutoConfig.from_pretrained("squad_finetuned")
tokenizer = AutoTokenizer.from_pretrained("squad_finetuned")
model = AutoModelForQuestionAnswering.from_pretrained("squad_finetuned").to(device)

tokenized_bio = bio.map(preprocess_function, batched=True)

data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir = "./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bio,
    eval_dataset=tokenized_bio,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("squad_BioASQ_finetuned")
