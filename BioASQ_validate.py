from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, QuestionAnsweringPipeline
import os
import torch
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

bio = pd.read_pickle("BioASQ_test.pkl")
bio = bio.reset_index()
# print(bio.head(10))

config = AutoConfig.from_pretrained("squad_BioASQ_finetuned")
tokenizer = AutoTokenizer.from_pretrained("squad_BioASQ_finetuned")
model = AutoModelForQuestionAnswering.from_pretrained("squad_BioASQ_finetuned").to(device)

# Create question answering pipeline
nlp = QuestionAnsweringPipeline(model=model, framework="pt", num_workers=2, batch_size=16, tokenizer=tokenizer, device=device)

# Get validation metrics
from evaluate import load
squad_metric = load("squad")

contexts = bio["context"].tolist()
questions = bio["question"].tolist()
bio_answers = bio["answers"].tolist()

# Reformat the answers to meet SQUAD format
print("Answering questions...")
predictions = nlp(context=contexts, question=questions)

references = []
predictions_formatted = []


for i in range(len(bio)):
	id = bio["id"][i]
	prediction_text = predictions[i]["answer"]
	predictions_formatted.append({
		"id": id,
		"prediction_text": prediction_text,
	})
	
	references.append({
		"id": id,
		"answers": bio_answers[i]
	})
results = squad_metric.compute(predictions=predictions_formatted, references=references)
print(results)
