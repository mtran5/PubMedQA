from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, QuestionAnsweringPipeline
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

squad = load_dataset("squad")
config = AutoConfig.from_pretrained("squad_finetuned")
tokenizer = AutoTokenizer.from_pretrained("squad_finetuned")
model = AutoModelForQuestionAnswering.from_pretrained("squad_finetuned").to(device)

# Create question answering pipeline
nlp = QuestionAnsweringPipeline(model=model, framework="pt", num_workers=2, batch_size=16, tokenizer=tokenizer, device=device)

contexts = squad["validation"]["context"][:5]
questions = squad["validation"]["question"][:5]
answers = nlp(context=contexts, question=questions)

for context, question, answer in zip(contexts, questions, answers):
    #print(context)
    print(question)
    print(answer["answer"])


# Get validation metrics
from evaluate import load
squad_metric = load("squad")

selected = 1000

contexts = squad["validation"]["context"][:selected]
questions = squad["validation"]["question"][:selected]

# Reformat the answers to meet SQUAD format
print("Answering questions")
predictions = nlp(context=contexts, question=questions)
predictions_formatted = [{"id": squad["validation"]["id"][i], "prediction_text": predictions[i]["answer"]} for i in range(selected)]

references = [{"id": squad["validation"]["id"][i], "answers": squad["validation"]["answers"][i]} for i in range(selected)]

results = squad_metric.compute(predictions=predictions_formatted, references=references)
print(results)
