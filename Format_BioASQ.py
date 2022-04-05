# Reformat BioASQ for HuggingFace Trainer
import json
import os
import pandas as pd

# Make sure we're working in the same directory
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

result = {}

contexts = []
questions = []
answers = []
ids = []

with open('BioASQ-dev.txt', 'r') as fr:
    for line in fr:
        res = json.loads(line)
        contexts += [res["context"]]
        qas = res["qas"][0]	
        questions.append(qas["question"])
        ids.append(qas["qid"])	
		
		# starting characters for the answers
        answer_start = []
        for answer in qas["detected_answers"]:
            for char_span in answer["char_spans"]:
                answer_start.append(char_span[0])
        answer = {
            "answer_start": answer_start,
            "text": qas["answers"]	
        }
        answers.append(answer)

df = pd.DataFrame(zip(contexts, questions, answers, ids), columns=["context", "question", "answers", "id"])
print(df.answers.head(10))

# Split into train and test dataset
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=5)

print(len(train))
print(len(test))

# Reset indices after shuffling
train = train.reset_index()
test = test.reset_index()

train.to_pickle("BioASQ_train.pkl")
test.to_pickle("BioASQ_test.pkl")
