from pymed import PubMed
import gensim
import spacy
import pickle

file = open("query_text.pickle",'rb')
corpus = pickle.load(file)
file.close()

# Only select articles with abstracts
corpus = filter(lambda x: x.abstract is not None, corpus)

abstracts = [article.abstract for article in corpus]
"""
Passage retrieval from corpus
"""
def PassageRetrieval(query, passages):
    # preprocess text
    nlp = spacy.load("en_core_web_sm")
    for passage in passages:
        passage = nlp(passage)
        # Remove stop words
        passage = filter(lambda x: not x.is_stop, passage)
        # Lemmatize texts
        passage = [token.lemma_ for token in passage]
        print(passage)

PassageRetrieval(query="", passages=abstracts)