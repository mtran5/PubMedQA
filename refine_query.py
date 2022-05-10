# Refine queries
import spacy
class Query():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    def create_query(self, query):
        query = self.nlp(query)

        # Remove stop words
        tokens = filter(lambda x: x.is_stop, query)
        # Select only several types of POS
        tokens = filter(lambda x: x.pos_ in {"PROPN", "NUM", "VERB", "NOUN", "ADJ"}, query)
        # TODO: Add query expansion

        query = map(lambda x: x.text, tokens)
        return " ".join(query)

Q = Query()
a = Q.create_query("What is the link between bacteria and ulcer?")
b = Q.create_query("Does playing football cause CTE?")

print(a)
print(b)