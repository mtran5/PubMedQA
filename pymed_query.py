from pymed import PubMed
import pickle

class Search():
    def __init__(self):
        my_email = "" # Enter your email here
        self.pubmed = PubMed(tool="PubMed QA Tool", email=my_email)

    def search(self, query_text):
        results = self.pubmed.query(query_text, max_results=100)
        results_list = list(results)
        with open('query_text.pickle', 'wb') as f:
            pickle.dump(results_list, f)

s = Search()
s.search("link bacteria ulcer")
