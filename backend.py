import pickle
import pandas as pd
import numpy as np
from collections import Counter
from inverted_index_gcp import InvertedIndex
from similarity_functions import BM25_score, tokenize

class BackendClass:
    def __init__(self):
        print("--- LOADING BACKEND ---")
        self.index = InvertedIndex.read_index('/home/puzik7399/postings_gcp', 'index')
        
        self.titles = {}
        try:
            print("Loading Titles...")
            with open('/home/puzik7399/id_title/even_id_title_dict.pkl', 'rb') as f:
                self.titles.update(pickle.load(f))
            with open('/home/puzik7399/id_title/uneven_id_title_dict.pkl', 'rb') as f:
                self.titles.update(pickle.load(f))
        except FileNotFoundError:
            print("WARNING: Title dictionaries not found!")

        self.pagerank = {}
        try:
            print("Loading PageRank...")
            with open('/home/puzik7399/pagerank.pkl', 'rb') as f:
                self.pagerank = pickle.load(f)
        except:
            print("WARNING: PageRank file not found.")

        print("--- BACKEND READY ---")

    def get_title(self, doc_id):
        return self.titles.get(doc_id, str(doc_id))

    def search(self, query):
        tokens = tokenize(query)
        if not tokens: return []

        bm25_scores = BM25_score(tokens, self.index)
        
        final_scores = {}
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        
        for doc_id, bm_val in bm25_scores.items():
            norm_bm25 = bm_val / max_bm25
            
            title_text = self.titles.get(doc_id, "").lower()
            title_hits = sum(1 for token in tokens if token in title_text)
            title_score = (title_hits / len(tokens)) if tokens else 0
            
            pr_val = self.pagerank.get(doc_id, 0)
            norm_pr = pr_val * 100000 

            # Formula: 50% BM25, 40% Title, 10% PageRank
            final_scores[doc_id] = (norm_bm25 * 0.5) + (title_score * 0.4) + (norm_pr * 0.1)

        sorted_res = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        return [(str(doc_id), self.get_title(doc_id)) for doc_id, score in sorted_res]

    def search_body(self, query): return self.search(query)
    def search_title(self, query): return self.search(query)
    def get_pagerank(self, ids): return [self.pagerank.get(int(i),0) for i in ids]
    def get_pageview(self, ids): return [0] * len(ids)