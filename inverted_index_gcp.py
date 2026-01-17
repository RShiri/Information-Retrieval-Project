import pickle
import os
from pathlib import Path
from collections import defaultdict
from contextlib import closing

TUPLE_SIZE = 6
TF_MASK = 2**16 - 1

class InvertedIndex:
    def __init__(self, docs={}):
        self.df = Counter()
        self.term_total = Counter()
        self.posting_locs = defaultdict(list)

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    def read_a_posting_list(self, base_dir, w):
        posting_list = []
        if w not in self.posting_locs:
            return posting_list
        
        locs = self.posting_locs[w]
        for file_name, offset in locs:
            with open(Path(base_dir) / file_name, 'rb') as f:
                f.seek(offset)
                n_bytes = self.df[w] * TUPLE_SIZE
                b = f.read(n_bytes)
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
        return posting_list