from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex #
import nltk
from nltk.corpus import stopwords
import re
import os

# --- Configuration & Initialization ---
PROJECT_ID = 'ir-project-2026' #
BUCKET_NAME = 'ir-project-2026-databucket' 

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))

# Global Index Loading
try:
    # This expects title_index.pkl and title_index.bin to be in your bucket
    title_index = InvertedIndex.read_index('.', 'title_index', BUCKET_NAME)
except Exception as e:
    print(f"Index loading failed: {e}")
    title_index = None

# Staff-provided Tokenizer (Assignment 3)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """Standard tokenizer: lowercase, no stemming, remove stopwords"""
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [token for token in tokens if token not in english_stopwords]

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION (Placeholder for full search)
    res = backend.serch_body(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION (TF-IDF implementation)
    res = backend.search_body(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    
    # BEGIN SOLUTION
    tokens = tokenize(query)
    distinct_tokens = set(tokens)
    scores = {} 
    
    for token in distinct_tokens:
        if title_index and token in title_index.df:
            # Read posting list from GCP Bucket
            posting_list = title_index.read_a_posting_list('.', token, BUCKET_NAME)
            for doc_id, _ in posting_list:
                scores[doc_id] = scores.get(doc_id, 0) + 1
                
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Return as list of (doc_id, title)
    res = [(str(doc_id), "Wiki Title") for doc_id, score in sorted_res]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = backend.search_anchor(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = backend.get_pagerank(wiki_ids)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = backend.get_pageview(wiki_ids)
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # Running on 0.0.0.0:8080 as per firewall rules
    app.run(host='0.0.0.0', port=8080, debug=True)
