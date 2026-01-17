import math
from collections import Counter

# הגדרות ברירת מחדל ל-BM25
k1 = 1.5
b = 0.75

def BM25_score(query_tokens, index):
    """
    מחשב ציון BM25 לכל מסמך רלוונטי עבור השאילתה.
    """
    score_per_doc = Counter()
    
    # נתונים גלובליים מהאינדקס
    N = len(index.DL) # מספר המסמכים הכולל
    AVGDL = index.avg_dl # אורך מסמך ממוצע

    for token in query_tokens:
        if token in index.df:
            # 1. חישוב IDF
            df = index.df[token]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            # 2. מעבר על כל המסמכים שמכילים את המילה (Posting List)
            posting_list = index.read_a_posting_list("", token, index.bucket_name)
            
            for doc_id, tf in posting_list:
                doc_len = index.DL.get(doc_id, AVGDL)
                
                # 3. חישוב הציון הסופי למילה במסמך
                numerator = idf * tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / AVGDL))
                score = numerator / denominator
                
                score_per_doc[doc_id] += score
                
    return score_per_doc

def tokenize(text):
    """פונקציית טוקניזציה פשוטה אך יעילה"""
    return [token.group() for token in RE_WORD.finditer(text.lower())]

import re
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)