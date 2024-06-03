import pickle
from typing import List
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cwd = Path.cwd()  # Current working directory
tfidf_model2 = cwd / 'Files' / 'tfidf_model2.pickle'
tfidf_matrix2 = cwd / 'Files' / 'tfidf_matrix2.pickle'
document_id_mapping2 = cwd / 'Files' / 'document_id_mapping2.pickle'

class TfidfEngine2:
    def __init__(self, text_preprocessor):
        self.text_preprocessor = text_preprocessor
        self.tfidf_matrix = None
        self.tfidf_model = None
        self.document_id_mapping = {}

    def train_model(self, documents):
        document_texts = [doc['text'] for doc in documents]
        vectorizer = TfidfVectorizer(preprocessor=self.text_preprocessor.preprocess, tokenizer=self.text_preprocessor.tokenizer)
        tfidf_matrix = vectorizer.fit_transform(document_texts)
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_model = vectorizer
        self.save_model(documents)

    def save_model(self, documents):
        with open(tfidf_model2, 'wb') as f_model:
            pickle.dump(self.tfidf_model, f_model)
        with open(tfidf_matrix2, 'wb') as f_matrix:
            pickle.dump(self.tfidf_matrix, f_matrix)
        with open(document_id_mapping2, 'wb') as f_mapping:
            pickle.dump({doc['id']: doc['text'] for doc in documents}, f_mapping)

    def load_model(self):
        with open(tfidf_model2, 'rb') as f_model:
            self.tfidf_model = pickle.load(f_model)
        with open(tfidf_matrix2, 'rb') as f_matrix:
            self.tfidf_matrix = pickle.load(f_matrix)
        with open(document_id_mapping2, 'rb') as f_mapping:
            self.document_id_mapping = pickle.load(f_mapping)

    def query(self, query_text):
        preprocessed_query = self.text_preprocessor.preprocess(query_text)
        query_vector = self.tfidf_model.transform([preprocessed_query])
        return query_vector
    
    def rank_documents(self, query_vector):
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(-cosine_similarities)
        return ranked_indices, cosine_similarities

    def get_results(self, query_text):
        query_vector = self.query(query_text)
        ranked_indices, similarities = self.rank_documents(query_vector)
        result_ids = []
        for idx in ranked_indices[:10]:  # Top 10 results
            if similarities[idx] >= 0.35:
                result_ids.append(list(self.document_id_mapping.keys())[idx])
        unordered_results = [{'_id': doc_id, 'text': self.document_id_mapping[doc_id]} for doc_id in result_ids]
        return unordered_results