import pickle
from typing import List
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
cwd = Path.cwd()  # Current working directory
_word_embedding_model_wiki = cwd / 'Files_embedding' / 'word_embedding_model_wiki.pickle'
_document_vectors_wiki = cwd / 'Files_embedding' / 'document_vectors_wiki.pickle'
_document_id_mapping_vector_wiki = cwd / 'Files_embedding' / 'document_id_mapping_vector_wiki.pickle'

class WordEmbeddingEngineWiki:
    def __init__(self, vector_size, sg, workers, epochs, text_processor, text_tokenizer):
        self.vector_size = vector_size
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.text_processor = text_processor
        self.text_tokenizer = text_tokenizer
        self.word_embedding_model = None
        self.documents_vectors = None
        self.document_id_mapping = {}

    def init_sentences(self, documents):
        sentences = []
        for doc_id, document in documents.items():
            sentences.append(self.text_tokenizer(self.text_processor.preprocess(document)))
            self.document_id_mapping[doc_id] = document
        return sentences

    def train_model(self, documents):
        sentences = self.init_sentences(documents)
        model = Word2Vec(sentences,
                         vector_size=self.vector_size,
                         sg=self.sg,
                         workers=self.workers,
                         epochs=self.epochs)

        self.word_embedding_model = model
        self.documents_vectors = self.vectorize_documents(sentences)
        self.save_model()

    def vectorize_documents(self, sentences):
        documents_vectors = []
        for sentence in sentences:
            zero_vector = np.zeros(self.vector_size)
            vectors = []
            for token in sentence:
                if token in self.word_embedding_model.wv:
                    try:
                        vectors.append(self.word_embedding_model.wv[token])
                    except KeyError:
                        vectors.append(np.random(self.vector_size))
            if vectors:
                vectors = np.asarray(vectors)
                avg_vec = vectors.mean(axis=0)
                documents_vectors.append(avg_vec)
            else:
                documents_vectors.append(zero_vector)
        return documents_vectors

    def save_model(self):
        with open(_word_embedding_model_wiki, 'wb') as f_model:
            pickle.dump(self.word_embedding_model, f_model)
        with open(_document_vectors_wiki, 'wb') as f_vectors:
            pickle.dump(self.documents_vectors, f_vectors)
        with open(_document_id_mapping_vector_wiki, 'wb') as f_mapping:
            pickle.dump(self.document_id_mapping, f_mapping)

    def load_model(self):
        with open(_word_embedding_model_wiki, 'rb') as f_model:
            self.word_embedding_model = pickle.load(f_model)
        with open(_document_vectors_wiki, 'rb') as f_vectors:
            self.documents_vectors = pickle.load(f_vectors)
        with open(_document_id_mapping_vector_wiki, 'rb') as f_mapping:
            self.document_id_mapping = pickle.load(f_mapping)

    def get_query_vector(self, query_text):
        preprocessed_query = self.text_processor.preprocess(query_text)
        tokens = self.text_tokenizer(preprocessed_query)
        vectors = [self.word_embedding_model.wv[token] for token in tokens if token in self.word_embedding_model.wv]
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            return avg_vec
        else:
            return np.zeros(self.vector_size)

    def get_results(self, query_text):
        query_vector = self.get_query_vector(query_text)
        similarities = cosine_similarity([query_vector], self.documents_vectors).flatten()
        ranked_indices = np.argsort(-similarities)
        result_ids = []
        for idx in ranked_indices[:10]:  # Top 10 results
            if similarities[idx] >= 0.35:
                result_ids.append(list(self.document_id_mapping.keys())[idx])
        unordered_results = [{'_id': doc_id, 'text': self.document_id_mapping[doc_id]} for doc_id in result_ids]
        return unordered_results