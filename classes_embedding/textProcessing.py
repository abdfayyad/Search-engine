import re
import string
import pickle
from typing import List
import numpy as np
from nltk import tokenize, pos_tag, download
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Callable



class LemmatizerWithPOSTagger(WordNetLemmatizer):
    def __init__(self):
        pass

    def _get_wordnet_pos(self, tag: str) -> str:
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(self, word: str, pos: str = "n") -> str:
        return super().lemmatize(word, self._get_wordnet_pos(pos))

class TextPreprocessor():

    def __init__(self, tokenizer: Callable = None) -> None:
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            self.tokenizer = tokenize.word_tokenize

        self.stopwords_tokens = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = LemmatizerWithPOSTagger()

    def tokenize(self, text: str)-> List[str]:
        tokens =self.tokenizer(text)
        return tokens
    
    def to_lower(self, tokens: List[str]) -> List[str]:
        lower_tokens = []
        for token in tokens:
            lower_token = str(np.char.lower(token))
            lower_tokens.append(lower_token)
        return lower_tokens

    
    def remove_markers(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(re.sub(r'\u00AE', '', token))
        return new_tokens

    def remove_punctuation(self, tokens: List[str]) ->  List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(token.translate(str.maketrans('', '', string.punctuation)))
        return new_tokens




    def rplace_under_score_with_space(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(re.sub(r'_', ' ', token))
        return new_tokens

    def remove_stop_words(self,tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            if token not in self.stopwords_tokens and len(token) > 1:
                new_tokens.append(token)
        return new_tokens

    def remove_apostrophe(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(str(np.char.replace(token, "'", " ")))
        return new_tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(self.stemmer.stem(token))
        return new_tokens
    
    
    def normalize_appreviations(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        resolved_terms = {}
        for token in tokens:

            if len(token) >= 2:
                synsets = wordnet.synsets(token)
                if synsets:
                    resolved_term = synsets[0].lemmas()[0].name()
                    resolved_terms[token] = resolved_term

        for abbreviation, resolved_term in resolved_terms.items():
            for i in range(len(tokens)):
                if tokens[i] == abbreviation:
                    tokens[i] = resolved_term
                    break

        return tokens
    
    def lemmatizing(self, tokens: List[str]) -> List[str]:
        tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, pos) for token, pos in tagged_tokens]
        return lemmatized_tokens


    def preprocess(self, text: str) -> str:
        operations = [
            self.to_lower,
            self.remove_punctuation,
            self.remove_apostrophe,
            self.remove_stop_words,
            self.remove_markers,
            self.stemming,
            self.lemmatizing,
            self.normalize_appreviations, 
            self.to_lower,
            self.rplace_under_score_with_space
        ]
        text_tokens=self.tokenize(text)
        for op in operations:
              text_tokens=op(text_tokens)
    
        new_text=""
        new_text = ' '.join(text_tokens)
            
        return new_text