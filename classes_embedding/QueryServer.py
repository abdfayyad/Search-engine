from model_wiki import WordEmbeddingEngineWiki
from model_antique import WordEmbeddingEngineAntique
from textProcessing import TextPreprocessor, LemmatizerWithPOSTagger
from flask import Flask, request, jsonify
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

# Initialize TextPreprocessor
text_preprocessor = TextPreprocessor()

# Initialize TfidfEngine with the TextPreprocessor
word_embedding_engine_wiki = WordEmbeddingEngineWiki(
    vector_size=500, sg=1, workers=4, epochs=35,
    text_processor=text_preprocessor,
    text_tokenizer=tokenize.word_tokenize
)
word_embedding_engine_antique = WordEmbeddingEngineAntique(
    vector_size=500, sg=1, workers=4, epochs=35,
    text_processor=text_preprocessor,
    text_tokenizer=tokenize.word_tokenize
)

# Load the trained models
word_embedding_engine_wiki.load_model()
word_embedding_engine_antique.load_model()

def process_query(query, dataset):
    results = []
    if dataset == "1":
        unordered_results = word_embedding_engine_wiki.get_results(query)
        for result in unordered_results:
            doc_id = result['_id']
            doc_text = result['text']
            score = float(cosine_similarity(
                [word_embedding_engine_wiki.get_query_vector(query)], 
                [word_embedding_engine_wiki.documents_vectors[list(word_embedding_engine_wiki.document_id_mapping.keys()).index(doc_id)]]
            ).flatten()[0])
            results.append({
                "id": doc_id,
                "text": doc_text,
                "score": score
            })
    else:
        unordered_results = word_embedding_engine_antique.get_results(query)
        for result in unordered_results:
            doc_id = result['_id']
            doc_text = result['text']
            score = float(cosine_similarity(
                [word_embedding_engine_antique.get_query_vector(query)], 
                [word_embedding_engine_antique.documents_vectors[list(word_embedding_engine_antique.document_id_mapping.keys()).index(doc_id)]]
            ).flatten()[0])
            results.append({
                "id": doc_id,
                "text": doc_text,
                "score": score
            })
    return results

  

@app.route("/query", methods=["POST"])
def query():
    query_input = request.json.get('input')
    dataset = request.json.get('dataset')
    results = process_query(query_input, dataset)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5013)
