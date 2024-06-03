from tfidf2 import TfidfEngine2
from tfidf1 import TfidfEngine1
from textProcessing import TextPreprocessor, LemmatizerWithPOSTagger
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize TextPreprocessor
text_preprocessor = TextPreprocessor()

# Initialize TfidfEngine with the TextPreprocessor
tfidf_engine1 = TfidfEngine1(text_preprocessor)
tfidf_engine2 = TfidfEngine2(text_preprocessor)

# Load the trained models
tfidf_engine1.load_model()
tfidf_engine2.load_model()

def process_query(query, dataset):
    results = []
    if dataset == "1":
        query_vector = tfidf_engine1.query(query)
        ranked_indices, similarities = tfidf_engine1.rank_documents(query_vector)
        for idx in ranked_indices[:20]:  # Show top 20 results
            doc_id = list(tfidf_engine1.document_id_mapping.keys())[idx]
            doc_text = tfidf_engine1.document_id_mapping[doc_id]
            results.append({
                "id": doc_id,
                "text": doc_text,
                "score": similarities[idx]
            })
    else:
        query_vector = tfidf_engine2.query(query)
        ranked_indices, similarities = tfidf_engine2.rank_documents(query_vector)
        for idx in ranked_indices[:20]:  # Show top 20 results
            doc_id = list(tfidf_engine2.document_id_mapping.keys())[idx]
            doc_text = tfidf_engine2.document_id_mapping[doc_id]
            results.append({
                "id": doc_id,
                "text": doc_text,
                "score": similarities[idx]
            })
    return results

@app.route("/query", methods=["POST"])
def query():
    query_input = request.json.get('input')
    dataset = request.json.get('dataset')
    results = process_query(query_input, dataset)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5003)
