from flask import Flask, request, render_template,jsonify
import requests
import ir_datasets
from fast_autocomplete import AutoComplete
app = Flask(__name__)
words = {}
main_dataset = ir_datasets.load("antique/train")
for query in main_dataset.queries_iter():
    words[query.text] = {}

main_dataset = ir_datasets.load("wikir/en1k/training")
for query in main_dataset.queries_iter():
    words[query.text] = {}
@app.route("/", methods=["GET", "POST"])
def Serve():
    if request.method == "GET":
        return render_template("temp.html")
    elif request.method == "POST":
        response = requests.post("http://localhost:5003/query", json={"input": request.form.get("input"), "dataset": request.form.get("dataset")})
        results = response.json().get("results")
        return render_template("temp.html", results=results, query=request.form.get("input"), count=len(results))

@app.route('/auto-complete')
def autocomplete():
    autocomplete = AutoComplete(words=words)
    search = request.args.get("q")
    results = autocomplete.search(word=search, max_cost=5, size=5)
    final = [" ".join(result) for result in results]
    response = jsonify(auto_complete=final)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run(port=5000)
