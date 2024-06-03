# information_retrieval
create first search engine using tfidf and word embedding

# the data set i use it :
Antique/train
Antique/test
Wikir/en1k/training

# folders:
-antique_test: i train model on this data set use tfidf vecorizor and word embedding on Word2Vec from Gensim library and TfidfVectorizer from Sklearn library
-antiqueEmbedding: i train the model useing word embedding on Word2Vec from Gensim library
-classes:have four python  fils
                -textProcessing.py text processing on document and query
                -tfidf1.py : train and save and load model and index for the wiki data set use tfidf 
                -tfidf2.py :train and save and load model and index for the antique data set use ifidf
                -QueryServer.py :this file is server do on port 5003 to receive the query and process it and send the document appropriate for it 
-classes_embedding:have four python  fils
                -textProcessing.py text processing on document and query
                -tfidf1.py : train and save and load model  for the wiki data set use word embedding 
                -tfidf2.py :train and save and load model for the antique data set use word embedding
                -QueryServer.py :this file is server do on port 5013 to receive the query and process it and send the document appropriate for it 
Files: i use this folder to save the models and index for the tfidf  in it
Files_embedding: i use this folder to save the models for the word embedding  in it
static:this folder have css fils and js file for the ui 
templates: this folder have html files for the ui
wiki&antique_without_wordembdding:this folder have a two file ipynb to find the evaluation for this search engine using tfidf vectorizor 
wikiEmbedding:i train the model useing word embedding on Word2Vec from Gensim library

# files:
start_server.sh :for start the servers that do on porst 5000 5003 5010 5013 
kill.sh :for stop the servers that do on porst 5000 5003 5010 5013 
main_server.py:this  file do on port 5000  to deal with ui to receive the query from the ui and send the result to the ui useing matching with  tfidf model 
main_server_embedding.py:this  file do on port 5010  to deal with ui to receive the query from the ui and send the result to the ui useing matching with  word embedding model 
