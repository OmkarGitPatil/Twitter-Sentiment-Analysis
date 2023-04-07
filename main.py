from flask import Flask,request,render_template,url_for
import pickle as pkl

import numpy as np
import pandas as pd
import time

from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost

from sklearn.metrics import accuracy_score


app = Flask('__main__')

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\svm_model.pkl','rb') as file1:
    svm_model=pkl.load(file1)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\xgb_model.pkl','rb') as file2:
    xgb_model=pkl.load(file2)
    
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\rfc_model.pkl','rb') as file3:
    rfc_model=pkl.load(file3)
    
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\mnb_model.pkl','rb') as file4:
    mnb_model=pkl.load(file4)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\tfidf.pkl','rb') as file5:
    tfidf=pkl.load(file5)
    
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\cv.pkl','rb') as file6:
    cv=pkl.load(file6)



@app.route('/')
def render():
    return render_template('index.html')

@app.route('/input', methods=['GET','POST'])
def predict():
    
    input_dict = request.form
    tweet=list(input_dict.values())[0]
    inverted_comma="'"

    if  inverted_comma in tweet:
        modified_tweet=tweet.replace(inverted_comma,'')
    else:
        modified_tweet=tweet
      
        
    def tokenization():
      words=word_tokenize(modified_tweet)
      return words
    tokens=tokenization()


    def cleaning():
      clean_text=[i for i in tokens if i not in punctuation]
      return clean_text
    clean_text=cleaning()
    

    def normalize():
      normal_text=[i.lower() for i in clean_text]
      return normal_text
    normal_text=normalize()
    

    stop=stopwords.words('english')
    def stop_removal():
      stop_text=[i for i in normal_text if i not in stop]
      return stop_text
    stop_text=stop_removal()
    

    lemma=WordNetLemmatizer()
    def lemmatization():
      l1=[]
      for i in stop_text:
          word=lemma.lemmatize(i)
          l1.append(word)
      return l1
    l1=lemmatization()


    def string():
      strings=' '.join(l1)
      return strings
    final_tweet=string()

    matrix=tfidf.transform([final_tweet]).A
    prediction=svm_model.predict(matrix)

    if prediction == 0:
       tweet_nature = 'Irrelevent'
    if prediction == 1:
       tweet_nature = 'Negative'
    if prediction == 2:
       tweet_nature = 'Neutral'
    if prediction == 3:
       tweet_nature = 'Positive'
       
    
    return render_template('display.html',dict1_values = modified_tweet, output=tweet_nature)

if __name__ == '__main__':
    app.run(debug=True)