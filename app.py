import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
app = Flask(__name__)

lem = WordNetLemmatizer()
tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        msg = request.form['message']
        review = re.sub('[^a-zA-Z]',' ',msg)
        review = review.lower()
        review = review.split()
        review = [lem.lemmatize(words) for words in review if not words in stopwords.words('english')]
        review = [' '.join(review)]
        review = tfidf.transform(review).toarray()
        pred = model.predict(review)
    return render_template('predict.html',msg=pred)
    
if __name__ == "__main__":
    app.run(debug=True)