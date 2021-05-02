from flask import Flask,render_template,url_for,request
import numpy as np
#import pickle
import pickle5 as pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from numpy import array


app=Flask(__name__)
model = load_model("lstmModel.h5")


# with open(path_to_protocol5, "rb") as fh:
#   data = pickle.load(fh)
with open('tokenizer2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    max_length = 200
    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        #tokenizer.fit_on_texts(data)
        enc = tokenizer.texts_to_sequences(data)
        enc = pad_sequences(enc, maxlen=max_length, dtype='int32', value=0)
        my_prediction = model.predict(array([enc][0]))[0]
        #class1 = model.predict_classes(array([enc][0]))[0]
        sentiment = model.predict(enc)[0]
        print(my_prediction)
        print(review)
        print(data)
        #neg = np.argmax(sentiment)
        print(sentiment)
        if (np.argmax(sentiment) == 0):
            sentimennya = 0
            # neg = sentiment
            # sentiment = neg
            print('Sentimen: Negatif')
        elif (np.argmax(sentiment) == 1):
            sentimennya = 1
            # net = sentiment
            # sentiment = net
            print('Sentimen: Netral')
        else:
            sentimennya = 2
            # pos = sentiment
            # sentiment = pos
            print('Sentimen: Positif')
       
    return render_template('result.html',prediction = sentimennya)


# def muncul():
#     max_length = 200
#     if request.method == 'POST':
#         inputan = request.form['review']
#         #data = [review]
#         return render_template('result.html', teks=inputan)



if __name__ == '__main__':
    app.run(debug=True)
    