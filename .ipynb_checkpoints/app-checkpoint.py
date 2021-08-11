from flask import Flask,render_template,url_for,request, redirect, Response
import numpy as np
#import pickle
import pickle5 as pickle
import pandas as pd
#import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import model_from_json
from numpy import array


app=Flask(__name__)
#model = tf.create_model()
model = load_model("lstmModel.h5")
model.load_weights("geaNlp_weight_model.h5")


# with open(path_to_protocol5, "rb") as fh:
#   data = pickle.load(fh)
with open('tokenizer2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def table():
    df = pd.read_csv('hasil_label.csv', index_col=0)
    positif1 = df.loc[df['nilai'] > 15].head()
    negatif1 = df.loc[df['nilai'] < -25].head()
    netral1 = df.loc[df['nilai'] == -1].head()
    headings = ("Tweet", "Nilai", "Sentimen")
    tuples1 = [tuple(x) for x in positif1.values]
    tuples2 = [tuple(x) for x in negatif1.values]
    tuples3 = [tuple(x) for x in netral1.values]
    senti_count = df['sentimen'].value_counts()
    senti_count2=list(zip(senti_count,senti_count.index))
    senti_count2=tuple(zip(senti_count,senti_count.index))
    senti_count2 = [tuple(str(x) for x in tup) for tup in senti_count2]
    senti_count2 = [(sub[1], sub[0]) for sub in senti_count2]
    return render_template('home.html', sentimen=senti_count, tabel=df, headings = headings, 
                            positif=tuples1, negatif=tuples2, netral=tuples3, sentimen2=senti_count2)


def default():
    return redirect('/home.html')

@app.route('/home.html')
def home():
    df = pd.read_csv('hasil_label.csv', index_col=0)
    positif1 = df.loc[df['nilai'] > 15].head()
    negatif1 = df.loc[df['nilai'] < -25].head()
    netral1 = df.loc[df['nilai'] == -1].head()
    headings = ("Tweet", "Nilai", "Sentimen")
    tuples1 = [tuple(x) for x in positif1.values]
    tuples2 = [tuple(x) for x in negatif1.values]
    tuples3 = [tuple(x) for x in netral1.values]
    senti_count = df['sentimen'].value_counts()
    senti_count2=list(zip(senti_count,senti_count.index))
    senti_count2=tuple(zip(senti_count,senti_count.index))
    senti_count2 = [tuple(str(x) for x in tup) for tup in senti_count2]
    senti_count2 = [(sub[1], sub[0]) for sub in senti_count2]
    return render_template('home.html', sentimen=senti_count, tabel=df, headings = headings, 
                            positif=tuples1, negatif=tuples2, netral=tuples3, sentimen2=senti_count2)

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
        #print(my_prediction)
        #print(review)
        #print(data)
        #neg = np.argmax(sentiment)
        print(sentiment)
        if (np.argmax(sentiment) == 0):
            sentimennya = 0
            # neg = sentiment
            # sentiment = neg
            #print('Sentimen: Negatif')
        elif (np.argmax(sentiment) == 1):
            sentimennya = 1
            # net = sentiment
            # sentiment = net
            #print('Sentimen: Netral')
        else:
            sentimennya = 2
            # pos = sentiment
            # sentiment = pos
            #print('Sentimen: Positif')
       
    return render_template('result.html',prediction = sentimennya, teks=review)

@app.route('/style.css',methods=['GET'])
def stylecss():
    read_file = open("static/style.css", "r")
    opens = read_file.read()
    return Response(opens, mimetype='text/css')


if __name__ == '__main__':
    app.run(debug=True)
    