from flask import Flask,render_template,url_for,request, redirect
import numpy as np
#import pickle
import pickle5 as pickle
#import pandas as pd
#import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import model_from_json
import random
from numpy import array
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,
                          Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


#<link rel="stylesheet" href="{{url_for('static', filename='style.css')}}">
app=Flask(__name__)
#model = tf.create_model()
model = load_model("lstmModel.h5")
model.load_weights("geaNlp_weight_model.h5")


# with open(path_to_protocol5, "rb") as fh:
#   data = pickle.load(fh)
with open('tokenizer2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def chart():
    fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
    counts = [5, 3, 4, 2, 4, 6]

    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

    p = figure(x_range=fruits, plot_height=250, toolbar_location=None, title="Fruit counts")
    p.vbar(x='fruits', top='counts', width=0.9, source=source, legend_field="fruits",
           line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    return render_template("home.html", bars_count=p)

def default():
    return redirect('/home.html')

@app.route('/home.html')
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


# def muncul():
#     if request.method == 'POST':
#         inputan = request.form['review']
#         #data = [review]
#         return render_template('result.html', teks=inputan)



if __name__ == '__main__':
    app.run(debug=True)
    