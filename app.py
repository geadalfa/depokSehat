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
import json
import plotly
#import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
#from charts.bar_chart import plot_chart
import plotly.graph_objs as go
import plotly.offline as pyo
from flask import Markup


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
    labels = []
    values = []
    df = pd.read_csv('hasil_label.csv', index_col=0)
    positif1 = df.loc[df['nilai'] > 15].head()
    negatif1 = df.loc[df['nilai'] < -25].head()
    netral1 = df.loc[df['nilai'] == -1].head()
    headings = ("Tweet", "Nilai", "Sentimen")
    tuples1 = [tuple(x) for x in positif1.values]
    tuples2 = [tuple(x) for x in negatif1.values]
    tuples3 = [tuple(x) for x in netral1.values]
    #kolom2 = df[['cleaned_tweets', 'sentimen']]
    senti_count = df['sentimen'].value_counts()
    senti_count2=list(zip(senti_count,senti_count.index))
    senti_count2=tuple(zip(senti_count,senti_count.index))
    kolom2 = [(sub[1], sub[0]) for sub in senti_count2]
    for row in kolom2:
        labels.append(row[0])
        values.append(row[1])
    senti_count2 = [tuple(str(x) for x in tup) for tup in senti_count2]
    senti_count2 = [(sub[1], sub[0]) for sub in senti_count2]
    return render_template('home.html', sentimen=senti_count, tabel=df, headings = headings, labels=labels, values=values, 
                            positif=tuples1, negatif=tuples2, netral=tuples3, sentimen2=senti_count2, set=zip(values, labels))


def diagram():
    df = pd.read_csv("C:/Users/Alfa/Program Skripsi/countries of the world.csv")
    trace1 = go.Bar(x=df["Country"][0:20], y=df["GDP ($ per capita)"])
    layout = go.Layout(title="GDP of the Country", xaxis=dict(title="Country"),
                       yaxis=dict(title="GDP Per Capita"), )
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    chart_div_string = pyo.offline.plot(fig, include_plotlyjs=False, output_type='div')
    chart_div_for_use_in_jinja_template = Markup(chart_div_string)
    #fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('home.html', chart=chart_div_for_use_in_jinja_template)
    # count = 500
    # xScale = np.linspace(0, 100, count)
    # yScale = np.random.randn(count)
 
    # # Create a trace
    # trace = go.Scatter(
    #     x = xScale,
    #     y = yScale
    # )
 
    # data = [trace]
    # graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    # return render_template('index1.html',
    #                            graphJSON=graphJSON)

    # labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    # values = [4500, 2500, 1053, 500]
    # # Use `hole` to create a donut-like pie chart
    # fig = go.Pie(labels=labels, values=values, hole=.3)
    # fig = [fig]
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # return render_template('home.html', graphJSON=graphJSON)

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
    kolom2 = df[['cleaned_tweets', 'sentimen']]
    senti_count = df['sentimen'].value_counts()
    senti_count2=list(zip(senti_count,senti_count.index))
    senti_count2=tuple(zip(senti_count,senti_count.index))
    senti_count2 = [tuple(str(x) for x in tup) for tup in senti_count2]
    senti_count2 = [(sub[1], sub[0]) for sub in senti_count2]
    return render_template('home.html', sentimen=senti_count, tabel=df, headings = headings, 
                            positif=tuples1, negatif=tuples2, netral=tuples3, sentimen2=senti_count2)
    #return render_template('home.html')

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

# def muncul():
#     if request.method == 'POST':
#         inputan = request.form['review']
#         #data = [review]
#         return render_template('result.html', teks=inputan)



if __name__ == '__main__':
    app.run(debug=True)
    