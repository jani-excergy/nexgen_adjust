from flask import Flask,render_template,request,jsonify
import datetime
import pickle
import numpy as np
import pandas as pd


app= Flask(__name__)

@app.route('/',methods=['GET','POST'])


def prediction():
    if request.method=="POST":
        date = datetime.datetime.strptime( request.form['day'],'%Y-%m-%d')
        df=pd.DataFrame([date],columns=['Created Date'])
        df['year'] = df['Created Date'].dt.year
        df['month'] = df['Created Date'].dt.month
        df['day'] = df['Created Date'].dt.month
        df['weekday'] = df['Created Date'].dt.dayofweek
        df['quarter'] = df['Created Date'].dt.quarter

        # cycling encoding
        def encode(data, col, max_val):
            data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
            data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
            return data

        encode(df, 'month', 12)
        encode(df, 'day', 31)
        encode(df, 'weekday', 7)
        encode(df, 'quarter', 4)

        df = df.drop(['month', 'day', 'weekday', 'quarter'], axis=1)

        df['year'] = df['year'].map({2018: 0, 2019: 1, 2020: 2, 2021: 3})
        df = df.drop(['Created Date','month_sin', 'day_cos', 'weekday_sin','quarter_sin', ], axis=1)


        leads = request.form.get("le")
        tech = request.form.get("te")
        ticket = request.form.get("ti")
        month_cos=df['month_cos']
        weekday_cos=df['weekday_cos']
        quarter_cos=df['quarter_cos']

        #req_json = request.json
        #leads = req_json['leads']
        #tech = req_json['tech']
        #ticket = req_json['ticket']
        #month_cos=req_json['month_cos']
        #weekday_cos=req_json['weekday_cos']
        #quarter_cos=req_json['quarter_cos']




        loadmodel_1 = pickle.load(open('adjust_revenue.pkl', 'rb'))
        pred_result= loadmodel_1.predict([leads,tech,ticket,month_cos,weekday_cos,quarter_cos])
        pred_result=str(pred_result)
        return (pred_result)

    return render_template('index_2.html')

if __name__=="__main__":
    app.run(debug=True)