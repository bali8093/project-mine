from flask import Flask,request,jsonify
import pickle
import numpy as np


model =pickle.load(open('RandomForest.pkl','rb'))
app =Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/crop',methods=['POST'])
def crop():
    n = request.form.get('n')
    p = request.form.get('p')
    k = request.form.get('k')
    temp = request.form.get('temp')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rain_fall = request.form.get('rain_fall')

    input_query = np.array([[n,p,k,temp,humidity,ph,rain_fall]])
    result = model.predict(input_query)[0]


    return jsonify(str(result))

if __name__=='__main__':
    app.run(debug=True)