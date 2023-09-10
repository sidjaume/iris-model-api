from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import pickle as pkl
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime



app = Flask(__name__)

# Crea una conexión a la base de datos
engine = create_engine('postgresql://postgres:admin123@iris-model.cbne5hhkognj.eu-central-1.rds.amazonaws.com/postgres?')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        def error():
            return "DATA ERROR"
        
        feat1 = request.form.get('feat1', None)
        feat2 = request.form.get('feat2', None)
        feat3 = request.form.get('feat3', None)
        feat4 = request.form.get('feat4', None)
        
        
        if feat1 is None or feat2 is None or feat3 is None or feat4 is None:
            return error()

        try:
            feat1 = float(feat1)
            feat2 = float(feat2)
            feat3 = float(feat3)
            feat4 = float(feat4)
            
        except ValueError:
            return error()
        
        if feat1 > 6.0:
            feat1_ = 1.
        else:
            feat1_ = 0.

        raw = np.array([[feat1_, feat2, feat3, feat4]])

        with open('./models/scaler.pkl', 'rb') as archivo_entrada:
            scaler = pkl.load(archivo_entrada)
        numeros = scaler.transform(raw)

        with open('./models/iris_model.pkl', 'rb') as archivo_entrada:
            modelo_importado = pkl.load(archivo_entrada)
        data = load_iris()
        prediction = data.target_names[modelo_importado.predict(numeros)[0]]

        time = str(datetime.now())

        cols = {
            'sepal length':feat1,
            'sepal width' : feat2,
            'petal length':feat3,
            'petal width' : feat4,
            'type' : prediction,
            'time' : time
            }
        
        df = pd.DataFrame(cols, index=[int(datetime.now().timestamp())])
        df.to_sql(name="predictions",if_exists='append',con=engine)
        
        return render_template('index.html',
                        tuprima=str(prediction),
                        inp=str([feat1, feat2, feat3, feat4])
        )
    return render_template("index.html")
    
@app.route('/v0/get_logs', methods=["GET"])
def get_logs():
    return jsonify(pd.read_sql_query("select * from predictions", con = engine).to_dict("records"))

@app.route('/v0/del_logs', methods=["DELETE"])
def del_logs():    

    conn = psycopg2.connect('postgresql://postgres:admin123@iris-model.cbne5hhkognj.eu-central-1.rds.amazonaws.com/postgres?')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return 'Table deleted succesfully!'

@app.route("/v0/table", methods=["GET", "POST"])
def get_table():

    if request.method == 'POST':

        del_logs()
        return render_template('simple.html')
    import requests
    url = "https://iris-model-api-62w3-dev.fl0.io/v0/get_logs"

    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload).json()
    df = pd.DataFrame.from_dict(response)
    titles = df.columns

    return render_template('simple.html', titles = titles, tables=[df.to_html(classes='data', header="true", index = False, justify='center', border = 5)])


if __name__ == '__main__':
    app.run(debug=True)