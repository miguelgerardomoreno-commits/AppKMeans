from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

kmeans = load('modelo_kmeans.pkl')

def get_group_label(label):
    mapping = {0: 'Grupo 1', 1: 'Grupo 2'}
    return mapping.get(int(label), 'Desconocido')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        ingresos = float(request.form.get('ingresos', '0').replace(',', '.'))
        gastos = float(request.form.get('gastos', '0').replace(',', '.'))
    except ValueError:
        return render_template('index.html', error='Valores inválidos')

    X_new = np.array([[ingresos, gastos]])
    label = kmeans.predict(X_new)[0]
    group = get_group_label(label)
    return render_template('index.html', ingresos=ingresos, gastos=gastos, group=group)




if __name__ == '__main__':
    app.run(debug=True)