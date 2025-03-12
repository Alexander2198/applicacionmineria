from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler  # Importar scaler

app = Flask(__name__)

# Cargar el modelo, encoders y scaler
modelo_rf = joblib.load('/app/models/Nmodelo_xgboost_optimizado99.pkl')

with open('/app/encoders/Nencoders_xgboost99.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('/app/scalers/Nscaler_y_xgboost99.pkl', 'rb') as f:
    scaler_y = pickle.load(f)  # Cargar el scaler para desescalar la predicción

# Definir columnas categóricas y características del modelo
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Tracción', 'Combustible']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Motor', 'Tracción', 'Combustible']

# Preparar las opciones para los combo boxes a partir de los encoders (las clases conocidas)
dropdown_options = {}
for col in categorical_cols:
    dropdown_options[col] = list(encoders[col].classes_)

# Cargar los datos para obtener las marcas y modelos
df_data = pd.read_csv('/app/data/data_filtrado.csv')

# Crear un mapeo Marca -> Modelos
marca_modelo_map = defaultdict(list)
for _, row in df_data[['Marca', 'Modelo']].drop_duplicates().iterrows():
    marca = row['Marca']
    modelo = row['Modelo']
    marca_modelo_map.setdefault(marca, []).append(modelo)

# Convertir el mapeo a JSON para enviarlo al frontend
marca_modelo_json = json.dumps(dict(marca_modelo_map))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}

    if request.method == "POST":
        # Recoger los datos del formulario
        data = {
            'Marca': request.form.get('Marca'),
            'Modelo': request.form.get('Modelo'),
            'Provincia': request.form.get('Provincia'),
            'Año': request.form.get('Año'),
            'Kilometraje': request.form.get('Kilometraje'),
            'Transmisión': request.form.get('Transmisión'),
            'Motor': request.form.get('Motor'),
            'Tracción': request.form.get('Tracción'),
            'Combustible': request.form.get('Combustible')
        }
        form_data = data

        # Crear un DataFrame con los datos ingresados
        df_input = pd.DataFrame([data], columns=features)

        # Convertir campos numéricos
        df_input['Año'] = pd.to_numeric(df_input['Año'], errors='coerce')
        df_input['Kilometraje'] = pd.to_numeric(df_input['Kilometraje'], errors='coerce')
        df_input['Motor'] = pd.to_numeric(df_input['Motor'], errors='coerce')

        # Aplicar los encoders a las columnas categóricas
        for col in categorical_cols:
            # Si el valor ingresado no está en las clases conocidas, se asigna el primer valor
            if df_input.loc[0, col] not in encoders[col].classes_:
                df_input.loc[0, col] = encoders[col].classes_[0]
            df_input[col] = encoders[col].transform(df_input[col].astype(str))

        # Asegurar que las columnas están en el orden correcto
        df_input = df_input[features]

        # Hacer la predicción (la salida estará escalada)
        prediction_scaled = modelo_rf.predict(df_input)

        # Desescalar la predicción para obtener el valor real
        prediction = scaler_y.inverse_transform(np.array(prediction_scaled).reshape(-1, 1))[0][0]

        print("Predicción escalada:", prediction_scaled)
        print("Predicción real:", prediction)

    return render_template("index.html",
                           dropdown_options=dropdown_options,
                           prediction=prediction,
                           form_data=form_data,
                           marca_modelo_json=marca_modelo_json)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
