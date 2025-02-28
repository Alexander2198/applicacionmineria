from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import pandas as pd
import json
from collections import defaultdict
import os

app = Flask(__name__)

# Rutas absolutas dentro del contenedor
MODEL_PATH = "../notebooks/models/modelo_xgboost.pkl"
ENCODER_PATH = "../notebooks/ncoders/encoders_xgboost.pkl"
DATA_PATH = "../notebooks/data/archivo_unido_FINAL3.csv"

# Cargar el modelo entrenado y los encoders guardados
modelo_rf = joblib.load(MODEL_PATH)

with open(ENCODER_PATH, 'rb') as f:
    encoders = pickle.load(f)

# Definir columnas categóricas y el orden de las características
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']

# Preparar las opciones para los combo boxes
dropdown_options = {col: list(encoders[col].classes_) for col in categorical_cols}

# Generar un mapeo de Marca a Modelos
if os.path.exists(DATA_PATH):
    df_data = pd.read_csv(DATA_PATH)
    marca_modelo_map = defaultdict(list)
    for index, row in df_data[['Marca', 'Modelo']].drop_duplicates().iterrows():
        marca = row['Marca']
        modelo = row['Modelo']
        marca_modelo_map.setdefault(marca, []).append(modelo)
    marca_modelo_json = json.dumps(dict(marca_modelo_map))
else:
    marca_modelo_json = json.dumps({})

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}
    
    if request.method == "POST":
        try:
            # Recoger los datos del formulario
            data = {
                'Marca': request.form.get('Marca'),
                'Modelo': request.form.get('Modelo'),
                'Provincia': request.form.get('Provincia'),
                'Año': request.form.get('Año'),
                'Kilometraje': request.form.get('Kilometraje'),
                'Transmisión': request.form.get('Transmisión'),
                'Dirección': request.form.get('Dirección'),
                'Motor': request.form.get('Motor'),
                'Tracción': request.form.get('Tracción'),
                'Color': request.form.get('Color'),
                'Combustible': request.form.get('Combustible')
            }
            form_data = data

            # Crear un DataFrame con los datos ingresados
            df_input = pd.DataFrame([data], columns=features)
            df_input[['Año', 'Kilometraje', 'Motor']] = df_input[['Año', 'Kilometraje', 'Motor']].apply(pd.to_numeric, errors='coerce')

            # Aplicar los encoders a las columnas categóricas
            for col in categorical_cols:
                if df_input.loc[0, col] not in encoders[col].classes_:
                    df_input.loc[0, col] = encoders[col].classes_[0]  # Asigna un valor válido
                df_input[col] = encoders[col].transform(df_input[col].astype(str))

            # Hacer la predicción
            prediction = modelo_rf.predict(df_input)[0]
        except Exception as e:
            prediction = f"Error en la predicción: {str(e)}"
    
    return render_template("index.html",
                           dropdown_options=dropdown_options,
                           prediction=prediction,
                           form_data=form_data,
                           marca_modelo_json=marca_modelo_json)

@app.route("/predict", methods=["POST"])
def predict():
    """ API REST para hacer predicciones con JSON """
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data], columns=features)
        df_input[['Año', 'Kilometraje', 'Motor']] = df_input[['Año', 'Kilometraje', 'Motor']].apply(pd.to_numeric, errors='coerce')

        # Aplicar Label Encoding
        for col in categorical_cols:
            if df_input.loc[0, col] not in encoders[col].classes_:
                df_input.loc[0, col] = encoders[col].classes_[0]  # Asignar un valor válido
            df_input[col] = encoders[col].transform(df_input[col].astype(str))

        prediction = modelo_rf.predict(df_input)[0]
        return jsonify({"precio_estimado": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
