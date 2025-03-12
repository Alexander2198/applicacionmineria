from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import json
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

app = Flask(_name_)

# Cargar el modelo entrenado y los encoders guardados
modelo_rf = joblib.load('/app/models/Nmodelo_xgboost_optimizado99.pkl')

# Cargar los encoders para variables categóricas
with open('/app/encoders/Nencoders_xgboost99.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Cargar el scaler para la variable objetivo (Precio)
with open('/app/scalers/Nscaler_y_xgboost99.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Cargar el scaler para las variables numéricas (Año, Kilometraje)
with open('/app/scalers/Nscaler_X_xgboost99.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

# Definir las columnas categóricas y numéricas
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Tracción', 'Combustible']
numerical_cols = ['Año', 'Kilometraje']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Motor', 'Tracción', 'Combustible']

# Preparar las opciones para los combo boxes a partir de los encoders
dropdown_options = {col: list(encoders[col].classes_) for col in categorical_cols}

# Cargar datos para el mapeo de Marca y Modelo
df_data = pd.read_csv('/app/data/data_filtrado.csv')

marca_modelo_map = defaultdict(list)
for index, row in df_data[['Marca', 'Modelo']].drop_duplicates().iterrows():
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
            if df_input.loc[0, col] not in encoders[col].classes_:
                df_input.loc[0, col] = encoders[col].classes_[0]  # Asigna un valor conocido si es desconocido
            df_input[col] = encoders[col].transform(df_input[col].astype(str))

        # Normalizar las variables numéricas antes de la predicción
        df_input[numerical_cols] = scaler_X.transform(df_input[numerical_cols])

        # Reordenar las columnas para coincidir con las del modelo
        df_input = df_input[features]

        # Realizar la predicción (en escala normalizada)
        precio_predicho_normalizado = modelo_rf.predict(df_input)[0]

        # Desnormalizar la predicción para obtener el valor real en la escala original
        precio_predicho = scaler_y.inverse_transform([[precio_predicho_normalizado]])[0][0]

        # Formatear la predicción para la interfaz
        prediction = round(precio_predicho, 2)


    return render_template("index.html",
                           dropdown_options=dropdown_options,
                           prediction=prediction,
                           form_data=form_data,
                           marca_modelo_json=marca_modelo_json)

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)