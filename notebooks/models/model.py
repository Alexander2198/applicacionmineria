import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle

# 1. Leer el dataset final
df = pd.read_csv('../data/data_FINAL2.csv')

# Convertir la variable objetivo "Precio" a numérico y eliminar registros sin precio
df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')
df = df.dropna(subset=['Precio'])

# 2. Seleccionar la variable objetivo y las variables predictoras
y = df['Precio']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 
            'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']
X = df[features]

# 3. Aplicar Label Encoding a las columnas categóricas y guardar los encoders
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
X_encoded = X.copy()
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    encoders[col] = le  # Guardamos el encoder para la columna

# Guardar los encoders en un archivo para usarlos en la predicción
with open('../encoders/encoders_xgboost.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# 4. Dividir el dataset en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=89)

# 5. Definir el modelo XGBoost con los mejores parámetros encontrados
best_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=82,
    colsample_bytree=0.7,
    gamma=0,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=700,
    subsample=0.8
)

# Entrenar el modelo
best_xgb.fit(X_train, y_train)

# 6. Evaluar el modelo en el conjunto de prueba
y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========\n")
print("Evaluación del modelo XGBoost:")
print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
print("\n========\n")

# 7. Guardar el modelo entrenado en un archivo
joblib.dump(best_xgb, '../models/modelo_xgboost.pkl')
print("✅ El modelo se ha guardado en '../models/modelo_xgboost.pkl'")
