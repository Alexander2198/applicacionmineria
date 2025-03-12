import pandas as pd
import xgboost as xgb
import pickle

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar el dataset
df = pd.read_csv('../data/data_filtrado.csv')

# 2. Convertir la variable objetivo "Precio" a numérico y eliminar registros sin precio
df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')
df = df.dropna(subset=['Precio'])

# 3. Seleccionar las variables predictoras y la variable objetivo
y = df['Precio']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Motor', 'Tracción', 'Combustible']
X = df[features]

# 4. Aplicar Label Encoding a las columnas categóricas y guardar los encoders
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Tracción', 'Combustible']
X_encoded = X.copy()
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    encoders[col] = le  # Guardamos el encoder

# Guardar los encoders en un archivo
with open('../encoders/Nencoders_xgboost99.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# 5. Normalizar la variable objetivo (Precio)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Guardar el escalador de Precio
with open('../scalers/Nscaler_y_xgboost99.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# 6. Dividir el dataset en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.10, random_state=42)

# 7. Definir modelo base
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# 8. Definir la rejilla de hiperparámetros
param_grid = {
    'n_estimators': [2500],
    'max_depth': [7],
    'learning_rate': [0.1],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'gamma': [0],
    'objetive':['reg:squarederror'],
    'alpha':[2],
    
}

# 9. Optimización de hiperparámetros con GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_regressor,
    param_grid=param_grid,
    scoring='r2',
    cv=3,  # Validación cruzada con 3 particiones
    verbose=2,
    n_jobs=-1  # Usa todos los procesadores disponibles
)

grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print("\n✅ Mejores hiperparámetros encontrados:", best_params)

# 10. Entrenar modelo con los mejores hiperparámetros
best_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    **best_params,
    random_state=42,
    n_jobs=-1
)

best_xgb.fit(X_train, y_train)

# 11. Evaluación del modelo (sin desnormalizar)
y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)

# 12. Calcular métricas de desempeño en escala normalizada
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# 13. Calcular la varianza entre entrenamiento y prueba
varianza_rmse = rmse_train - rmse_test
varianza_mae = mae_train - mae_test
varianza_r2 = r2_train - r2_test

# 14. Mostrar resultados en escala normalizada
print("\n======== RESULTADOS (Normalizados) ========\n")
print("🔹 *Entrenamiento:*")
print(f" - RMSE: {rmse_train:.3f}")
print(f" - MAE: {mae_train:.3f}")
print(f" - R²: {r2_train:.4f}")

print("\n🔹 *Prueba:*")
print(f" - RMSE: {rmse_test:.3f}")
print(f" - MAE: {mae_test:.3f}")
print(f" - R²: {r2_test:.4f}")

print("\n🔹 *Varianza (Diferencia entre entrenamiento y prueba):*")
print(f" - RMSE Varianza: {varianza_rmse:.3f}")
print(f" - MAE Varianza: {varianza_mae:.3f}")
print(f" - R² Varianza: {varianza_r2:.4f}")

# 15. Detectar sobreajuste
if abs(varianza_rmse) > 0.1 or abs(varianza_r2) > 0.05:
    print("\n⚠️ *Posible sobreajuste detectado* ⚠️")
else:
    print("\n✅ *El modelo generaliza bien*")

# 16. Guardar el modelo entrenado
joblib.dump(best_xgb, '../models/Nmodelo_xgboost_optimizado99.pkl')
print("\n✅ El modelo optimizado se ha guardado en 'models/Nmodelo_xgboost_optimizado99.pkl'")