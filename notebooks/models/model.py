import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle


df = pd.read_csv('../data/data_FINAL2.csv')


df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')
df = df.dropna(subset=['Precio'])


y = df['Precio']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 
            'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']
X = df[features]


categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
X_encoded = X.copy()
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    encoders[col] = le  # Guardamos el encoder para la columna

# Guardar los encoders en un archivo para usarlo en la predicción
with open('../encoders/encoders_xgboost2.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# 4. Dividir el dataset en conjuntos de ntrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=89)


best_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.8,  # Aumentamos el número de variables usadas en cada árbol
    gamma=0.1,  # Permitimos más divisiones
    learning_rate=0.025,  # Reducimos la tasa de aprendizaje para mejorar precisión
    max_depth=4,  # Árboles más profundos para capturar más patrones
    n_estimators=700,  # Más árboles para mejorar aprendizaje
    subsample=0.8,  # Usamos más datos por cada árbol
    alpha=0.5,  # Reducimos la regularización L1 para permitir más flexibilidad
    lambda_=1.0,  # Reducimos la regularización L2 para evitar que los pesos sean demasiado pequeños
    random_state=42,
    n_jobs=-1
)
# Entrena el modelo
best_xgb.fit(X_train, y_train)


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
print("✅ El modelo se ha guardado en '../models/modelo_xgboost_optimizado.pkl'")
