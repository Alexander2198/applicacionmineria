# Imagen base de Python 3.10
FROM python:3.10

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar dependencias y paquetes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación
COPY web_app/ web_app/
COPY notebooks/models/ models/
COPY notebooks/encoders/ encoders/
COPY notebooks/data/ data/


# Exponer el puerto 5000 para Flask
EXPOSE 5000

# Ejecutar Flask con Python
CMD ["python", "web_app/app.py"]
