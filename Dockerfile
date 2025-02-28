# Imagen base
FROM python:3.10

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY requirements.txt requirements.txt
COPY web_app/ web_app/
COPY models/ models/
COPY encoders/ encoders/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto para Flask
EXPOSE 5000

# Ejecutar la aplicación
CMD ["python", "web_app/app.py"]
