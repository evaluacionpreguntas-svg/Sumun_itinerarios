
# Paso 1: Usar una imagen base oficial de Python.
# 'slim' es una versión ligera que hace la imagen final más pequeña.
FROM python:3.9-slim

# Paso 2: Establecer el directorio de trabajo dentro del contenedor.
# A partir de aquí, todos los comandos se ejecutan dentro de la carpeta /app.
WORKDIR /app

# Paso 3: Copiar el archivo de requerimientos y luego instalar las librerías.
# Se hace en dos pasos para aprovechar el caché de Docker. Si no cambias tus
# requerimientos, esta capa se reutilizará, haciendo las construcciones futuras más rápidas.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Paso 4: Copiar el resto del código de tu aplicación (app.py) al contenedor.
COPY . .

# Paso 5: Exponer el puerto en el que Cloud Run ejecutará el contenedor.
# Aunque Cloud Run usa la variable $PORT, es una buena práctica documentarlo.
EXPOSE 8080

# Paso 6: El comando para iniciar la aplicación en formato "shell".
# Este formato permite que la variable $PORT sea reemplazada por su valor (ej. 8080)
# antes de que el comando se ejecute, solucionando el error.
ENTRYPOINT streamlit run app.py --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false
