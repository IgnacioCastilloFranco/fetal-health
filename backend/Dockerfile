# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

# Configuración del entorno Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar backend requirements e instalar dependencias Python
COPY backend/requirements.txt .
# Instalar primero las dependencias más pesadas con timeout extendido
RUN pip install --no-cache-dir --default-timeout=1000 \
    numpy==2.2.3 \
    pandas==2.2.3 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    imbalanced-learn==0.12.4 \
    xgboost==2.0.3
# Luego instalar el resto de dependencias
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Crear usuario no-root para seguridad
RUN useradd --create-home appuser
# Asegurar permisos completos para appuser
RUN chown -R appuser:appuser /app && chmod -R 755 /app

# Crear directorios para volúmenes con permisos apropiados
RUN mkdir -p /app/models /app/data /app/reports && \
    chown -R appuser:appuser /app/models /app/data /app/reports && \
    chmod -R 777 /app/models /app/data /app/reports

# Copiar el código del proyecto
COPY --chown=appuser:appuser . .

# Cambiar al usuario no-root
USER appuser

# Exponer el puerto 8000
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI
#CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
