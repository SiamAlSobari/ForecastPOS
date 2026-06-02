# Gunakan base image Python yang stabil dan ringan
FROM python:3.11-slim

# Atur environment variables agar Python tidak menulis file .pyc dan output langsung dicetak ke log
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tentukan working directory di dalam container
WORKDIR /app

# Instal dependensi sistem yang diperlukan (misal: curl untuk healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Instal dependensi Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Salin file dan folder yang diperlukan saja ke dalam container
COPY app /app/app
COPY main.py /app/

# Port yang diekspos oleh container
EXPOSE 8080

# Jalankan aplikasi menggunakan main.py
CMD ["python", "main.py"]
