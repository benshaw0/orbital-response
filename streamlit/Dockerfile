FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY models models
COPY images_masks images_masks/
COPY ml_logic ml_logic
COPY app_ui_ben.py app_ui_ben.py

RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

CMD streamlit run app_ui_ben.py --port $PORT
