FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
COPY fast_api.py /fast_api.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn fast_api:app --host 0.0.0.0 --port $PORT

# CMD ["streamlit", "run", "fast_api.py", "--server.port=8080", "--server.address=0.0.0.0"]

