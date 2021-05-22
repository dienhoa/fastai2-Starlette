FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc libsndfile1\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY app app/

EXPOSE 5000

CMD ["python", "app/server.py"]
