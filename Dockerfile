FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc libsndfile1\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

RUN mkdir -pv /var/lib/knap/product

COPY app app/

#RUN python app/server.py

EXPOSE 5000

CMD ["python", "app/server.py", "serve"]
