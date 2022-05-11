FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /src

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "./app/__main__.py"]
